import os
import sys
import json
import math
import random
import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.axislayer import AxisLayerFK
from trimesh import Trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from DRO_Grasp.utils.func_utils import farthest_point_sampling
from DRO_Grasp.utils.mesh_utils import load_link_geometries
from DRO_Grasp.utils.rotation import *


class HandModel:
    def __init__(
        self,
        robot_name,
        urdf_path,
        meshes_path,
        links_pc_path,
        device,
        link_num_points=512
    ):
        self.robot_name = robot_name
        self.urdf_path = urdf_path
        self.meshes_path = meshes_path
        self.device = device

        if robot_name == 'mano':
            self.mano_layer = ManoLayer(rot_mode="axisang",
                                        center_idx=0,
                                        mano_assets_root="assets/mano",
                                        use_pca=False,
                                        flat_hand_mean=True)
            self.axisFK = AxisLayerFK(mano_assets_root="assets/mano")

        self.pk_chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        if os.path.exists(links_pc_path):  # In case of generating robot links pc, the file doesn't exist.
            links_pc_data = torch.load(links_pc_path, map_location=device)
            self.links_pc = links_pc_data['filtered']
            self.links_pc_original = links_pc_data['original']
        else:
            self.links_pc = None
            self.links_pc_original = None

        self.meshes = load_link_geometries(robot_name, self.urdf_path, self.pk_chain.get_link_names())

        self.vertices = {}
        removed_links = json.load(open(os.path.join(ROOT_DIR, 'data_utils/removed_links.json')))[robot_name]
        for link_name, link_mesh in self.meshes.items():
            if link_name in removed_links:  # remove links unrelated to contact
                continue
            v = link_mesh.sample(link_num_points)
            self.vertices[link_name] = v

        self.frame_status = None

    def get_joint_orders(self):
        return [joint.name for joint in self.pk_chain.get_joints()]

    def update_status(self, q):
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    # def get_transformed_links_pc(self, q=None, links_pc=None):
    #     """
    #     Use robot link pc & q value to get point cloud.

    #     :param q: (6 + DOF,), joint values (euler representation)
    #     :param links_pc: {link_name: (N_link, 3)}, robot links pc dict, not None only for get_sampled_pc()
    #     :return: point cloud: (N, 4), with link index
    #     """
    #     if q is None:
    #         q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
    #     self.update_status(q)
    #     if links_pc is None:
    #         links_pc = self.links_pc

    #     all_pc_se3 = []
    #     for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
    #         if not torch.is_tensor(link_pc):
    #             link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
    #         n_link = link_pc.shape[0]
    #         se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)
    #         homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
    #         link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
    #         link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
    #         index_tensor = torch.full([n_link, 1], float(link_index), device=q.device)
    #         link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=1)
    #         all_pc_se3.append(link_pc_se3_index)
    #     all_pc_se3 = torch.cat(all_pc_se3, dim=0)

    #     return all_pc_se3
    
    def get_transformed_links_pc(self, q=None, links_pc=None, only_palm=False):
        """
        Use robot link pc & q value to get point cloud.

        :param q: (6 + DOF,), joint values (euler representation)
        :param links_pc: {link_name: (N_link, 3)}, robot links pc dict, not None only for get_sampled_pc()
        :return: point cloud: (N, 4), with link index
        """
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        if links_pc is None:
            links_pc = self.links_pc

        all_pc_se3 = []
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            if only_palm and link_index != 0:
                continue
            n_link = link_pc.shape[0]
            se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)
            homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
            link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
            index_tensor = torch.full([n_link, 1], float(link_index), device=q.device)
            link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=1)
            all_pc_se3.append(link_pc_se3_index)
        all_pc_se3 = torch.cat(all_pc_se3, dim=0)

        return all_pc_se3
    
    def get_sampled_pc(self, q=None, num_points=512):
        """
        :param q: (9 + DOF,), joint values (rot6d representation)
        :param num_points: int, number of sampled points
        :return: ((N, 3), list), sampled point cloud (numpy) & index
        """
        if q is None:
            q = self.get_canonical_q()

        sampled_pc = self.get_transformed_links_pc(q, self.vertices)
        return farthest_point_sampling(sampled_pc, num_points)
    
    def get_sampled_pc_fast(self, q=None, num_points=512):
        """
        :param q: (9 + DOF,), joint values (rot6d representation)
        :param num_points: int, number of sampled points
        :return: ((N, 3), list), sampled point cloud (numpy) & index
        """
        if q is None:
            q = self.get_canonical_q()

        sampled_pc = self.get_transformed_links_pc(q, self.vertices)
        indices = torch.randperm(65536)[:num_points]
        sampled_pc = sampled_pc[indices]
        return sampled_pc

    def get_canonical_q(self):
        """ For visualization purposes only. """
        lower, upper = self.pk_chain.get_joint_limits()
        canonical_q = torch.tensor(lower) * 0.75 + torch.tensor(upper) * 0.25
        canonical_q[:6] = 0
        return canonical_q

    def get_initial_q(self, q=None, max_angle: float = math.pi / 6):
        """
        Compute the robot initial joint value q based on the target grasp.
        Root translation is not considered since the point cloud will be normalized to zero-mean.

        :param q: (6 + DOF,) or (9 + DOF,), joint values (euler/rot6d representation)
        :param max_angle: float, maximum angle of the random rotation
        :return: initial q: (6 + DOF,), euler representation
        """
        if q is None:  # random sample root rotation and joint values
            # print("Randomly sample initial q.")
            q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

            q_initial[3:6] = (torch.rand(3) * 2 - 1) * torch.pi
            q_initial[5] /= 2

            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[6:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
        else:
            if len(q) == self.dof:
                q = q_euler_to_q_rot6d(q)
            q_initial = q.clone()

            # compute random initial rotation
            direction = - q_initial[:3] / torch.norm(q_initial[:3])
            angle = torch.tensor(random.uniform(0, max_angle), device=q.device)  # sample rotation angle
            axis = torch.randn(3).to(q.device)  # sample rotation axis
            axis -= torch.dot(axis, direction) * direction  # ensure orthogonality
            axis = axis / torch.norm(axis)
            random_rotation = axisangle_to_matrix(axis, angle).to(q.device)
            rotation_matrix = random_rotation @ rot6d_to_matrix(q_initial[3:9])
            q_initial[3:9] = matrix_to_rot6d(rotation_matrix)

            # compute random initial joint values
            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[9:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
            # q_initial[9:] = torch.zeros_like(q_initial[9:], dtype=q.dtype, device=q.device)

            q_initial = q_rot6d_to_q_euler(q_initial)

        return q_initial
    
    def get_fixed_initial_q(self):
        """ For training purposes only. """
        q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

        q_initial[3:6] = torch.ones(3) * torch.pi / 30
        q_initial[5] /= 2

        lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
        lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
        upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
        portion = 0.85
        q_initial[6:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
        return q_initial   

    def get_trimesh_q(self, q):
        """ Return the hand trimesh object corresponding to the input joint value q. """
        self.update_status(q)

        scene = trimesh.Scene()
        for link_name in self.vertices:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        parts = {}
        for link_name in self.meshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            part_mesh = self.meshes[link_name].copy().apply_transform(mesh_transform_matrix)
            parts[link_name] = part_mesh

        return_dict = {
            'visual': trimesh.Trimesh(vertices=all_vertices, faces=all_faces),
            'parts': parts
        }
        return return_dict

    def get_trimesh_se3(self, transform, index):
        """ Return the hand trimesh object corresponding to the input transform. """
        scene = trimesh.Scene()
        for link_name in transform:
            mesh_transform_matrix = transform[link_name][index].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    
    def get_mano_pc(self, q, translation, hand_pose, hand_shape):
        """ 
        Return the mano hand point cloud corresponding to the input joint value q.
            :param q: (6 + DOF,), joint values (euler representation)
            :param translation: torch.tensor, (3,), translation of the hand
            :param hand_pose: torch.tensor, (1, 48), mano hand pose
            :param hand_shape: torch.tensor, (1, 10), mano hand shape
        """
        assert self.robot_name == 'mano', f"Only support mano hand model, but got {self.robot_name}"

        rpy = q[3:6].clone()
        robot_joint_order = self.get_joint_orders()
        joint_to_q = dict(zip(robot_joint_order, q))

        composed_ee = torch.zeros((1, 16, 3))  
        composed_ee[:, 0] = torch.zeros_like(rpy).unsqueeze(0)  # (1, 3)

        composed_ee[:, 13] = torch.tensor([0, joint_to_q["j_thumb1y"], joint_to_q["j_thumb1x"]]).unsqueeze(0)  # thumb1
        composed_ee[:, 14] = torch.tensor([0, 0, joint_to_q["j_thumb2"]]).unsqueeze(0)
        composed_ee[:, 15] = torch.tensor([0, 0, joint_to_q["j_thumb3"]]).unsqueeze(0)

        composed_ee[:, 1] = torch.tensor([0, joint_to_q["j_index1y"], joint_to_q["j_index1x"]]).unsqueeze(0)  # index1
        composed_ee[:, 2] = torch.tensor([0, 0, joint_to_q["j_index2"]]).unsqueeze(0)
        composed_ee[:, 3] = torch.tensor([0, 0, joint_to_q["j_index3"]]).unsqueeze(0)

        composed_ee[:, 4] = torch.tensor([0, joint_to_q["j_middle1y"], joint_to_q["j_middle1x"]]).unsqueeze(0)  # middle1
        composed_ee[:, 5] = torch.tensor([0, 0, joint_to_q["j_middle2"]]).unsqueeze(0)
        composed_ee[:, 6] = torch.tensor([0, 0, joint_to_q["j_middle3"]]).unsqueeze(0)

        composed_ee[:, 10] = torch.tensor([0, joint_to_q["j_ring1y"], joint_to_q["j_ring1x"]]).unsqueeze(0)  # ring1
        composed_ee[:, 11] = torch.tensor([0, 0, joint_to_q["j_ring2"]]).unsqueeze(0)
        composed_ee[:, 12] = torch.tensor([0, 0, joint_to_q["j_ring3"]]).unsqueeze(0)

        composed_ee[:, 7] = torch.tensor([0, joint_to_q["j_pinky1y"], joint_to_q["j_pinky1x"]]).unsqueeze(0)  # pinky1
        composed_ee[:, 8] = torch.tensor([0, 0, joint_to_q["j_pinky2"]]).unsqueeze(0)
        composed_ee[:, 9] = torch.tensor([0, 0, joint_to_q["j_pinky3"]]).unsqueeze(0)

        composed_aa = self.axisFK.compose_our(composed_ee).clone()  # (B=1, 16, 3)
        composed_aa = composed_aa.reshape(1, -1)  # (1, 48)
        composed_aa = torch.cat([hand_pose[:, :3], composed_aa[:, 3:]], dim=1)

        mano_output: MANOOutput = self.mano_layer(composed_aa, hand_shape)
        hand_verts = mano_output.verts.squeeze(0)   # (NV, 3)
        hand_verts = hand_verts + translation[None, :]
        
        return self.get_mano_pc_from_verts(hand_verts)

    def get_mano_pc_from_verts(self, verts: torch.Tensor):
        """
            Return the mano hand point cloud corresponding to the input vertices.
            :param verts: torch.tensor, (NV, 3), mano hand vertices
        """
        robot_pc_target = farthest_point_sampling(verts, 512)
        return robot_pc_target


def create_hand_model(
    robot_name,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_points=512
):
    json_path = os.path.join(ROOT_DIR, 'data/data_urdf/robot/urdf_assets_meta.json')
    urdf_assets_meta = json.load(open(json_path))
    urdf_path = os.path.join(ROOT_DIR, urdf_assets_meta['urdf_path'][robot_name])
    meshes_path = os.path.join(ROOT_DIR, urdf_assets_meta['meshes_path'][robot_name])
    links_pc_path = os.path.join(ROOT_DIR, f'data/PointCloud/robot/{robot_name}.pt')
    hand_model = HandModel(robot_name, urdf_path, meshes_path, links_pc_path, device, num_points)
    return hand_model
