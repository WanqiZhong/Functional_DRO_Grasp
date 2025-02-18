import os
import sys
import json
import math
import hydra
import random
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
from oikit.oi_shape import OakInkShape
from tqdm import tqdm
import glob
import numpy as np
from urdfpy import URDF
from math import pi
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
import pickle as pk
import open3d as o3d
from DRO_Grasp.utils.hand_model import create_hand_model, HandModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.func_utils import farthest_point_sampling

class OakInkDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        robot_names: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = 'random',
        mano_assets_root: str = "assets/mano",
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names 
        assert self.robot_names == ['mano'], "OakInkDataset only supports mano"
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.mano_assets_root = mano_assets_root

        self.hands = {}
        self.dofs = []
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(math.sqrt(self.hands[robot_name].dof))

        assert os.path.exists(os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_object_pcs.pt')), "Please run generate pc to generate object pcs package first"

        split_json_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/split_train_validate_objects.json')
        dataset_split = json.load(open(split_json_path))
        self.object_names = dataset_split['train'] if is_train else dataset_split['validate']
        if debug_object_names is not None:
            print("!!! Using debug objects !!!")
            self.object_names = debug_object_names

        dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all.pt')
        metadata = torch.load(dataset_path)['metadata']
        self.metadata = [m for m in metadata if m[5] in self.object_names and m[6] in self.robot_names]
        if not self.is_train:
            self.combination = [m for m in self.metadata if m[6] in self.robot_names and m[5] in self.object_names]
            self.combination = sorted(self.combination)

        print("Loading OakInkShape pcs...")
        self.object_pcs = torch.load(os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_object_pcs.pt'))
        # else:
        #     print("Loading OakInkShape metadata...")
        #     self.object_pcs = {}
        #     for grasp_item in tqdm(self.metadata):
        #         if grasp_item[7] not in self.object_pcs:
        #             name = grasp_item[5].split('+')
        #             obj_mesh_path = list(
        #             glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{grasp_item[7]}.obj')) +
        #             glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{grasp_item[7]}.ply')))
        #             assert len(obj_mesh_path) == 1
        #             obj_path = obj_mesh_path[0]

        #             # if obj_path.endswith('.ply'):
        #             mesh = o3d.io.read_triangle_mesh(obj_path)
        #             vertices = np.asarray(mesh.vertices)
        #             bbox_center = (vertices.min(0) + vertices.max(0)) / 2
        #             mesh.vertices = o3d.utility.Vector3dVector(vertices - bbox_center)
        #             pcd = mesh.sample_points_uniformly(number_of_points=self.num_points)
        #             object_pc = np.asarray(pcd.points)
        #             # else:  
        #             #     obj_trimesh = trimesh.load(obj_path, process=False, force='mesh', skip_materials=True)
        #             #     bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
        #             #     obj_trimesh.vertices -= bbox_center
        #             #     object_pc, _ = obj_trimesh.sample(self.num_points, return_index=True)
        #             self.object_pcs[grasp_item[7]] = torch.tensor(object_pc, dtype=torch.float32)   

        #     # torch.save(self.object_pcs, os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_all_object_pcs.pt'))

    def __getitem__(self, index):
        """
        Train: sample a batch of data
        Validate: get (robot, object) from index, sample a batch of data
        """
        if self.is_train:
            robot_name_batch = []
            object_name_batch = []
            object_id_batch = []
            # robot_links_pc_batch = []
            robot_pc_initial_batch = []
            robot_pc_target_batch = []
            object_pc_batch = []
            dro_gt_batch = []
            # initial_q_batch = []
            # target_q_batch = []
            for idx in range(self.batch_size):  

                robot_name = random.choice(self.robot_names)
                robot_name_batch.append(robot_name)
                hand: HandModel = self.hands[robot_name]
                metadata_robot = [m for m in self.metadata if m[6] == robot_name]

                hand_pose, hand_shape, tsl, target_q, intent, object_name, robot_name, object_id, real_object_id, hand_verts = random.choice(metadata_robot)
                # target_q_batch.append(target_q)
                object_name_batch.append(object_name)
                object_id_batch.append(object_id)
                # robot_links_pc_batch.append(hand.links_pc)

                if self.object_pc_type == 'fixed':
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt')
                    object_pc = torch.load(object_path)[:, :3]
                if self.object_pc_type == 'random':
                    indices = torch.randperm(65536)[:self.num_points]
                    object_pc = self.object_pcs[object_id][indices]
                    object_pc += torch.randn(object_pc.shape) * 0.002
                else:  # 'partial', remove 50% points
                    indices = torch.randperm(65536)[:self.num_points * 2]
                    object_pc = self.object_pcs[object_id][indices]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    _, indices = torch.sort(proj)
                    indices = indices[self.num_points:]
                    object_pc = object_pc[indices]

                object_pc_batch.append(object_pc)

                robot_pc_target, _ = hand.get_mano_pc_from_verts(torch.tensor(hand_verts))
                robot_pc_target_batch.append(robot_pc_target)
                # target_q_batch.append(target_q)
                
                initial_q = hand.get_initial_q(target_q)
                # initial_q_batch.append(initial_q)
                robot_pc_initial, _ = hand.get_mano_pc(initial_q, tsl, hand_pose.unsqueeze(0), hand_shape.unsqueeze(0))
                robot_pc_initial_batch.append(robot_pc_initial)

                dro = torch.cdist(robot_pc_target, object_pc, p=2)
                dro_gt_batch.append(dro)

            robot_pc_initial_batch = torch.stack(robot_pc_initial_batch)
            robot_pc_target_batch = torch.stack(robot_pc_target_batch)
            object_pc_batch = torch.stack(object_pc_batch)
            dro_gt_batch = torch.stack(dro_gt_batch)

            B, N = self.batch_size, self.num_points
            assert robot_pc_initial_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_initial_batch.shape}"
            assert robot_pc_target_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_target_batch.shape}"
            assert object_pc_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
            assert dro_gt_batch.shape == (B, N, N),\
                f"Expected: {(B, N, N)}, Actual: {dro_gt_batch.shape}"

            return {
                'robot_name': robot_name_batch,  # list(len = B): str
                'object_name': object_name_batch,  # list(len = B): 
                'object_id': object_id_batch,  # list(len = B): 
                'robot_links_pc': None,
                'robot_pc_initial': robot_pc_initial_batch,
                'robot_pc_target': robot_pc_target_batch, 
                'object_pc': object_pc_batch,
                'dro_gt': dro_gt_batch,
                'initial_q': None,
                'target_q': None
            }
        else:  # validate
            hand_pose, hand_shape, tsl, target_q, intent, object_name, robot_name, object_id, real_object_id, hand_verts = random.choice(metadata_robot)
            hand = self.hands[robot_name]

            initial_q_batch = torch.zeros([self.batch_size, hand.dof], dtype=torch.float32)
            robot_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)
            object_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)

            for batch_idx in range(self.batch_size):
                initial_q = hand.get_initial_q()
                robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3]

                if self.object_pc_type == 'partial':
                    indices = torch.randperm(65536)[:self.num_points * 2]
                    object_pc = self.object_pcs[object_id][indices]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    _, indices = torch.sort(proj)
                    indices = indices[self.num_points:]
                    object_pc = object_pc[indices]
                else:
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt')
                    object_pc = torch.load(object_path)[:, :3]

                initial_q_batch[batch_idx] = initial_q
                robot_pc_batch[batch_idx] = robot_pc
                object_pc_batch[batch_idx] = object_pc

            B, N, DOF = self.batch_size, self.num_points, len(hand.pk_chain.get_joint_parameter_names())
            assert initial_q_batch.shape == (B, DOF), \
                f"Expected: {(B, DOF)}, Actual: {initial_q_batch.shape}"
            assert robot_pc_batch.shape == (B, N, 3), \
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_batch.shape}"
            assert object_pc_batch.shape == (B, N, 3), \
                f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"

            return {
                'robot_name': robot_name,  # str
                'object_name': object_name,  # str
                'initial_q': initial_q_batch,
                'robot_pc': robot_pc_batch,
                'object_pc': object_pc_batch
            }

    def __len__(self):
        if self.is_train:
            return math.ceil(len(self.metadata) / self.batch_size)
        else:
            return len(self.combination)        


def custom_collate_fn(batch):
    return batch[0]


def create_dataloader(cfg, is_train):
    dataset = OakInkDataset(
        batch_size=cfg.batch_size,
        robot_names=cfg.robot_names,
        is_train=is_train,
        debug_object_names=cfg.debug_object_names,
        object_pc_type=cfg.object_pc_type
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=cfg.num_workers,
        shuffle=is_train
    )
    return dataloader
