import os
import sys
import json
import pickle
import torch
import hashlib
import trimesh
import numpy as np
from tqdm import tqdm
from oikit.oi_shape import OakInkShape 
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
from trimesh import Trimesh
# import pyvista as pv
import multiprocessing
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.hand_model import create_hand_model
from manotorch.utils.geometry import matrix_to_euler_angles
import argparse
from collections import defaultdict
from openai import OpenAI
import re
from prepare_contact_area import process_meta
import clip
from typing import Dict, List, Tuple
import open3d as o3d
from oikit.oak_base import ObjectAffordanceKnowledge as OAK
from oikit.oak_base import OakBase

INTENT_MAP = {
    '0001': 'use',
    '0002': 'hold',
    '0003': 'liftup',
    '0004': 'handover',
}


FINGER_NAMES = {
    0: "Thumb",
    1: "Index",
    2: "Middle",
    3: "Ring",
    4: "Pinky",
}


OAKINK_DIR = '/data/zwq/data/oakink'  
TARGET_DIR = '/data/zwq/code/DRO_Grasp/data/OakInkDataset'
REAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'object_id.json')
VIRTUAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'virtual_object_id.json')

def run_urdf_animation(robot, robot_cfg):
    robot.animate(cfg_trajectory=robot_cfg)


def show_plotter(mesh, color):
    if not isinstance(mesh, (list, tuple)):
        mesh = [mesh]
    if not isinstance(color, (list, tuple)):
        color = [color]

    pl = pv.Plotter(off_screen=False, polygon_smoothing=True)
    pl.set_background('white')

    for i, m in enumerate(mesh):
        pl.add_mesh(m, color=color[i], name=f"mesh_{i}")
    
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)


def check_origin_pose(composed_aa, tsl, axisFK:AxisLayerFK, mano_layer):
        
        zero_shape = torch.zeros((1, 10))
        mano_output_recomputed: MANOOutput = mano_layer(composed_aa, zero_shape)
        new_hand_verts = mano_output_recomputed.verts.squeeze(0).numpy()  # (NV, 3)
        new_hand_verts = new_hand_verts + tsl[None, :]
        new_hand_faces_np = mano_layer.th_faces  # (NF, 3)
        new_mesh = Trimesh(new_hand_verts, new_hand_faces_np)

        return new_mesh, new_hand_verts


def check_origin(ee, tsl, generated_pose, axisFK:AxisLayerFK, mano_layer):
        
        composed_aa = axisFK.compose(ee).clone()  # (B=1, 16, 3)
        composed_aa = composed_aa.reshape(1, -1)  # (1, 48)
        composed_aa = torch.cat([generated_pose[:,:3], composed_aa[:, 3:]], dim=1)

        zero_shape = torch.zeros((1, 10))
        mano_output_recomputed: MANOOutput = mano_layer(composed_aa, zero_shape)
        new_hand_verts = mano_output_recomputed.verts.squeeze(0).numpy()  # (NV, 3)
        new_hand_verts = new_hand_verts + tsl[None, :]
        new_hand_faces_np = mano_layer.th_faces  # (NF, 3)
        new_mesh = Trimesh(new_hand_verts, new_hand_faces_np)

        return new_mesh, new_hand_verts

def check(robot_value_full, urdf_joint_order, tsl, generated_pose, shape, mode, axisFK:AxisLayerFK, mano_layer):
        rpy = robot_value_full[3:6].clone()
        q_values = robot_value_full[6:].clone()  

        composed_ee = torch.zeros((1, 16, 3))  
        joint_to_q = dict(zip(urdf_joint_order, q_values))

        if mode == "standard":
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

        elif mode == "extended":
            composed_ee[:, 0] = torch.tensor(rpy).unsqueeze(0)  # (1, 3)

            composed_ee[:, 13] = torch.tensor([joint_to_q["j_thumb1z"], joint_to_q["j_thumb1y"], joint_to_q["j_thumb1x"]]).unsqueeze(0)  # thumb1
            composed_ee[:, 14] = torch.tensor([0, 0, joint_to_q["j_thumb2"]]).unsqueeze(0)
            composed_ee[:, 15] = torch.tensor([0, 0, joint_to_q["j_thumb3"]]).unsqueeze(0)

            composed_ee[:, 1] = torch.tensor([joint_to_q["j_index1z"], joint_to_q["j_index1y"], joint_to_q["j_index1x"]]).unsqueeze(0)  # index1
            composed_ee[:, 2] = torch.tensor([0, 0, joint_to_q["j_index2"]]).unsqueeze(0)
            composed_ee[:, 3] = torch.tensor([0, 0, joint_to_q["j_index3"]]).unsqueeze(0)

            composed_ee[:, 4] = torch.tensor([joint_to_q["j_middle1z"], joint_to_q["j_middle1y"], joint_to_q["j_middle1x"]]).unsqueeze(0)  # middle1
            composed_ee[:, 5] = torch.tensor([0, 0, joint_to_q["j_middle2"]]).unsqueeze(0)
            composed_ee[:, 6] = torch.tensor([0, 0, joint_to_q["j_middle3"]]).unsqueeze(0)

            composed_ee[:, 10] = torch.tensor([joint_to_q["j_ring1z"], joint_to_q["j_ring1y"], joint_to_q["j_ring1x"]]).unsqueeze(0)  # ring1
            composed_ee[:, 11] = torch.tensor([0, 0, joint_to_q["j_ring2"]]).unsqueeze(0)
            composed_ee[:, 12] = torch.tensor([0, 0, joint_to_q["j_ring3"]]).unsqueeze(0)

            composed_ee[:, 7] = torch.tensor([joint_to_q["j_pinky1z"], joint_to_q["j_pinky1y"], joint_to_q["j_pinky1x"]]).unsqueeze(0)  # pinky1
            composed_ee[:, 8] = torch.tensor([0, 0, joint_to_q["j_pinky2"]]).unsqueeze(0)
            composed_ee[:, 9] = torch.tensor([0, 0, joint_to_q["j_pinky3"]]).unsqueeze(0)

        elif mode == "all":
            composed_ee[:, 0] = torch.tensor(rpy).unsqueeze(0)  # (1, 3)
            base_idx = [13, 1, 4, 10, 7]
            for i, joint in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
                for j, joint_idx in enumerate(["1", "2", "3"]):
                    composed_ee[:, base_idx[i] + j] = torch.tensor([joint_to_q[f"j_{joint}{joint_idx}z"], joint_to_q[f"j_{joint}{joint_idx}y"], joint_to_q[f"j_{joint}{joint_idx}x"]]).unsqueeze(0)
                    print(f"composed_ee[:, {base_idx[i] + j}] = j_{joint}{joint_idx}z, j_{joint}{joint_idx}y, j_{joint}{joint_idx}x")
        else:
            raise ValueError(f"mode={mode} is not supported")

        composed_aa = axisFK.compose_our(composed_ee).clone()  # (B=1, 16, 3)
        composed_aa = composed_aa.reshape(1, -1)  # (1, 48)
        composed_aa = torch.cat([generated_pose[:,:3], composed_aa[:, 3:]], dim=1)

        # zero_shape = torch.zeros((1, 10))
        mano_output_recomputed: MANOOutput = mano_layer(composed_aa, shape)
        new_hand_verts = mano_output_recomputed.verts.squeeze(0).numpy()  # (NV, 3)
        new_hand_verts = new_hand_verts + tsl[None, :]
        new_hand_faces_np = mano_layer.th_faces  # (NF, 3)
        new_mesh = Trimesh(new_hand_verts, new_hand_faces_np)

        return new_mesh, new_hand_verts, composed_ee


def process_robot_value(mode, joint_order, urdf_joint_order, ee_reordered):

    urdf_idx = 0
    robot_value = {}
    if mode == "standard":
        assert len(urdf_joint_order) == 20, f"len(urdf_joint_order)={len(urdf_joint_order)} is not 20"
        robot_value_array = torch.zeros(20)
        for idx_joint, joint in enumerate(joint_order):
                # the joint includes "1" means it has two joint values
                if "1" in joint: 
                    bend_value = ee_reordered[idx_joint, 1].item()
                    robot_value[urdf_joint_order[urdf_idx]] = bend_value
                    robot_value_array[urdf_idx] = bend_value
                    urdf_idx += 1

                    spread_value = ee_reordered[idx_joint, -1].item()
                    robot_value[urdf_joint_order[urdf_idx]] = spread_value
                    robot_value_array[urdf_idx] = spread_value
                    urdf_idx += 1
                else:
                    target_value = ee_reordered[idx_joint, -1].item()
                    robot_value[urdf_joint_order[urdf_idx]] = target_value
                    robot_value_array[urdf_idx] = target_value
                    urdf_idx += 1
    elif mode == "extended":
        assert len(urdf_joint_order) == 25, f"len(urdf_joint_order)={len(urdf_joint_order)} is not 25"
        robot_value_array = torch.zeros(25)
        for idx_joint, joint in enumerate(joint_order):
            # the joint includes "1" means it has two joint values
            if "1" in joint: 
                for idx in range(3):
                    spread_value = ee_reordered[idx_joint, idx].item()
                    robot_value[urdf_joint_order[urdf_idx]] = spread_value
                    robot_value_array[urdf_idx] = spread_value
                    urdf_idx += 1
            else:
                target_value = ee_reordered[idx_joint, -1].item()
                robot_value[urdf_joint_order[urdf_idx]] = target_value
                robot_value_array[urdf_idx] = target_value
                urdf_idx += 1
    elif mode == "all":
        assert len(urdf_joint_order) == 45, f"len(urdf_joint_order)={len(urdf_joint_order)} is not 45"
        robot_value_array = torch.zeros(45)
        for idx_joint, joint in enumerate(joint_order):
            for idx in range(3):
                spread_value = ee_reordered[idx_joint, idx].item()
                robot_value[urdf_joint_order[urdf_idx]] = spread_value
                robot_value_array[urdf_idx] = spread_value
                urdf_idx += 1
    else:
        raise ValueError(f"mode={mode} is not supported")

    return robot_value, robot_value_array

def process_joint_order(mode):

    joint_order = [
        "j_thumb1", "j_thumb2", "j_thumb3",
        "j_index1", "j_index2", "j_index3",
        "j_middle1", "j_middle2", "j_middle3",
        "j_ring1", "j_ring2", "j_ring3",
        "j_pinky1", "j_pinky2", "j_pinky3"
    ]
     
    if mode == "standard":
        urdf_joint_order = [
            "j_thumb1y", "j_thumb1x", "j_thumb2", "j_thumb3",
            "j_index1y", "j_index1x", "j_index2", "j_index3",
            "j_middle1y", "j_middle1x", "j_middle2", "j_middle3",
            "j_ring1y", "j_ring1x", "j_ring2", "j_ring3",
            "j_pinky1y", "j_pinky1x", "j_pinky2", "j_pinky3"
        ]
    elif mode == "extended":
        urdf_joint_order = [
            "j_thumb1z", "j_thumb1y", "j_thumb1x", "j_thumb2", "j_thumb3",
            "j_index1z", "j_index1y", "j_index1x", "j_index2", "j_index3",
            "j_middle1z", "j_middle1y", "j_middle1x", "j_middle2", "j_middle3",
            "j_ring1z", "j_ring1y", "j_ring1x", "j_ring2", "j_ring3",
            "j_pinky1z", "j_pinky1y", "j_pinky1x", "j_pinky2", "j_pinky3"
        ]   
    elif mode == "all":     
        urdf_joint_order = []
        for joint in ["thumb", "index", "middle", "ring", "pinky"]:
            for joint_idx in ["1", "2", "3"]:
                for joint_value in ["z", "y", "x"]:
                    urdf_joint_order.append(f"j_{joint}{joint_idx}{joint_value}")
    else:
        raise ValueError(f"mode={mode} is not supported")

    return joint_order, urdf_joint_order

def preprocess_oakink(mode="standard", category="all", split="all", dataset_save_path=None, 
         quick_load=True, save_pc=True, num_samples=65536, debug_show=False, remove_error=True):

    if debug_show:
        category = "teapot"

    
    oakink = OakInkShape(
        data_split=split,  
        category=category,
        mano_assets_root="assets/mano",  
        use_cache=False, 
        use_downsample_mesh=False, 
        preload_obj=False,  
        vis_handover=False 
    )

    if quick_load:
        save_pc = False
        oakink = oakink.grasp_list
        print("Quick load mode is on, save_pc is set to False (quick load use none pc load method to speed up)")

    if save_pc:
        pc_save_path =  os.path.join(ROOT_DIR, f"data/PointCloud/oakink/")
        os.makedirs(pc_save_path, exist_ok=True)  
        print(f"Start saving pc to {pc_save_path}")

    if not debug_show:
        assert dataset_save_path is not None, "Not in debug mode, dataset_save_path should be provided"

    dataset = {
        'info': {},
        'metadata': []
    }

    robot_name = 'mano'
    dataset['info'][robot_name] = {
        'robot_name': robot_name,
        'num_total': 0,
        'num_upper_object': 0,
        'num_per_object': {}
    }

    mano_layer = ManoLayer(rot_mode="axisang",
                           center_idx=0,
                           mano_assets_root="assets/mano",
                           use_pca=False,
                           flat_hand_mean=True)
    hand_faces = mano_layer.th_faces  
    axisFK = AxisLayerFK(mano_assets_root="assets/mano")

    joint_order, urdf_joint_order = process_joint_order(mode)
    re_index = [0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]

    for idx in tqdm(range(len(oakink)), desc="Processing grasps"):
        grasp = oakink[idx]

        pose = grasp["hand_pose"]  
        shape = grasp["hand_shape"]  
        tsl = grasp["hand_tsl"]  
        intent_id = grasp["action_id"]  
        intent = INTENT_MAP.get(intent_id, 'unknown')
        object_key = f"oakink+{grasp['cate_id']}"
        object_id = grasp["obj_id"] 
        parent_object_id = "-1" if not grasp["is_virtual"] else grasp["raw_obj_id"]
        read_hand_vertices = grasp["verts"]

        if object_key not in dataset['info'][robot_name]['num_per_object']:
            dataset['info'][robot_name]['num_per_object'][object_key] = {'all': 0}

        if object_id not in dataset['info'][robot_name]['num_per_object'][object_key]:
            dataset['info'][robot_name]['num_per_object'][object_key][object_id] = 0

        dataset['info'][robot_name]['num_per_object'][object_key][object_id] += 1
        dataset['info'][robot_name]['num_per_object'][object_key]['all'] += 1
        dataset['info'][robot_name]['num_total'] += 1

        hand_pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)  # (1, pose_dim)
        hand_shape_tensor = torch.tensor(shape, dtype=torch.float32).unsqueeze(0)  # (1, shape_dim)
        mano_output: MANOOutput = mano_layer(hand_pose_tensor, hand_shape_tensor)

        generated_hand_pose = mano_output.full_poses
        T_g_p = mano_output.transforms_abs  # (B=1, 16, 4, 4)
        T_g_a, R, ee = axisFK.forward(T_g_p)
        origin_ee = ee.clone()
        T_g_a = T_g_a.squeeze(0)  # (16, 4, 4)
        ee = ee.squeeze(0)  # (16, 3)
        ee_reordered = ee[re_index, :][1:, :]  # (15, 3)
        global_ee = matrix_to_euler_angles(T_g_p[:, 0, :3, :3], convention='XYZ')  # (1, 3)        

        hand_verts = mano_output.verts.squeeze(0).numpy()  # (NV, 3)
        hand_verts = hand_verts + tsl[None, :]

        xyz = T_g_p[0, 0, :3, 3].to(torch.float32) + torch.tensor(tsl)
        rpy = global_ee[0, :].to(torch.float32)  # (3,)

        robot_value, robot_value_array = process_robot_value(mode, joint_order, urdf_joint_order, ee_reordered)
        robot_value_full = torch.cat([xyz, rpy, robot_value_array], dim=0) 

        if save_pc:
            cate_path = os.path.join(pc_save_path, grasp['cate_id'])
            os.makedirs(cate_path, exist_ok=True)
            obj_save_path = os.path.join(cate_path, f"{object_id}.pt")

            if os.path.exists(obj_save_path):
                print(f"Object pc already exists: {obj_save_path}, skip...")
                continue

            obj_mesh = grasp["obj_mesh"]  
            object_pc, _ = obj_mesh.sample(num_samples, return_index=True)
            object_pc = torch.tensor(object_pc, dtype=torch.float32)
            assert object_pc.dim() == 2, f"object_pc.dim()={object_pc.dim()} must be 2"
            print(f"Saving object pc to {obj_save_path}")
            torch.save(object_pc, obj_save_path)

        # debug only >>>>>>
        # robot_value_full = torch.zeros_like(robot_value_full)
        # generated_hand_pose = torch.zeros_like(generated_hand_pose)

        # urdf_joint_order = [
        #     "j_thumb1y", "j_thumb1x", "j_thumb2", "j_thumb3",
        #     "j_index1y", "j_index1x", "j_index2", "j_index3",
        #     "j_middle1y", "j_middle1x", "j_middle2", "j_middle3",
        #     "j_ring1y", "j_ring1x", "j_ring2", "j_ring3",
        #     "j_pinky1y", "j_pinky1x", "j_pinky2", "j_pinky3"
        # ]

        # robot_value_full = torch.tensor([
        #     0, 0, 0, 0, 0, 0,
        #     1.2, 1.2, 0, 0, 
        #     0, 0, 0, 0, 
        #     0, 0, 0, 0,
        #     0, 0, 0, 0,
        #     0, 0, 0, 0,
        # ])

        # robot_cfg = {
        #     joint_name: [joint_value,joint_value] for joint_name, joint_value in zip(urdf_joint_order, robot_value_full[6:])
        # }
        # <<<<<< debug only

        if debug_show:
            
            read_hand_mesh = Trimesh(read_hand_vertices, hand_faces)
            mesh = Trimesh(hand_verts, hand_faces)

            # origin_hand_pose_mesh, origin_hand_pose_verts = check_origin_pose(generated_hand_pose, tsl, axisFK, mano_layer)
            origin_hand_mesh, origin_hand_verts = check_origin(origin_ee, tsl, generated_hand_pose, axisFK, mano_layer)
            genereted_hand_mesh, generated_hand_verts, composed_ee = check(robot_value_full, urdf_joint_order, tsl, generated_hand_pose, hand_shape_tensor, mode, axisFK, mano_layer)
            # genereted_hand_mesh_2, generated_hand_verts, composed_ee = check(robot_value_full, urdf_joint_order, tsl, torch.zeros_like(generated_hand_pose), hand_shape_tensor, mode, axisFK, mano_layer)
            
            show_robot = []
            show_color = []
            show_robot.append(read_hand_mesh)
            show_color.append("red")
            # show_robot.append(mesh)
            # show_color.append("green")

            if mode in ["standard"]:
                hand = create_hand_model(robot_name)
                robot_value_full = torch.cat([robot_value_full[:6], robot_value_full[7:8], robot_value_full[6:7], robot_value_full[8:]])
                robot_trimesh = hand.get_trimesh_q(robot_value_full)["visual"]
                robot_trimesh.vertices = robot_trimesh.vertices
                show_robot.append(robot_trimesh)
                show_color.append("blue")

            # print("origin", origin_ee)
            # print("composed", composed_ee)
            # print(origin_ee == composed_ee)

            for i in range(5):
                print(urdf_joint_order[4*i:4*(i+1)])
                print(robot_value_full[6+4*i:6+4*(i+1)])

            # urdf_process = multiprocessing.Process(target=run_urdf_animation, args=(robot, robot_cfg))
            # urdf_process.start()

            show_plotter(show_robot, color=show_color)

            # urdf_process.join()

        assert xyz.shape == (3,), f"xyz.shape={xyz.shape} is not (3,)"
        assert rpy.shape == (3,), f"rpy.shape={rpy.shape} is not (3,)"

        dataset['metadata'].append((
            torch.tensor(pose, dtype=torch.float32),   # 0
            torch.tensor(shape, dtype=torch.float32),  # 1
            torch.tensor(tsl, dtype=torch.float32),    # 2
            robot_value_full,                          # (q) 3
            intent,                                    # 4
            object_key,                                # 5
            robot_name,                                # 6
            object_id,                            # 7
            parent_object_id,                          # 8
            hand_verts,                                # 9
        ))


        dataset['info'][robot_name]['num_upper_object'] = max(
            dataset['info'][robot_name]['num_upper_object'],
            dataset['info'][robot_name]['num_per_object'][object_key]['all']
        )

    dataset['version'] = {
        'metadata': '1.1.0',
        'description': 'Metadata with valid q parameter (with translation). [Update to 1.2.0 for valid pc entries]'
    }

    if remove_error:
        remove_error_metadata(dataset, dataset_save_path)
        remove_lack_coacd_metadata(dataset, dataset_save_path)
    else:
        try:
            os.makedirs(os.path.dirname(dataset_save_path), exist_ok=True)
            torch.save(dataset, dataset_save_path)
            print(f"Dataset successfully saved to {dataset_save_path}")
        except Exception as e:
            print(f"Failed to save dataset: {dataset_save_path}, error: {e}")
            exit(1)    

    update_simple_language_embedding(dataset, dataset_save_path)

def remove_error_metadata(dataset=None, dataset_path=None):
    '''
        Remove metadata with invalid object point clouds from the dataset.
        Update metadata version from 1.1.0 to 1.2.0.
    '''    

    log_dir = os.path.join(ROOT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'remove_error_metadata.log')
    
    if dataset_path is None:
        dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all.pt')
    if dataset is None:
        dataset = torch.load(dataset_path)

    dataset_save_path = dataset_path

    object_pcs_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_object_pcs.pt')
    object_pcs = torch.load(object_pcs_path)

    info = dataset.get('info', {})
    metadata = dataset.get('metadata', [])

    invalid_objects = set()
    removed_samples = 0

    unique_objects = set((sample[5], sample[7]) for sample in metadata)
    print(f"Checking {len(unique_objects)} unique objects...")

    for object_key, object_id in tqdm(unique_objects, desc="Validating point clouds"):
        try:
            prefix, category = object_key.split('+')
        except ValueError:
            print(f"Invalid object_key format: {object_key}. Expected format: 'oakink+<category>'")
            invalid_objects.add((object_key, object_id))
            continue
        
        try:
            object_pc = object_pcs[object_id]
            if not isinstance(object_pc, torch.Tensor) or object_pc.shape != (65536, 3):
                print(f"Invalid point cloud shape for {object_key}-{object_id}: {object_pc.shape}")
                invalid_objects.add((object_key, object_id))
                continue
            if not torch.isfinite(object_pc).all():
                print(f"Point cloud contains non-finite values: {object_key}-{object_id}")
                invalid_objects.add((object_key, object_id))
        except Exception as e:
            print(f"Error loading point cloud for {object_key}-{object_id}: {e}")
            invalid_objects.add((object_key, object_id))
    
    new_metadata = [
        sample for sample in metadata
        if (sample[5], sample[7]) not in invalid_objects
    ]
    removed_samples = len(metadata) - len(new_metadata)
    
    for robot_name, robot_info in info.items():
        num_per_object = robot_info.get('num_per_object', {})
        for object_key, object_id in invalid_objects:
            if object_id in object_pcs:
                del object_pcs[object_id]
            if object_key in num_per_object and object_id in num_per_object[object_key]:
                removed_count = num_per_object[object_key][object_id]
                robot_info['num_total'] -= removed_count
                num_per_object[object_key]['all'] -= removed_count
                del num_per_object[object_key][object_id]
        
        robot_info['num_upper_object'] = 0
        for object_key, object_data in num_per_object.items():
            robot_info['num_upper_object'] = max(
                robot_info['num_upper_object'],
                robot_info['num_per_object'][object_key]['all']
            )

    dataset['metadata'] = new_metadata
    dataset['version'] = {
        'metadata': '1.2.0',
        'description': 'Metadata with valid pc entries. [Check to 1.2.0_COACD for COACD compatibility]'
    }

    torch.save(dataset, dataset_save_path)
    print(f"Updated dataset saved to {dataset_save_path}")
    torch.save(object_pcs, object_pcs_path)
    print(f"Updated object point clouds saved to {object_pcs_path}")
    
    with open(log_file, 'w') as f:
        f.write(f"-"*50 + "\n")
        f.write(f"pdated dataset saved to {dataset_save_path}")
        for object_key, object_id in invalid_objects:
            f.write(f"{object_key} - {object_id}\n")
        f.write(f"Total removed samples: {removed_samples}")
    print(f"Log of removed objects saved to {log_file}")

def remove_lack_coacd_metadata(dataset=None, dataset_path=None):
    '''
        Remove metadata with invalid object point clouds from the dataset.
        Update metadata version from 1.2.0 to 1.2.0_COACD.
    '''

    log_dir = os.path.join(ROOT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'remove_lack_coacd_metadata.log')    
    
    if dataset_path is None:
        dataset_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_dataset_standard_all.pt')
    if dataset is None:
        dataset = torch.load(dataset_path)

    dataset_save_path = os.path.splitext(dataset_path)[0] + '_coacd.pt'

    object_data_dir = os.path.join(ROOT_DIR, 'data', 'data_urdf', 'object', 'oakink')
    info = dataset.get('info', {})
    metadata = dataset.get('metadata', [])
    
    invalid_objects = set()
    for object_name in tqdm(os.listdir(object_data_dir), desc="Checking object data"):
        object_path = os.path.join(object_data_dir, object_name)
        if not os.path.isdir(object_path):
            continue  
        for file in os.listdir(object_path):
            if file.startswith('coacd_') and file.endswith('.obj'):
                object_id = file[len('coacd_'):-len('.obj')]
                object_key = f"oakink+{object_name}"
                if 'valid_objects' not in locals():
                    valid_objects = set()
                valid_objects.add((object_key, object_id))
    
    unique_objects = set((sample[5], sample[7]) for sample in metadata)
    invalid_objects = unique_objects - valid_objects
    
    new_metadata = [
        sample for sample in metadata
        if (sample[5], sample[7]) not in invalid_objects
    ]
    removed_samples = len(metadata) - len(new_metadata)
    print(f"Removed {removed_samples} samples.")
    
    for robot_name, robot_info in info.items():
        num_per_object = robot_info.get('num_per_object', {})
        for object_key, object_id in invalid_objects:
            if object_key in num_per_object and object_id in num_per_object[object_key]:
                removed_count = num_per_object[object_key][object_id]
                robot_info['num_total'] -= removed_count
                num_per_object[object_key]['all'] -= removed_count
                del num_per_object[object_key][object_id]
            
        robot_info['num_upper_object'] = 0
        for object_key, object_data in num_per_object.items():
            robot_info['num_upper_object'] = max(
                robot_info['num_upper_object'],
                robot_info['num_per_object'][object_key]['all']
            )

    dataset['metadata'] = new_metadata
    dataset['version']['metadata'] = '1.2.0_COACD'
    dataset['version']['description'] = 'Metadata with valid pc entries and COACD compatibility.'

    torch.save(dataset, dataset_save_path)
    print(f"Updated dataset saved to {dataset_save_path}")

    with open(log_file, 'w+') as f:
        f.write(f"-"*50 + "\n")
        f.write(f"pdated dataset saved to {dataset_save_path}")
        for object_key, object_id in invalid_objects:
            f.write(f"{object_key} - {object_id}\n")
        f.write(f"\nTotal invalid objects: {len(invalid_objects)}\n")
        f.write(f"Total removed samples: {removed_samples}")


def update_simple_language_embedding(batch_size = 500, dataset=None, dataset_path=None):
        
    client = OpenAI(api_key="your-key")

    log_dir = os.path.join(ROOT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'simple_language_embedding.log')
    
    if dataset_path is None:
        dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all.pt')
    if dataset is None:
        dataset = torch.load(dataset_path)

    dataset_save_path = dataset_path
    metadata = dataset.get('metadata', [])
    sentences = []
    indices = []

    for idx in tqdm(range(len(metadata)), desc="Preparing sentences"):
        sample = metadata[idx]
        # intent_desc = sample[4]
        # object_key = sample[5]
    
        # try:
        #     _, category = object_key.split('+', 1)
        # except ValueError:
        #     print(f"Invalid object_key format: '{object_key}'. Skipping sample {idx}.")
        #     continue
    
        # article = 'an' if category[0].lower() in 'aeiou' else 'a' 
        # sentence = f"{intent_desc.capitalize()} {article} {category}."

        sentence = sample[10]
        sentences.append(sentence)
        indices.append(idx)
    
        if len(sentences) == batch_size:

            embeddings = None

            while embeddings is None:
                embeddings = get_embedding(sentences, client)
                if embeddings is None:
                    print("Embedding is None. Try again.")

            assert len(embeddings) == len(sentences), "Number of embeddings does not match number of sentences."
    
            for i, embedding in enumerate(embeddings):
                embedding_tensor = torch.tensor(embedding.embedding, dtype=torch.float32)
                # new_sample = metadata[indices[i]] + (sentences[i], embedding_tensor)
                new_sample = metadata[indices[i]] + (embedding_tensor, )
                metadata[indices[i]] = new_sample
    
            sentences = []
            indices = []
    
    if len(sentences) > 0:

        embeddings = None

        while embeddings is None:
            embeddings = get_embedding(sentences, client)
            if embeddings is None:
                print("Embedding is None. Try again.")

        assert len(embeddings) == len(sentences), "Number of embeddings does not match number of sentences."
    
        for i, embedding in enumerate(embeddings):
            embedding_tensor = torch.tensor(embedding.embedding, dtype=torch.float32)
            # new_sample = metadata[indices[i]] + (sentences[i], embedding_tensor)
            new_sample = metadata[indices[i]] + (embedding_tensor, )
            metadata[indices[i]] = new_sample

    current_version = dataset.get('version', {}).get('metadata', '1.0.0')
    new_version = increment_version(current_version, 3)
    version_description = dataset.get('version', {}).get('description', '')

    new_description = version_description
    if 'sentence and embedding' not in version_description.lower():
        if version_description:
            new_description += " Added descriptive sentences and their embeddings."
        else:
            new_description = "Added descriptive sentences and their embeddings."

    

    if 'retarget' in dataset['version'].keys():
        dataset['version'] = {
            'metadata': new_version,
            'description': new_description,
            'retarget': dataset['version']['retarget']
        }
    else:
        dataset['version'] = {
            'metadata': new_version,
            'description': new_description,
        }

    if dataset_path is not None:
        torch.save(dataset, dataset_save_path)
        print(f"Updated dataset saved to '{dataset_save_path}'.")


def update_complex_openai_language_embedding(batch_size = 500, dataset=None, dataset_path=None):
        
    client = OpenAI(api_key="your-key")

    log_dir = os.path.join(ROOT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'simple_language_embedding.log')
    
    if dataset_path is None:
        dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all.pt')
    if dataset is None:
        dataset = torch.load(dataset_path)

    dataset_save_path = dataset_path
    metadata = dataset.get('metadata', [])
    sentences = []
    indices = []

    for idx in tqdm(range(len(metadata)), desc="Preparing sentences"):
        sample = metadata[idx][:11]
        sentence = sample[10]
        sentences.append(sentence)
        indices.append(idx)
    
        if len(sentences) == batch_size:

            embeddings = None

            while embeddings is None:
                embeddings = get_embedding(sentences, client)
                if embeddings is None:
                    print("Embedding is None. Try again.")

            assert len(embeddings) == len(sentences), "Number of embeddings does not match number of sentences."
    
            for i, embedding in enumerate(embeddings):
                embedding_tensor = torch.tensor(embedding.embedding, dtype=torch.float32)
                new_sample = metadata[indices[i]] + (embedding_tensor,)
                metadata[indices[i]] = new_sample
    
            sentences = []
            indices = []
    
    if len(sentences) > 0:

        embeddings = None

        while embeddings is None:
            embeddings = get_embedding(sentences, client)
            if embeddings is None:
                print("Embedding is None. Try again.")

        assert len(embeddings) == len(sentences), "Number of embeddings does not match number of sentences."
    
        for i, embedding in enumerate(embeddings):
            embedding_tensor = torch.tensor(embedding.embedding, dtype=torch.float32)
            new_sample = metadata[indices[i]] + (embedding_tensor,)
            metadata[indices[i]] = new_sample

    current_version = dataset.get('version', {}).get('metadata', '1.0.0')
    new_version = increment_version(current_version, 3)
    version_description = dataset.get('version', {}).get('description', '')

    new_description = version_description
    if 'sentence and embedding' not in version_description.lower():
        if version_description:
            new_description += " Added contact-based sentences and OpenAI embeddings."
        else:
            new_description = "Added descriptive sentences and OpenAI embeddings."
    print(f"New description: {new_description}")

    if 'retarget' in dataset['version'].keys():
        dataset['version'] = {
            'metadata': new_version,
            'description': new_description,
            'retarget': dataset['version']['retarget']
        }
    else:
        dataset['version'] = {
            'metadata': new_version,
            'description': new_description,
        }

    if dataset_path is not None:
        torch.save(dataset, dataset_save_path)
        print(f"Updated dataset saved to '{dataset_save_path}'.")


def sanitize_region_name(region: str) -> str:
    region = region.replace("_", " ")
    region = re.sub(r"[^a-zA-Z0-9\s]", "", region)
    region = re.sub(r"\s+", " ", region).strip()
    return region

def build_detailed_sentence(intent_desc: str,
                            category_clean: str,
                            finger2part: Dict[int, str],
                            long_sentence: bool = False
                            ) -> str:
    """
    Constructs a detailed English sentence based on the intent, object category,
    and finger-to-part contact mappings.

    Args:
        intent_desc (str): Description of the intent (e.g., "use").
        category_clean (str): Cleaned category name of the object (e.g., "teapot").
        finger2part (Dict[int, str]): Mapping from finger indices to object parts.
        long_sentence (bool): Whether to generate a long sentence with more details.

    Returns:
        str: A detailed English sentence describing finger contacts.
    """
    article = 'an' if category_clean and category_clean[0].lower() in 'aeiou' else 'a'
    base_sentence = f"{intent_desc.capitalize()} {article} {category_clean}."

    # Group fingers by their contact regions
    region_to_fingers = {}
    for finger_idx, region_name in finger2part.items():
        cleaned_region = sanitize_region_name(region_name)
        region_to_fingers.setdefault(cleaned_region, []).append(FINGER_NAMES[finger_idx])

    # Sort regions by the number of fingers contacting them (ascending)
    sorted_regions = sorted(region_to_fingers.items(), key=lambda item: len(item[1]))

    if long_sentence:
        detail_phrases = []
        for region, fingers in sorted_regions:
            # Modify finger names: add 'finger' suffix except for 'Thumb'
            formatted_fingers = []
            for finger in fingers:
                if finger != "Thumb":
                    formatted_fingers.append(f"{finger.lower()} finger")
                else:
                    formatted_fingers.append(f"{finger}")

            # Construct the phrase based on the number of fingers
            if len(formatted_fingers) >= 3:
                if len(formatted_fingers) == 3:
                    phrase = f"{formatted_fingers[0]}, {formatted_fingers[1]}, and {formatted_fingers[2]} are contacting the {region}"
                else:
                    # For 4 or 5 fingers
                    all_but_last = ", ".join(formatted_fingers[:-1])
                    phrase = f"{all_but_last} and {formatted_fingers[-1]} are contacting the {region}"
            elif len(formatted_fingers) == 2:
                phrase = f"{formatted_fingers[0]} and {formatted_fingers[1]} are contacting the {region}"
            else:
                # Single finger
                phrase = f"{formatted_fingers[0]} is contacting the {region}"
            
            detail_phrases.append(phrase.capitalize())

        # Combine all detail phrases into one part of the sentence
        detailed_part = ". ".join(detail_phrases) + "."
        final_sentence = base_sentence + " " + detailed_part
    else:
        # Short sentence: only mention the regions with the most fingers
        most_fingers_region, most_fingers_fingers = sorted_regions[-1]
        final_sentence = base_sentence.replace(".", f" by the {most_fingers_region}.")
        # print(f"base_sentence: {final_sentence}")

    return final_sentence


def get_clip_embeddings(sentences: List[str], clip_model, long_embedding=False, device="cuda") -> torch.Tensor:
    text_tokens = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)
    text_emb = text_emb.squeeze()
    return text_emb.cpu()


def update_complex_clip_language_embedding(
    batch_size=500,
    dataset=None,
    dataset_path=None,
    dimension=512,
    use_exist_sentences=False,
    long_sentences=False,
    long_embedding=False,
):
    
    with open(REAL_META_FILE, 'r', encoding='utf-8') as f:
        real_meta_file = json.load(f)

    with open(VIRTUAL_META_FILE, 'r', encoding='utf-8') as f:
        virtual_meta_file = json.load(f)

    oakink_dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all.pt')
    oakink_data = torch.load(oakink_dataset_path)
    oakink_metadata = oakink_data.get('metadata', [])

    obj_metadata = {**real_meta_file, **virtual_meta_file}

    oakbase = OakBase()

    all_cates: List[str] = list(oakbase.categories.keys())
    obj_nameid_metadata = {}

    for cate in all_cates:
        for obj in oakbase.get_objs_by_category(cate):
            obj_nameid_metadata[obj.obj_id] = obj


    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dimension == 512:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="/data/zwq/code/CLIP")
    elif dimension == 768:
        clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, download_root="/data/zwq/code/CLIP")
    else:
        raise ValueError(f"Invalid dimension: {dimension}")

    clip_model = clip_model.float()
    clip_model.eval()

    if dataset is None and dataset_path is None:
        raise ValueError("Must provide either 'dataset' or 'dataset_path' to load the dataset.")
    if dataset is None:
        dataset = torch.load(dataset_path)

    metadata = dataset.get('metadata', [])
    if not metadata:
        print("No metadata found in dataset.")
        return

    if oakbase is None:
        raise ValueError("oakbase (OakBase instance) cannot be None. Please provide a valid OakBase object.")

    all_cates: List[str] = list(oakbase.categories.keys())

    sentences = []
    indices = []


    if use_exist_sentences:
        for idx in tqdm(range(len(metadata)), desc="Generating contact-based sentences"):
            
            sample = metadata[idx]
            sentence = sample[10]
            sentences.append(sentence)
            indices.append(idx)

            if len(sentences) >= batch_size:
                embeddings = get_clip_embeddings(sentences, clip_model, long_embedding, device=device)
                for i, emb in enumerate(embeddings):
                    new_sample = metadata[indices[i]] + (emb, )
                    metadata[indices[i]] = new_sample

                sentences.clear()
                indices.clear()

        if len(sentences) > 0:
            embeddings = get_clip_embeddings(sentences, clip_model, long_embedding, device=device)
            for i, emb in enumerate(embeddings):
                new_sample = metadata[indices[i]] + (emb, )
                metadata[indices[i]] = new_sample
        
    else:
        for idx in tqdm(range(len(metadata)), desc="Generating contact-based sentences"):

            sample = metadata[idx][:10]
            if len(sample) < 10:
                continue

            hand_pose, hand_shape, tsl, target_q, intent, object_name, robot_name, object_id, real_object_id, hand_verts = sample[:10]

            result = process_meta(
                meta=sample[:10],
                obj_metadata=obj_metadata,
                obj_nameid_metadata=obj_nameid_metadata,
                all_cates=all_cates
            )
            if "error" in result:
                continue

            closest_parts_dict = result.get("closest_parts", {})
            if not closest_parts_dict:
                continue

            tip_id_map = {745: 0, 317: 1, 444: 2, 556: 3, 673: 4}
            finger2part = {}
            for tip_id, part_name in closest_parts_dict.items():
                if tip_id in tip_id_map:
                    finger_idx = tip_id_map[tip_id]
                    finger2part[finger_idx] = part_name

            category_clean = object_name
            if '+' in object_name:
                _, category_clean = object_name.split('+', 1)

            sentence = build_detailed_sentence(intent, category_clean, finger2part, long_sentences)
            sentences.append(sentence)
            indices.append(idx)

            if len(sentences) >= batch_size:
                embeddings = get_clip_embeddings(sentences, clip_model, device=device)
                for i, emb in enumerate(embeddings):
                    sen = sentences[i]
                    new_sample = metadata[indices[i]][:10] + (sen, emb)
                    metadata[indices[i]] = new_sample

                sentences.clear()
                indices.clear()

        if len(sentences) > 0:
            embeddings = get_clip_embeddings(sentences, clip_model, device=device)
            for i, emb in enumerate(embeddings):
                sen = sentences[i]
                new_sample = metadata[indices[i]][:10] + (sen, emb)
                metadata[indices[i]] = new_sample

    dataset['metadata'] = metadata
    current_version = dataset.get('version', {}).get('metadata', '1.0.0')
    new_version = increment_version(current_version, None, 2) 
    version_description = dataset.get('version', {}).get('description', '')

    new_description = version_description
    if 'contact-based sentence' not in version_description.lower():
        if version_description:
            new_description += f" Added contact-based sentences and CLIP embeddings ({dimension} dimension)."
        else:
            new_description = f"Added contact-based sentences and CLIP embeddings ({dimension} dimension)."
    
    print(f"New description: {new_description}")
    print(f"New version: {new_version}")

    if 'retarget' in dataset['version'].keys():
        dataset['version'] = {
            'metadata': new_version,
            'description': new_description,
            'retarget': dataset['version']['retarget']
        }
    else:
        dataset['version'] = {
            'metadata': new_version,
            'description': new_description,
        }

    if dataset_path is not None:
        torch.save(dataset, dataset_path)
        print(f"Updated dataset saved to '{dataset_path}'.")

    
def get_embedding(text, client, model="text-embedding-3-small", dimensions=256):
    try:
        response = client.embeddings.create(
            input=text,
            model=model,
            dimensions=dimensions
        )
        return response.data
    except Exception as e:
        print(f"Get embedding error: {e}")
        return None
    
def increment_version(current_version, new_minor=None, new_major=None):
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(.*)$', current_version)
    if not match:
        return '1.0.0'

    major, minor, patch, suffix = match.groups()
    minor = str(int(minor) + 1) if new_minor is None else str(new_minor)
    if new_major is not None:
        major = str(new_major)
        minor = '0'
    new_version = f"{major}.{minor}.{patch}{suffix}"
    return new_version

if __name__ == "__main__":
      
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_show', action='store_true')
    parser.add_argument('--mode', default="standard")
    parser.add_argument('--split', default="all")
    parser.add_argument('--category', default="all")
    parser.add_argument('--func', default="update_batch_process", choices=["preprocess", "remove_error_metadata", "remove_lack_coacd_metadata", "update_simple_language_embedding", "update_simple_language_embedding_with_contact", "update_complex_openai_language_embedding", "update_batch_process"])
    args = parser.parse_args()

    # if args.category != "all":
    #     dataset_save_path =  os.path.join(ROOT_DIR, f"data/OakInkDataset/{args.category}_oakink_dataset_{args.mode}_{args.split}.pt")
    # else:
    #     dataset_save_path =  os.path.join(ROOT_DIR, f"data/OakInkDataset/oakink_dataset_{args.mode}_{args.split}.pt")

    # dataset_save_path = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_shadowhand.pt"
    dataset_save_path = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand.pt"
    
    print(f"dataset_save_path is {dataset_save_path}")

    if args.func == "preprocess":
        print(f"Start to save to {dataset_save_path}")
        preprocess_oakink(debug_show=args.debug_show, category=args.category, mode=args.mode, split=args.split, dataset_save_path=dataset_save_path)
    elif args.func == "remove_error_metadata":
        remove_error_metadata(dataset_path=dataset_save_path)
    elif args.func == "remove_lack_coacd_metadata":
        remove_lack_coacd_metadata(dataset_path=dataset_save_path)
    elif args.func == "update_simple_language_embedding":
        update_simple_language_embedding(dataset_path=dataset_save_path)
    elif args.func == "update_simple_language_embedding_with_contact":
        update_complex_clip_language_embedding(dataset_path=dataset_save_path)
    elif args.func == "update_complex_openai_language_embedding":
        update_complex_openai_language_embedding(dataset_path=dataset_save_path)
    elif args.func == "update_batch_process":
        update_complex_clip_language_embedding(dataset_path=dataset_save_path, dimension=768, use_exist_sentences=False, long_sentences=False)
        update_complex_openai_language_embedding(dataset_path=dataset_save_path)
        update_complex_clip_language_embedding(dataset_path=dataset_save_path, dimension=512, use_exist_sentences=True, long_sentences=False)
    else:
        raise ValueError(f"func={args.func} is not supported")


