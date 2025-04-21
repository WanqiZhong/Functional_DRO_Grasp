from collections import defaultdict
import os
import sys
import time
import torch
import viser
import numpy as np
import open3d as o3d
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import trimesh
import glob
from utils.func_utils import get_contact_map, get_aligned_distance, get_euclidean_distance
from utils.hand_model import create_hand_model, HandModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# robot_names = ['leaphand', 'shadowhand', 'allegro']
robot_names = ['shadowhand']
object_names = [
    "oakink+teapot",
    "oakink+lotion_pump",
    "oakink+cylinder_bottle",
    "oakink+mug",
    "oakink+bowl",
    "oakink+cup",
    "oakink+knife",
    "oakink+pen",
    "oakink+bottle",
    "oakink+headphones"
]


robot_metadata = defaultdict(lambda: defaultdict(list))
hands = dict()
point_cloud_dataset = torch.load(os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_object_pcs_with_normals.pt'))
for robot_name in robot_names:
    hand = create_hand_model(robot_name, torch.device('cpu'), 8192)
    hands[robot_name] = hand
    dataset_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', f'teapot_oakink_dataset_standard_all_retarget_to_{robot_name}_contact_map_eu.pt')
    # dataset_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', f'oakink_dataset_standard_all_retarget_to_{robot_name}_valid_dro.pt')  
    metadata = torch.load(dataset_path, map_location=torch.device('cpu'))['metadata']
    for object_name in object_names:
        robot_metadata[robot_name][object_name] = [item for item in metadata if item[5] == object_name and item[6] == robot_name]


def get_object_mesh(object_name, object_id, scale_factor=1.0):
    name = object_name.split('+')
    obj_mesh_path = list(
    glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.obj')) +
    glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.ply')))
    assert len(obj_mesh_path) == 1
    object_path = obj_mesh_path[0]

    if object_path.endswith('.ply'):
        object_trimesh = o3d.io.read_triangle_mesh(object_path)
        vertices = np.asarray(object_trimesh.vertices)
        triangles = np.asarray(object_trimesh.triangles)
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
        vertices = np.asarray(vertices - bbox_center)            
        object_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        
    else:  
        object_trimesh = trimesh.load(object_path, process=False, force='mesh', skip_materials=True)
        bbox_center = (object_trimesh.vertices.min(0) + object_trimesh.vertices.max(0)) / 2
        object_trimesh.vertices -= bbox_center
        object_trimesh= trimesh.Trimesh(vertices=object_trimesh.vertices, faces=object_trimesh.faces, process=False)

    return object_trimesh.apply_scale(float(scale_factor))


def get_hand_point_cloud(hand:HandModel, q, num_points=512):
    sampled_pc, _ = hand.get_sampled_pc(q, num_points=num_points)
    sampled_pc = sampled_pc.cpu().numpy()
    return sampled_pc[:, :3]

def get_object_point_cloud(object_id, object_pcs, num_points=512, random=False, scale_factor=1.0):
    if object_id not in object_pcs:
        print(f'Object {object_id} not found!')
        return None

    indices = torch.randperm(65536)[:num_points]
    object_pc = np.array(object_pcs[object_id])
    object_pc = object_pc[indices]
    object_pc = torch.tensor(object_pc)
    # change track array to numpy array
    if random:
        object_pc += torch.randn(object_pc.shape) * 0.002
    object_pc = object_pc * float(scale_factor)
    object_pc = object_pc.numpy()

    return object_pc


def get_full_object_point_cloud(object_id, object_pcs, scale_factor=1.0):
    if object_id not in object_pcs:
        print(f'Object {object_id} not found!')
        return None

    object_pc = np.array(object_pcs[object_id])
    object_pc = object_pc * float(scale_factor)

    return object_pc

def on_update(robot_idx, object_idx, grasp_idx, num_points=512, gamma=1.0, method='important'):
    robot_name = robot_names[robot_idx]
    object_name = object_names[object_idx]    
    metadata_curr = robot_metadata[robot_name][object_name]

    grasp_item = metadata_curr[grasp_idx % len(metadata_curr)]
    object_name = grasp_item[5]
    object_id = grasp_item[7]
    scale_factor = grasp_item[8] if isinstance(grasp_item[8], float) else 1.0
    q = grasp_item[3]

    hand = hands[robot_name]
    hand_pc = get_hand_point_cloud(hand, q, num_points)
    object_pc_normals = get_full_object_point_cloud(object_id, point_cloud_dataset, scale_factor=scale_factor)

    obj_normals = object_pc_normals[:, 3:]
    obj_pc = object_pc_normals[:, :3]

    # Check if contact_map already exists in metadata
    if isinstance(grasp_item[-1], np.ndarray) and grasp_item[-1].dtype == np.uint8:
        contact_map = grasp_item[-1].astype(np.float32) / 255.0
        print("[Info] Loaded precomputed contact map.")
    else:
        # Fallback to computing contact map
        obj_tensor = torch.from_numpy(obj_pc).unsqueeze(0).float()
        normals_tensor = torch.from_numpy(obj_normals).unsqueeze(0).float()
        hand_tensor = torch.from_numpy(hand_pc).unsqueeze(0).float()

        distance, distance_idx = get_euclidean_distance(obj_tensor, hand_tensor)
        contact_map = get_contact_map(distance)
        contact_map = contact_map[0].cpu().numpy()
        print("[Info] Computed contact map on the fly.")

    # Random get number of object points
    if method == "full":
        pass

    elif method == "random":
        indices = torch.randperm(contact_map.shape[0])[:num_points]
        contact_map = contact_map[indices]
        obj_pc = obj_pc[indices]
        threshold_mask = contact_map > 0
        high_indices = np.where(threshold_mask)[0]
        print(f"[Info] {len(high_indices)} points above threshold.")
        
    elif method == "highest":
        indices = torch.randperm(contact_map.shape[0])[:2048]
        contact_map = contact_map[indices]
        obj_pc = obj_pc[indices]

        threshold_mask = contact_map > 0.1
        high_indices = np.where(threshold_mask)[0]
        print(f"[Info] {len(high_indices)} points above threshold.")
        high_values = contact_map[high_indices]

        sorted_idx = np.argsort(-high_values)  
        high_indices = high_indices[sorted_idx]

        np.random.shuffle(high_indices)

        if len(high_indices) >= num_points // 2:
            high_indices = high_indices[:num_points // 2]
            print(f"[Info] {len(high_indices)} points selected from high indices.")
       
        low_indices = np.where(~threshold_mask)[0]
        supplement = np.random.choice(low_indices, min(num_points - len(high_indices), len(low_indices)), replace=False)
        indices = np.concatenate([high_indices, supplement])

        contact_map = contact_map[indices]
        obj_pc = obj_pc[indices]

    elif method == "balanced":

        indices = torch.randperm(contact_map.shape[0])[:2048]
        contact_map = contact_map[indices]
        obj_pc = obj_pc[indices]

        high_mask = contact_map > 0.2
        high_idx  = np.where(high_mask)[0]
        print(f"[Info] {len(high_idx)} points above threshold.")
        low_idx   = np.where(~high_mask)[0]

        half = num_points * 3 // 4

        if len(high_idx) >= half:
            high_pick = np.random.choice(high_idx, half, replace=False)
        else:
            high_pick = high_idx 

        need = num_points - len(high_pick)
        if need > 0 and len(low_idx) > 0:
            low_pick = np.random.choice(low_idx, min(need, len(low_idx)), replace=False)
        else:
            low_pick = np.array([], dtype=int)

        indices = np.concatenate([high_pick, low_pick])
        np.random.shuffle(indices)

        contact_map = contact_map[indices]
        obj_pc      = obj_pc[indices]

    elif method == "important":
        indices = torch.randperm(contact_map.shape[0])[:2048]
        contact_map = contact_map[indices]
        obj_pc = obj_pc[indices]
        print(contact_map == 0)
        
        probs = contact_map.copy()
        probs = np.maximum(probs, 0.002)  
        
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones_like(probs) / len(probs)
        
        selected_indices = np.random.choice(
            np.arange(len(contact_map)), 
            size=min(num_points, len(contact_map)),
            replace=False,
            p=probs
        )
        
        contact_map = contact_map[selected_indices]
        obj_pc = obj_pc[selected_indices]
        
        print(f"[Info] Importance sampling selected {len(selected_indices)} points.")



    cmap_rgba = plt.get_cmap('jet')(contact_map)
    colors = (cmap_rgba[:, :3] * 255).astype(np.uint8)

    # Visualization
    server.scene.reset()
    server.scene.add_point_cloud(
        'hand_pc',
        hand_pc,
        point_size=0.001,
        point_shape="circle",
        colors=(102, 192, 255)
    )
    server.scene.add_mesh_simple(
        'hand_mesh',
        hand.get_trimesh_q(q)['visual'].vertices,
        hand.get_trimesh_q(q)['visual'].faces,
        color=(239, 132, 167),
        opacity=0.5,
    )
    server.scene.add_mesh_simple(
        'object_mesh',
        get_object_mesh(object_name, object_id, scale_factor).vertices,
        get_object_mesh(object_name, object_id, scale_factor).faces,
        color=(102, 192, 255),
        opacity=0.5,
    )
    server.scene.add_point_cloud(
        'contact_map',
        obj_pc,
        point_size=0.001,
        point_shape='circle',
        colors=colors
    )

def update_visualization(_):
    on_update(robot_slider.value, object_slider.value, grasp_slider.value)

server = viser.ViserServer(host='127.0.0.1', port=8080)
robot_slider = server.gui.add_slider(
    label='Robot',
    min=0,
    max=len(robot_names) - 1,
    step=1,
    initial_value=0
)
object_slider = server.gui.add_slider(
    label='Object',
    min=0,
    max=len(object_names) - 1,
    step=1,
    initial_value=0
)
grasp_slider = server.gui.add_slider(
    label='Grasp',
    min=0,
    max=199,  
    step=1,
    initial_value=0
)

robot_slider.on_update(update_visualization)
object_slider.on_update(update_visualization)
grasp_slider.on_update(update_visualization)

while True:
    time.sleep(1)