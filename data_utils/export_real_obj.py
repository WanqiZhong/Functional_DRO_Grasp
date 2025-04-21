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

from utils.hand_model import create_hand_model, HandModel

robot_names = ['leaphand']
object_names = [
    "oakink+lotion_pump",
    "oakink+cylinder_bottle",
    "oakink+mug",
    "oakink+teapot",
    "oakink+bowl",
    "oakink+cup",
    "oakink+knife",
    "oakink+pen",
    "oakink+bottle",
    "oakink+headphones"
]

grasp_indices = [0, 94, 135, 51, 121, 93, 93, 106, 106, 85]

dataset_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_dataset_standard_all_retarget_to_leaphand.pt')  
# point_cloud_dataset = torch.load(os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_object_pcs.pt'))
metadata = torch.load(dataset_path, map_location=torch.device('cpu'))['metadata']

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

output_dir = os.path.join(ROOT_DIR, 'real_object')
os.makedirs(output_dir, exist_ok=True)


for object_name, grasp_idx in zip(object_names, grasp_indices):
    metas = [m for m in metadata if m[5] == object_name and m[6] == 'leaphand']
    if not metas:
        print(f"[Warning] No metadata for {object_name}")
        continue

    if grasp_idx >= len(metas):
        print(f"[Warning] Grasp index {grasp_idx} out of range for {object_name}, using last available.")
        grasp_item = metas[-1]
    else:
        grasp_item = metas[grasp_idx]

    object_id    = grasp_item[7]
    scale_factor = float(grasp_item[8])

    mesh = get_object_mesh(object_name, object_id, scale_factor)

    safe_name = object_name.replace('+', '_')
    filename = f"{safe_name}_{object_id}.obj"
    save_path = os.path.join(output_dir, filename)

    mesh.export(save_path, file_type='obj')
    print(f"Saved: {save_path}")