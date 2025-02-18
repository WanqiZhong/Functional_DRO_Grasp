import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
import time
import trimesh
import torch
import viser
from utils.hand_model import create_hand_model
import glob
import open3d as o3d
import numpy as np
import manotorch
from manotorch.manolayer import ManoLayer, MANOOutput

filtered = True


robot_names = ['shadowhand']
object_names = [
     "oakink+teapot",
]
hand = create_hand_model(robot_names[0])

if "oakink" in object_names[0]:
    if "mano" in robot_names[0]:
        dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all.pt')
    else:
        dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_names[0]}.pt')
elif filtered:
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
else:
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset/cmap_dataset.pt')
    
dataset = torch.load(dataset_path, map_location=torch.device('cpu'))
metadata = dataset['metadata']
mano_compatibility = dataset['version']['metadata'] in ['1.2.0', '1.2.0_COACD', '1.1.0']
assert 'mano' in robot_names and mano_compatibility, "if only mano is visualized, the dataset must be in mano format"

if mano_compatibility:
    mano_layer = ManoLayer(rot_mode="axisang",
                            center_idx=0,
                            mano_assets_root="assets/mano",
                            use_pca=False,
                            flat_hand_mean=True)

def on_update(robot_idx=0, object_idx=0, grasp_idx=0):
    robot_name = robot_names[robot_idx]
    object_name = object_names[object_idx]
    if "oakink" in object_name:
        if "mano" in robot_name:
            metadata_curr = [m  for m in metadata if m[5] == object_name and m[6] == robot_name]
        else:
            metadata_curr = [m  for m in metadata if m[5] == object_name and m[6] == robot_name]
    elif filtered:
        metadata_curr = [m[0] for m in metadata if m[1] == object_name and m[2] == robot_name]
    else:
        metadata_curr = [m[1] for m in metadata if m[2] == object_name and m[3] == robot_name]
    if len(metadata_curr) == 0:
        print('No metadata found!')
        return
    grasp_item = metadata_curr[grasp_idx % len(metadata_curr)]
    if "oakink" in object_name:
        object_id = grasp_item[7]
        hand_tsl = grasp_item[2].numpy()
        q = grasp_item[3]
    else:
        q = grasp_item

    print(f"joint values: {q}")
    print(f"object: {object_id}")
    print(f"intent: {grasp_item[4]}")

    robot_trimesh = hand.get_trimesh_q(q)["visual"]

    if "oakink" in object_name:
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
            vertices = o3d.utility.Vector3dVector(vertices - bbox_center)
            object_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        else:  
            object_trimesh = trimesh.load(object_path, process=False, force='mesh', skip_materials=True)
            bbox_center = (object_trimesh.vertices.min(0) + object_trimesh.vertices.max(0)) / 2
            object_trimesh.vertices -= bbox_center
        hand_faces = mano_layer.th_faces  
    else:
        name = object_name.split('+')
        object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')  # visual mesh
        # object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/coacd_allinone.obj')  # collision mesh
        object_trimesh = trimesh.load_mesh(object_path)




on_update()