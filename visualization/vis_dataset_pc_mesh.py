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

dataset_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_dataset_standard_all_retarget_to_leaphand.pt')  
point_cloud_dataset = torch.load(os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_object_pcs.pt'))
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

def on_update(robot_idx, object_idx, grasp_idx, num_points=512):

    robot_name = robot_names[robot_idx]
    object_name = object_names[object_idx]    
    metadata_curr = [m for m in metadata if m[5] == object_name and m[6] == robot_name]
    if len(metadata_curr) == 0:
        print('No metadata found!')
        return
    
    grasp_item = metadata_curr[grasp_idx % len(metadata_curr)]
    object_name = grasp_item[5]
    object_id = grasp_item[7]
    scale_factor = grasp_item[8]
    q = grasp_item[3]

    print(f"Joint values: {q}")
    print(f"Object key: {object_name}")
    print(f"Object id: {object_id}")
    
    hand = create_hand_model(robot_name, torch.device('cpu'), num_points)

    hand_pc = get_hand_point_cloud(hand, q)
    object_pc = get_object_point_cloud(object_id, point_cloud_dataset, scale_factor=scale_factor)

    hand_mesh = hand.get_trimesh_q(q)['visual']
    object_mesh = get_object_mesh(object_name, object_id, scale_factor)

    if object_pc is None: 
        return
    
    server.scene.add_point_cloud(
        'object_pc',
        object_pc,
        point_size=0.002,
        point_shape="circle",
        colors=(239, 132, 167)
    )
    
    server.scene.add_point_cloud(
        'hand_pc',
        hand_pc,  
        point_size=0.002,
        point_shape="circle",
        colors=(102, 192, 255)
    )

    server.scene.add_mesh_simple(
        'hand_mesh',
        hand_mesh.vertices,
        hand_mesh.faces,
        color=(239, 132, 167),
        opacity=1,
    )

    server.scene.add_mesh_simple(
        'object_mesh',
        object_mesh.vertices,
        object_mesh.faces,
        color=(102, 192, 255),
        opacity=1,
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