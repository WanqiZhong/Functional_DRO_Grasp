import os
import sys
import time
import torch
import viser
import numpy as np
import open3d as o3d
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model, HandModel

robot_names = ['mano']
object_names = [
    "oakink+teapot",
]

dataset_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_dataset_standard_all.pt')  
point_cloud_dataset = torch.load(os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_object_pcs.pt'))
point_cloud_path = torch.load(os.path.join(ROOT_DIR, 'data', 'PointCloud', 'oakink'))
metadata = torch.load(dataset_path, map_location=torch.device('cpu'))['metadata']

def get_hand_point_cloud(hand:HandModel, q, num_points=512):
    sampled_pc, _ = hand.get_sampled_pc(q, num_points=num_points)
    sampled_pc = sampled_pc.cpu().numpy()
    return sampled_pc[:, :3]

def get_object_point_cloud(object_name, object_pcs, num_points=512, random=False):
    if object_name not in object_pcs:
        print(f'Object {object_name} not found!')
        return None

    indices = torch.randperm(65536)[:num_points]
    object_pc = np.array(object_pcs[object_name])
    object_pc = object_pc[indices]
    object_pc = torch.tensor(object_pc)
    # change track array to numpy array
    if random:
        object_pc += torch.randn(object_pc.shape) * 0.002
    object_pc = object_pc.numpy()
    return object_pc

def on_update(robot_idx, object_idx, grasp_idx, num_points=512):

    robot_name = robot_names[robot_idx]
    object_type = object_names[object_idx]    
    metadata_curr = [m for m in metadata if m[5] == object_type and m[6] == robot_name]
    if len(metadata_curr) == 0:
        print('No metadata found!')
        return
    
    grasp_item = metadata_curr[grasp_idx % len(metadata_curr)]
    object_type = grasp_item[5]
    object_name = grasp_item[7]
    q = grasp_item[3]
    intent = grasp_item[4]
    print(f"Joint values: {q}")
    print(f"Object type: {object_type}")
    print(f"Object name: {object_name}")
    print(f"Intent: {intent}")
    
    hand = create_hand_model(robot_name, torch.device('cpu'), num_points)
    hand_pc = get_hand_point_cloud(hand, q)
    object_pc = get_object_point_cloud(object_name, point_cloud_dataset)
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