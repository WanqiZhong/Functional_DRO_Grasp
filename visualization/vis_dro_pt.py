import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import time
import trimesh
import torch
import viser
import glob
import open3d as o3d
import numpy as np
from utils.hand_model import create_hand_model
from utils.rotation import quaternion_to_euler

# CONVERTED_DATASET_PATH = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand_valid_dro.pt'
CONVERTED_DATASET_PATH = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_teapot_retarget_to_shadowhand_no_valid_dro.pt'

def load_object_mesh(object_key, object_id):
    """Load object mesh based on object key and ID"""
    name = object_key.split('+')
    obj_mesh_path = list(
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.obj')) +
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.ply')))
    
    if len(obj_mesh_path) == 0:
        print(f"Warning: No mesh found for {object_key} with ID {object_id}")
        return None
        
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
    
    return object_trimesh


print(f"Loading converted dataset from {CONVERTED_DATASET_PATH}")
converted_data = torch.load(CONVERTED_DATASET_PATH, map_location=torch.device('cpu'))
converted_metadata = converted_data['metadata']

print(f"Loaded {len(converted_metadata)} items from dataset")

# Create hand model
hand = create_hand_model('shadowhand')

# Setup visualization server
server = viser.ViserServer(host='127.0.0.1', port=8080)

def on_update(item_idx, bodex_idx):
    """Update visualization when slider changes"""
    # Clear previous visualization
    server.scene.reset()
    
    converted_item = converted_metadata[item_idx]
    
    converted_robot_value = converted_item[3]  # Converted robot value (hand pose)
    object_key = converted_item[5]  # e.g., 'oakink+teapot'
    object_id = converted_item[7]  # e.g., 'C90001'


    print(f"converted_robot_value aft: {converted_robot_value.shape}")
    print(f"Showing item {item_idx}: {object_key} (ID: {object_id})")
    
    # Get object mesh
    object_trimesh = load_object_mesh(object_key, object_id)
    if object_trimesh is not None:
        server.scene.add_mesh_simple(
            'object',
            object_trimesh.vertices,
            object_trimesh.faces,
            color=(239, 132, 167),
            opacity=1
        )
    
    # Create hand meshes for original and converted poses
    converted_hand_mesh = hand.get_trimesh_q(converted_robot_value)["visual"]

    # Add converted hand mesh (right side)
    server.scene.add_mesh_simple(
        'converted_hand',
        converted_hand_mesh.vertices,  
        converted_hand_mesh.faces,
        color=(192, 102, 255),  # Purple color
        opacity=0.8
    )

    # server.scene.add_label(
    #     'function',
    #     f'{converted_item[10]}',
    #     wxyz=(1, 0, 0, 0),
    #     position=(1, 1, 1)
    # )
    
# Add slider to control which item is displayed
item_slider = server.gui.add_slider(
    label='Dataset Item',
    min=0,
    max=len(converted_metadata) - 1,
    step=1,
    initial_value=0
)

bodex_slider = server.gui.add_slider(
    label='Bodex Item',
    min=0,
    max=39,
    step=1,
    initial_value=0
)


# Connect slider to update function 
item_slider.on_update(lambda _: on_update(item_slider.value, bodex_slider.value))
bodex_slider.on_update(lambda _: on_update(item_slider.value, bodex_slider.value))

# Initial update
on_update(0, 0)

print("Visualization server started at http://127.0.0.1:8080")
print("Use the slider to browse through different items")

# Keep server running
while True:
    time.sleep(1)