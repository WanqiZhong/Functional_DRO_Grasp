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

def torch_quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    quaternions = torch.as_tensor(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def rotation_matrix_y(angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

def rotation_matrix_x(angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])

def transform_matrix(translation, rotation_matrix):
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform

def correct_shadowhand_wrist(robot_pose, wrj1_angle=0.0, wrj2_angle=0.0):

    position = robot_pose[:3]
    quat = robot_pose[3:7]
    
    R_palm = torch_quaternion_to_matrix(torch.tensor(quat)).cpu().numpy()

    R_wrj1_axis = rotation_matrix_x(wrj1_angle)
    t_wrj1 = np.array([0.0, 0.0, 0.034])

    R_wrj2_axis = rotation_matrix_y(wrj2_angle)
    t_wrj2 = np.array([0.00, -0.01, 0.21301])
    
    T_wrj1 = transform_matrix(t_wrj1, R_wrj1_axis)
    T_wrj2 = transform_matrix(t_wrj2, R_wrj2_axis)
    
    global_transform = np.eye(4)
    global_transform[:3, :3] = R_palm
    
    T_palm_to_wrist = global_transform @ T_wrj1
    T_wrist_to_forearm = T_palm_to_wrist @ T_wrj2
    
    corrected_position = position - T_wrist_to_forearm[:3, 3]
    
    return np.concatenate([corrected_position, quat, robot_pose[7:]])

# Configuration
# Change these paths to match your dataset locations
ORIGINAL_DATASET_PATH = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand.pt'
CONVERTED_DATASET_PATH = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand_mujoco.pt'

# ORIGINAL_DATASET_PATH = os.path.join(ROOT_DIR, '/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_shadowhand.pt')
# CONVERTED_DATASET_PATH = os.path.join(ROOT_DIR, '/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_shadowhand_mujoco.pt')

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

# Load datasets
print(f"Loading original dataset from {ORIGINAL_DATASET_PATH}")
original_data = torch.load(ORIGINAL_DATASET_PATH, map_location=torch.device('cpu'))
original_metadata = original_data['metadata']

print(f"Loading converted dataset from {CONVERTED_DATASET_PATH}")
converted_data = torch.load(CONVERTED_DATASET_PATH, map_location=torch.device('cpu'))
converted_metadata = converted_data['metadata']

print(f"Loaded {len(original_metadata)} items from dataset")

# Create hand model
hand = create_hand_model('shadowhand')

# Setup visualization server
server = viser.ViserServer(host='127.0.0.1', port=8080)

def on_update(item_idx):
    """Update visualization when slider changes"""
    # Clear previous visualization
    server.scene.reset()
    
    # Get current item from both datasets
    original_item = original_metadata[item_idx]
    converted_item = converted_metadata[item_idx]
    
    # Extract relevant information
    original_robot_value = original_item[3]  # Original robot value (hand pose)
    converted_robot_value = converted_item[3]  # Converted robot value (hand pose)
    object_key = original_item[5]  # e.g., 'oakink+teapot'
    object_id = original_item[7]  # e.g., 'C90001'

    converted_robot_value = torch.tensor(correct_shadowhand_wrist(converted_robot_value))
    euler = quaternion_to_euler(torch.cat([converted_robot_value[4:7], converted_robot_value[3:4]]))
    converted_robot_value = torch.cat([converted_robot_value[:3], euler, torch.tensor([0.0, 0.0]), converted_robot_value[12:], converted_robot_value[7:12]], axis=-1).float()

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
    original_hand_mesh = hand.get_trimesh_q(original_robot_value)["visual"]
    converted_hand_mesh = hand.get_trimesh_q(converted_robot_value)["visual"]
    
    # Add original hand mesh (left side)
    server.scene.add_mesh_simple(
        'original_hand',
        original_hand_mesh.vertices,  
        original_hand_mesh.faces,
        color=(102, 192, 255),  # Blue color
        opacity=0.8
    )
    
    # Add converted hand mesh (right side)
    server.scene.add_mesh_simple(
        'converted_hand',
        converted_hand_mesh.vertices,  
        converted_hand_mesh.faces,
        color=(255, 102, 102),  # Red color
        opacity=0.8
    )
    
# Add slider to control which item is displayed
item_slider = server.gui.add_slider(
    label='Dataset Item',
    min=0,
    max=len(original_metadata) - 1,
    step=1,
    initial_value=0
)

# Connect slider to update function
item_slider.on_update(lambda _: on_update(item_slider.value))

# Initial update
on_update(0)

print("Visualization server started at http://127.0.0.1:8080")
print("Use the slider to browse through different items")

# Keep server running
while True:
    time.sleep(1)