import torch
import numpy as np
import os
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

# Paths
# CONVERTED_DATASET_PATH = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand_valid_bodex.pt'
# OUTPUT_DATASET_PATH = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand_valid_dro.pt'
CONVERTED_DATASET_PATH = '/data/zwq/code/BODex/sim_shadow/fc/oakink_multi_teapot_step_0001_max/graspdata/combined_results.pt'
OUTPUT_DATASET_PATH = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_teapot_retarget_to_shadowhand_no_valid_dro.pt'

# Load the dataset
print(f"Loading dataset from {CONVERTED_DATASET_PATH}")
converted_data = torch.load(CONVERTED_DATASET_PATH, map_location=torch.device('cpu'))
converted_metadata = converted_data['metadata']
# format_data = converted_data['version']['format']
# assert format_data == 'bodex', "Only bodex format is supported"

# Create a new metadata list
new_metadata = []

# Process each item
print("Processing items...")
for item_idx in range(len(converted_metadata)):
    # Get current item and its bodex items
    item_data = converted_metadata[item_idx]
    converted_item = item_data[0]
    bodex_item = item_data[1][0, 1, :]
    converted_item = list(converted_item)
        
    # Transform bodex item
    bodex_robot_value = torch.tensor(correct_shadowhand_wrist(bodex_item))
    euler = quaternion_to_euler(torch.cat([bodex_robot_value[4:7], bodex_robot_value[3:4]]))
    bodex_robot_value = torch.cat([
        bodex_robot_value[:3],            # First 3 values
        euler,                           # Euler angles
        torch.tensor([0.0, 0.0]),        # Two zeros
        bodex_robot_value[12:],          # From index 12 to end
        bodex_robot_value[7:12]          # From index 7 to 12
    ], axis=-1).float()
        
    converted_item[3] = bodex_robot_value
    converted_item = tuple(converted_item)
    new_metadata.append(converted_item)

# Replace the metadata in the converted_data
converted_data['metadata'] = new_metadata
converted_data['version']['format'] = 'dro'

# Save the modified dataset
print(f"Saving modified dataset to {OUTPUT_DATASET_PATH}")
torch.save(converted_data, OUTPUT_DATASET_PATH)
print("Conversion completed successfully.")