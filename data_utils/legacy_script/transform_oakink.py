import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import os
from tqdm import tqdm

# URDF joint order and rotation axes
urdf_joint_order = [
    'virtual_joint_x',    # 0
    'virtual_joint_y',    # 1
    'virtual_joint_z',    # 2
    'virtual_joint_roll', # 3
    'virtual_joint_pitch',# 4
    'virtual_joint_yaw',  # 5
    'j_thumb1y',          # 6
    'j_thumb1z',          # 7
    'j_thumb2',           # 8
    'j_thumb3',           # 9
    'j_index1y',          #10
    'j_index1x',          #11
    'j_index2',           #12
    'j_index3',           #13
    'j_middle1y',         #14
    'j_middle1x',         #15
    'j_middle2',          #16
    'j_middle3',          #17
    'j_ring1y',           #18
    'j_ring1x',           #19
    'j_ring2',            #20
    'j_ring3',            #21
    'j_pinky1y',          #22
    'j_pinky1x',          #23
    'j_pinky2',           #24
    'j_pinky3'            #25
]

# Joint axis mapping
joint_axis_map = {
    'virtual_joint_x': 'x',
    'virtual_joint_y': 'y',
    'virtual_joint_z': 'z',
    'virtual_joint_roll': 'x',
    'virtual_joint_pitch': 'y',
    'virtual_joint_yaw': 'z',
    'j_thumb1y': 'y',
    'j_thumb1z': 'z',
    'j_thumb2': 'x',
    'j_thumb3': 'x',
    'j_index1y': 'y',
    'j_index1x': 'x',
    'j_index2': 'x',
    'j_index3': 'x',
    'j_middle1y': 'y',
    'j_middle1x': 'x',
    'j_middle2': 'x',
    'j_middle3': 'x',
    'j_ring1y': 'y',
    'j_ring1x': 'x',
    'j_ring2': 'x',
    'j_ring3': 'x',
    'j_pinky1y': 'y',
    'j_pinky1x': 'x',
    'j_pinky2': 'x',
    'j_pinky3': 'x'
}


def transform_oakink(source_file):
    dataset = torch.load(source_file)
    metadata = dataset['metadata']

    new_metadata = []

    # Iterate over each sample in metadata
    for idx, sample in enumerate(tqdm(metadata)):
        pose = sample[0].numpy()    # (48,)
        shape = sample[1].numpy()   # Unused
        tsl = sample[2].numpy()     # (3,)

        # Check pose shape
        if pose.shape != (48,):
            print(f"Sample {idx} has pose shape {pose.shape}, expected (48,), skipping.")
            continue

        pose = pose.reshape(16, 3)  # (16, 3)

        q_virtual_prismatic = tsl  # (3,)
        global_axis_angle = pose[0]  # (3,)
        global_rot = R.from_rotvec(global_axis_angle)
        global_euler = global_rot.as_euler('xyz', degrees=False)  # (3,)
        q_virtual_revolute = global_euler  # (3,)

        q = np.concatenate([q_virtual_prismatic, q_virtual_revolute])  # (6,)

        joint_angles = []

        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        finger_pose_indices = {
            'thumb': range(1, 4),
            'index': range(4, 7),
            'middle': range(7, 10),
            'ring': range(10, 13),
            'pinky': range(13, 16)
        }

        for finger in finger_names:
            finger_axis_angles = pose[list(finger_pose_indices[finger])]  # (3, 3)
            finger1_axis_angle = finger_axis_angles[0]
            finger1_rot = R.from_rotvec(finger1_axis_angle)
            if finger == 'thumb':
                finger1_euler = finger1_rot.as_euler('xyz', degrees=False)
                thumb1y_angle = finger1_euler[1]  # Rotation around y-axis
                thumb1z_angle = finger1_euler[2]  # Rotation around z-axis
                joint_angles.extend([thumb1y_angle, thumb1z_angle])
                for i in range(1, 3):
                    joint_rot = R.from_rotvec(finger_axis_angles[i])
                    joint_euler = joint_rot.as_euler('xyz', degrees=False)
                    angle = joint_euler[0] 
                    joint_angles.append(angle)
            else:
                finger1_euler = finger1_rot.as_euler('xyz', degrees=False)
                finger1y_angle = finger1_euler[1]  # Rotation around y-axis
                finger1x_angle = finger1_euler[0]  # Rotation around x-axis
                joint_angles.extend([finger1y_angle, finger1x_angle])
                for i in range(1, 3):
                    joint_rot = R.from_rotvec(finger_axis_angles[i])
                    joint_euler = joint_rot.as_euler('xyz', degrees=False)
                    angle = joint_euler[0]  
                    joint_angles.append(angle)

        q = np.concatenate([q, joint_angles])  # (26,)

        if q.shape[0] != 26:
            print(f"Sample {idx} q vector length {q.shape[0]}, expected 26, skipping.")
            continue

        new_metadata.append((
            q, 
            sample[3],  # intent
            sample[4],  # object_name
            sample[5],  # robot_name
            sample[6],  # object_id
            sample[7]   # parent_object_id
        ))

        new_dataset = {
            'info': dataset['info'],
            'metadata': new_metadata
        }

        torch.save(new_dataset, os.path.join(os.path.dirname(source_file),'modified_oakink_dataset.pt')) 

if __name__ == "__main__":
    transform_oakink("/data/zwq/code/DRO-Grasp/data/OakInkDataset/oakink_dataset.pt")