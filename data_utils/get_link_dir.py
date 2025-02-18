import os
import torch
from urdfpy import URDF
import numpy as np
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def extract_link_dirs(urdf_path, robot_name):
    robot = URDF.load(urdf_path)
    
    link_dirs = {}
    
    for joint in robot.joints:
        if joint.joint_type == 'fixed':
            continue  
        
        axis = np.array(joint.axis)  # 获取关节的轴向        
        origin = joint.origin
        if origin is not None:
            rotation = origin[:3, :3]  # 提取旋转矩阵 R
        else:
            raise ValueError("Origin is None!")
        
        axis_world = rotation @ axis
        axis_world_normalized = axis_world / np.linalg.norm(axis_world)
        
        link_dir = torch.tensor(axis_world_normalized, dtype=torch.float32)
        link_dirs[joint.name] = link_dir
    
    return link_dirs

if __name__ == "__main__":
    urdf_path = os.path.join(ROOT_DIR, 'data/data_urdf/robot/mano_v2/mano_description_extended.urdf')  
    robot_name = 'mano'
    link_dirs = extract_link_dirs(urdf_path, robot_name)
    
    for joint_name, dir_vector in link_dirs.items():
        print(f"Joint: {joint_name}, Link Direction: {dir_vector.tolist()}")