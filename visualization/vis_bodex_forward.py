import os
import sys
import time
import trimesh
import numpy as np
import viser
import torch
from utils.hand_model import create_hand_model, HandModel
from utils.rotation import quaternion_to_euler, matrix_to_euler
import pickle

# BODEX_GRASP_DATA_PATH = "/data/zwq/code/BODex/src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata/binoculars_o42103/scale100_pose000_grasp.npy"
GRASP_DATA_PATH = "/data/zwq/code/DexGraspBench/output/origin_dro_gold_eval_500_shadow/evaluation/contactdb/apple/0.npy"
# DRO_OBJECT_PATH = "/data/zwq/code/DRO_Grasp/data/data_urdf/object/oakink/teapot/coacd_C13001.obj"
HAND_OBJECT_PATH = "/data/zwq/code/DexGraspBench/output/origin_dro_gold_eval_500_shadow/vis_obj/contactdb/apple/0_obj.obj"
OBJECT_PATH = "/data/zwq/code/DexGraspBench/output/origin_dro_gold_eval_500_shadow/vis_obj/contactdb/apple/0_grasp_0.obj"
# OAKINK_PATH = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_shadowhand.pt"
OAKINK_PATH = "/data/zwq/code/DRO_Grasp/output/model_origin/res_diffuser_dro_predict_q.pkl"

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


def apply_transform_to_points(points, transform):
    """应用变换矩阵到点集"""
    # 转换为齐次坐标
    homogeneous_points = np.ones((points.shape[0], 4))
    homogeneous_points[:, :3] = points
    
    # 应用变换
    transformed_points = (transform @ homogeneous_points.T).T
    
    # 转回3D坐标
    return transformed_points[:, :3]

def torch_quaternion_to_matrix(quaternions):

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

def rotation_matrix_z(angle_rad):
    """创建Z轴旋转矩阵"""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

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

def correct_shadowhand_wrist_2(robot_pose, wrj2_angle=0.0):

    position = robot_pose[:3]
    quat = robot_pose[3:7]
    
    R_palm = torch_quaternion_to_matrix(torch.tensor(quat)).cpu().numpy()

    R_wrj2_axis = rotation_matrix_y(wrj2_angle)
    t_wrj2 = np.array([0.00, -0.01, 0.21301])
    
    T_wrj2 = transform_matrix(t_wrj2, R_wrj2_axis)
    
    global_transform = np.eye(4)
    global_transform[:3, :3] = R_palm
    
    T_wrist_to_forearm = global_transform @ T_wrj2
    
    corrected_position = position - T_wrist_to_forearm[:3, 3]
    
    return np.concatenate([corrected_position, quat, robot_pose[7:]])

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
    
    # total_offset = T_wrist_to_forearm[:3, 3] - position    
    corrected_position = position - T_wrist_to_forearm[:3, 3]
    # corrected_position = position - T_palm_to_wrist[:3, 3]

    # T_palm = transform_matrix(position, R_palm)
    
    # 计算从palm到forearm的完整变换
    # T_palm_to_forearm = T_palm @ T_wrj1 @ T_wrj2
    
    # 计算校正变换 (从Mujoco模型转换到URDF模型)
    # 逆向应用变换，从palm回溯到forearm
    # T_correction = T_wrj1 @ T_wrj2
    
    # 应用校正变换
    # T_palm_corrected = T_palm @ T_correction
    
    # # 提取校正后的位置和旋转
    # corrected_position = T_palm_corrected[:3, 3]
    # corrected_rotation = T_palm_corrected[:3, :3]
    # corrected_quat = torch_matrix_to_quaternion(corrected_rotation)
    
    # return np.concatenate([corrected_position, corrected_quat])
    
    return np.concatenate([corrected_position, quat, robot_pose[7:]])

def main():
    # Create viser server
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    print("Viser server started at http://127.0.0.1:8080")
    
    # Load grasp data
    try:
        grasp_data = np.load(GRASP_DATA_PATH, allow_pickle=True).item()
        print(f"Loaded grasp data with keys: {list(grasp_data.keys())}")
    except Exception as e:
        print(f"Error loading grasp data: {e}")
        grasp_data = {}

    # Load BODex grasp data
    try:
        bodex_grasp_data = np.load(BODEX_GRASP_DATA_PATH, allow_pickle=True).item()
        print(f"Loaded BODex grasp data with keys: {list(bodex_grasp_data.keys())}")
    except Exception as e:
        print(f"Error loading BODex grasp data: {e}")
        bodex_grasp_data = {}
    
    # Create hand model
    # robot_name = "shadowhand"
    robot_name = "shadowhand"
    hand = create_hand_model(robot_name)
    print("hand: ", hand.get_joint_orders())
    
    # Extract and process grasp pose 
    q = None
    if 'grasp_data' in grasp_data:
        q = grasp_data['grasp_data']
    elif 'grasp_qpos' in grasp_data:
        q = grasp_data['grasp_qpos']
    elif 'squeeze_qpos' in grasp_data:
        q = grasp_data['squeeze_qpos']

    # Extract q grasp pose
    if robot_name == "shadowhand":
        # q = torch.tensor(q)
        # quaternion = quaternion_to_euler(q[3:7])
        # reorder_q = torch.cat([q[:3], quaternion, q[7:]], axis=-1)

        # bodex_q = torch.tensor(bodex_grasp_data['robot_pose'][0, 0, 1]) 
        # quaternion = quaternion_to_euler(torch.cat([bodex_q[4:7], bodex_q[3:4]]))
        # quaternion_2 = matrix_to_euler(torch_quaternion_to_matrix(bodex_q[3:7]))
        # reorder_bodex_q = torch.cat([bodex_q[:3], quaternion_2, bodex_q[12:], bodex_q[7:12]], axis=-1)
        # reorder_bodex_q = reorder_bodex_q.float()

        q = torch.tensor(q)
        quaternion = quaternion_to_euler(torch.cat([q[4:7], q[3:4]]))
        q = torch.tensor(correct_shadowhand_wrist_2(q))
        reorder_q = torch.cat([q[:3], quaternion, torch.tensor([0.0, 0.0]), q[7:]], axis=-1)
        reorder_q = reorder_q.float()

        # Extract BODex grasp pose
        # bodex_q = bodex_grasp_data['robot_pose'][0, 0, 1]
        # bodex_q = torch.tensor(correct_shadowhand_wrist(bodex_q))
        # quaternion = quaternion_to_euler(torch.cat([bodex_q[4:7], bodex_q[3:4]]))
        # reorder_bodex_q = torch.cat([bodex_q[:3], quaternion, torch.tensor([0.0, 0.0]), bodex_q[12:], bodex_q[7:12]], axis=-1)
        # reorder_bodex_q = reorder_bodex_q.float()
        reorder_bodex_q = None


        # oakink_q = torch.load(OAKINK_PATH)[0][3]
        # oakink_q = None
        pkl_data = pickle.load(open(OAKINK_PATH, 'rb'))['results']
        oakink_q = pkl_data[3]['predict_q']

    elif robot_name == "test_shadowhand":
        q = torch.tensor(q)
        reorder_q = torch.cat([q[:7], torch.tensor([0.0, 0.0]), q[7:]], axis=-1)

        # Extract BODex grasp pose
        bodex_q = torch.tensor(bodex_grasp_data['robot_pose'][0, 0, 1])
        bodex_q = torch.tensor(correct_shadowhand_wrist(bodex_q))
        reorder_bodex_q = torch.cat([bodex_q[:7], torch.tensor([0.0, 0.0]), bodex_q[12:], bodex_q[7:12]], axis=-1).float()
    
    # Get object scale and pose if available
    obj_scale = grasp_data.get('obj_scale', 1.0)
    obj_pose = grasp_data.get('obj_pose', None)
    
    # Setup scene
    visualize_scene(server, hand, reorder_q, reorder_bodex_q, oakink_q, robot_name, obj_scale, obj_pose)
    # Keep the server running
    while True:
        time.sleep(1)

def visualize_scene(server, hand: HandModel, q, bodex_q, oakink_q, robot_name, obj_scale=1.0, obj_pose=None):
    """Visualize the hand and objects in the scene"""
    # Clear previous visualization if any
    server.scene.reset()
    
    if robot_name == "shadowhand":
        # Visualize hand model if q is available
        if q is not None:
            try:
                robot_trimesh = hand.get_trimesh_q(q)["visual"]
                
                server.scene.add_mesh_simple(
                    'robot',
                    robot_trimesh.vertices,
                    robot_trimesh.faces,
                    color=(102, 192, 255),
                    opacity=0.8
                )
                print("Added hand model to scene")
            except Exception as e:
                print(f"Error visualizing hand model: {e}")
        else:
            print("Warning: No grasp pose data found")

        # Visualize BODex hand model if bodex_q is available
        if bodex_q is not None:
            try:
                robot_trimesh = hand.get_trimesh_q(bodex_q)["visual"]
                
                server.scene.add_mesh_simple(
                    'bodex_robot',
                    robot_trimesh.vertices,
                    robot_trimesh.faces,
                    color=(102, 192, 255),
                    opacity=0.8
                )
                print("Added BODex hand model to scene")    
            except Exception as e:
                print(f"Error visualizing BODex hand model: {e}")
        else:
            print("Warning: No BODex grasp pose data found")

        if oakink_q is not None:
            try:
                robot_trimesh = hand.get_trimesh_q(oakink_q)["visual"]
                server.scene.add_mesh_simple(
                    'oakink_robot',
                    robot_trimesh.vertices, 
                    robot_trimesh.faces,
                    color=(102, 192, 255),
                    opacity=0.8
                )
                print("Added Oakink hand model to scene")
            except Exception as e:
                print(f"Error visualizing Oakink hand model: {e}")
        else:
            print("Warning: No Oakink grasp pose data found")

    elif robot_name == "test_shadowhand":
        # Visualize hand model if q is available
        if q is not None:
            try:
                robot_trimesh = hand.get_trimesh_q(q[7:])["visual"]
                robot_trimesh.apply_translation(q[:3])
                robot_trimesh.apply_transform(
                    trimesh.transformations.quaternion_matrix(q[3:7])
                )
                server.scene.add_mesh_simple(
                    'robot',
                    robot_trimesh.vertices,
                    robot_trimesh.faces,
                    color=(102, 192, 255),
                    opacity=0.8
                )
                print("Added hand model to scene")
            except Exception as e:
                print(f"Error visualizing hand model: {e}") 
        else:
            print("Warning: No grasp pose data found")

        # Visualize BODex hand model if bodex_q is available
        if bodex_q is not None:
            try:
                robot_trimesh = hand.get_trimesh_q(bodex_q[7:])["visual"]
                robot_trimesh.apply_translation(bodex_q[:3])
                robot_trimesh.apply_transform(
                    trimesh.transformations.quaternion_matrix(bodex_q[3:7])
                )
                server.scene.add_mesh_simple(
                    'bodex_robot',
                    robot_trimesh.vertices,
                    robot_trimesh.faces,
                    color=(102, 192, 255),
                    opacity=0.8
                )
                print("Added BODex hand model to scene")
            except Exception as e:
                print(f"Error visualizing BODex hand model: {e}")
        else:
            print("Warning: No BODex grasp pose data found")

    # Load and visualize object mesh
    try:
        object_trimesh = trimesh.load_mesh(OBJECT_PATH)
        
        # Apply object scale if available
        if obj_scale != 1.0:
            object_trimesh.apply_scale(obj_scale)
        
        # Apply object pose if available
        if obj_pose is not None:
            # Assuming obj_pose is [position, quaternion]
            if len(obj_pose) >= 7:  # Has position and orientation
                pos = obj_pose[:3]
                quat = obj_pose[3:7]
                object_trimesh.apply_transform(
                    trimesh.transformations.quaternion_matrix(quat)
                )
                object_trimesh.apply_translation(pos)
        
        server.scene.add_mesh_simple(
            'object',
            object_trimesh.vertices,
            object_trimesh.faces,
            color=(239, 132, 167),
            opacity=1
        )
        print("Added object mesh to scene")
    except Exception as e:
        print(f"Error loading object mesh: {e}")
    
    # Load and visualize hand-object interaction mesh
    try:
        hand_object_trimesh = trimesh.load_mesh(HAND_OBJECT_PATH)
        server.scene.add_mesh_simple(
            'hand_object',
            hand_object_trimesh.vertices,
            hand_object_trimesh.faces,
            color=(192, 192, 255),
            opacity=0.6
        )
        print("Added hand-object interaction mesh to scene")
    except Exception as e:
        print(f"Error loading hand-object mesh: {e}")

    # Load and visualize object mesh
    try:
        dro_object_trimesh = trimesh.load_mesh(DRO_OBJECT_PATH)
        server.scene.add_mesh_simple(
            'dro_object',
            dro_object_trimesh.vertices,
            dro_object_trimesh.faces,
            color=(239, 132, 167),
            opacity=1
        )
        print("Added DRO object mesh to scene")
    except Exception as e:
        print(f"Error loading DRO object mesh: {e}")
    

if __name__ == "__main__":
    main()