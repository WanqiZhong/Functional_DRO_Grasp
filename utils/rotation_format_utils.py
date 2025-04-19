from DRO_Grasp.utils.rotation_utils import torch_matrix_to_quaternion, torch_quaternion_to_matrix
import numpy as np
import torch
from DRO_Grasp.utils.rotation import transform_matrix, rotation_matrix_x, rotation_matrix_y, euler_to_matrix, matrix_to_euler
from scipy.spatial.transform import Rotation as R

def shadowhand_from_oakink_to_bodex(robot_pose, wrj1_angle=0.0, wrj2_angle=0.0):

    position = robot_pose[:3]
    euler = robot_pose[3:6]
    
    R_palm = euler_to_matrix(torch.tensor(euler).float()).cpu().numpy()

    R_wrj1_axis = rotation_matrix_x(wrj1_angle)
    t_wrj1 = np.array([0.0, 0.0, 0.034])  

    R_wrj2_axis = rotation_matrix_y(wrj2_angle)
    t_wrj2 = np.array([0.00, -0.01, 0.21301])
    
    T_wrj1 = transform_matrix(t_wrj1, R_wrj1_axis)
    T_wrj2 = transform_matrix(t_wrj2, R_wrj2_axis)
    
    global_transform = np.eye(4)
    global_transform[:3, :3] = R_palm
    global_transform[:3, 3] = position
    
    T_forearm_to_wrist = global_transform @ T_wrj2
    T_wrist_to_palm = T_forearm_to_wrist @ T_wrj1

    quat = torch_matrix_to_quaternion(torch.tensor(T_wrist_to_palm[:3, :3]))

    return np.concatenate([T_wrist_to_palm[:3, 3], quat, robot_pose[6:]])


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
    
    corrected_position = position - T_wrist_to_forearm[:3, 3]
    
    return np.concatenate([corrected_position, quat, robot_pose[7:]])



def leaphand_from_oakink_to_bodex(robot_pose: np.ndarray) -> np.ndarray:
    """
    Convert LeapHand pose from base-rooted URDF (e.g., DRO) to palm-rooted URDF (e.g., BODex),
    applying the fixed transform between base and palm_lower.

    Args:
        pose_euler (np.ndarray): A pose array of shape (N,) where:
            - pose[:3] is the translation (x, y, z) of the base in world coordinates.
            - pose[3:6] is the orientation in Euler angles (roll, pitch, yaw) in radians.
            - pose[6:] are the remaining joint values.

    Returns:
        np.ndarray: Transformed pose array of shape (N,) where:
            - pose[:3] is the palm_lower position in world coordinates.
            - pose[3:7] is the orientation in quaternion (qw, qx, qy, qz).
            - pose[7:] are the same remaining joint values as input.
    """
    _offset_t = np.array([0.0, 0.038, 0.098])       
    _offset_R = rotation_matrix_y(-1.57079)         
    T_base_to_palm = transform_matrix(_offset_t, _offset_R)
    T_palm_to_base = np.linalg.inv(T_base_to_palm)

    pos  = robot_pose[:3]
    q    = robot_pose[3:6]    
    tail = robot_pose[6:]

    Rb = euler_to_matrix(torch.tensor(q)).cpu().numpy()  
    Tb = transform_matrix(pos, Rb)                       

    Tp = Tb @ T_base_to_palm
    new_pos  = Tp[:3, 3]
    new_R    = Tp[:3, :3]
    new_q    = torch_matrix_to_quaternion(torch.tensor(new_R)).cpu().numpy()

    return np.concatenate([new_pos, new_q, tail], axis=0)

def leaphand_from_bodex_to_oakink(pose_quat: np.ndarray) -> np.ndarray:
    """
    Convert LeapHand pose from palm-rooted URDF (e.g., BODex) to base-rooted URDF (e.g., DRO),
    applying the fixed transform between palm_lower and base.
    Args:
        pose_quat (np.ndarray): A pose array of shape (N,) where:
            - pose[:3] is the translation (x, y, z) of the palm_lower in world coordinates.
            - pose[3:7] is the orientation in quaternion (qw, qx, qy, qz).
            - pose[7:] are the remaining joint values.
    Returns:
        np.ndarray: Transformed pose array of shape (N,) where:
            - pose[:3] is the base position in world coordinates.
            - pose[3:6] is the euler angle.
            - pose[6:] are the same remaining joint values as input.
    """

    _offset_t = np.array([0.0, 0.038, 0.098])       
    _offset_R = rotation_matrix_y(-1.57079)         
    T_base_to_palm = transform_matrix(_offset_t, _offset_R)
    T_palm_to_base = np.linalg.inv(T_base_to_palm)

    pos  = pose_quat[:3]
    q    = pose_quat[3:7]
    tail = pose_quat[7:]

    Rp = torch_quaternion_to_matrix(torch.tensor(q)).cpu().numpy()
    Tp = transform_matrix(pos, Rp)

    Tb = Tp @ T_palm_to_base

    new_pos = Tb[:3, 3]
    new_R   = Tb[:3, :3]
    new_q   = matrix_to_euler(torch.tensor(new_R)).cpu().numpy()
    
    return np.concatenate([new_pos, new_q, tail], axis=0)


def leaphand_from_bodex_to_mujoco(robot_pose: np.ndarray) -> np.ndarray:
    """
    Convert LeapHand root pose from palm-rooted URDF (e.g., BODex) to Mujoco frame (e.g., DexGraspBench).

    This function applies the following transformation:
        - A 180-degree rotation around the X-axis applied to orientation.
        - A translation offset of 0.1m along Z in the local frame after rotation.

    Args:
        robot_pose (np.ndarray): A pose array of shape (7+J), where:
            - [:3] are the root positions (x, y, z) in world coordinates.
            - [3:7] are root orientations as quaternions (qw, qx, qy, qz).
            - [7:] are any remaining joint values.

    Returns:
        np.ndarray: The corrected pose array with:
            - Root pose transformed from palm_root URDF frame to Mujoco frame.
            - All other joint values unchanged.
    """

    R_palm = torch_quaternion_to_matrix(
        torch.tensor(robot_pose[3:7], dtype=torch.float32)
    )  
    delta_rot = torch_quaternion_to_matrix(
        torch.tensor([0, 1, 0, 0], dtype=torch.float32)
    )  

    R_mj = R_palm @ delta_rot.transpose(-1, -2)
    robot_pose[3:7] = torch_matrix_to_quaternion(R_mj).cpu().numpy()

    # Apply the translation: pos += R_mj @ [0, 0, 0.1]
    offset = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float32)
    world_offset = (R_mj @ offset).cpu().numpy()  
    robot_pose[:3] -= world_offset

    return robot_pose

def leaphand_from_mujoco_to_bodex(robot_pose: np.ndarray) -> np.ndarray:
    """
    Convert LeapHand root pose from Mujoco frame (eg. DexGraspBench) back to palm-rooted URDF (e.g., BODex)

    This function reverses the following transformation:
        - A 180-degree rotation around the X-axis applied to orientation.
        - A translation offset of 0.1m along Z in the local frame after rotation.

    Args:
        robot_pose (np.ndarray): A pose array of shape (7+J), where:
            - [:3] are the root positions (x, y, z) in world coordinates.
            - [3:7] are root orientations as quaternions (qw, qx, qy, qz).
            - [7:] are any remaining joint values.

    Returns:
        np.ndarray: The corrected pose array with:
            - Root pose transformed from Mujoco frame back to palm_root URDF frame.
            - All other joint values unchanged.
    """

    R_mj = torch_quaternion_to_matrix(
        torch.tensor(robot_pose[3:7], dtype=torch.float32)
    )  
    # Reverse the translation: pos += R_palm @ [0, 0, 0.1]
    offset = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float32)
    world_offset = (R_mj @ offset).cpu().numpy()  # (..., 3)
    robot_pose[:3] += world_offset

    delta_rot = torch_quaternion_to_matrix(
        torch.tensor([0, 1, 0, 0], dtype=torch.float32)
    )  
    R_palm = R_mj @ delta_rot
    robot_pose[3:7] = torch_matrix_to_quaternion(R_palm).cpu().numpy()

    return robot_pose

def allegro_from_mujoco_to_oakink(robot_pose: np.ndarray) -> np.ndarray:
    """
    Convert Allegro root pose from Mujoco frame (eg. DexGraspBench) back to base-rooted URDF (e.g., DRO/BODex)
    Args:
        robot_pose (np.ndarray): A pose array of shape (7+J), where:
            - [:3] are the root positions (x, y, z) in world coordinates.
            - [3:7] are root orientations as quaternions (qw, qx, qy, qz).
            - [7:] are any remaining joint values.

    Returns:
        np.ndarray: The corrected pose array with:
            - Root pose transformed from Mujoco frame back to base_root URDF frame.
            - All other joint values unchanged.
    """

    R_mj = torch_quaternion_to_matrix(
        torch.tensor(robot_pose[3:7], dtype=torch.float32)
    )  
    delta_rot = torch_quaternion_to_matrix(
        torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    )  

    R_base = R_mj @ delta_rot
    euler = matrix_to_euler(R_base).cpu().numpy()
    
    return np.concatenate([robot_pose[:3], euler, robot_pose[7:]], axis=0)


# def leaphand_order_from_oakink_to_bodex(qpos: np.ndarray) -> np.ndarray:
#     """
#     Convert LeapHand joint order from Oakink to Bodex.
#     Args:
#         qpos (np.ndarray): A pose array of shape (N,) where:
#             - qpos is the joint values in oakink (eg. DRO) order.
#     Returns:
#         np.ndarray: Transformed pose array of shape (N,) where:
#             - qpos is the joint values in bodex (eg. BODex) order.
#     """

#     original_order = ['1', '0', '2', '3', '5', '4', '6', '7', '9', '8', '10', '11', '12', '13', '14', '15']
#     original_named = [f'j{i}' for i in original_order]
#     target_order = ['j12', 'j13', 'j14', 'j15', 'j1', 'j0', 'j2', 'j3', 'j9', 'j8', 'j10', 'j11', 'j5', 'j4', 'j6', 'j7']
#     permutation_indices = [original_named.index(j) for j in target_order]
#     qpos_reordered = [qpos[i] for i in permutation_indices]

#     return np.array(qpos_reordered, dtype=qpos.dtype)

# def leaphand_order_from_bodex_to_oakink(qpos: np.ndarray) -> np.ndarray:
#     """
#     Convert LeapHand joint order from Bodex to Oakink, supports batch input.
    
#     Args:
#         qpos (np.ndarray): An array of shape (..., 16) representing joint values 
#                            in bodex order, where the last dimension is joints.
    
#     Returns:
#         np.ndarray: Transformed array of same shape, but with joint order converted to Oakink.
#     """
#     # Mapping from Bodex joint order to Oakink joint order
#     original_order = ['j12', 'j13', 'j14', 'j15', 'j1', 'j0', 'j2', 'j3',
#                       'j9', 'j8', 'j10', 'j11', 'j5', 'j4', 'j6', 'j7']
#     target_order = ['1', '0', '2', '3', '5', '4', '6', '7',
#                     '9', '8', '10', '11', '12', '13', '14', '15']
#     target_named = [f'j{i}' for i in target_order]

#     # Compute permutation indices to rearrange the last axis
#     permutation_indices = [original_order.index(j) for j in target_named]
#     permutation_indices = np.array(permutation_indices)

#     # Use np.take to index along the last axis
#     return np.take(qpos, permutation_indices, axis=-1)







# # Fixed local offset from base → palm_lower
# _offset_t = np.array([0.0, 0.038, 0.098], dtype=np.float32)
# _angle   = -np.pi/2
# _axis    = np.array([0.0, 1.0, 0.0], dtype=np.float32)
# _s, _c   = np.sin(_angle/2), np.cos(_angle/2)
# _q_offset = np.array([_axis[0]*_s,
#                       _axis[1]*_s,
#                       _axis[2]*_s,
#                       _c], dtype=np.float32)    # [qx,qy,qz,qw]

# def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
#     """Hamilton product (x,y,z,w)·(x,y,z,w) → (x,y,z,w)."""
#     x1,y1,z1,w1 = q1;  x2,y2,z2,w2 = q2
#     return np.array([
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2,
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#     ], dtype=q1.dtype)

# def _quat_conjugate(q: np.ndarray) -> np.ndarray:
#     """Inverse of a unit quaternion [x,y,z,w]."""
#     return np.array([-q[0], -q[1], -q[2], q[3]], dtype=q.dtype)

# def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
#     """
#     Rotate vector v by quaternion q (x,y,z,w).
#     v' = v + 2·w*(u×v) + 2*(u×(u×v)),  u=q[:3], w=q[3]
#     """
#     u = q[:3];  w = q[3]
#     return v + 2*w*np.cross(u, v) + 2*np.cross(u, np.cross(u, v))


# # ———— 1) base-rooted Euler → palm-rooted quaternion —————————————————————————————————
# def leaphand_from_base_to_palm_quat(pose_euler: np.ndarray) -> np.ndarray:
#     """
#     Convert a LeapHand pose whose root is 'base' (Euler) into
#     a pose whose root is 'palm_lower' (quaternion).

#     Args:
#         pose_euler (np.ndarray): shape (7+J,)
#           - [0:3]   = base position (x,y,z)
#           - [3:6]   = base orientation in Euler angles (roll,pitch,yaw)
#           - [6:]    = remaining joint values

#     Returns:
#         np.ndarray shape (7+J,):
#           - [0:3]   = palm_lower position in world
#           - [3:7]   = palm_lower orientation quaternion (x,y,z,w)
#           - [7:]    = same joint values
#     """
#     pos   = pose_euler[:3]
#     euler = pose_euler[3:6]
#     tail  = pose_euler[6:]

#     # 1) Euler → quaternion for the base frame
#     q_b = R.from_euler('XYZ', euler).as_quat()  # [x,y,z,w]

#     # 2) apply the fixed local rotation offset
#     q_p = _quat_mul(q_b, _q_offset)

#     # 3) rotate the translation offset into world, then add
#     offs_world = _quat_rotate_vector(q_b, _offset_t)
#     new_pos    = pos + offs_world

#     return np.concatenate([new_pos, q_p, tail], axis=0)


# # ———— 2) palm-rooted quaternion → base-rooted Euler —————————————————————————————————
# def leaphand_from_palm_to_base_euler(pose_quat: np.ndarray) -> np.ndarray:
#     """
#     Convert a LeapHand pose whose root is 'palm_lower' (quaternion) into
#     a pose whose root is 'base' (Euler).

#     Args:
#         pose_quat (np.ndarray): shape (7+J,)
#           - [0:3]   = palm_lower position (x,y,z)
#           - [3:7]   = palm_lower orientation quaternion (x,y,z,w)
#           - [7:]    = remaining joint values

#     Returns:
#         np.ndarray shape (7+J,):
#           - [0:3]   = base position in world
#           - [3:6]   = base orientation in Euler angles (roll,pitch,yaw)
#           - [6:]    = same joint values
#     """
#     pos  = pose_quat[:3]
#     q_p  = pose_quat[3:7]
#     tail = pose_quat[7:]

#     # 1) undo the orientation offset
#     q_b = _quat_mul(q_p, _quat_conjugate(_q_offset))

#     # 2) undo the translation offset
#     offs_world = _quat_rotate_vector(q_b, _offset_t)
#     base_pos   = pos - offs_world

#     # 3) quaternion → Euler for the base frame
#     base_euler = R.from_quat(q_b).as_euler('XYZ')

#     return np.concatenate([base_pos, base_euler, tail], axis=0)

# def leaphand_from_base_to_palm_euler(pose_euler: torch.Tensor) -> torch.Tensor:
#     """
#     pose_euler: [x,y,z, roll,pitch,yaw, ...tail_joints] (torch.Tensor)
#     """
#     pos   = pose_euler[:3]    
#     euler = pose_euler[3:6]   
#     tail  = pose_euler[6:]    

#     Rb = euler_to_matrix(euler)                      
#     Tb = torch.eye(4, device=euler.device)
#     Tb[:3,:3] = Rb
#     Tb[:3, 3] = pos

#     To = torch.eye(4, device=euler.device)
#     To[:3,:3] = torch.from_numpy(_offset_R).to(euler.device).float()
#     To[:3, 3] = torch.from_numpy(_offset_t).to(euler.device).float()

#     Tp = Tb @ To

#     new_pos   = Tp[:3, 3]
#     new_euler = matrix_to_euler(Tp[:3, :3])
#     return torch.cat([new_pos, new_euler, tail], dim=0)


# def leaphand_from_palm_to_base_euler(pose_euler: torch.Tensor) -> torch.Tensor:
#     """
#     pose_euler: [x,y,z, roll,pitch,yaw, ...] (palm_lower root)
#     """
#     pos   = pose_euler[:3]
#     euler = pose_euler[3:6]
#     tail  = pose_euler[6:]

#     Rp = euler_to_matrix(euler)
#     Tp = torch.eye(4, device=euler.device)
#     Tp[:3,:3] = Rp
#     Tp[:3, 3] = pos

#     Ti = torch.from_numpy(T_palm_to_base).to(euler.device).float() 
#     Tb = Tp @ Ti

#     new_pos   = Tb[:3, 3]
#     new_euler = matrix_to_euler(Tb[:3, :3])
#     return torch.cat([new_pos, new_euler, tail], dim=0)