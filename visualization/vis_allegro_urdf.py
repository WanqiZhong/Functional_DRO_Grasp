import time
import torch
import viser
from utils.hand_model import create_hand_model, HandModel
from utils.rotation import quaternion_to_euler
import trimesh
from DRO_Grasp.utils.rotation import matrix_to_euler, euler_to_matrix, rotation_matrix_x, rotation_matrix_y, transform_matrix


GRASP_DATA_PATH = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_allegro.pt"
robot1_name = "allegro"
robot2_name = "allegro_bodex"


import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

_offset_t = np.array([0.0, 0.038, 0.098])       
_offset_R = rotation_matrix_y(-1.57079)         
T_base_to_palm = transform_matrix(_offset_t, _offset_R)
T_palm_to_base = np.linalg.inv(T_base_to_palm)

def leaphand_from_base_to_palm_euler(pose_euler: torch.Tensor) -> torch.Tensor:
    """
    pose_euler: [x,y,z, roll,pitch,yaw, ...tail_joints] (torch.Tensor)
    """
    pos   = pose_euler[:3]    
    euler = pose_euler[3:6]   
    tail  = pose_euler[6:]    

    Rb = euler_to_matrix(euler)                      
    Tb = torch.eye(4, device=euler.device)
    Tb[:3,:3] = Rb
    Tb[:3, 3] = pos

    To = torch.eye(4, device=euler.device)
    To[:3,:3] = torch.from_numpy(_offset_R).to(euler.device).float()
    To[:3, 3] = torch.from_numpy(_offset_t).to(euler.device).float()

    Tp = Tb @ To

    new_pos   = Tp[:3, 3]
    new_euler = matrix_to_euler(Tp[:3, :3])
    return torch.cat([new_pos, new_euler, tail], dim=0)


def leaphand_from_palm_to_base_euler(pose_euler: torch.Tensor) -> torch.Tensor:
    """
    pose_euler: [x,y,z, roll,pitch,yaw, ...] (palm_lower root)
    """
    pos   = pose_euler[:3]
    euler = pose_euler[3:6]
    tail  = pose_euler[6:]

    Rp = euler_to_matrix(euler)
    Tp = torch.eye(4, device=euler.device)
    Tp[:3,:3] = Rp
    Tp[:3, 3] = pos

    Ti = torch.from_numpy(T_palm_to_base).to(euler.device).float() 
    Tb = Tp @ Ti

    new_pos   = Tb[:3, 3]
    new_euler = matrix_to_euler(Tb[:3, :3])
    return torch.cat([new_pos, new_euler, tail], dim=0)


# # ———— 2️⃣ Quaternion 版 ———————————————————————————

def visualize_scene(server, hand1: HandModel, hand2: HandModel, q1, q2):
    server.scene.reset()

    mesh = hand1.get_trimesh_q(q1)["visual"]
    server.scene.add_mesh_simple(
        name="hand_original",
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=(102, 192, 255),
        opacity=0.8
    )
    print("Added leaphand")

    mesh = hand2.get_trimesh_q(q2)["visual"]
    server.scene.add_mesh_simple(
        name="hand_reordered",
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=(255, 192, 102),
        opacity=0.8
    )
    print("Added leaphand_bodex")

    palm_mesh = trimesh.load("data/data_urdf/robot/leaphand/meshes/visual/palm_lower.obj")
    server.scene.add_mesh_simple(
        name="palm_lower",
        vertices=palm_mesh.vertices,
        faces=palm_mesh.faces,
        color=(150, 255, 150),
        opacity=0.6
    )
    print("Added palm_lower.obj")

    pip_mesh = trimesh.load("/data/zwq/code/BODex/src/curobo/content/assets/robot/leap_description/meshes/leap_simplified/palm_lower.stl")
    server.scene.add_mesh_simple(
        name="palm_lower_stl",
        vertices=pip_mesh.vertices,
        faces=pip_mesh.faces,
        color=(255, 102, 204),
        opacity=0.6
    )
    print("Added pip.stl")

def main():
    
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    print("Viser server started at http://127.0.0.1:8080")

    hand1 = create_hand_model(robot1_name)
    hand2 = create_hand_model(robot2_name)

    print(f"{robot1_name} joint order:", hand1.get_joint_orders())
    print(f"{robot2_name} joint order:", hand2.get_joint_orders())

    # try:
    grasp_data = torch.load(GRASP_DATA_PATH)
    print(f"Loaded grasp data with keys: {list(grasp_data.keys())}")

    qpos_raw = grasp_data['metadata'][0][3][6:]
    # qpos_reordered = [qpos_raw[i] for i in permutation_indices]

    qpos_raw = torch.cat([torch.randn(6), torch.tensor(qpos_raw)], dim=0)
    # qpos_raw = leaphand_from_palm_to_base_euler(qpos_raw)
    qpos_reordered = torch.tensor(qpos_raw).float()

    visualize_scene(server, hand1, hand2, qpos_raw, qpos_reordered)

    # except Exception as e:
    #     print(f"Error loading or processing grasp data: {e}")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()