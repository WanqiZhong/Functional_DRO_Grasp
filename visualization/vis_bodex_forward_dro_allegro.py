import os
import sys
import time
import trimesh
import numpy as np
import viser
import torch
from utils.hand_model import create_hand_model, HandModel
from utils.rotation import quaternion_to_euler, matrix_to_euler
from utils.rotation_format_utils import allegro_from_mujoco_to_oakink

robot_name = "allegro"
BENCH_DATA_PATH = "/data/zwq/code/DexGraspBench/output/oakink_teapot_step020_max_allegro_bodex/succgrasp/teapot/o13104/5.npy"
BODEX_DATA_PATH = "/data/zwq/code/BODex/src/curobo/content/assets/output/sim_allegro/fc/oakink_teapot_step015_max/graspdata/o13104/scale190_pose000_6grasp.npy"
# DRO_MUJOCO_DATA_PATH = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_allegro_mujoco.pt"
DRO_DATA_PATH = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_allegro_bodex_valid.pt"
DRO_OBJECT_PATH = "/data/zwq/code/DRO_Grasp/data/data_urdf/object/oakink/teapot/o13104.obj"
HAND_VSI_OBJ_PATH = "/data/zwq/code/DexGraspBench/output/oakink_teapot_step020_max_allegro_bodex/vis_obj/teapot/o13104/5_grasp_0.obj"
OBJECT_VIS_OBJ_PATH = "/data/zwq/code/DexGraspBench/output/oakink_teapot_step020_max_allegro_bodex/vis_obj/teapot/o13104/5_obj.obj"

def main():
    # Create viser server
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    print("Viser server started at http://127.0.0.1:8080")
    
    hand = create_hand_model(robot_name)
    print("hand: ", hand.get_joint_orders())

    hand_bodex = create_hand_model(f"{robot_name}_bodex")
    print("hand_bodex: ", hand_bodex.get_joint_orders())

    # Load grasp data
    try:
        grasp_data = np.load(BENCH_DATA_PATH, allow_pickle=True).item()
        print(f"Loaded grasp data with keys: {list(grasp_data.keys())}")
        bench_q = grasp_data['grasp_qpos']
        bench_q = allegro_from_mujoco_to_oakink(bench_q)
        bench_q = torch.tensor(bench_q)
        bench_q = bench_q.float()
        bench_obj_scale = grasp_data.get('obj_scale', 1.0)
    except Exception as e:
        print(f"Error loading grasp data: {e}")
        bench_q = None
        bench_obj_scale = 1.0

    # Load BODex grasp data
    try:
        bodex_grasp_data = np.load(BODEX_DATA_PATH, allow_pickle=True).item()
        print(f"Loaded BODex grasp data with keys: {list(bodex_grasp_data.keys())}")
        bodex_q = bodex_grasp_data['robot_pose'][0]
        bodex_q = bodex_q[0][1]
        bodex_q = torch.tensor(bodex_q)
        bodex_q = torch.cat([bodex_q[:3], quaternion_to_euler(torch.cat([bodex_q[4:7], bodex_q[3:4]])), bodex_q[7:]])
        bodex_q = bodex_q.float()
    except Exception as e:
        print(f"Error loading BODex grasp data: {e}")
        bodex_q = None

    # Load DRO grasp data
    try:
        mujoco_dataset_data = torch.load(DRO_MUJOCO_DATA_PATH)
        print(f"Loaded DRO mujoco dataset with keys: {list(mujoco_dataset_data.keys())}")
        mujoco_dataset_q = mujoco_dataset_data['metadata'][0][3]
        mujoco_dataset_q = torch.tensor(mujoco_dataset_q)
        mujoco_dataset_q = torch.cat([mujoco_dataset_q[:3], quaternion_to_euler(torch.cat([mujoco_dataset_q[4:7], mujoco_dataset_q[3:4]])), mujoco_dataset_q[7:]])
        mujoco_dataset_q = mujoco_dataset_q.float()
        dataset_obj_scale = float(mujoco_dataset_data['metadata'][0][8])
    except Exception as e:
        print(f"Error loading DRO mujoco dataset: {e}")
        mujoco_dataset_q = None
        dataset_obj_scale = 1.0

    # Load DRO valid data
    try:
        dro_dataset_data = torch.load(DRO_DATA_PATH)
        print(f"Loaded DRO valid dataset with keys: {list(dro_dataset_data.keys())}")
        dro_dataset_q = dro_dataset_data['metadata'][0][3].float()
        print(f"DRO valid q: {dro_dataset_q}")
        dro_obj_scale = float(dro_dataset_data['metadata'][0][8])
    except Exception as e:
        print(f"Error loading DRO valid dataset: {e}")
        dro_dataset_q = None
        dro_obj_scale = 1.0
    
    visualize_scene(server, hand, bench_q, bodex_q, mujoco_dataset_q, dro_dataset_q, robot_name, bench_obj_scale, dataset_obj_scale)
    # Keep the server running
    while True:
        time.sleep(1)

def visualize_scene(server, hand: HandModel, bench_q, bodex_q, mujoco_dataset_q, dro_dataset_q, robot_name, bench_obj_scale, dataset_obj_scale):
    """Visualize the hand and objects in the scene"""
    # Clear previous visualization if any
    server.scene.reset()
    
    # Visualize hand model if q is available
    if bench_q is not None:
        try:
            robot_trimesh = hand.get_trimesh_q(bench_q)["visual"]
            
            server.scene.add_mesh_simple(
                'bench_robot',
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

            # origin_robot_trimesh = hand_bodex.get_trimesh_q(original_bodex_q)["visual"]
            # server.scene.add_mesh_simple(
            #     'original_bodex_robot',
            #     origin_robot_trimesh.vertices,
            #     origin_robot_trimesh.faces,
            #     color=(102, 192, 255),
            #     opacity=0.8
            # ) 
            # print("Added original BODex hand model to scene")

        except Exception as e:
            print(f"Error visualizing BODex hand model: {e}")
    else:
        print("Warning: No BODex grasp pose data found")

    if mujoco_dataset_q is not None:
        try:
            robot_trimesh = hand.get_trimesh_q(mujoco_dataset_q)["visual"]
            server.scene.add_mesh_simple(
                'dataset_robot',
                robot_trimesh.vertices, 
                robot_trimesh.faces,
                color=(102, 192, 255),
                opacity=0.8
            )
            print("Added Oakink hand model to scene")
        except Exception as e:
            print(f"Error visualizing Oakink hand model: {e}")
    else:
        print("Warning: No Oakink mujoco grasp pose data found")

    if dro_dataset_q is not None:
        try:
            robot_trimesh = hand.get_trimesh_q(dro_dataset_q)["visual"]
            server.scene.add_mesh_simple(
                'dro_robot',
                robot_trimesh.vertices,
                robot_trimesh.faces,
                color=(102, 192, 255),
                opacity=0.8
            )
            print("Added DRO hand model to scene")
        except Exception as e:
            print(f"Error visualizing DRO hand model: {e}")

    # Load and visualize object mesh
    try:
        object_trimesh = trimesh.load_mesh(OBJECT_VIS_OBJ_PATH)
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
        hand_object_trimesh = trimesh.load_mesh(HAND_VSI_OBJ_PATH)
        server.scene.add_mesh_simple(
            'hand_obj_vis',
            hand_object_trimesh.vertices,
            hand_object_trimesh.faces,
            color=(192, 192, 255),
            opacity=0.6
        )
        print("Added hand vis obj mesh to scene")
    except Exception as e:
        print(f"Error loading hand vis obj mesh: {e}")

    # Load and visualize object mesh
    try:
        dro_object_trimesh = trimesh.load_mesh(DRO_OBJECT_PATH)

        if dataset_obj_scale != 1.0:
            dro_object_trimesh.apply_scale(dataset_obj_scale)

        server.scene.add_mesh_simple(
            'object_obj_vis',
            dro_object_trimesh.vertices,
            dro_object_trimesh.faces,
            color=(239, 132, 167),
            opacity=1
        )
        print("Added DRO mesh to scene")
    except Exception as e:
        print(f"Error loading DRO mesh: {e}")
    

if __name__ == "__main__":
    main()