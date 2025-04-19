import time
import torch
import viser
from utils.hand_model import create_hand_model, HandModel
from utils.rotation import quaternion_to_euler
import trimesh
from DRO_Grasp.utils.rotation import matrix_to_euler, euler_to_matrix, rotation_matrix_x, rotation_matrix_y, transform_matrix
from DRO_Grasp.utils.rotation_format_utils import leaphand_order_from_bodex_to_oakink, leaphand_order_from_oakink_to_bodex,leaphand_from_oakink_to_bodex, leaphand_from_bodex_to_oakink


GRASP_DATA_PATH = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_leaphand.pt"
robot1_name = "leaphand"
robot2_name = "leaphand_bodex"

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


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

    mesh = hand2.get_trimesh_q(q2)["visual"]
    server.scene.add_mesh_simple(
        name="hand_reordered",
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=(255, 192, 102),
        opacity=0.8
    )

def main():
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    print("Viser server started at http://127.0.0.1:8080")

    hand1 = create_hand_model(robot1_name)
    hand2 = create_hand_model(robot2_name)

    print(f"{robot1_name} joint order:", hand1.get_joint_orders())
    print(f"{robot2_name} joint order:", hand2.get_joint_orders())

    grasp_data = torch.load(GRASP_DATA_PATH)
    print(f"Loaded grasp data with keys: {list(grasp_data.keys())}")
    metadata = grasp_data["metadata"]
    num_samples = len(metadata)

    slider = server.gui.add_slider(
        label="Sample Index",
        initial_value=0,
        min=0,
        max=num_samples - 1,
        step=1,
    )

    def update_scene(index):
        qpos_raw = metadata[index][3]
        qpos_reordered = leaphand_from_oakink_to_bodex(qpos_raw)
        qpos_reordered[7:] = leaphand_order_from_oakink_to_bodex(qpos_reordered[7:])
        qpos_reordered = torch.from_numpy(qpos_reordered).float()
        qpos_raw = leaphand_from_bodex_to_oakink(qpos_reordered.numpy())
        
        qpos_reordered = torch.cat([
            qpos_reordered[:3],
            quaternion_to_euler(torch.cat([qpos_reordered[4:7], qpos_reordered[3:4]])),
            qpos_reordered[7:]
        ])

        qpos_raw[6:] = leaphand_order_from_bodex_to_oakink(qpos_raw[6:])
        qpos_raw = torch.from_numpy(qpos_raw).float()

        visualize_scene(server, hand1, hand2, qpos_raw, qpos_reordered)

    # 初始化展示
    update_scene(0)

    # 当 slider 被拖动时更新可视化
    @slider.on_update
    def _on_slider_change(event):
        update_scene(slider.value)

    # 保持服务运行
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()