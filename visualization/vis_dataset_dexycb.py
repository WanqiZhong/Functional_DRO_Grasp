import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import time
import trimesh
import torch
import viser
from utils.hand_model import create_hand_model
import glob
import open3d as o3d
import numpy as np
import manotorch
from manotorch.manolayer import ManoLayer, MANOOutput
import manopth
from mano_layer import MANOLayer  
from urdfpy import URDF, Mesh, Box, Cylinder, Sphere
import urdfpy



filtered = True
robot_names = ['shadowhand']
object_names = [
    # "oakink+gamecontroller",
    #  "oakink+toothbrush",
    #  "oakink+wineglass",
    #  "oakink+cup",
    #  "oakink+mouse",
    #  "oakink+binoculars",
    #  "oakink+lightbulb",
    #  "oakink+lotion_pump",
    #  "oakink+squeezable",
    #  "oakink+hammer",
    #  "oakink+pen",
    #  "oakink+pincer",
    #  "oakink+mug",
    #  "oakink+screwdriver",
    #  "oakink+banana",
    #  "oakink+stapler",
    #  "oakink+fryingpan",
    #  "oakink+bowl",
    #  "oakink+phone",
    #  "oakink+scissors",
    #  "oakink+flashlight",
    #  "oakink+eyeglasses",
     "oakink+teapot",
    #  "oakink+power_drill",
    #  "oakink+wrench",
    #  "oakink+trigger_sprayer",
    #  "oakink+donut",
    #  "oakink+cylinder_bottle",
    #  "oakink+apple",
    #  "oakink+bottle",
    #  "oakink+cameras",
    #  "oakink+knife",
    #  "oakink+headphones"
]


dataset_path = os.path.join("/data/zwq/code/dex-retargeting/example/position_retargeting/data/shadow_log_12.pt")
dataset = torch.load(dataset_path, map_location=torch.device('cpu'))
metadata = dataset['metadata']
mano_layer = ManoLayer( rot_mode="axisang",
                        mano_assets_root="assets/mano",
                        use_pca=False,
                        flat_hand_mean=True)
robot_name = robot_names[0]
metadata_curr = metadata[robot_name if robot_name != 'shadowhand' else 'shadow']

def transform_verts(vertex, camera_pose):
    camera_mat = camera_pose.to_transformation_matrix()
    vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
    vertex = np.ascontiguousarray(vertex)
    return vertex

def load_shadowhand_urdf(urdf_path, joint_values):
    """
    加载 ShadowHand 的 URDF 文件，应用关节值，并提取合并后的顶点和面数据。

    :param urdf_path: URDF 文件的路径。
    :param joint_values: 包含关节名称及其对应值的字典。
    :param mesh_dir: 如果网格文件路径是相对的，可以指定网格目录。
    :return: (vertices, faces) 元组，其中 vertices 是 (N, 3) 的 numpy 数组，faces 是 (M, 3) 的 numpy 数组。
    """
    # 加载 URDF 文件
    robot = URDF.load(urdf_path)

    # 设置关节值
    robot.q = joint_values

    # 计算每个连杆的变换矩阵（前向运动学）
    link_transforms = robot.link_fk()  # 返回一个字典 {link_name: 4x4 矩阵}

    all_meshes = []

    for link in robot.links:
        if link.visuals:
            for visual in link.visuals:
                geometry = visual.geometry
                origin = visual.origin  # 4x4 变换矩阵

                # 获取连杆的变换
                link_transform = link_transforms.get(link.name, np.eye(4))

                # 合并变换：连杆变换 @ 视觉变换
                combined_transform = link_transform @ origin

                # 根据几何类型创建 trimesh 对象
                if isinstance(geometry, Mesh):
                    for mesh in geometry.meshes:
                        transformed_mesh = mesh.copy()
                        transformed_mesh.apply_transform(combined_transform)
                        all_meshes.append(transformed_mesh)
                elif isinstance(geometry, Box):
                    # Box 的 size 是 (x, y, z)
                    box_mesh = trimesh.creation.box(extents=geometry.size)
                    box_mesh.apply_transform(combined_transform)
                    all_meshes.append(box_mesh)
                elif isinstance(geometry, Cylinder):
                    # Cylinder 的 radius 和 length
                    cylinder_mesh = trimesh.creation.cylinder(radius=geometry.radius, height=geometry.length, sections=32)
                    # 默认 cylinder 沿 Z 轴，确保与 URDF 一致
                    cylinder_mesh.apply_transform(combined_transform)
                    all_meshes.append(cylinder_mesh)
                elif isinstance(geometry, Sphere):
                    # Sphere 的 radius
                    sphere_mesh = trimesh.creation.icosphere(radius=geometry.radius, subdivisions=3)
                    sphere_mesh.apply_transform(combined_transform)
                    all_meshes.append(sphere_mesh)
                else:
                    print(f"警告：未处理的几何类型: {type(geometry)}")

    if not all_meshes:
        raise ValueError("没有找到任何可视化网格。请检查 URDF 文件和关节值。")

    combined = trimesh.util.concatenate(all_meshes)

    return combined


def on_update(grasp_idx):
    if len(metadata_curr) == 0:
        print('No metadata found!')
        return
    grasp_item = metadata_curr[grasp_idx % len(metadata_curr)]
    hand_pose, hand_shape, hand_translation, q, _, _, _, _, _, verts, camera_pose, joint_info = grasp_item

    # print(f"joint values: {q}")
    # q = torch.cat((q[:3], q[4:5], q[3:4], q[5:6], q[6:]), dim=0)
    hand = create_hand_model(robot_name)
    robot_trimesh = hand.get_trimesh_q(q)["visual"]

    # q_joint = {}
    # for joint_name, q_value in zip(hand.get_joint_orders(), q):
    #     q_joint[joint_name] = q_value

    # combined_mesh = load_shadowhand_urdf(hand.urdf_path, q_joint)

    # create from verts
    # robot_trimesh = trimesh.Trimesh(vertices=verts)
    # robot = URDF.load(hand.urdf_path)
    # robot.show(q_joint)

    server.scene.add_mesh_simple(
        'robot',
        # (torch.from_numpy(np.array(robot_trimesh.vertices)) + hand_tsl[None, :]).numpy(),
        robot_trimesh.vertices,
        robot_trimesh.faces,
        color=(102, 192, 255),
        opacity=0.8
    )

    # server.scene.add_mesh_simple(
    #     'robot_2',
    #     # (torch.from_numpy(np.array(robot_trimesh.vertices)) + hand_tsl[None, :]).numpy(),
    #     combined_mesh.vertices,
    #     combined_mesh.faces,
    #     color=(192, 102, 255),
    #     opacity=0.8
    # )

    server.scene.add_point_cloud(
        'human_pc',
        verts,
        colors=(192, 192, 255),
        point_size=0.002,
        point_shape="circle",
    )

    mano_outer_layer = MANOLayer('right', betas=hand_shape.numpy())
    vertex, joint = mano_outer_layer(hand_pose.unsqueeze(0), hand_translation.unsqueeze(0))
    
    mano_torch_layer = ManoLayer(mano_assets_root='visualization/manopth/mano/',use_pca=True, flat_hand_mean=False, ncomps=45)
    mano_output = mano_torch_layer(grasp_item[0].unsqueeze(0), grasp_item[1].unsqueeze(0))
    t_vertex, t_joint = mano_output.verts, mano_output.joints
    t_vertex[0] += hand_translation[None, :]

    t_vertex_ = t_vertex[0].numpy()
    vertex_ = vertex[0].numpy()
    # t_vertex_ = transform_verts(t_vertex[0].numpy(), camera_pose)
    # vertex_ = transform_verts(vertex[0].numpy(), camera_pose)

    server.scene.add_mesh_simple(
        'robot_verts',
        # (vertex[0] + hand_tsl[None, :]).numpy(),
        vertex_,
        mano_outer_layer.f.cpu().numpy(),
        color=(102, 192, 255),
        opacity=0.8
    )

    server.scene.add_mesh_simple(
        'robot_verts_2',
        # ().numpy(),
        t_vertex_,
        mano_torch_layer.th_faces.cpu().numpy(),
        color=(192, 102, 255),
        opacity=0.8
    )


server = viser.ViserServer(host='127.0.0.1', port=8080)

grasp_slider = server.gui.add_slider(
    label='grasp',
    min=0,
    max=68,
    step=1,
    initial_value=0
)
grasp_slider.on_update(lambda _: on_update(grasp_slider.value))

while True:
    time.sleep(1)
