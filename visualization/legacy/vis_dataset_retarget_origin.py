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
filtered = True
# robot_names = ['shadowhand']
# robot_names = ['allegro']
robot_names = ['leaphand']
object_names = [
     "oakink+teapot",
]

if "oakink" in object_names[0]:
    if "mano" in robot_names[0]:
        dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all.pt')
    else:
        dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_{robot_names[0]}.pt')
        # dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_names[0]}.pt')
elif filtered:
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
else:
    dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset/cmap_dataset.pt')

print("Load from ", dataset_path)
dataset = torch.load(dataset_path, map_location=torch.device('cpu'))
metadata = dataset['metadata']
mano_compatibility = dataset['version']['metadata'] in ['1.3.0', '1.2.0', '1.2.0_COACD', '1.1.0', '2.0.0']
print("Dataset version: ", dataset['version']['metadata'])
assert 'mano' not in robot_names or mano_compatibility, "if only mano is visualized, the dataset must be in mano format"
print(f"Mano compatibility: {mano_compatibility}")

if mano_compatibility:
    mano_layer = ManoLayer( rot_mode="axisang",
                            mano_assets_root="assets/mano",
                            use_pca=False,
                            flat_hand_mean=True)

def on_update(robot_idx, object_idx, grasp_idx):
    robot_name = robot_names[robot_idx]
    object_name = object_names[object_idx]
    if "oakink" in object_name:
        metadata_curr = [m  for m in metadata if m[5] == object_name and m[6] == robot_name]
    elif filtered:
        metadata_curr = [m[0] for m in metadata if m[1] == object_name and m[2] == robot_name]
    else:
        metadata_curr = [m[1] for m in metadata if m[2] == object_name and m[3] == robot_name]
    if len(metadata_curr) == 0:
        print('No metadata found!')
        return
    grasp_item = metadata_curr[grasp_idx % len(metadata_curr)]
    if "oakink" in object_name:
        hand_pose, hand_shape, hand_tsl, q, _, _, _, object_id, _, hand_verts_pc = grasp_item[:10]
        hand_tsl = hand_tsl.numpy()
        hand_verts_pc = hand_verts_pc
    else:
        q = grasp_item


    if "oakink" in object_name:
        name = object_name.split('+')
        obj_mesh_path = list(
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.obj')) +
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.ply')))
        assert len(obj_mesh_path) == 1
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

        hand_faces = mano_layer.th_faces  
    else:
        name = object_name.split('+')
        object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')  # visual mesh
        object_trimesh = trimesh.load_mesh(object_path)

    hand = create_hand_model(robot_name)
    robot_trimesh = hand.get_trimesh_q(q)["visual"]

    server.scene.add_mesh_simple(
        'object',
        object_trimesh.vertices,
        object_trimesh.faces,
        color=(239, 132, 167),
        opacity=1
    )

    # server.scene.add_point_cloud(
    #     'hand_pc',
    #     hand_verts_pc,
    #     colors=(192, 192, 255),
    #     point_size=0.002,
    #     point_shape="circle",
    # )

    # server.scene.add_mesh_simple(
    #     'robot',
    #     # (torch.from_numpy(np.array(robot_trimesh.vertices)) + hand_tsl[None, :]).numpy(),
    #     robot_trimesh.vertices,
    #     robot_trimesh.faces,
    #     color=(102, 192, 255),
    #     opacity=0.8
    # )

    if mano_compatibility:

        mano_outer_layer = MANOLayer('right', betas=grasp_item[1].numpy())
        vertex, joint = mano_outer_layer(grasp_item[0].unsqueeze(0), torch.zeros(1, 3))
        
        mano_torch_layer = ManoLayer(mano_assets_root='visualization/manopth/mano/',use_pca=False, flat_hand_mean=True, center_idx=0)
        mano_output = mano_torch_layer(grasp_item[0].unsqueeze(0), grasp_item[1].unsqueeze(0))
        t_vertex, t_joint = mano_output.verts, mano_output.joints

        # server.scene.add_mesh_simple(
        #     'robot_verts',
        #     # (vertex[0] + hand_tsl[None, :]).numpy(),
        #     vertex[0].numpy(),
        #     mano_outer_layer.f.cpu().numpy(),
        #     color=(102, 192, 255),
        #     opacity=0.8
        # )

        server.scene.add_mesh_simple(
            'robot_verts_2',
            (t_vertex[0] + hand_tsl[None, :]).numpy(),
            # t_vertex[0].numpy(),
            mano_torch_layer.th_faces.cpu().numpy(),
            color=(192, 102, 255),
            opacity=0.8
        )

server = viser.ViserServer(host='127.0.0.1', port=8080)

robot_slider = server.gui.add_slider(
    label='robot',
    min=0,
    max=len(robot_names) - 1,
    step=1,
    initial_value=0
)
object_slider = server.gui.add_slider(
    label='object',
    min=0,
    max=len(object_names) - 1,
    step=1,
    initial_value=0
)
grasp_slider = server.gui.add_slider(
    label='grasp',
    min=0,
    max=199,
    step=1,
    initial_value=0
)
robot_slider.on_update(lambda _: on_update(robot_slider.value, object_slider.value, grasp_slider.value))
object_slider.on_update(lambda _: on_update(robot_slider.value, object_slider.value, grasp_slider.value))
grasp_slider.on_update(lambda _: on_update(robot_slider.value, object_slider.value, grasp_slider.value))

while True:
    time.sleep(1)