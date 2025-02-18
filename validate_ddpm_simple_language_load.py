import os
import sys
import time
import pickle
import warnings
import torch
import viser
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from data_utils.CombineRetargetDataset import create_dataloader
from model.network import create_network_larger_transformer_openai, create_network_larger_transformer_openai_dgcnn
import hydra
import open3d as o3d
import numpy as np
import glob
from gorilla.config import Config
from OpenAD.utils.model_builder import build_model


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

server = viser.ViserServer(host='127.0.0.1', port=8080)

def process_affordance(model, point_cloud_data, affordance):
    """Process point cloud through model"""
    model.eval()
    with torch.no_grad():
        # Ensure data is in correct format (B, 3, N)
        if len(point_cloud_data.shape) == 2:
            point_cloud_data = point_cloud_data.unsqueeze(0)
        
        point_cloud_data = point_cloud_data.permute(0, 2, 1)
        afford_pred = model(point_cloud_data, affordance)
        afford_pred = afford_pred.permute(0, 2, 1).cpu().numpy()
        afford_pred = np.argmax(afford_pred, axis=2)
    
    return afford_pred

def create_affordance_color_map(affordance_names):
    """
    Create a fixed color mapping for each affordance
    """
    base_colors = [
        [1.0, 0.0, 0.0],    # Red
        [0.0, 1.0, 0.0],    # Green
        [0.0, 0.0, 1.0],    # Blue
        [1.0, 1.0, 0.0],    # Yellow
        [1.0, 0.0, 1.0],    # Magenta
        [0.0, 1.0, 1.0],    # Cyan
        [0.5, 0.0, 0.0],    # Dark Red
        [0.0, 0.5, 0.0],    # Dark Green
        [0.0, 0.0, 0.5],    # Dark Blue
        [0.5, 0.5, 0.0],    # Olive
        [0.5, 0.0, 0.5],    # Purple
        [0.0, 0.5, 0.5],    # Teal
        [1.0, 0.5, 0.0],    # Orange
        [0.5, 0.0, 1.0],    # Violet
        [0.0, 1.0, 0.5],    # Spring Green
    ]
    
    num_affordances = len(affordance_names)
    if num_affordances > len(base_colors):
        extra_colors = np.random.rand(num_affordances - len(base_colors), 3)
        colors = np.vstack([base_colors, extra_colors])
    else:
        colors = np.array(base_colors[:num_affordances])
    
    color_map = {name: color for name, color in zip(affordance_names, colors)}
    return color_map

def visualize_results(point_cloud, affordance_pred, affordance_names, scene, color_map):
    """Visualize results with viser"""
    
    points = point_cloud.reshape(-1, 3)
    predictions = affordance_pred.flatten()
    
    for i, afford_name in enumerate(affordance_names):
        
        afford_mask = predictions == i
        if not afford_mask.any():
            continue
            
        afford_points = points[afford_mask]
        
        color = color_map[afford_name]
        
        scene.add_point_cloud(
            name=f"{afford_name}",
            points=afford_points.cpu().numpy(),
            colors=color,  
            point_size=0.002,
            point_shape="circle",
        )

def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def get_object_vertices(object_name, object_id):

    name = object_name.split('+')
    obj_mesh_path = list(
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.obj')) +
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.ply'))
    )
    if len(obj_mesh_path) == 0:
        print(f"No mesh file found for object ID {object_id}")
        return
    assert len(obj_mesh_path) == 1, f"Multiple mesh files found for object ID {object_id}"
    obj_path = obj_mesh_path[0]
    object_mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(object_mesh.vertices)

    return vertices, np.asarray(object_mesh.triangles)

def _get_object_pc(object_name, object_id, robot_name):
    name = object_name.split('+')
    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt') if (robot_name == 'mano' or robot_name == 'retarget_shadowhand') \
        else os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
    object_pc = torch.load(object_path)[:, :3]
    return object_pc

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_retarget_language_larger_transformer_openad_dgcnn")
def main(cfg):
    print("******************************** [Config] ********************************")    
    device = torch.device(f'cuda:{cfg.gpu}')
    batch_size = cfg.dataset.batch_size
    print(f"Device: {device}, Batch size: {batch_size}")

    os.makedirs(os.path.join(ROOT_DIR, 'validate_output'), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/{cfg.name}.log')
    print('Log file:', log_file_name)
    validate_epoch = cfg.validate_epochs[0]
    print(f"************************ Validating epoch {validate_epoch} ************************")
    with open(log_file_name, 'a') as f:
        print(f"************************ Validating epoch {validate_epoch} ************************", file=f)

    # # Load network
    # network = create_network_larger_transformer_openai_dgcnn(cfg.model, mode='validate').to(device)
    # network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{validate_epoch}.pth", map_location=device))
    # network.eval()

    result_path = os.path.join(ROOT_DIR, "results_with_predict_q.pkl")
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_18-07-23_oakink_all_dof/eval/final/2025-01-09_22-02-54/res_diffuser_100.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_18-07-23_oakink_all_dof/eval/final/2025-01-09_22-24-16/res_diffuser_1000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_18-07-23_oakink_all_dof/eval/final_validate_data/2025-01-10_02-03-02/res_diffuser_1000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-15_04-21-46_robot_rotmat_oakink/eval/final/2025-01-18_20-40-45/res_diffuser_3000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-15_04-21-46_robot_rotmat_oakink/eval/final_validate_data/2025-01-18_23-25-25/res_diffuser_3000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_15-39-57_6dof_oakink/eval/final/2025-01-09_22-02-57/res_diffuser_100.pkl"
    results = load_results(result_path)['results']

    # Filter out results with chamfer distance >= 5
    # results = [result for result in results if result['chamfer_value'] >= 3]
    print(len(results))

    openad_config = "/data/zwq/code/OpenAD/config/openad_pn2/full_shape_cfg_downsample_oakink_combine.py"
    openad_checkpoint = "/data/zwq/code/OpenAD/log/openad_pn2/OPENAD_PN2_FULL_SHAPE_Downsample_Combine_New_Spilt/best_model.t7"

    openad_cfg = Config.fromfile(openad_config)

    if "oakinknet" in openad_cfg.data.dataset_name:
        dataset_file = os.path.join(openad_cfg.data.data_root, 'oakink_full_segmentation_avg_5000.pt')
        all_dataset = torch.load(dataset_file)
        all_label = list(all_dataset['label'])
        openad_cfg.training_cfg.train_affordance = all_label
        print(openad_cfg.training_cfg.train_affordance)

    openad_model = build_model(openad_cfg).to(device)
    color_map = create_affordance_color_map(openad_cfg.training_cfg.train_affordance)

    _, exten = os.path.splitext(openad_checkpoint)
    if exten == '.t7':
        openad_model.load_state_dict(torch.load(openad_checkpoint))
    elif exten == '.pth':
        check = torch.load(openad_checkpoint) 
        openad_model.load_state_dict(check['model_state_dict'])  

    def on_update(grasp_id):

        server.scene.reset()
        data = results[grasp_id]
        
        # Create hand model
        robot_name = data['robot_name'][0]
        hand = create_hand_model(robot_name.split('_')[1] if "retarget" in robot_name else robot_name, device)

        # Get robot point cloud from ddpm_qpos
        initial_q = hand.get_fixed_initial_q()

        if "oakink_all_dof" not in result_path:
            initial_q = torch.cat([data['ddpm_qpos'][0].float().to(device), initial_q[6:]])
        else:
            initial_q = torch.cat([data['ddpm_qpos'][0].float()[:6].to(device), initial_q[6:]])

        initial_q = initial_q.unsqueeze(0)

        # initial_q = data['nofix_initial_q'].to(device)
        robot_pc_initial = hand.get_transformed_links_pc(initial_q)[..., :3].unsqueeze(0).to(device)

        print(f"Grasp ID: {grasp_id}, Robot: {robot_name}, Object: {data['object_name'][0]}")
        print(f"Grasp sentence: {data['complex_language_sentence'][0]}")
        
        # Get object point cloud
        object_pc = _get_object_pc(
            data['object_name'][0], 
            data['object_id'][0], 
            data['robot_name'][0]
        ).to(device).unsqueeze(0)
        object_mesh, object_faces = get_object_vertices(data['object_name'][0], data['object_id'][0])

        predict_q = data['predict_q'][0].to(device).unsqueeze(0)
        robot_predict_mesh = hand.get_trimesh_q(predict_q)["visual"]

        # Get meshes for visualization
        robot_initial_mesh = hand.get_trimesh_q(initial_q)["visual"]
        robot_target_mesh = hand.get_trimesh_q(data['target_q'][0])["visual"]

        affordance_pred = process_affordance(openad_model, object_pc.squeeze(0), openad_cfg.training_cfg.train_affordance)

        server.scene.reset()
        visualize_results(
            object_pc,
            affordance_pred,
            openad_cfg.training_cfg.train_affordance,
            server.scene,
            color_map
        )

        # Visualize in viser
        server.scene.add_mesh_simple(
            name='ddpm_initial_hand',
            vertices=robot_initial_mesh.vertices,
            faces=robot_initial_mesh.faces,
            color=(192, 102, 255),
            opacity=0.5
        )

        server.scene.add_mesh_simple(
            name='target_hand',
            vertices=robot_target_mesh.vertices,
            faces=robot_target_mesh.faces,
            color=(255, 102, 192),
            opacity=0.5
        )

        # server.scene.add_point_cloud(
        #     name='object_pc',
        #     points=
        #     colors=(102, 192, 255),
        #     point_size=0.003,
        #     point_shape='circle'
        # )
        server.scene.add_mesh_simple(
            name='object_mesh',
            vertices=object_mesh,
            faces=object_faces,
            color=(102, 192, 255),
            opacity=0.5
        )

       
        server.scene.add_mesh_simple(
            name='predict_hand',
            vertices=robot_predict_mesh.vertices,
            faces=robot_predict_mesh.faces,
            color=(102, 192, 255),
            opacity=0.5
        )

        # Add chamfer distance to the scene
        server.scene.add_label(
            'chamfer_distance',
            f'Chamfer distance: {data["chamfer_value"]:0.4f}',
            position=(-0.2, 0.2, 0.2)
        )

        # Add grasp sentence to the scene
        server.scene.add_label(
            'grasp_sentence',
            f'Grasp {data["object_name"][0]}: {data["complex_language_sentence"][0]}',
            position=(0.2, 0.2, 0.2)
        )
        


    # Create slider for interaction
    grasp_slider = server.gui.add_slider(
        label='Grasp',
        min=0,
        max=len(results)-1,
        step=1,
        initial_value=0
    )
    
    def slider_update_callback(_):
        grasp_id = int(grasp_slider.value)
        on_update(grasp_id)

    grasp_slider.on_update(slider_update_callback)
    print("GUI initialized. Ready for interaction.")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()