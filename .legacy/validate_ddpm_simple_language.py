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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

server = viser.ViserServer(host='127.0.0.1', port=8080)

def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def _get_object_pc(object_name, object_id, robot_name):
    name = object_name.split('+')
    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt') if (robot_name == 'mano' or robot_name == 'retarget_shadowhand') \
        else os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
    object_pc = torch.load(object_path)[:, :3]
    return object_pc

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_retarget_language_larger_transformer_openad_dgcnn_1")
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

    # Load network
    network = create_network_larger_transformer_openai_dgcnn(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{validate_epoch}.pth", map_location=device))
    network.eval()

    result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_03-22-31_6dof_oakink/eval/final_validate_data/2025-01-10_02-03-00/res_diffuser_1000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_18-07-23_oakink_all_dof/eval/final/2025-01-09_22-02-54/res_diffuser_100.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_18-07-23_oakink_all_dof/eval/final/2025-01-09_22-24-16/res_diffuser_1000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_18-07-23_oakink_all_dof/eval/final_validate_data/2025-01-10_02-03-02/res_diffuser_1000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-15_04-21-46_robot_rotmat_oakink/eval/final/2025-01-18_20-40-45/res_diffuser_3000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-15_04-21-46_robot_rotmat_oakink/eval/final_validate_data/2025-01-18_23-25-25/res_diffuser_3000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_15-39-57_6dof_oakink/eval/final/2025-01-09_22-02-57/res_diffuser_100.pkl"
    results = load_results(result_path)['results']

    print(len(results))

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
        print(robot_pc_initial.shape)

        print(f"Grasp ID: {grasp_id}, Robot: {robot_name}, Object: {data['object_name'][0]}")
        print(f"Grasp sentence: {data['complex_language_sentence'][0]}")
        
        # Get object point cloud
        object_pc = _get_object_pc(
            data['object_name'][0], 
            data['object_id'][0], 
            data['robot_name'][0]
        ).to(device).unsqueeze(0)

        # Network inference
        with torch.no_grad():
            dro = network(
                robot_pc_initial,
                object_pc,
                language_emb=data['complex_openai_embedding'][0].to(device)
                # language_emb=data['complex_language_embedding_openai_256'].to(device)
            )['dro'].detach()

        # Post-processing
        mlat_pc = multilateration(dro, object_pc)
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
        optim_transform = process_transform(hand.pk_chain, transform)

        # Optimization
        layer = create_problem(hand.pk_chain, optim_transform.keys())
        predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
        robot_predict_mesh = hand.get_trimesh_q(predict_q)["visual"]

        # Get meshes for visualization
        robot_initial_mesh = hand.get_trimesh_q(initial_q)["visual"]
        robot_target_mesh = hand.get_trimesh_q(data['target_q'][0])["visual"]

        # Visualize in viser
        server.scene.add_mesh_simple(
            name='initial_mesh',
            vertices=robot_initial_mesh.vertices,
            faces=robot_initial_mesh.faces,
            color=(192, 102, 255),
            opacity=0.5
        )

        server.scene.add_mesh_simple(
            name='target_mesh',
            vertices=robot_target_mesh.vertices,
            faces=robot_target_mesh.faces,
            color=(255, 102, 192),
            opacity=0.5
        )

        server.scene.add_point_cloud(
            name='object_pc',
            points=object_pc[0].cpu().numpy(),
            colors=(102, 192, 255),
            point_size=0.003,
            point_shape='circle'
        )

       
        server.scene.add_mesh_simple(
            name='predict_mesh',
            vertices=robot_predict_mesh.vertices,
            faces=robot_predict_mesh.faces,
            color=(102, 192, 255),
            opacity=0.5
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