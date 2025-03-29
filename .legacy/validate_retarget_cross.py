import os
import sys
import time
import warnings
from termcolor import cprint
import hydra
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from model.network import create_network
from data_utils.CombineDataset import create_dataloader
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac
import matplotlib.pyplot as plt
import viser
from mpl_toolkits.mplot3d import Axes3D

server = viser.ViserServer(host='127.0.0.1', port=8080)

def visualize_point_clouds(point_clouds, labels=None, colors=None, title="Point Clouds"):
    """
    Visualizes multiple 3D point clouds in a single plot.
    
    Args:
        point_clouds (list[torch.Tensor or np.ndarray]): List of point clouds, each with shape (N, 3).
        labels (list[str]): List of labels for each point cloud.
        colors (list[str]): List of colors for each point cloud.
        title (str): Title of the plot.
    """
    if labels is None:
        labels = [f"PointCloud {i}" for i in range(len(point_clouds))]
    if colors is None:
        colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Default colors for up to 6 point clouds

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, pc in enumerate(point_clouds):
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().detach().numpy()
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors[i % len(colors)], s=1, label=labels[i])

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


@hydra.main(version_base="1.2", config_path="configs", config_name="validate_retarget_cross")
# @hydra.main(version_base="1.2", config_path="configs", config_name="validate_origin")
# @hydra.main(version_base="1.2", config_path="configs", config_name="validate_retarget_intent")
def main(cfg):
    print("******************************** [Config] ********************************")
    print("dataset_name:", cfg.dataset.dataset_name)
    
    device = torch.device(f'cuda:{cfg.gpu}')
    batch_size = cfg.dataset.batch_size
    print(f"Device: {device}")
    print('Name:', cfg.name)

    os.makedirs(os.path.join(ROOT_DIR, 'validate_output'), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/{cfg.name}.log')
    print('Log file:', log_file_name)
    validate_epoch = cfg.validate_epochs[0]
    print(f"************************ Validating epoch {validate_epoch} ************************")
    with open(log_file_name, 'a') as f:
        print(f"************************ Validating epoch {validate_epoch} ************************", file=f)

    network = create_network(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{validate_epoch}.pth", map_location=device))
    network.eval()

    is_train = True
    dataloader = create_dataloader(cfg.dataset, is_train=is_train)

    time_list = []

    def on_update(grasp_id):
    
        try:

            data = dataloader.dataset[grasp_id]
            if is_train:
                robot_name = data['robot_name'][0]
                # object_name = data['object_name'][0]

                hand = create_hand_model(robot_name, device)

                initial_q = data['initial_q'][0].to(device).unsqueeze(0)
                robot_pc = data['robot_pc_initial'][0].to(device).unsqueeze(0)
                object_pc = data['object_pc'][0].to(device).unsqueeze(0)
                target_pc = data['robot_pc_target'][0].to(device).unsqueeze(0)
                target_q = data['target_q'][0].to(device).unsqueeze(0)
                
            else:
                robot_name = data['robot_name']
                # object_name = data['object_name']

                hand = create_hand_model(robot_name, device)

                initial_q = data['initial_q'].to(device)
                robot_pc = data['robot_pc'].to(device)
                object_pc = data['object_pc'].to(device)


            with torch.no_grad():
                dro = network(
                    robot_pc,
                    object_pc,
                    target_pc=target_pc,
                    # intent=torch.tensor([0], dtype=torch.int32, device=device).reshape(-1, 1),
                    visualize=False
                )['dro'].detach()

            mlat_pc = multilateration(dro, object_pc)
            transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
            optim_transform = process_transform(hand.pk_chain, transform)

            layer = create_problem(hand.pk_chain, optim_transform.keys())
            start_time = time.time()
            predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
            end_time = time.time()
            print(f"Optimization time: {end_time - start_time:.4f} s")
            time_list.append(end_time - start_time)

            # robot_predict_pc = hand.get_transformed_links_pc(predict_q)[:, :3].cpu().numpy()
            robot_predict_mesh = hand.get_trimesh_q(predict_q)["visual"]

            # robot_initial_pc = data['robot_pc_initial'][0].numpy()
            robot_initial_mesh = hand.get_trimesh_q(initial_q)["visual"]

            # robot_target_pc = data['robot_pc_target'][0].numpy()
            # robot_target_mesh = hand.get_trimesh_q(target_q)["visual"]

            server.scene.add_mesh_simple(
                name='initial_mesh',
                vertices=robot_initial_mesh.vertices,
                faces=robot_initial_mesh.faces,
                color=(192, 102, 255),
                opacity=0.5
            )

            server.scene.add_mesh_simple(
                name='predict_mesh',
                vertices=robot_predict_mesh.vertices,
                faces=robot_predict_mesh.faces,
                color=(102, 192, 255),
                opacity=0.5
            )

            # server.scene.add_mesh_simple(
            #     name='target_mesh',
            #     vertices=robot_target_mesh.vertices,
            #     faces=robot_target_mesh.faces,
            #     color=(255, 102, 192),
            #     opacity=0.5
            # )

            # server.scene.add_point_cloud(
            #     name='initial_pc',
            #     points=robot_initial_pc,
            #     colors=(255, 192, 102),
            #     point_size=0.003,
            #     point_shape='circle'
            # )

            # server.scene.add_point_cloud(
            #     name='predict_pc',
            #     points=robot_predict_pc,
            #     colors=(192, 102, 255),
            #     point_size=0.003,
            #     point_shape='circle'
            # )

            server.scene.add_point_cloud(
                name='object_pc',
                points=object_pc[0].cpu().numpy(),
                colors=(102, 192, 255),
                point_size=0.003,
                point_shape='circle'
            )

            # server.scene.add_point_cloud(
            #     name='target_pc',
            #     points=robot_target_pc,
            #     colors=(255, 192, 102),
            #     point_size=0.003,
            #     point_shape='circle'
            # )

        except Exception as e:
            print(e)

    grasp_slider = server.gui.add_slider(
        label='Grasp',
        min=0,
        max=99,
        step=1,
        initial_value=0
    )
        
    def slider_update_callback(_):
        grasp_id = int(grasp_slider.value)
        on_update(grasp_id)

    grasp_slider.on_update(slider_update_callback)
    print("GUI sliders initialized. Ready for interaction.")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()
