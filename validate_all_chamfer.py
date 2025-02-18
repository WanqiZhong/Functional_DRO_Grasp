# Validate and calcualte Chamfer Distance for the DRO-Grasp model with language embedding.

import os
import sys
import time
import warnings
import torch
import hydra
import numpy as np
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from model.network import create_network
from data_utils.CombineRetargetDataset import create_dataloader
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac

def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer Distance between two point clouds.

    Args:
        pc1 (torch.Tensor): Point cloud 1 of shape (N, 3).
        pc2 (torch.Tensor): Point cloud 2 of shape (M, 3).

    Returns:
        float: Chamfer Distance.
    """
    pc1 = pc1.unsqueeze(0)  # Shape: (1, N, 3)
    pc2 = pc2.unsqueeze(0)  # Shape: (1, M, 3)
    
    # Compute pairwise distances
    dist1 = torch.cdist(pc1, pc2, p=2)  # Shape: (1, N, M)
    dist2 = dist1.transpose(1, 2)        # Shape: (1, M, N)
    
    # For each point in pc1, find the minimum distance to pc2
    min_dist1 = torch.min(dist1, dim=2)[0]  # Shape: (1, N)
    # For each point in pc2, find the minimum distance to pc1
    min_dist2 = torch.min(dist2, dim=2)[0]  # Shape: (1, M)
    
    # Chamfer Distance is the sum of both minimum distances
    chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
    return chamfer_dist.item()

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_retarget_language")
# @hydra.main(version_base="1.2", config_path="configs", config_name="validate_origin")
# @hydra.main(version_base="1.2", config_path="configs", config_name="validate_retarget_intent")
def main(cfg):
    print("******************************** [Config] ********************************")
    print("dataset_name:", cfg.dataset.dataset_name)
    
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
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
    dataloader = create_dataloader(cfg.dataset, is_train=is_train, fix_sample=True, fixed_initial_q=False)

    chamfer_distances = []

    fixed_sample_indices = list(range(100))  

    total_samples = len(dataloader.dataset)
    num_samples = len(fixed_sample_indices)
    if num_samples > total_samples:
        fixed_sample_indices = list(range(total_samples))
        num_samples = total_samples
        print(f"Dataset contains only {total_samples} samples. Processing all available samples.")

    print(f"Processing {num_samples} samples with fixed indices: {fixed_sample_indices}")

    for grasp_id in fixed_sample_indices:
        try:
            data = dataloader.dataset[grasp_id]
            if is_train:
                robot_name = data['robot_name'][0]
                # object_name = data['object_name'][0]

                hand = create_hand_model(robot_name.split('_')[1] if "retarget" in robot_name else robot_name, device)

                initial_q = data['initial_q'][0].to(device).unsqueeze(0)
                robot_pc_initial = data['robot_pc_initial'][0].to(device).unsqueeze(0)
                object_pc = data['object_pc'][0].to(device).unsqueeze(0)
                retrieve_hand_pc_target = data['cross_pc_target'][0].to(device).unsqueeze(0)
                retrieve_object_pc = data['cross_object_pc'][0].to(device).unsqueeze(0)
                intent = data['intent'][0].to(device).unsqueeze(0)
                language_embedding = data['language_embedding'][0].to(device).unsqueeze(0)
                target_q = data['target_q'][0].to(device).unsqueeze(0)
            else:
                robot_name = data['robot_name']
                # object_name = data['object_name']
                hand = create_hand_model(robot_name.split('_')[1] if "retarget" in robot_name else robot_name, device)

                initial_q = data['initial_q'].to(device)
                robot_pc_initial = data['robot_pc_initial'].to(device).unsqueeze(0)
                object_pc = data['object_pc'].to(device).unsqueeze(0)
                target_q = data['target_q'].to(device).unsqueeze(0)

            with torch.no_grad():
                dro = network(
                    robot_pc_initial,
                    object_pc,
                    target_pc=retrieve_hand_pc_target,
                    cross_object_pc=retrieve_object_pc,
                    intent=intent,
                    language_emb=language_embedding,
                )['dro'].detach()

            mlat_pc = multilateration(dro, object_pc)
            transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
            optim_transform = process_transform(hand.pk_chain, transform)

            layer = create_problem(hand.pk_chain, optim_transform.keys())
            start_time = time.time()
            predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
            end_time = time.time()
            optimization_time = end_time - start_time
            print(f"Sample {grasp_id}: Optimization time: {optimization_time:.4f} s")


            robot_predict_pc = hand.get_transformed_links_pc(predict_q)[:, :3].cpu()
            robot_target_pc = data['robot_pc_target'][0]
            
            # 假设 predict_q 和 target_q 的形状为 (1, N, 3)，调整为 (N, 3)
            chamfer = chamfer_distance(robot_target_pc, robot_predict_pc)
            chamfer_distances.append(chamfer)
            print(f"Sample {grasp_id}: Chamfer Distance: {chamfer:.6f}")

        except Exception as e:
            print(f"Error processing sample {grasp_id}: {e}")

    if chamfer_distances:
        average_chamfer = sum(chamfer_distances) / len(chamfer_distances)
        print(f"\nAverage Chamfer Distance over {len(chamfer_distances)} samples: {average_chamfer:.6f}")
        with open(log_file_name, 'a') as f:
            print(f"Average Chamfer Distance over {len(chamfer_distances)} samples: {average_chamfer:.6f}", file=f)
    else:
        print("No Chamfer Distances were computed.")

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()