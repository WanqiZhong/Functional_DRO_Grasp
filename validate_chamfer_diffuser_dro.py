import os
import sys
import time
import pickle
import warnings
import numpy as np
import torch
from chamfer_distance import ChamferDistance as chamfer_dist
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from model.network import create_network_larger_transformer_clip_add_dgcnn
import hydra
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def _get_object_pc(object_names, object_ids, robot_names, batch_size):
    object_pcs = []
    for i in range(batch_size):
        name = object_names[i].split('+')
        object_path = os.path.join(
            ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_ids[i]}.pt'
        ) if (name[0] == 'oakink') else os.path.join(
            ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt'
        )
        object_pc = torch.load(object_path)[:, :3]
        object_pcs.append(object_pc)
    return torch.stack(object_pcs)

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_clip_512_add_dgcnn")
def main(cfg):
    print("******************************** [Config] ********************************")    
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    batch_size = cfg.dataset.batch_size
    print(f"Device: {device}, Batch size: {batch_size}")

    os.makedirs(os.path.join(ROOT_DIR, 'validate_output'), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/{cfg.name}.log')
    print('Log file:', log_file_name)
    validate_epoch = cfg.validate_epochs[0]
    print(f"************************ Validating epoch {validate_epoch} ************************")
    with open(log_file_name, 'a') as f:
        print(f"************************ Validating epoch {validate_epoch} ************************", file=f)

    # Load the network model
    network = create_network_larger_transformer_clip_add_dgcnn(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{validate_epoch}.pth", map_location=device), strict=False)
    network.eval()

    result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-04-13_17-25-08_oakink_rotmat_custom_robot_pn2_object_pn2_three_split_ss_shadow/eval/final_validate_data/2025-04-18_16-54-47/res_diffuser_k3.pkl"

    res = load_results(result_path)
    results = res['results']
    print(f"Loading results from: {result_path}")
    print(f"Total data batches: {len(results)}")

    # Get k_sample from the data structure
    if results and 'ddpm_qpos' in results[0] and results[0]['ddpm_qpos'] is not None:
        k_sample = results[0]['ddpm_qpos'].shape[1]
        print(f"Detected k_sample: {k_sample}")
    else:
        k_sample = 3  # Default fallback
        print(f"Using default k_sample: {k_sample}")

    # Initialize lists to store Chamfer distances for each k
    chamfer_per_k = [[] for _ in range(k_sample)]
    robot_name = results[0]['robot_name'][0]
    hand = create_hand_model(robot_name.split('_')[1] if "retarget" in robot_name else robot_name, device)

    # Create k separate result structures for DRO predictions
    k_results = [{
        'method': f'diffuser_dro@k{k+1}',
        'desc': f'Diffuser DRO model with k={k+1}',
        'results': []
    } for k in range(k_sample)]

    for batch_idx, batch_data in enumerate(tqdm(results)):
        if 'ddpm_qpos' not in batch_data or batch_data['ddpm_qpos'] is None:
            print(f"Skipping batch {batch_idx} - no ddpm_qpos found")
            continue
            
        # Get the number of samples in this batch
        batch_size = batch_data['ddpm_qpos'].shape[0]
        
        # Get ground truth data for the entire batch
        gt_qs = batch_data['target_q']
        gt_pts_list = []
        for i in range(batch_size):
            gt_pts_list.append(hand.get_transformed_links_pc(gt_qs[i].unsqueeze(0))[..., :3])
        gt_pts = torch.stack(gt_pts_list).float().to(device)
        
        object_pcs = _get_object_pc(
            batch_data['object_name'],
            batch_data['object_id'],
            batch_data['robot_name'],
            batch_size
        ).to(device)
        
        # Process each k value separately
        for k in range(k_sample):
            # Extract the k-th predictions for the batch
            diffuser_qposes = batch_data['ddpm_qpos'][:, k]
            
            # Initialize empty lists
            predict_qs = []
            initial_qs = []
            chamfer_values = []            
            robot_pc_initials = []

            # Process each batch sample for this k
            for sample_idx in range(batch_size):
                # Create initial_q for each sample
                initial_q = hand.get_initial_q()
                diffuser_qpos = diffuser_qposes[sample_idx]
                
                initial_q = torch.cat([torch.tensor(diffuser_qpos).float().to(device)[:6], initial_q[6:]])
                initial_qs.append(initial_q)
                initial_q = initial_q.unsqueeze(0)

                robot_pc_initials.append(hand.get_transformed_links_pc(initial_q)[..., :3])
            
            initial_qs = torch.stack(initial_qs).to(device)
            robot_pc_initials = torch.stack(robot_pc_initials).to(device)
            # Prepare language embeddings in batch
            language_embs = torch.tensor(batch_data['complex_language_embedding_clip_512']).to(device)
            
            # Forward pass through the network (batch processing)
            with torch.no_grad():
                dros = network(
                    robot_pc_initials,
                    object_pcs,
                    language_emb=language_embs
                )['dro'].detach()
            
            mlat_pc = multilateration(dros, object_pcs)
            transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
            optim_transform = process_transform(hand.pk_chain, transform)
            layer = create_problem(hand.pk_chain, optim_transform.keys())
            predict_q = optimization(hand.pk_chain, layer, initial_qs, optim_transform)

            # Compute predicted point clouds for each sample
            pred_pts_list = []
            for i in range(batch_size):
                pred_pts_list.append(hand.get_transformed_links_pc(predict_q[i])[..., :3])
            pred_pts = torch.stack(pred_pts_list).float().to(device)
            
            # Calculate Chamfer distances in batch
            try:
                d1, d2, _, _ = chamfer_dist(x=pred_pts, y=gt_pts)
            except:
                d1, d2, _, _ = chamfer_dist()(x=pred_pts, y=gt_pts)

            chamfer_values = (d1.sum(dim=1) + d2.sum(dim=1)).detach().cpu().numpy()
            
            # Sum along appropriate dimensions for each sample
            for sample_idx in range(batch_size):
                chamfer_value = chamfer_values[sample_idx].item()
                chamfer_per_k[k].append(chamfer_value)
                
                # Create individual sample data for this k
                sample_data = {key: batch_data[key][sample_idx].cpu().numpy() if isinstance(batch_data[key], torch.Tensor) else 
                                    batch_data[key][sample_idx] for key in batch_data if batch_data[key] is not None}
                
                sample_data['predict_q'] = predict_q[sample_idx].cpu().numpy()
                sample_data['ddpm_initial_q'] = initial_qs[sample_idx].cpu().numpy()
                sample_data['chamfer_value'] = chamfer_value
                sample_data['batch_idx'] = batch_idx
                sample_data['sample_idx'] = sample_idx
                sample_data['k_value'] = k
                
                k_results[k]['results'].append(sample_data)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed batch {batch_idx}, k={k+1}, mean Chamfer: {np.mean(chamfer_values):.6f}")

    # Calculate statistics for each k and save separate files
    output_dir = os.path.dirname(result_path)
    
    for k in range(k_sample):
        # Filter out NaN values
        valid_values = [data['chamfer_value'] for data in k_results[k]['results'] 
                        if 'chamfer_value' in data and not np.isnan(data['chamfer_value'])]
        
        if valid_values:
            mean_chamfer = sum(valid_values) / len(valid_values)
            std_chamfer = np.std(valid_values) if len(valid_values) > 1 else 0.0
        else:
            mean_chamfer = float('nan')
            std_chamfer = float('nan')
        
        chamfer_stats = {
            'mean': float(mean_chamfer),
            'std': float(std_chamfer),
            'valid_samples': len(valid_values),
            'invalid_samples': len(k_results[k]['results']) - len(valid_values)
        }
        
        print(f"K={k+1}: Mean Chamfer={mean_chamfer:.6f}, Std={std_chamfer:.6f}, "
              f"Valid={len(valid_values)}, Invalid={len(k_results[k]['results']) - len(valid_values)}")
        
        # Add stats to the results
        k_results[k]['chamfer_stats'] = chamfer_stats
        
        # Save this k's results to a separate file
        k_file_path = os.path.join(output_dir, f"res_diffuser_dro_k{k+1}_chamfer_batch.pkl")
        with open(k_file_path, 'wb') as f:
            pickle.dump(k_results[k], f)
        print(f"Results for k={k+1} saved to: {k_file_path}")

    # Calculate overall statistics across all k values
    all_means = [k_results[k]['chamfer_stats']['mean'] for k in range(k_sample) 
                if not np.isnan(k_results[k]['chamfer_stats']['mean'])]
    
    if all_means:
        overall_mean = sum(all_means) / len(all_means)
        overall_std = np.std(all_means) if len(all_means) > 1 else 0.0
    else:
        overall_mean = float('nan')
        overall_std = float('nan')
    
    overall_stats = {
        'mean': float(overall_mean),
        'std': float(overall_std),
        'means_per_k': [float(k_results[k]['chamfer_stats']['mean']) for k in range(k_sample)],
        'all_k_stats': [k_results[k]['chamfer_stats'] for k in range(k_sample)]
    }
    
    print(f"Overall: Mean of k means={overall_mean:.6f}, Std across k={overall_std:.6f}")
    
    # Save overall statistics
    stats_path = os.path.join(output_dir, f"dro_chamfer_stats_all_k_batch.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(overall_stats, f)
    print(f"Overall DRO Chamfer statistics saved to: {stats_path}")

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()