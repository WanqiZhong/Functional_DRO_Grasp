import os
import sys
import time
import pickle
import warnings
import numpy as np
import torch
from chamfer_distance import ChamferDistance
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
import hydra
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def _get_object_pc(object_name, object_id, robot_name):
    name = object_name.split('+')
    object_path = os.path.join(
        ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt'
    ) if (robot_name == 'mano' or robot_name == 'retarget_shadowhand') else os.path.join(
        ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt'
    )
    object_pc = torch.load(object_path)[:, :3]
    return object_pc

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_clip_512_add_dgcnn")
def main(cfg):
    print("******************************** [Config] ********************************")    
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    batch_size = cfg.dataset.batch_size
    print(f"Device: {device}, Batch size: {batch_size}")

    os.makedirs(os.path.join(ROOT_DIR, 'validate_output'), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/{cfg.name}.log')
    print('Log file:', log_file_name)

    result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-04-13_17-25-11_oakink_rotmat_all_robot_pn2_object_pn2_three_split_ss_shadow/eval/final_validate_data/2025-04-18_16-53-04/res_diffuser_k3.pkl"
   
    res = load_results(result_path)
    results = res['results']
    print(f"Total samples: {len(results)}")

    # Get k_sample from the data structure by examining the first batch
    if results and 'ddpm_qpos' in results[0] and results[0]['ddpm_qpos'] is not None:
        k_sample = results[0]['ddpm_qpos'].shape[1]
        print(f"Detected k_sample: {k_sample}")
    else:
        k_sample = 3  # Default fallback
        print(f"Using default k_sample: {k_sample}")

    # Initialize Chamfer distance calculator
    try:
        chamfer_fn = ChamferDistance()
    except:
        chamfer_fn = ChamferDistance

    # Initialize lists to store Chamfer distances for each k
    chamfer_per_k = [[] for _ in range(k_sample)]
    robot_name = 'shadowhand'
    hand = create_hand_model(robot_name.split('_')[1] if "retarget" in robot_name else robot_name, device)

    # Create k separate result structures
    k_results = [{
        'method': f'diffuser@k{k+1}',
        'desc': f'Diffuser model with k={k+1}',
        'results': []
    } for k in range(k_sample)]

    for batch_idx, batch_data in enumerate(tqdm(results)):
        if 'ddpm_qpos' not in batch_data or batch_data['ddpm_qpos'] is None:
            print(f"Skipping batch {batch_idx} - no ddpm_qpos found")
            continue
            
        # Get predictions for this batch
        batch_size = batch_data['ddpm_qpos'].shape[0]
        
        # Handle ground truth data for the entire batch at once
        try:
            gt_q = torch.stack(batch_data['target_q']).float().to(device)
            gt_pts = []
            for i in range(batch_size):
                gt_pts.append(hand.get_transformed_links_pc(gt_q[i])[..., :3])
            gt_pts = torch.stack(gt_pts).float().to(device)
        except Exception as e:
            print(f"Error processing ground truth for batch {batch_idx}: {e}")
            continue
        
        # Process each k sample for the entire batch
        for k in range(k_sample):
            try:
                # Extract the k-th prediction for all samples in batch
                predict_q = torch.tensor(batch_data['ddpm_qpos'][:, k]).float().to(device)
                
                # Calculate the point cloud for the predicted poses (batch)
                pred_pts = []
                for i in range(batch_size):
                    pred_pts.append(hand.get_transformed_links_pc(predict_q[i])[..., :3])
                pred_pts = torch.stack(pred_pts).float().to(device)

                print(f"Batch {batch_idx}, k={k+1}: Predicted points shape: {pred_pts.shape}, GT points shape: {gt_pts.shape}")
                
                # Calculate Chamfer distance for the entire batch at once
                try:
                    d1, d2, _, _ = chamfer_fn(x=pred_pts, y=gt_pts)
                except:
                    d1, d2, _, _ = chamfer_fn()(x=pred_pts, y=gt_pts)
                
                # Sum distances for each sample in the batch
                chamfer_values = (d1.sum(dim=1) + d2.sum(dim=1)).detach().cpu().numpy()
                
                # Store results for each sample in the batch
                for sample_idx in range(batch_size):
                    chamfer_value = chamfer_values[sample_idx].item()
                    chamfer_per_k[k].append(chamfer_value)
                    
                    # Create individual sample data for this k
                    sample_data = {key: batch_data[key][sample_idx].cpu().numpy() if isinstance(batch_data[key], torch.Tensor) else 
                                    batch_data[key][sample_idx] for key in batch_data if batch_data[key] is not None}
                    
                    sample_data['predict_q'] = predict_q[sample_idx].cpu().numpy()
                    sample_data['chamfer_value'] = chamfer_value
                    sample_data['batch_idx'] = batch_idx
                    sample_data['sample_idx'] = sample_idx
                    sample_data['k_value'] = k
                    
                    k_results[k]['results'].append(sample_data)
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}, k={k+1}: {e}")
                # Add samples with error indicator
                for sample_idx in range(batch_size):
                    sample_data = {key: batch_data[key][sample_idx].cpu().numpy() if isinstance(batch_data[key], torch.Tensor) else 
                                    batch_data[key][sample_idx] for key in batch_data if batch_data[key] is not None}
                    
                    sample_data['predict_q'] = batch_data['ddpm_qpos'][sample_idx][k] if k < batch_data['ddpm_qpos'].shape[1] else None
                    sample_data['chamfer_value'] = float('nan')
                    sample_data['error'] = str(e)
                    sample_data['batch_idx'] = batch_idx
                    sample_data['sample_idx'] = sample_idx
                    
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
        k_file_path = os.path.join(output_dir, f"res_diffuser_k{k+1}_chamfer.pkl")
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
    stats_path = os.path.join(output_dir, f"chamfer_stats_all_k.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(overall_stats, f)
    print(f"Overall Chamfer statistics saved to: {stats_path}")

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()