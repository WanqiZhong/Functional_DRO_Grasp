import os
import sys
import time
import warnings
from termcolor import cprint
import hydra
import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from DRO_Grasp.data_utils.CombineDataset import create_dataloader
from DRO_Grasp.utils.multilateration import multilateration
from DRO_Grasp.utils.se3_transform import compute_link_pose
from DRO_Grasp.utils.optimization import *
from DRO_Grasp.utils.hand_model import create_hand_model
from DRO_Grasp.validation.validate_utils import validate_isaac_multi
import matplotlib.pyplot as plt
import pickle

def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def check_urdf(object_name, object_id):
    name = object_name.split('+')
    object_urdf_path = f'{name[0]}/{name[1]}/coacd_decomposed_object_one_link_{object_id}.urdf'
    data_urdf_path = os.path.join(ROOT_DIR, 'data/data_urdf')
    if not os.path.exists(os.path.join(data_urdf_path, 'object', object_urdf_path)):
        return False
    return True

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_retarget_language_larger_transformer_clip_512_dgcnn_add")
def main(cfg):
    print("******************************** [Config] ********************************")
    print("dataset_name:", cfg.dataset.dataset_name)
    
    device = torch.device(f'cuda:{cfg.gpu}')
    batch_size = 50  # Set batch size to 50 as requested
    print(f"Device: {device}")
    print('Name:', cfg.name)

    os.makedirs(os.path.join(ROOT_DIR, 'validate_output'), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/{cfg.name}.log')
    print('Log file:', log_file_name)
    
    validate_epoch = cfg.validate_epochs[0]
    print(f"************************ Validating epoch {validate_epoch} ************************")
    with open(log_file_name, 'a') as f:
        print(f"************************ Validating epoch {validate_epoch} ************************", file=f)
        
    dataset_name = 'predict'

    if dataset_name == 'predict':
        result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-02-17_21-34-30_oakink_rotmat_custom_robot_pn2_object_pn2_new_spilt_short_sentence/eval/final_validate_data/2025-02-19_16-36-29/res_diffuser_dro_predict_q.pkl"
        # result_path = "/data/zwq/code/DRO_Grasp/results_with_predict_q_10000.pkl"
        # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-02-16_09-58-23_oakink_rotmat_all_custom_robot_pn2_object_pn2_new_spilt_short_sentence/eval/final_validate_data/2025-02-20_18-55-58/res_diffuser_dro_predict_q.pkl"
        # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-02-16_09-58-23_oakink_rotmat_all_custom_robot_pn2_object_pn2_new_spilt_short_sentence/eval/final_validate_data/2025-02-20_18-55-58/res_diffuser_only_predict_q.pkl"
        print(f"Loading results from: {result_path}")
        results = load_results(result_path)['results'][:100]
    elif dataset_name == 'oakink':
        result_path = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand.pt"
        results = torch.load(result_path)['metadata'][100:200]
    elif dataset_name == 'cmap':
        result_path = "/data/zwq/code/DRO_Grasp/data/CMapDataset/cmap_dataset.pt"
        results = torch.load(result_path)['metadata'][:100]

    # Group data by robot_name to process in batches
    robot_data_map = {}
    
    for i, data in enumerate(results):
        if dataset_name == 'predict':
            robot_name = data['robot_name'][0].replace('retarget_', '')
            object_name = data['object_name'][0]
            object_id = data['object_id'][0]
            # predict_q = data['predict_q'][0]
            predict_q = data['target_q'][0]
            if not check_urdf(object_name, object_id):
                cprint(f"Object {object_name} with id {object_id} not found, skipping...", 'red')
                continue
        elif dataset_name == 'oakink':
            hand_pose, hand_shape, tsl, predict_q, intent, object_name, robot_name, object_id = results[i][:8]
            if not check_urdf(object_name, object_id):
                cprint(f"Object {object_name} with id {object_id} not found, skipping...", 'red')
                continue
        elif dataset_name == 'cmap':
            predict_q = data[1]
            object_name = data[2]
            robot_name = data[3]
            object_id = None

        if robot_name not in robot_data_map:
            robot_data_map[robot_name] = []
            
        robot_data_map[robot_name].append({
            'robot_name': robot_name,
            'object_name': object_name,
            'object_id': object_id,
            'predict_q': predict_q
        })
    
    # Process each robot's data in batches
    for robot_name, data_list in robot_data_map.items():
        cprint(f"Processing robot: {robot_name}", 'magenta')
        
        # Process in batches of 50
        all_success_q = []
        time_list = []
        success_num = 0
        total_num = 0
        vis_info = []
        
        for batch_start in range(0, len(data_list), batch_size):
            batch_end = min(batch_start + batch_size, len(data_list))
            current_batch = data_list[batch_start:batch_end]
            current_batch_size = len(current_batch)
            
            # Prepare batch data
            batch_object_names = [item['object_name'] for item in current_batch]
            batch_object_ids = [item['object_id'] for item in current_batch]
            batch_predict_qs = torch.stack([item['predict_q'] for item in current_batch]).to(device)
            
            cprint(f"Batch {batch_start//batch_size + 1}: Processing {current_batch_size} objects", 'light_blue')
            
            # Start timer
            start_time = time.time()
            
            # Validate batch
            batch_success, batch_isaac_q = validate_isaac_multi(
                robot_name, 
                batch_object_names, 
                batch_predict_qs, 
                gpu=cfg.gpu, 
                dataset_name=dataset_name, 
                object_ids=batch_object_ids
            )
            
            # Calculate time
            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            
            # Process results
            batch_succ_num = batch_success.sum().item() if batch_success is not None else 0
            success_q = batch_predict_qs[batch_success] if batch_success is not None and batch_success.sum() > 0 else torch.tensor([])
            
            if len(success_q) > 0:
                all_success_q.append(success_q)
            
            # Store visualization info
            for i in range(current_batch_size):
                vis_info.append({
                    'robot_name': robot_name,
                    'object_name': batch_object_names[i],
                    'predict_q': batch_predict_qs[i].unsqueeze(0),
                    'success': batch_success[i] if batch_success is not None else None,
                    'isaac_q': batch_isaac_q[i] if batch_isaac_q is not None else None,
                    'object_id': batch_object_ids[i]
                })
            
            # Log batch results
            success_rate = (batch_succ_num / current_batch_size) * 100 if current_batch_size > 0 else 0
            cprint(f"Batch result: {batch_succ_num}/{current_batch_size} ({success_rate:.2f}%)", 'green')
            with open(log_file_name, 'a') as f:
                f.write(f"Batch {batch_start//batch_size + 1}: {batch_succ_num}/{current_batch_size} ({success_rate:.2f}%), Time: {elapsed_time:.2f}s\n")
            
            success_num += batch_succ_num
            total_num += current_batch_size
        
        # Calculate overall statistics for this robot
        if all_success_q:
            all_success_q = torch.cat(all_success_q, dim=0)
            diversity_std = torch.std(all_success_q, dim=0).mean()
        else:
            diversity_std = 0.0
            
        times = np.array(time_list)
        time_mean = np.mean(times) if len(times) > 0 else 0
        time_std = np.std(times) if len(times) > 0 else 0
        success_rate = (success_num / total_num) * 100 if total_num > 0 else 0
        
        # Log overall results for this robot
        cprint(f"[{robot_name}] Overall Result: {success_num}/{total_num} ({success_rate:.2f}%)", 'yellow', end=' ')
        cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ')
        cprint(f"Time: (mean) {time_mean:.2f}s, (std) {time_std:.2f}s", 'blue')
        
        with open(log_file_name, 'a') as f:
            f.write(f"[{robot_name}] Result: {success_num}/{total_num} ({success_rate:.2f}%) Std: {diversity_std:.3f} Time: (mean) {time_mean:.2f}s, (std) {time_std:.2f}s\n")
    
    # Save visualization info
    vis_info_file = f'{cfg.name}_oakink_epoch{validate_epoch}'
    os.makedirs(os.path.join(ROOT_DIR, 'vis_info'), exist_ok=True)
    torch.save(vis_info, os.path.join(ROOT_DIR, f'vis_info/{vis_info_file}.pt'))

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()