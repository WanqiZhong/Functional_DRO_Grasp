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

from data_utils.CombineDataset import create_dataloader
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac
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
        

    dataset_name = 'predict'

    if dataset_name == 'predict':
    
        # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-02-16_09-58-23_oakink_rotmat_all_custom_robot_pn2_object_pn2_new_spilt_short_sentence/eval/final_validate_data/2025-02-20_17-35-30/res_diffuser_dro_predict_q.pkl"
        
        result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-02-17_21-34-30_oakink_rotmat_custom_robot_pn2_object_pn2_new_spilt_short_sentence/eval/final_validate_data/2025-02-19_16-36-29/res_diffuser_dro_predict_q.pkl"

        print(f"Loading results from: {result_path}")
        results = load_results(result_path)['results'][:20]

    elif dataset_name == 'oakink':

        result_path = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand.pt"
        # result_path = "/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_allegro.pt"
        results = torch.load(result_path)['metadata']

    elif dataset_name == 'cmap':

        result_path = "/data/zwq/code/DRO_Grasp/data/CMapDataset/cmap_dataset.pt"
        results = torch.load(result_path)['metadata']

    
    global_robot_name = None
    hand = None
    all_success_q = []
    time_list = []
    success_num = 0
    total_num = 0
    vis_info = []
    
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

        elif dataset_name == 'cmap':
            predict_q = data[1]
            object_name = data[2]
            robot_name = data[3]
            object_id = None

        cprint(f"[{robot_name}/{object_name}/{object_id}]", 'light_blue', end=' ')
        if robot_name != global_robot_name:
            if global_robot_name is not None:
                all_success_q = torch.cat(all_success_q, dim=0)
                diversity_std = torch.std(all_success_q, dim=0).mean()
                times = np.array(time_list)
                time_mean = np.mean(times)
                time_std = np.std(times)

                success_rate = success_num / total_num * 100
                cprint(f"[{global_robot_name}]", 'magenta', end=' ')
                cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ')
                cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ')
                cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue')
                with open(log_file_name, 'a') as f:
                    f.write(f"[{global_robot_name}] Result: {success_num}/{total_num}({success_rate:.2f}%) Std: {diversity_std:.3f} Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s\n")
                
                all_success_q = []
                time_list = []
                success_num = 0
                total_num = 0
            global_robot_name = robot_name

        predict_q = predict_q.to(device).unsqueeze(0)
        success, isaac_q = validate_isaac(robot_name, object_name, predict_q, gpu=cfg.gpu, dataset_name=dataset_name, object_id=object_id)
        succ_num = success.sum().item() if success is not None else -1
        success_q = predict_q[success]
        all_success_q.append(success_q)
        
        vis_info.append({
            'robot_name': robot_name,
            'object_name': object_name,
            'predict_q': predict_q,
            'success': success,
            'isaac_q': isaac_q,
            'object_id': object_id
        })
        
        cprint(f"[{robot_name}/{object_name}]", 'light_blue', end=' ')
        cprint(f"Result: {succ_num}/{batch_size}({succ_num / batch_size * 100:.2f}%)", 'green')
        with open(log_file_name, 'a') as f:
            f.write(f"[{robot_name}/{object_name}] Result: {succ_num}/{batch_size}({succ_num / batch_size * 100:.2f}%)\n")
            
        success_num += succ_num
        total_num += batch_size

    all_success_q = torch.cat(all_success_q, dim=0)
    diversity_std = torch.std(all_success_q, dim=0).mean()
    times = np.array(time_list)
    time_mean = np.mean(times)
    time_std = np.std(times)
    success_rate = success_num / total_num * 100
    
    cprint(f"[{global_robot_name}]", 'magenta', end=' ')
    cprint(f"Result: {success_num}/{total_num}({success_rate:.2f}%)", 'yellow', end=' ')
    cprint(f"Std: {diversity_std:.3f}", 'cyan', end=' ')
    cprint(f"Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s", 'blue')
    with open(log_file_name, 'a') as f:
        f.write(f"[{global_robot_name}] Result: {success_num}/{total_num}({success_rate:.2f}%) Std: {diversity_std:.3f} Time: (mean) {time_mean:.2f} s, (std) {time_std:.2f} s\n")
    
    vis_info_file = f'{cfg.name}_epoch{validate_epoch}'
    os.makedirs(os.path.join(ROOT_DIR, 'vis_info'), exist_ok=True)
    torch.save(vis_info, os.path.join(ROOT_DIR, f'vis_info/{vis_info_file}.pt'))

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()
