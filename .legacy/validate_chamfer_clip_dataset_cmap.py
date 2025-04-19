import os
import sys
import time
import pickle
import warnings
import torch
from chamfer_distance import ChamferDistance as chamfer_dist
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from data_utils.CombineRetargetDatasetMulti import create_dataloader
from utils.optimization import *
from model.network import *
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

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_origin")
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

    network = create_network(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{validate_epoch}.pth", map_location=device), strict=False)
    network.eval()

    dataloader = create_dataloader(cfg.dataset, is_train=False, fixed_initial_q=False)

    chamfer_list = []
    results = []
    robot_name = 'shadowhand'
    max_samples = 25
    hand = create_hand_model(robot_name.split('_')[1] if "retarget" in robot_name else robot_name, device)

    for batch_data in tqdm(dataloader):

        if(len(results) >= max_samples):
            break

        initial_q = torch.stack(batch_data['initial_q']).to(device)
        robot_pc_initial = batch_data['robot_pc_initial'].to(device)
        object_pc = batch_data['object_pc'].to(device)
        gt_q = torch.stack(batch_data['target_q']).to(device)

        with torch.no_grad():
            dro = network(
                robot_pc_initial,
                object_pc
            )['dro'].detach()

        mlat_pc = multilateration(dro, object_pc)
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
        optim_transform = process_transform(hand.pk_chain, transform)

        layer = create_problem(hand.pk_chain, optim_transform.keys())
        predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)

        batch_data['predict_q'] = predict_q.cpu()

        for item_idx in range(len(batch_data['initial_q'])):
            item = {}
            for key in batch_data:
                try:
                    item_data = batch_data[key][item_idx]
                    if isinstance(item_data, torch.Tensor):
                        item[key] = item_data.cpu().clone()
                    else:
                        item[key] = item_data
                except Exception as e:
                    pass           

            pred_pts = hand.get_transformed_links_pc(predict_q[item_idx])[..., :3].to(device).unsqueeze(0)
            gt_pts = hand.get_transformed_links_pc(gt_q[item_idx])[..., :3].to(device).unsqueeze(0)
            try:
                d1, d2, _, _ = chamfer_dist(x=pred_pts, y=gt_pts)
            except:
                d1, d2, _, _ = chamfer_dist()(x=pred_pts, y=gt_pts)

            chamfer_value = (d1.sum() + d2.sum()).cpu().item()
            chamfer_list.append(chamfer_value)

            item['chamfer_value'] = chamfer_value
            results.append(item)

        print(f"Processed {len(batch_data['initial_q'])} samples, Avg Chamfer Distance: {sum(chamfer_list)/len(chamfer_list)}")

    average_chamfer = sum(chamfer_list) / len(chamfer_list) if chamfer_list else float('nan')
    print(f"Overall Average Chamfer Distance: {average_chamfer}")

    new_result_path = os.path.join(f"output/{cfg.name}/res_diffuser_dro_predict_q.pt")
    torch.save(results, new_result_path)
    print(f"Updated results with predict_q saved to: {new_result_path}")

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()
