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
from utils.optimization import *
from data_utils.CombineRetargetDataset import create_dataloader
from model.network import create_network_larger_transformer_openai, create_network_larger_transformer_openai_dgcnn, create_network_larger_transformer_openai_openad_dgcnn
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

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_retarget_language_larger_transformer_openad_opaddgcnn")
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

    # 加载网络模型
    network = create_network_larger_transformer_openai_openad_dgcnn(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{validate_epoch}.pth", map_location=device))
    network.eval()


    result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_03-22-31_6dof_oakink/eval/final_validate_data/2025-01-10_02-03-00/res_diffuser_1000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-19_00-37-49_robot_rotmat_openad_oakink/eval/final_validate_data/2025-01-20_18-15-45/res_diffuser_100.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_18-07-23_oakink_all_dof/eval/final_validate_data/2025-01-10_02-03-02/res_diffuser_1000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-15_04-21-46_robot_rotmat_oakink/eval/final/2025-01-18_20-40-45/res_diffuser_3000.pkl"
    # result_path = "/data/zwq/code/Scene-Diffuser/outputs/2025-01-15_04-21-46_robot_rotmat_oakink/eval/final_validate_data/2025-01-18_23-25-25/res_diffuser_3000.pkl"

    results = load_results(result_path)['results'][:100]
    print(f"Total samples: {len(results)}")

    chamfer_list = []
    robot_name = 'shadowhand'
    hand = create_hand_model(robot_name.split('_')[1] if "retarget" in robot_name else robot_name, device)


    for idx, data in enumerate(tqdm(results)):
        # 创建手部模型

        # 初始化手部姿态
        initial_q = hand.get_fixed_initial_q()
        if "oakink_all_dof" not in result_path:
            initial_q = torch.cat([data['ddpm_qpos'][0].float().to(device), initial_q[6:]])
        else:
            initial_q = torch.cat([data['ddpm_qpos'][0].float()[:6].to(device), initial_q[6:]])
        initial_q = initial_q.unsqueeze(0)

        robot_pc_initial = hand.get_transformed_links_pc(initial_q)[..., :3].unsqueeze(0).to(device)

        # 获取物体点云
        object_pc = _get_object_pc(
            data['object_name'][0], 
            data['object_id'][0], 
            data['robot_name'][0]
        ).to(device).unsqueeze(0)

        # 网络推理
        with torch.no_grad():
            dro = network(
                robot_pc_initial,
                object_pc,
                # language_emb=data['complex_language_embedding_openai_256'].to(device)
                language_emb=data['complex_openai_embedding'].to(device)
            )['dro'].detach()

        # 后处理与优化
        mlat_pc = multilateration(dro, object_pc)
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
        optim_transform = process_transform(hand.pk_chain, transform)

        layer = create_problem(hand.pk_chain, optim_transform.keys())
        predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)

        # 将计算得到的 predict_q 保存到当前样本的字典中
        data['predict_q'] = predict_q.cpu()

        # 计算 Chamfer 距离
        pred_pts = hand.get_transformed_links_pc(predict_q)[..., :3].unsqueeze(0).to(device)
        gt_q = data['target_q'][0].unsqueeze(0).to(device)
        gt_pts = hand.get_transformed_links_pc(gt_q)[..., :3].unsqueeze(0).to(device)
        d1, d2, _, _ = chamfer_dist(x=pred_pts, y=gt_pts)
        chamfer_value = (d1.sum() + d2.sum()).item()
        chamfer_list.append(chamfer_value)

        data['chamfer_value'] = chamfer_value


        print(f"Processed sample {idx+1}/{len(results)}, Chamfer distance: {chamfer_value}")

    average_chamfer = sum(chamfer_list) / len(chamfer_list) if chamfer_list else float('nan')
    print(f"Average Chamfer Distance over {len(results)} samples: {average_chamfer}")

    new_result_path = os.path.join(ROOT_DIR, 'results_with_predict_q.pkl')
    with open(new_result_path, 'wb') as f:
        pickle.dump({'results': results}, f)
    print(f"Updated results with predict_q saved to: {new_result_path}")

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()