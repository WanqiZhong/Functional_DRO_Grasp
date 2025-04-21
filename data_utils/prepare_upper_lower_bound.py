import os
import torch
import numpy as np
from tqdm import tqdm
from DRO_Grasp.utils.rotation import euler_to_matrix  # (B,3) -> (B,3,3)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def process_dataset(path):
    data = torch.load(path, map_location='cpu')
    metadata = data['metadata']

    trans_list = []
    rotmat_list = []

    for m in metadata:
        q = m[3]  # target_q: [x, y, z, roll, pitch, yaw]
        trans = q[:3].unsqueeze(0)  # (1, 3)
        rpy   = q[3:6].unsqueeze(0)  # (1, 3)

        R = euler_to_matrix(rpy)[0]  # (3, 3)
        trans_list.append(trans.numpy()[0])
        rotmat_list.append(R.numpy().reshape(-1))  # flatten to (9,)

    trans_all = np.stack(trans_list, axis=0)
    rot_all   = np.stack(rotmat_list, axis=0)

    return trans_all, rot_all


def format_tensor(name, array):
    lines = ", ".join([f"{x:.8f}" for x in array])
    return f"{name} = torch.tensor([{lines}])"


robot_names = ['shadowhand', 'leaphand', 'allegro']
output_lines = ["import torch\n"]

# 全体累计
all_trans_all = []
all_rot_all = []

for robot in tqdm(robot_names, desc="Processing robots"):
    path = os.path.join(
        ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot}_valid_dro.pt'
    )
    trans_all, rot_all = process_dataset(path)

    # 累加用于全体统计
    all_trans_all.append(trans_all)
    all_rot_all.append(rot_all)

    # 当前 robot 的 min/max
    trans_min = trans_all.min(axis=0)
    trans_max = trans_all.max(axis=0)
    rot_min = rot_all.min(axis=0)
    rot_max = rot_all.max(axis=0)

    output_lines.append(f"\n# === {robot.upper()} ===")
    output_lines.append(format_tensor(f"_global_trans_lower_{robot}", trans_min))
    output_lines.append(format_tensor(f"_global_trans_upper_{robot}", trans_max))
    output_lines.append(format_tensor(f"_rot_matrix_lower_{robot}", rot_min))
    output_lines.append(format_tensor(f"_rot_matrix_upper_{robot}", rot_max))

# 拼接全体数据
all_trans_all = np.concatenate(all_trans_all, axis=0)
all_rot_all = np.concatenate(all_rot_all, axis=0)

trans_min = all_trans_all.min(axis=0)
trans_max = all_trans_all.max(axis=0)
rot_min   = all_rot_all.min(axis=0)
rot_max   = all_rot_all.max(axis=0)

output_lines.append("\n# === ALL ROBOTS COMBINED ===")
output_lines.append(format_tensor("_global_trans_lower_all", trans_min))
output_lines.append(format_tensor("_global_trans_upper_all", trans_max))
output_lines.append(format_tensor("_rot_matrix_lower_all", rot_min))
output_lines.append(format_tensor("_rot_matrix_upper_all", rot_max))

out_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/robot_transform_bounds.py')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    f.write("\n".join(output_lines))
print(f"Bounds saved to {out_path}")

