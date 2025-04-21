import torch
import numpy as np
import os
import sys
from collections import defaultdict
import time
from tqdm import tqdm
from utils.func_utils import get_contact_map, get_euclidean_distance, get_aligned_distance, timed
from utils.hand_model import create_hand_model, HandModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:5")

BATCH_SIZE = 1
NUM_HAND_POINTS = 8192
NUM_OBJ_POINTS = 65536 


# @timed
def get_hand_point_cloud(hand:HandModel, q, num_points=512):
    sampled_pc = hand.get_sampled_pc_fast(q, num_points=num_points)
    return sampled_pc[:, :3]


def main():
    robot_names = ['shadowhand']
    point_cloud_dataset = torch.load(os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_object_pcs_with_normals.pt'))

    for robot_name in robot_names:
        print(f"Processing {robot_name}...")
        path_name = f'oakink_dataset_standard_all_retarget_to_{robot_name}_valid_dro.pt'
        # path_name = f'teapot_oakink_dataset_standard_all_retarget_to_{robot_name}.pt'
        dataset_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', path_name)
        dataset = torch.load(dataset_path, map_location='cpu')
        metadata = dataset['metadata']

        hand = create_hand_model(robot_name, device=device, num_points=NUM_HAND_POINTS)
        updated_metadata = []

        for i in tqdm(range(0, len(metadata), BATCH_SIZE), desc=f"Processing {robot_name}"):
            batch_items = metadata[i:i+BATCH_SIZE]
            batch_hand_pc = []
            batch_obj_pc = []
            batch_obj_normals = []
            valid_indices = []

            for j, item in enumerate(batch_items):
                object_name = item[5]
                object_id = item[7]
                scale_factor = item[8] if isinstance(item[8], float) else 1.0
                q = item[3]

                if object_id not in point_cloud_dataset:
                    print(f"[Skip] Object {object_id} not found.")
                    updated_metadata.append(item)
                    continue

                hand_pc = get_hand_point_cloud(hand, q, NUM_HAND_POINTS).to(device)  # [N, 3]

                object_pc_normals = np.array(point_cloud_dataset[object_id]) * float(scale_factor)
                obj_pc = object_pc_normals[:, :3]
                obj_normals = object_pc_normals[:, 3:]

                obj_tensor = torch.from_numpy(obj_pc).unsqueeze(0).to(device).float()       # [1, N, 3]
                normal_tensor = torch.from_numpy(obj_normals).unsqueeze(0).to(device).float()  # [1, N, 3]

                batch_hand_pc.append(hand_pc.unsqueeze(0))           # [1, N, 3]
                batch_obj_pc.append(obj_tensor)
                batch_obj_normals.append(normal_tensor)
                valid_indices.append(j)

            if len(batch_hand_pc) == 0:
                continue

            # Stack batch
            hand_tensor = torch.cat(batch_hand_pc, dim=0)          # [B, Nh, 3]
            obj_tensor = torch.cat(batch_obj_pc, dim=0)            # [B, No, 3]
            normals_tensor = torch.cat(batch_obj_normals, dim=0)   # [B, No, 3]

            start = time.perf_counter()
            # distance, _ = get_aligned_distance(obj_tensor, hand_tensor, normals_tensor)
            distance, _ = get_euclidean_distance(obj_tensor, hand_tensor)
            contact_map = get_contact_map(distance)

            # Move contact map to CPU and compress
            contact_map_np = contact_map.cpu().numpy()  # [B, No]

            for k, j in enumerate(valid_indices):
                compressed = (contact_map_np[k] * 255).astype(np.uint8)
                item_list = list(batch_items[j])
                item_list.append(compressed)
                updated_metadata.append(tuple(item_list))

        # Save dataset
        output_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', path_name.replace('.pt', '_contact_map_eu.pt'))
        dataset['metadata'] = updated_metadata
        torch.save(dataset, output_path)
        print(f"Saved updated dataset to {output_path}")

if __name__ == "__main__":
    main()