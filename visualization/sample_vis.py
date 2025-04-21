import sys
import os
import hydra
import random
import json
import copy
from tqdm import tqdm
from collections import defaultdict
import torch
import pickle
import numpy as np

# load一下data，看一下一共有几个类别，每个类别选3个object
DRO_PATH = "/home/mzy/code"
sys.path.append(DRO_PATH)
from DRO_Grasp.data_utils import create_dataloader
from DRO_Grasp.utils.hand_model import create_hand_model
from DRO_Grasp.utils.multilateration import multilateration
from DRO_Grasp.utils.se3_transform import compute_link_pose
from DRO_Grasp.utils.optimization import *

SCENE_DIFFUSER_PATH = "/home/mzy/code/Scene-Diffuser-DRO"
sys.path.append(SCENE_DIFFUSER_PATH)
from sampler import DiffuserSampler

def set_global_seed(seed: int):
    random.seed(seed)              # Python random
    torch.manual_seed(seed)        # PyTorch CPU
    torch.cuda.manual_seed(seed)   # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs
    
set_global_seed(42)

SAVE_DIR = "functional_test/"
SAVE_PATH_K2_RES = os.path.join(SAVE_DIR, "functional_test_m1.pt")
SAVE_PATH_K3_RES = os.path.join(SAVE_DIR, "functional_test_m2.pt")
SAVE_PATH_MIX = os.path.join(SAVE_DIR, "functional_test_mix.pt") # in use

def add_idx_to_sample():
    # path_1 = "/home/mzy/code/DRO_Grasp/res_diffuser_k2_chamfer.pkl"
    # path_2 = "/home/mzy/code/DRO_Grasp/res_diffuser_k3_chamfer.pkl"
    path_1 = "/home/mzy/code/DRO_Grasp/res_diffuser_k3_chamfer_batch.pkl"
    path_2 = "/home/mzy/code/DRO_Grasp/res_diffuser_k3_chamfer.pkl"

    with open(path_1, "rb") as f:
        data_1 = pickle.load(f)
        
    with open(path_2, "rb") as f2:
        data_2 = pickle.load(f2)
        
    data_1 = data_1['results']
    data_2 = data_2['results']

    # Preprocess the data so they have same id
    for i in range(len(data_1)):
        data_1[i]['vis_sample_idx'] = i
        data_2[i]['vis_sample_idx'] = i
    
    # Check index
    for i in range(len(data_1)):
        print("data1: ", data_1[i]['vis_sample_idx'])
        print("data2: ", data_2[i]['vis_sample_idx'])
        
    torch.save(data_1, os.path.join(SAVE_DIR, "data_1_with_idx.pt"))
    torch.save(data_2, os.path.join(SAVE_DIR, "data_2_with_idx.pt"))

# add_idx_to_sample() # Can be commeted out after first run

path_idx_1 = "/home/mzy/code/DRO_Grasp/functional_test/data_1_with_idx.pt"
path_idx_2 = "/home/mzy/code/DRO_Grasp/functional_test/data_2_with_idx.pt"

data_1 = torch.load(path_idx_1)
data_2 = torch.load(path_idx_2)

# Collect the id list, organize by category and object_id
category_sample_map_dict = defaultdict(lambda: defaultdict(list))
category_to_object_stats = defaultdict(lambda: defaultdict(int))
for i in range(len(data_2)):
    current_sample = data_2[i]
    category = current_sample['object_name'].split('+')[1]
    object_id = current_sample['object_id']
    category_sample_map_dict[category][object_id].append(i)
    category_to_object_stats[category][object_id] += 1

all_selected_sample_ids = defaultdict(lambda: defaultdict(list))
for category, object_id_map in category_sample_map_dict.items():
    object_ids = list(object_id_map.keys()) # 当前category下所有的object id
    num_object_id_to_select = min(3, len(object_ids))
    num_sample_to_select_per_object_id = 2
    selected_object_ids = random.sample(object_ids, k=num_object_id_to_select)
    
    for object_id in selected_object_ids:
        sample_ids = object_id_map[object_id]
        # select the one with different instructions
        visited_instructions = []
        for sample_id in sample_ids:
            if data_1[sample_id]['complex_language_sentence'] not in visited_instructions:
                all_selected_sample_ids[category][object_id].append(sample_id)
                visited_instructions.append(data_1[sample_id]['complex_language_sentence'])
            if len(all_selected_sample_ids[category][object_id]) >= num_sample_to_select_per_object_id:
                break

# Map the all_selected_sample_ids dict into a list containing actual samples for data_1 and data_2
print("=====all_selected_sample_ids======")
print(all_selected_sample_ids)
print(len(all_selected_sample_ids))
print(len(all_selected_sample_ids.items()))
print("==================================")

all_selected_sample_ids_data_1 = []
all_selected_sample_ids_data_2 = []
for category in all_selected_sample_ids.keys():
    for object_id in all_selected_sample_ids[category].keys():
        for idx in all_selected_sample_ids[category][object_id]:
            all_selected_sample_ids_data_1.append(data_1[idx])
            all_selected_sample_ids_data_2.append(data_2[idx])

print("len data_1 selected: ", len(all_selected_sample_ids_data_1))
print("len data_2 selected: ", len(all_selected_sample_ids_data_2))

all_selected_sample_ids_mix = all_selected_sample_ids_data_1 + all_selected_sample_ids_data_2

# print("double check id added")
# print(all_selected_sample_ids_mix[0].keys())

# save to one file
torch.save(all_selected_sample_ids_mix, SAVE_PATH_MIX)

# Visualize the data's structure
for category, id_counts in category_to_object_stats.items():
    print(f"Class '{category}' has {len(id_counts)} unique objects:")
    for object_id, count in sorted(id_counts.items()):
        print(f"  - Object ID: {object_id} → {count} samples")
    print()

print(category_sample_map_dict)