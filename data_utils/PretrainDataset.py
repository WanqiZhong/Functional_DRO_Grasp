# Description: Dataset for pretraining model from open hand to closed hand to get a better encoder.
import os
import sys
import time
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from DRO_Grasp.utils.hand_model import create_hand_model, HandModel



class PretrainDataset(Dataset):
    def __init__(self, robot_names: list = None, use_multi_dex: bool = False):
        self.robot_names = robot_names if robot_names is not None \
            else ['barrett', 'allegro', 'shadowhand']

        self.dataset_len = 0
        self.robot_len = {}
        self.hands = {}
        self.dofs = []
        self.dataset = {}

        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(len(self.hands[robot_name].pk_chain.get_joint_parameter_names()))
            self.dataset[robot_name] = []

            if use_multi_dex:
                dataset_path = os.path.join(ROOT_DIR, f'data/MultiDex_filtered/{robot_name}/{robot_name}.pt')
                dataset = torch.load(dataset_path)
                metadata_split = dataset['metadata']
            else:

                if os.path.exists(os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}_valid_dro.pt')):
                    dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}_valid_dro.pt')
                elif os.path.exists(os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}.pt')):
                    dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}.pt')
                else:
                    raise FileNotFoundError(f"Dataset file not found for {robot_name}")

                dataset = torch.load(dataset_path)
                metadata = dataset['metadata']

                oakink_metadata_filtered = [m for m in metadata if m[6] == robot_name]
                object_to_metadata = defaultdict(list)

                for m in oakink_metadata_filtered:
                    object_name = m[5]
                    object_to_metadata[object_name].append(m)

                global_seed = 42
                metadata_split = []
                for object_name in sorted(object_to_metadata.keys()):
                    meta_list = object_to_metadata[object_name]
                    local_rng = random.Random(f"{global_seed}_{object_name}")
                    meta_list_shuffled = meta_list.copy()
                    local_rng.shuffle(meta_list_shuffled)
                    n_train = int(0.8 * len(meta_list_shuffled))
                    metadata_split.extend(meta_list_shuffled[:n_train])

            self.dataset[robot_name].extend(metadata_split)
            self.dataset_len += len(metadata_split)
            self.robot_len[robot_name] = len(metadata_split)

            print(f"Loaded {self.robot_len[robot_name]} samples for {robot_name}")

        print(f"Loaded {self.dataset_len} samples for all robots")

    def __getitem__(self, index):
        robot_name = random.choices(self.robot_names, weights=self.dofs, k=1)[0]

        hand:HandModel = self.hands[robot_name]
        dataset = self.dataset[robot_name]
        # target_q, _, _ = random.choice(dataset)
        target_q = random.choice(dataset)[3]

        robot_pc_1 = hand.get_transformed_links_pc(target_q)[:, :3]
        nofix_initial_q = hand.get_initial_q(target_q)
        robot_pc_2 = hand.get_transformed_links_pc(nofix_initial_q)[:, :3]

        return {
            'robot_pc_1': robot_pc_1,
            'robot_pc_2': robot_pc_2,
        }

    def __len__(self):
        return self.dataset_len


def create_dataloader(cfg):
    dataset = PretrainDataset(robot_names=cfg.robot_names)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True
    )
    return dataloader
