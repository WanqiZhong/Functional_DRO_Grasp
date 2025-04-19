# Description: Dataset for pretraining model from open hand to closed hand to get a better encoder.
import os
import sys
import time
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from DRO_Grasp.utils.hand_model import create_hand_model, HandModel


def split_oakink_by_object_id(oakink_metadata, min_objects_for_split=5, train_ratio=0.8, seed=42):
    """
    Split OakInk dataset by object ID within each object name category.
    Split into train (80%) and test+val (20%), then further split test+val into val and test (50% each).
    
    Args:
        oakink_metadata: List of metadata entries for OakInk dataset
        min_objects_for_split: Minimum number of object IDs needed to perform a split (otherwise all go to training)
        train_ratio: Ratio of object IDs to use for training
        seed: Random seed for reproducibility
        
    Returns:
        train_metadata: List of metadata entries for training
        val_metadata: List of metadata entries for validation
        test_metadata: List of metadata entries for testing
        split_stats: Dictionary with statistics about the split
    """
    # Group metadata by object name
    object_name_to_ids = defaultdict(set)
    for entry in oakink_metadata:
        object_name = entry[5].split('+')[1]  # Get the object name (category)
        object_id = entry[7]  # Get the object ID
        object_name_to_ids[object_name].add(object_id)
    
    # Prepare containers for results
    train_metadata = []
    val_metadata = []
    test_metadata = []
    split_stats = {
        "object_categories": {},
        "total": {
            "categories": 0,
            "categories_split": 0,
            "categories_all_train": 0,
            "object_ids": {
                "total": 0,
                "train": 0,
                "val": 0,
                "test": 0
            },
            "samples": {
                "total": 0,
                "train": 0,
                "val": 0,
                "test": 0
            }
        }
    }
    
    
    for object_name, object_ids in object_name_to_ids.items():
        object_ids_list = list(object_ids)
        category_stats = {
            "total_object_ids": len(object_ids_list),
            "train_object_ids": 0,
            "val_object_ids": 0,
            "test_object_ids": 0,
            "total_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "train_ids": [],
            "val_ids": [],
            "test_ids": []
        }
        
        # Decide on train/test split
        if len(object_ids_list) < min_objects_for_split:
            # If too few object IDs, put all in training
            train_ids = object_ids_list
            val_ids = []
            test_ids = []
            split_stats["total"]["categories_all_train"] += 1
        else:
            # Otherwise do the split - first into train and rest
            object_ids_list = sorted(object_ids_list)
            train_size = int(len(object_ids_list) * train_ratio)
            train_ids = object_ids_list[:train_size]
            rest_ids = object_ids_list[train_size:]
            
            # Then split the remaining 20% into val and test (10% each of the total)
            val_size = len(rest_ids) // 2
            val_ids = rest_ids[:val_size]
            test_ids = rest_ids[val_size:]
            
            split_stats["total"]["categories_split"] += 1
        
        # Update category statistics
        category_stats["train_object_ids"] = len(train_ids)
        category_stats["val_object_ids"] = len(val_ids)
        category_stats["test_object_ids"] = len(test_ids)
        category_stats["train_ids"] = train_ids
        category_stats["val_ids"] = val_ids
        category_stats["test_ids"] = test_ids
        
        # Assign metadata entries to train, val, or test based on their object IDs
        for entry in oakink_metadata:
            entry_object_name = entry[5].split('+')[1]
            entry_object_id = entry[7]
            
            if entry_object_name == object_name:
                category_stats["total_samples"] += 1
                if entry_object_id in train_ids:
                    train_metadata.append(entry)
                    category_stats["train_samples"] += 1
                elif entry_object_id in val_ids:
                    val_metadata.append(entry)
                    category_stats["val_samples"] += 1
                elif entry_object_id in test_ids:
                    test_metadata.append(entry)
                    category_stats["test_samples"] += 1
        
        # Store category statistics
        split_stats["object_categories"][object_name] = category_stats
        
        # Update total statistics
        split_stats["total"]["categories"] += 1
        split_stats["total"]["object_ids"]["total"] += len(object_ids_list)
        split_stats["total"]["object_ids"]["train"] += len(train_ids)
        split_stats["total"]["object_ids"]["val"] += len(val_ids)
        split_stats["total"]["object_ids"]["test"] += len(test_ids)
        split_stats["total"]["samples"]["total"] += category_stats["total_samples"]
        split_stats["total"]["samples"]["train"] += category_stats["train_samples"]
        split_stats["total"]["samples"]["val"] += category_stats["val_samples"]
        split_stats["total"]["samples"]["test"] += category_stats["test_samples"]
    
    return train_metadata, val_metadata, test_metadata, split_stats

class PretrainDataset(Dataset):
    def __init__(self, robot_names: list = None, use_multi_dex: bool = False, use_valid_data=True):
        self.robot_names = robot_names if robot_names is not None \
            else ['barrett', 'allegro', 'shadowhand']

        self.dataset_len = 0
        self.robot_len = {}
        self.hands = {}
        self.dofs = []
        self.dataset = defaultdict(list)

        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(len(self.hands[robot_name].pk_chain.get_joint_parameter_names()))

        if use_multi_dex:
            for robot_name in self.robot_names:
                dataset_path = os.path.join(ROOT_DIR, f'data/MultiDex_filtered/{robot_name}/{robot_name}.pt')
                dataset = torch.load(dataset_path)
                metadata_split = dataset['metadata']
                self.dataset[robot_name].extend(metadata_split)
                self.dataset_len += len(metadata_split)
                self.robot_len[robot_name] = len(metadata_split)
                print(f"Loaded {self.robot_len[robot_name]} samples for {robot_name}")
        else:
            all_oakink_metadata = []
            for robot_name in self.robot_names:                                
                if use_valid_data:
                    oakink_dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}_valid_dro.pt')
                else:
                    oakink_dataset_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}.pt')

                oakink_metadata = torch.load(oakink_dataset_path)['metadata']
                print(f"Loading OakInkDataset for retargeted {robot_name} with {len(oakink_metadata)} samples")
                all_oakink_metadata.extend(oakink_metadata)
            
            # Apply the new split logic
            train_metadata, val_metadata, test_metadata, split_stats = split_oakink_by_object_id(all_oakink_metadata)
            self.metadata = train_metadata

            # Save the split statistics to a JSON file
            split_stats_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/pretrain_object_id_split_stats.json')
            with open(split_stats_path, 'w') as f:
                json.dump(split_stats, f, indent=2)
            print(f"Split statistics saved to {split_stats_path}")
            
            random.seed(42)
            random.shuffle(self.metadata)

            for robot_name in self.robot_names:                                
                self.dataset[robot_name] = [m for m in self.metadata if m[6] == robot_name]
                self.dataset_len += len(self.dataset[robot_name])
                self.robot_len[robot_name] = len(self.dataset[robot_name])
                print(f"Loaded {self.robot_len[robot_name]} samples for {robot_name}")

        print(f"Loaded {self.dataset_len} samples for all robots")

    def __getitem__(self, index):
        robot_name = random.choices(self.robot_names, weights=self.dofs, k=1)[0]

        hand:HandModel = self.hands[robot_name]
        dataset = self.dataset[robot_name]
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
