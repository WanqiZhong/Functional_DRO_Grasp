import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import json
import math
import hydra
import random
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from DRO_Grasp.utils.hand_model import create_hand_model, HandModel
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



# Existing constant mappings
RATIO_MAP = {
    'shadowhand': 8,
    # 'allegro': 4,
    # 'barrett': 3
}

INTENT_MAP = {
    'use': 0,
    'hold': 1,
    'liftup': 2,
    'handover': 3
}

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

class CombineDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        robot_names: list = None,
        mode: str = 'train',  
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = 'random',
        cross_object: bool = True,
        data_ratio = None,
        fixed_initial_q: bool = False,
        only_palm = False,
        complex_language_type: str = 'openai_256',  # clip_768, openai_256, clip_512
        provide_pc: bool = True,
        use_dro: bool = True,
        dataset_name: str = 'oakink',
        use_valid_data: bool = False,
        use_contact_data: bool = False,
        use_small_data: bool = True, # for quicker debug, reduce the dataloading time
        contact_type: str = 'important',
        ddpm_contact_type: str = 'random',
        small_data_name: str = 'small',
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names if robot_names is not None else ['barrett', 'allegro', 'shadowhand']
        self.mode = mode
        self.is_train = (mode == 'train')  # For backward compatibility
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.use_fixed_initial_q = fixed_initial_q
        self.cross_object = cross_object
        self.only_palm = only_palm
        self.complex_language_type = complex_language_type
        self.provide_pc = provide_pc
        self.use_dro = use_dro
        self.dataset_name = dataset_name
        self.use_valid_data = use_valid_data
        self.use_contact_data = use_contact_data
        self.use_small_data = use_small_data
        self.contact_type = contact_type
        self.ddpm_contact_type = ddpm_contact_type
        self.small_data_name = small_data_name
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Load Hand (Both CMapDataset and OakInkDataset)
        self.hands = {}
        self.dofs = []
        self.robot_ratio = {}
        self.metadata = []

        # Seperate different robot in test mode
        if mode == 'test':
            assert len(robot_names) == 1, "In test mode, only one robot name is allowed"

        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(math.sqrt(self.hands[robot_name].dof))
            if data_ratio is not None and robot_name in data_ratio:
                self.robot_ratio[robot_name] = data_ratio[robot_name]
            else:
                self.robot_ratio[robot_name] = RATIO_MAP.get(robot_name, 1)  # Default to 1 if not found
            print(f"Robot {robot_name}: {self.robot_ratio[robot_name]}")
        
        # Create initial_q
        self.robot_fix_initial_q = {}
        self.robot_fix_initial_q_pc = {}

        for robot_name in self.robot_names:
            hand = self.hands[robot_name]
            robot_initial_q = hand.get_fixed_initial_q()
            self.robot_fix_initial_q[robot_name] = robot_initial_q
            robot_initial_q_pc = hand.get_transformed_links_pc(robot_initial_q, only_palm=self.only_palm)[:, :3]
            self.robot_fix_initial_q_pc[robot_name] = robot_initial_q_pc

        all_oakink_metadata = []

        # Load retargeted data if needed
        for robot_name in self.robot_names:
                            
            if self.use_small_data:
                oakink_dataset_path = os.path.join(self.ROOT_DIR, f'data/OakInkDataset/oakink_{self.small_data_name}_dataset_standard_all_retarget_to_{robot_name}_valid_dro_contact_map.pt')
            elif self.use_contact_data:
                oakink_dataset_path = os.path.join(self.ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}_valid_dro_contact_map.pt')
            elif self.use_valid_data:
                oakink_dataset_path = os.path.join(self.ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}_valid_dro.pt')
            else:
                oakink_dataset_path = os.path.join(self.ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}.pt')

            oakink_metadata = torch.load(oakink_dataset_path)['metadata']
            print(f"Loading OakInkDataset for retargeted {robot_name} with {len(oakink_metadata)} samples")
            all_oakink_metadata.extend(oakink_metadata)
        
        # Filter by debug objects if specified
        if debug_object_names is not None:
            print("!!! Using debug objects for OakInkDataset !!!")
            all_oakink_metadata = [m for m in all_oakink_metadata if m[5] in debug_object_names]
        
        # Apply the new split logic
        train_metadata, val_metadata, test_metadata, split_stats = split_oakink_by_object_id(all_oakink_metadata)
        
        # Save the split statistics to a JSON file
        split_stats_path = os.path.join(self.ROOT_DIR, 'data/OakInkDataset/object_id_split_stats.json')
        with open(split_stats_path, 'w') as f:
            json.dump(split_stats, f, indent=2)
        print(f"Split statistics saved to {split_stats_path}")
        
        # Assign the appropriate set based on mode flag
        if self.mode == 'train':
            self.metadata = train_metadata
            print(f"Using {len(train_metadata)} OakInk training samples")
        elif self.mode == 'val':
            self.metadata = val_metadata
            print(f"Using {len(val_metadata)} OakInk validation samples")
        else:  # test
            self.metadata = test_metadata
            print(f"Using {len(test_metadata)} OakInk testing samples")

        random.seed(42)
        random.shuffle(self.metadata)

        # Load OakInkShape point clouds
        print("Loading OakInkShape pcs...")
        if self.use_small_data:
            oakink_object_pcs_path = os.path.join(self.ROOT_DIR, f'data/OakInkDataset/oakink_{self.small_data_name}_object_pcs_with_normals.pt')
        elif self.use_contact_data:
            oakink_object_pcs_path = os.path.join(self.ROOT_DIR, 'data/OakInkDataset/oakink_object_pcs_with_normals.pt')
        else:
            oakink_object_pcs_path = os.path.join(self.ROOT_DIR, 'data/OakInkDataset/oakink_object_pcs.pt')
        self.object_pcs = torch.load(oakink_object_pcs_path)

        self.category_intent_to_entries = defaultdict(lambda: defaultdict(list))

        for entry in self.metadata:
            object_key = entry[5]  
            intent = entry[4]      
            object_name = object_key.split('+')[1] 
            self.category_intent_to_entries[object_name][intent].append(entry)

        self.metadata_robots = {}
        self.metadata_robots_objects = defaultdict(lambda: defaultdict(list))

        for robot_name in self.robot_names:
            metadata_robot = [m for m in self.metadata if m[6] == robot_name]

            self.metadata_robots[robot_name] = metadata_robot
            print(f"Robot split: {robot_name} has {len(metadata_robot)} samples")

            for entry in metadata_robot:
                object_id = entry[7]
                self.metadata_robots_objects[robot_name][object_id].append(entry)            

        # Load object intent embeddings
        self.object_intent_embeddings = torch.load(os.path.join(self.ROOT_DIR, 'data/OakInkDataset/clip_object_intent_embeddings.pt'))        

        print(f"CombineDataset: {len(self)} samples (Batch)")
        print(f"CombineDataset: {len(self.metadata)} samples")

    def __getitem__(self, index):
        """
        Train: sample a batch of data
        Validate: get (robot, object) from index, sample a batch of data
        """
        robot_name_batch = []
        object_name_batch = []
        object_id_batch = []
        robot_links_pc_batch = []
        robot_pc_initial_batch = []

        if self.use_dro:
            dro_gt_batch = []
            robot_pc_target_batch = []
        else:
            dro_gt_batch = None
            robot_pc_target_batch = None

        if self.provide_pc:
            robot_pc_nofix_initial_batch = []
        else:
            robot_pc_nofix_initial_batch = None

        initial_q_batch = []
        nofix_initial_q_batch = []
        target_q_batch = []
        intent_batch = []
        language_embedding_batch = []
        object_pc_batch = []
        object_pc_ddpm_batch = []
        contact_map_batch = []
        contact_map_ddpm_batch = []
        complex_language_embedding_batch = []
        complex_language_sentence_batch = []
        complex_language_embedding_clip_512_batch = []
        complex_language_embedding_clip_768_batch = []
        complex_language_embedding_openai_256_batch = []
        
        batch_robot_data = []
        if self.mode == 'train':
            robot_names = calculate_robot_counts(self.batch_size, self.robot_ratio)
            for idx, robot_name in enumerate(robot_names):
                batch_robot_data.append(random.choice(self.metadata_robots[robot_name]))
        else:
            start_idx = index * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.metadata))
            batch_robot_data = self.metadata[start_idx:end_idx]

        for idx, robot_data in enumerate(batch_robot_data):

            if self.dataset_name == 'oakink':

                if self.use_contact_data:
                    hand_pose, hand_shape, tsl, target_q, intent, object_name, robot_name, object_id, scale_factor, hand_verts, complex_sentence, complex_embedding_clip_768, complex_embedding_openai_256, complex_embedding_clip_512, contact_info = robot_data
                else:
                    hand_pose, hand_shape, tsl, target_q, intent, object_name, robot_name, object_id, scale_factor, hand_verts, complex_sentence, complex_embedding_clip_768,  complex_embedding_openai_256, complex_embedding_clip_512 = robot_data

                scale_factor = float(scale_factor)
                language_embedding = self.object_intent_embeddings[object_name][intent]
                intent = INTENT_MAP[intent]

                if self.complex_language_type == 'clip_768':
                    complex_embedding = complex_embedding_clip_768
                elif self.complex_language_type == 'openai_256':
                    complex_embedding = complex_embedding_openai_256
                elif self.complex_language_type == 'clip_512':
                    complex_embedding = complex_embedding_clip_512
                else:
                    raise ValueError(f"Invalid complex_language_type: {self.complex_language_type}")

                complex_language_embedding_batch.append(complex_embedding)
                complex_language_sentence_batch.append(complex_sentence)
                complex_language_embedding_clip_768_batch.append(complex_embedding_clip_768)
                complex_language_embedding_openai_256_batch.append(complex_embedding_openai_256)
                complex_language_embedding_clip_512_batch.append(complex_embedding_clip_512)
            else:
                scale_factor = 1.0
                target_q, object_name = robot_data
                object_id = None
                language_embedding = self.object_intent_embeddings[object_name]['hold']
                intent = INTENT_MAP['hold']              

            hand = self.hands[robot_name]

            if self.dataset_name == 'oakink':
                robot_links_pc_batch.append(hand.links_pc)
                object_id_batch.append(object_id)
            else:
                robot_links_pc_batch.append(hand.links_pc)
                object_id_batch.append(None)

            target_q_batch.append(target_q)
            robot_name_batch.append(robot_name)
            object_name_batch.append(object_name)

            if self.use_contact_data:
                object_pc_ddpm, contact_map_ddpm = self._get_contact_object_pc(contact_map=contact_info, object_name=object_name,contact_type=self.ddpm_contact_type, object_id=object_id, num_points=2048, scale_factor=scale_factor)
                contact_map_ddpm_batch.append(contact_map_ddpm)
                object_pc_ddpm_batch.append(object_pc_ddpm)
                if self.provide_pc:
                    object_pc, contact_map = self._get_contact_object_pc(contact_map=contact_info, object_name=object_name, contact_type=self.contact_type, object_id=object_id, scale_factor=scale_factor)
                    contact_map_batch.append(contact_map)
                    object_pc_batch.append(object_pc)
            else:
                object_pc_ddpm = self._get_object_pc(object_name, object_id, 2048, scale_factor)
                object_pc_ddpm_batch.append(object_pc_ddpm)
                if self.provide_pc:
                    object_pc = self._get_object_pc(object_name, object_id, scale_factor=scale_factor)
                    object_pc_batch.append(object_pc)


            initial_q = self.robot_fix_initial_q[robot_name]
            initial_q_batch.append(initial_q)

            nofix_initial_q = hand.get_initial_q(target_q)
            nofix_initial_q_batch.append(nofix_initial_q)

            robot_pc_initial = self.robot_fix_initial_q_pc[robot_name]
            robot_pc_initial_batch.append(robot_pc_initial)
            
            if self.use_dro:
                robot_pc_target = hand.get_transformed_links_pc(target_q, only_palm=self.only_palm)[:, :3]
                robot_pc_target_batch.append(robot_pc_target)
                dro = torch.cdist(robot_pc_target, object_pc, p=2)
                dro_gt_batch.append(dro)

            if self.provide_pc:
                robot_pc_nofix_initial = hand.get_transformed_links_pc(nofix_initial_q, only_palm=self.only_palm)[:, :3]
                robot_pc_nofix_initial_batch.append(robot_pc_nofix_initial)

            intent_batch.append(intent)
            language_embedding_batch.append(language_embedding)


        if self.provide_pc:
            robot_pc_nofix_initial_batch = torch.stack(robot_pc_nofix_initial_batch)

        if self.use_dro:
            robot_pc_target_batch = torch.stack(robot_pc_target_batch)
            dro_gt_batch = torch.stack(dro_gt_batch)
        
        intent_batch = torch.tensor(intent_batch, dtype=torch.int).reshape(-1, 1)
        language_embedding_batch = torch.stack(language_embedding_batch)
        robot_pc_initial_batch = torch.stack(robot_pc_initial_batch)
        object_pc_ddpm_batch = torch.stack(object_pc_ddpm_batch)
        object_pc_batch = torch.stack(object_pc_batch) if self.provide_pc else None

        if len(complex_language_embedding_batch) == len(batch_robot_data):
            complex_language_embedding_batch = torch.stack(complex_language_embedding_batch)
            complex_language_embedding_clip_768_batch = torch.stack(complex_language_embedding_clip_768_batch)
            complex_language_embedding_openai_256_batch = torch.stack(complex_language_embedding_openai_256_batch)
            complex_language_embedding_clip_512_batch = torch.stack(complex_language_embedding_clip_512_batch)
        
        if self.use_contact_data:
            contact_map_ddpm_batch = torch.stack(contact_map_ddpm_batch)
            if self.provide_pc:
                contact_map_batch = torch.stack(contact_map_batch)

        B = len(batch_robot_data)
        N = self.num_points

        # assert robot_pc_initial_batch.shape == (B, 512, 3),\
            # f"Expected: {(B, 512, 3)}, Actual: {robot_pc_initial_batch.shape}"
        # assert robot_pc_target_batch.shape == (B, 512, 3),\
            # f"Expected: {(B, 512, 3)}, Actual: {robot_pc_target_batch.shape}"
        # assert object_pc_ddpm.shape == (B, 2048, 3),\
        #     f"Expected: {(B, 2048, 3)}, Actual: {object_pc_ddpm_batch.shape}"
        # assert object_pc.shape == (B, N, 3),\
        #     f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
        # assert dro_gt_batch.shape == (B, N, N),\
            # f"Expected: {(B, N, N)}, Actual: {dro_gt_batch.shape}"

        initial_q_batch = initial_q_batch if self.use_fixed_initial_q else nofix_initial_q_batch
        robot_pc_initial_batch = robot_pc_initial_batch if self.use_fixed_initial_q else robot_pc_nofix_initial_batch

        return {
            'robot_name': robot_name_batch,  # list(len = B): str
            'object_name': object_name_batch,  # list(len = B): str
            'object_id': object_id_batch,  # list(len = B): str
            'robot_links_pc': robot_links_pc_batch,  # list(len = B): dict, {link_name: (N_link, 3)}
            'robot_pc_initial': robot_pc_initial_batch,
            'robot_pc_target': robot_pc_target_batch,
            'language_embedding': language_embedding_batch,
            'object_pc': object_pc_batch,
            'object_pc_ddpm': object_pc_ddpm_batch,
            'dro_gt': dro_gt_batch,
            'initial_q': initial_q_batch,
            'target_q': target_q_batch,
            'intent': intent_batch,
            'complex_language_embedding': complex_language_embedding_batch,
            'complex_language_embedding_clip_512': complex_language_embedding_clip_512_batch,
            'complex_language_embedding_clip_768': complex_language_embedding_clip_768_batch,
            'complex_language_embedding_openai_256': complex_language_embedding_openai_256_batch,
            'complex_language_sentence': complex_language_sentence_batch,
            'contact_map': contact_map_batch,
            'contact_map_ddpm': contact_map_ddpm_batch,
        }

    def __len__(self):
        return math.ceil(len(self.metadata) / self.batch_size)

        
    def _get_object_pc(self, object_name, object_id=None, num_points=None, scale_factor=1.0):
        if num_points is None:
            num_points = self.num_points
        if self.object_pc_type == 'fixed':
            name = object_name.split('+')
            object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt') if object_id is not None \
            else os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
            object_pc = torch.load(object_path)[:, :3]
        elif self.object_pc_type == 'random':
            indices = torch.randperm(65536)[:num_points]
            object_pc = self.object_pcs[object_id][indices] if object_id is not None \
            else self.object_pcs[object_name][indices]
            object_pc += torch.randn(object_pc.shape) * 0.002
        else:  # 'partial', remove 50% points
            indices = torch.randperm(65536)[:num_points * 2]
            object_pc = self.object_pcs[object_id][indices] if object_id is not None \
            else self.object_pcs[object_name][indices]
            direction = torch.randn(3)
            direction = direction / torch.norm(direction)
            proj = object_pc @ direction
            _, indices = torch.sort(proj)
            indices = indices[num_points:]
            object_pc = object_pc[indices]
        return object_pc * scale_factor
    

    def _get_contact_object_pc(self, contact_map, object_name, contact_type, object_id=None, num_points=None, scale_factor=1.0):
        if num_points is None:
            num_points = self.num_points
        if self.object_pc_type == 'fixed':
            name = object_name.split('+')
            object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt') if object_id is not None \
            else os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
            obj_pc = torch.load(object_path)[:, :3]
        elif self.object_pc_type == 'random':
            object_pc = self.object_pcs[object_id] if object_id is not None \
            else self.object_pcs[object_name]
            obj_pc = object_pc[:, :3]
        else:
            raise ValueError(f"Invalid object_pc_type: {self.object_pc_type}")
        
        contact_map = contact_map.astype(np.float32) / 255.0

        if contact_type == "random":
            indices = torch.randperm(contact_map.shape[0])[:2048]
            contact_map = contact_map[indices]
            obj_pc = obj_pc[indices]

        elif contact_type == "important":

            indices = torch.randperm(contact_map.shape[0])[:2048]
            contact_map = contact_map[indices]
            obj_pc = obj_pc[indices]
            
            probs = contact_map.copy()
            probs = np.maximum(probs, 0.002)  
            
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones_like(probs) / len(probs)
            
            selected_indices = np.random.choice(
                np.arange(len(contact_map)), 
                size=min(num_points, len(contact_map)),
                replace=False,
                p=probs
            )
        
            contact_map = contact_map[selected_indices]
            obj_pc = obj_pc[selected_indices]
        else:
            raise ValueError(f"Invalid contact_type: {self.contact_type}")

        contact_map = torch.from_numpy(contact_map)
        
        return obj_pc * scale_factor, contact_map


def calculate_robot_counts(batch_size, robot_ratio):

    total_ratio = sum(robot_ratio.values())
    expected_counts = {robot: (ratio / total_ratio) * batch_size for robot, ratio in robot_ratio.items()}
    
    counts = {robot: math.floor(count) for robot, count in expected_counts.items()}
    allocated = sum(counts.values())
    remaining = batch_size - allocated
    
    if remaining > 0:
        fractional_parts = {robot: expected_counts[robot] - counts[robot] for robot in robot_ratio}
        total_fraction = sum(fractional_parts.values())
        allocation_probs = {robot: fractional_parts[robot] / total_fraction for robot in robot_ratio}
        
        robots = list(robot_ratio.keys())
        probs = [allocation_probs[robot] for robot in robots]
        allocated_robots = random.choices(robots, weights=probs, k=remaining)
        for robot in allocated_robots:
            counts[robot] += 1
    
    robots_num = []
    for robot, count in counts.items():
        robots_num.extend([robot] * count)
    random.shuffle(robots_num)
    
    return robots_num

def custom_collate_fn(batch):
    return batch[0]


def create_dataloader(cfg, mode, fixed_initial_q=False):

    dataset = CombineDataset(
        batch_size=cfg.batch_size,
        robot_names=cfg.robot_names,
        mode=mode,
        debug_object_names=cfg.debug_object_names,
        object_pc_type=cfg.object_pc_type,
        data_ratio=cfg.ratio,
        fixed_initial_q=fixed_initial_q,
        complex_language_type=cfg.complex_language_type,
        dataset_name=cfg.dataset_name,
        use_valid_data=cfg.use_valid_data,
        use_contact_data=cfg.use_contact_data
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=cfg.num_workers,
        shuffle=(mode=='train'),
    )
    return dataloader
