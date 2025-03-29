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

RATIO_MAP = {
    'shadowhand': 8,
    # 'retarget_shadowhand': 5,
    # 'allegro': 4,
    # 'barrett': 3
}

INTENT_MAP = {
    'use': 0,
    'hold': 1,
    'liftup': 2,
    'handover': 3
}


class CombineDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        robot_names: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = 'random',
        cross_obejct: bool = True,
        data_ratio = None,
        fixed_initial_q: bool = False,
        fix_sample = False,
        only_palm = False,
        use_validatedata_in_train_mode = False,
        complex_language_type: str = 'openai_256',  # clip_768, openai_256, clip_512
        provide_pc: bool = True,
        use_dro: bool = True,
        use_valid_data: bool = False
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names if robot_names is not None else ['barrett', 'allegro', 'shadowhand']
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.use_fixed_initial_q = fixed_initial_q
        self.cross_object = cross_obejct
        self.fix_sample = fix_sample
        self.only_palm = only_palm
        self.complex_language_type = complex_language_type
        self.use_validatedata_in_train_mode = use_validatedata_in_train_mode
        self.provide_pc = provide_pc
        self.use_dro = use_dro
        self.use_valid_data = use_valid_data
        # Load Hand (Both CMapDataset and OakInkDataset)
        self.hands = {}
        self.dofs = []
        self.robot_ratio ={}
        self.oakink_metadata = []
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name.split('_')[1] if "retarget" in robot_name else robot_name, torch.device('cpu'), self.num_points)
            self.dofs.append(math.sqrt(self.hands[robot_name].dof))
            if data_ratio is not None and robot_name in data_ratio:
                self.robot_ratio[robot_name] = data_ratio[robot_name]
            else:
                self.robot_ratio[robot_name] = RATIO_MAP[robot_name]
            print(f"Robot {robot_name}: {self.robot_ratio[robot_name]}")
        
        # Create initial_q
        self.robot_fix_initial_q = {}
        self.robot_fix_initial_q_pc = {}

        for robot_name in self.robot_names:
            hand = self.hands[robot_name]
            robot_initial_q = hand.get_initial_q()
            self.robot_fix_initial_q[robot_name] = robot_initial_q
            robot_initial_q_pc = hand.get_transformed_links_pc(robot_initial_q, only_palm=self.only_palm)[:, :3]
            self.robot_fix_initial_q_pc[robot_name] = robot_initial_q_pc


        # Load CMapDataset
        print("Loading CMapDataset...")
        cmap_split_json_path = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered/split_train_validate_objects.json')
        cmap_dataset_split = json.load(open(cmap_split_json_path))
        if self.is_train:
            if self.use_validatedata_in_train_mode:
                print("!!! Using validate data in train mode !!!")
                self.cmap_object_names = cmap_dataset_split['validate']
            else:
                self.cmap_object_names = cmap_dataset_split['train']
        else:
            self.cmap_object_names = cmap_dataset_split['validate']
        # if debug_object_names is not None:
        #     print("!!! Using debug objects for CMapDataset !!!")
        #     self.cmap_object_names = debug_object_names

        cmap_dataset_path = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered/cmap_dataset.pt')
        cmap_metadata = torch.load(cmap_dataset_path)['metadata']
        self.cmap_metadata = [m for m in cmap_metadata if m[1] in self.cmap_object_names and m[2] in self.robot_names]

        # Load CMapDataset object_pcs
        print("Loading CMapDataset object_pcs...")
        self.cmap_object_pcs = {}
        if self.object_pc_type != 'fixed':
            for object_name in tqdm(self.cmap_object_names, desc="Loading CMapDataset object pcs"):
                name = object_name.split('+')
                mesh_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                pcd = mesh.sample_points_uniformly(65536)
                object_pc = np.asarray(pcd.points)
                self.cmap_object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)
        else:
            print("!!! Using fixed object pcs for CMapDataset !!!")

        # Load OakInkDataset
        if "mano" in self.robot_names:

            print("Loading OakInkDataset...")

            oakink_dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all.pt')
            oakink_metadata = torch.load(oakink_dataset_path)['metadata']
            oakink_metadata_filtered = [m for m in oakink_metadata if m[6] in self.robot_names]
            
            object_to_metadata = defaultdict(list)
            for m in oakink_metadata_filtered:
                object_name = m[5]
                object_to_metadata[object_name].append(m)

            if debug_object_names is not None:
                print("!!! Using debug objects for OakInkDataset !!!")
                object_to_metadata = {obj: metas for obj, metas in object_to_metadata.items() if obj in debug_object_names}
            else:
                object_to_metadata = dict(object_to_metadata)


            global_seed = 42
            oakink_metadata_split = []
            for object_name in sorted(object_to_metadata.keys()):
                meta_list = object_to_metadata[object_name]
                local_rng = random.Random(f"{global_seed}_{object_name}")
                meta_list_shuffled = meta_list.copy()
                local_rng.shuffle(meta_list_shuffled)
                n_train = int(0.8 * len(meta_list_shuffled))
                if is_train:
                    if self.use_validatedata_in_train_mode:
                        print("!!! Using validate data in train mode !!!")
                        oakink_metadata_split.extend(meta_list_shuffled[n_train:])
                    else:
                        oakink_metadata_split.extend(meta_list_shuffled[:n_train])
                else:
                    oakink_metadata_split.extend(meta_list_shuffled[n_train:])

            self.oakink_metadata = oakink_metadata_split


        if "retarget_shadowhand" in self.robot_names:

            print("Loading OakInkDataset...")

            if self.use_valid_data:
                oakink_dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand_valid_dro.pt')
            else:
                oakink_dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand.pt')

            oakink_metadata = torch.load(oakink_dataset_path)['metadata']
            oakink_metadata_filtered = [m for m in oakink_metadata if (m[6] in self.robot_names or m[6] == 'shadowhand')]
            
            object_to_metadata = defaultdict(list)
            for m in oakink_metadata_filtered:
                object_name = m[5]
                object_to_metadata[object_name].append(m)

            if debug_object_names is not None:
                print("!!! Using debug objects for OakInkDataset !!!")
                object_to_metadata = {obj: metas for obj, metas in object_to_metadata.items() if obj in debug_object_names}
            else:
                object_to_metadata = dict(object_to_metadata)

            global_seed = 42
            oakink_metadata_split = []
            for object_name in sorted(object_to_metadata.keys()):
                meta_list = object_to_metadata[object_name]
                local_rng = random.Random(f"{global_seed}_{object_name}")
                meta_list_shuffled = meta_list.copy()
                local_rng.shuffle(meta_list_shuffled)
                n_train = int(0.8 * len(meta_list_shuffled))
                if is_train:
                    if self.use_validatedata_in_train_mode:
                        print("!!! Using validate data in train mode !!!")
                        oakink_metadata_split.extend(meta_list_shuffled[n_train:])
                    else:
                        oakink_metadata_split.extend(meta_list_shuffled[:n_train])
                else:
                    oakink_metadata_split.extend(meta_list_shuffled[n_train:])

            self.oakink_metadata = oakink_metadata_split


        print("Loading OakInkShape pcs...")
        oakink_object_pcs_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_object_pcs.pt')
        self.oakink_object_pcs = torch.load(oakink_object_pcs_path)

        # Create metadata dict
        self.category_intent_to_entries = {}

        if self.oakink_metadata is not None:

            for entry in self.oakink_metadata:

                object_key = entry[5]  
                intent = entry[4]      
                object_name = object_key.split('+')[1] 
                
                if object_name not in self.category_intent_to_entries:
                    self.category_intent_to_entries[object_name] = {}
                
                if intent not in self.category_intent_to_entries[object_name]:
                    self.category_intent_to_entries[object_name][intent] = []
                
                self.category_intent_to_entries[object_name][intent].append(entry)

        # Metadata Robot
        self.metadata_robots = {}
        self.metadata_robots_objects = {}

        for robot_name in self.robot_names:

            if robot_name == 'mano':
                metadata_robot = [m for m in self.oakink_metadata if m[6] == robot_name]
            elif robot_name == 'retarget_shadowhand':
                metadata_robot = [m for m in self.oakink_metadata if m[6] == 'shadowhand']
            else:
                metadata_robot = [(m[0], m[1]) for m in self.cmap_metadata if m[2] == robot_name]
            self.metadata_robots[robot_name] = metadata_robot

            self.metadata_robots_objects[robot_name] = {}

            for entry in metadata_robot:
                if robot_name == "mano" or robot_name == "retarget_shadowhand":
                    object_id = entry[7]
                else:
                    object_id = entry[1]
                if object_id not in self.metadata_robots_objects[robot_name]:
                    self.metadata_robots_objects[robot_name][object_id] = []
                self.metadata_robots_objects[robot_name][object_id].append(entry)                

        # Combine object_pcs
        self.object_pcs = {}
        self.object_pcs.update(self.cmap_object_pcs)
        if "mano" in self.robot_names or "retarget_shadowhand" in self.robot_names:
            self.object_pcs.update(self.oakink_object_pcs)

        # Setup validation combinations
        if not self.is_train:
            self.combination = []
            for robot_name in self.robot_names:
                if robot_name == 'mano':
                    continue  # 'mano' is not supported in validate mode as per original code
                for object_name in self.cmap_object_names:
                    self.combination.append((robot_name, object_name))
            self.combination = sorted(self.combination)

        # Load object intent embeddings
        self.object_intent_embeddings = torch.load(os.path.join(ROOT_DIR, 'data/OakInkDataset/clip_object_intent_embeddings.pt'))        

        print(f"CombineDataset: {len(self)} samples")
        print(f"CombineDataset: {len(self.oakink_metadata)} oakink samples")

    def __getitem__(self, index):
        """
        Train: sample a batch of data
        Validate: get (robot, object) from index, sample a batch of data
        """
        if self.is_train:
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

            if self.use_dro and self.provide_pc:
                wrist_dro_gt_batch = []
                robot_pc_wrist_target_batch = []
            else:
                wrist_dro_gt_batch = None
                robot_pc_wrist_target_batch = None

            if self.provide_pc:
                robot_pc_nofix_initial_batch = []
                cross_pc_target_batch = []
            else:
                robot_pc_nofix_initial_batch = None
                cross_pc_target_batch = None

            initial_q_batch = []
            nofix_initial_q_batch = []
            target_q_batch = []
            wrist_target_q_batch = []
            intent_batch = []
            language_embedding_batch = []
            object_pc_batch = []
            object_pc_ddpm_batch = []
            cross_object_batch = []
            complex_language_embedding_batch = []
            complex_language_sentence_batch = []
            complex_language_embedding_clip_512_batch = []
            complex_language_embedding_clip_768_batch = []
            complex_language_embedding_openai_256_batch = []
            
            robot_names = calculate_robot_counts(self.batch_size, self.robot_ratio)
            # choose_object_name = random.choice(list(self.metadata_robots_objects[robot_names[0]]))
            # print(f"Choose Object: {choose_object_name}")

            for idx, robot_name in enumerate(robot_names):

                if self.fix_sample:
                    robot_name = "retarget_shadowhand" 
                    s_idx = 0

                robot_name_batch.append(robot_name)
                hand: HandModel = self.hands[robot_name]
                # print(robot_name + " " + choose_object_name)
                # metadata_robot = self.metadata_robots_objects[robot_name][choose_object_name]
                metadata_robot = self.metadata_robots[robot_name]

                if robot_name == 'mano' or robot_name == 'retarget_shadowhand':
                    # metadata_robot = [m for m in self.oakink_metadata if m[6] == robot_name]
                    if self.fix_sample:
                        hand_pose, hand_shape, tsl, target_q, intent, object_name, _, object_id, _, hand_verts, complex_sentence, complex_embedding_clip_768, complex_embedding_openai_256, complex_embedding_clip_512 = metadata_robot[s_idx]
                        s_idx += 1
                    else:
                        hand_pose, hand_shape, tsl, target_q, intent, object_name, _, object_id, _, hand_verts, complex_sentence, complex_embedding_clip_768,  complex_embedding_openai_256, complex_embedding_clip_512 = random.choice(metadata_robot)

                    cross_data = random.choice(self.category_intent_to_entries[object_name.split('+')[1]][intent])
                    language_embedding = self.object_intent_embeddings[object_name][intent]
                    intent = INTENT_MAP[intent]
                    _, _, _, _, _, cross_object_name, _, cross_object_id, _, cross_hand_verts, _, _, _, _ = cross_data

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
                    # metadata_robot = [(m[0], m[1]) for m in self.cmap_metadata if m[2] == robot_name]
                    # >>> Debug Only >>>
                    target_q, object_name = random.choice(metadata_robot)
                    # target_q, object_name = metadata_robot[0]
                    # # <<< End Debug <<<
                    object_id = None
                    language_embedding = self.object_intent_embeddings[object_name]['hold']
                    intent = INTENT_MAP['hold']              

                target_q_batch.append(target_q)
                object_name_batch.append(object_name)

                if robot_name == 'mano':
                    robot_links_pc_batch.append(None)
                    object_id_batch.append(object_id)
                elif robot_name == 'retarget_shadowhand':
                    robot_links_pc_batch.append(hand.links_pc)
                    object_id_batch.append(object_id)
                else:
                    robot_links_pc_batch.append(hand.links_pc)
                    object_id_batch.append(None)

                # Add original object
                object_pc = self._get_object_pc(object_name, object_id, robot_name)
                object_pc_ddpm = self._get_object_pc(object_name, object_id, robot_name, 2048)

                # Add initial and target robot pc
                # if robot_name == 'mano':
                #     robot_pc_target, _ = hand.get_mano_pc_from_verts(torch.tensor(hand_verts))
                #     robot_pc_target_batch.append(robot_pc_target)
                #     # target_q_batch.append(target_q)
                #     target_q_batch.append(None)

                #     initial_q = hand.get_fixed_initial_q()
                #     nofix_initial_q = hand.get_initial_q(target_q)

                #     # initial_q_batch.append(initial_q)
                #     initial_q_batch.append(None)
                #     robot_pc_initial, _ = hand.get_mano_pc(initial_q, tsl, hand_pose.unsqueeze(0), hand_shape.unsqueeze(0))
                #     robot_pc_initial_batch.append(robot_pc_initial)
                # else:

                initial_q = hand.get_initial_q()
                initial_q_batch.append(initial_q)

                nofix_initial_q = hand.get_initial_q(target_q)
                nofix_initial_q_batch.append(nofix_initial_q)

                wrist_target_q = torch.cat([target_q[:6], initial_q[6:]])
                wrist_target_q_batch.append(wrist_target_q)

                robot_pc_initial = hand.get_transformed_links_pc(initial_q)[:, :3]
                robot_pc_initial_batch.append(robot_pc_initial)
                
                if self.use_dro:
                    robot_pc_target = hand.get_transformed_links_pc(target_q, only_palm=self.only_palm)[:, :3]
                    robot_pc_target_batch.append(robot_pc_target)

                    if self.provide_pc:
                        robot_pc_wrist_target = hand.get_transformed_links_pc(wrist_target_q)[:, :3]
                        robot_pc_wrist_target_batch.append(robot_pc_wrist_target)

                if self.provide_pc:
                    robot_pc_nofix_initial = hand.get_transformed_links_pc(nofix_initial_q, only_palm=self.only_palm)[:, :3]
                    robot_pc_nofix_initial_batch.append(robot_pc_nofix_initial)

                if robot_name == 'mano':
                    cross_pc_target, _ = hand.get_mano_pc_from_verts(torch.tensor(cross_hand_verts))
                else:
                    if self.provide_pc:
                        cross_pc_target = hand.get_transformed_links_pc(target_q, only_palm=self.only_palm)[:, :3]
                        cross_pc_target_batch.append(cross_pc_target)

                    # this makes DRO object use the same object as the origin object
                    cross_object_name = object_name 
                    cross_object_id = object_id

                cross_object_pc = self._get_object_pc(cross_object_name, cross_object_id, robot_name)
                    
                if self.use_dro:    
                    dro = torch.cdist(robot_pc_target, object_pc, p=2)
                    dro_gt_batch.append(dro)

                if self.provide_pc and self.use_dro:
                    wrist_dro = torch.cdist(robot_pc_wrist_target, object_pc, p=2)
                    wrist_dro_gt_batch.append(wrist_dro)

                intent_batch.append(intent)
                object_pc_batch.append(object_pc)
                object_pc_ddpm_batch.append(object_pc_ddpm)
                cross_object_batch.append(cross_object_pc)
                language_embedding_batch.append(language_embedding)


            if self.provide_pc:
                robot_pc_target_batch = torch.stack(robot_pc_target_batch)
                robot_pc_nofix_initial_batch = torch.stack(robot_pc_nofix_initial_batch)
                robot_pc_wrist_target_batch = torch.stack(robot_pc_wrist_target_batch)
                cross_pc_target_batch = torch.stack(cross_pc_target_batch)

            if self.use_dro:
                dro_gt_batch = torch.stack(dro_gt_batch)

            if self.use_dro and self.provide_pc:
                wrist_dro_gt_batch = torch.stack(wrist_dro_gt_batch)
            
            intent_batch = torch.tensor(intent_batch, dtype=torch.int).reshape(-1, 1)
            object_pc_batch = torch.stack(object_pc_batch)
            object_pc_ddpm_batch = torch.stack(object_pc_ddpm_batch)
            cross_object_batch = torch.stack(cross_object_batch)
            language_embedding_batch = torch.stack(language_embedding_batch)
            nofix_initial_q_batch = torch.stack(nofix_initial_q_batch)
            initial_q_batch = torch.stack(initial_q_batch)
            target_q_batch = torch.stack(target_q_batch)
            robot_pc_initial_batch = torch.stack(robot_pc_initial_batch)

            B, N = self.batch_size, self.num_points

            if len(complex_language_embedding_batch) == B:
                complex_language_embedding_batch = torch.stack(complex_language_embedding_batch)
                complex_language_embedding_clip_768_batch = torch.stack(complex_language_embedding_clip_768_batch)
                complex_language_embedding_openai_256_batch = torch.stack(complex_language_embedding_openai_256_batch)
                complex_language_embedding_clip_512_batch = torch.stack(complex_language_embedding_clip_512_batch)
            else:
                print("Include non-oakink data, cannot stack complex language embedding")

            # assert robot_pc_initial_batch.shape == (B, N, 3),\
            #     f"Expected: {(B, N, 3)}, Actual: {robot_pc_initial_batch.shape}"
            # assert robot_pc_target_batch.shape == (B, N, 3),\
            #     f"Expected: {(B, N, 3)}, Actual: {robot_pc_target_batch.shape}"
            # assert object_pc_batch.shape == (B, N, 3),\
            #     f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
            # assert dro_gt_batch.shape == (B, N, N),\
            #     f"Expected: {(B, N, N)}, Actual: {dro_gt_batch.shape}"

            initial_q_batch = initial_q_batch if self.use_fixed_initial_q else nofix_initial_q_batch
            robot_pc_initial_batch = robot_pc_initial_batch if self.use_fixed_initial_q else robot_pc_nofix_initial_batch

            return {
                'robot_name': robot_name_batch,  # list(len = B): str
                'object_name': object_name_batch,  # list(len = B): str
                'object_id': object_id_batch,  # list(len = B): str
                'robot_links_pc': robot_links_pc_batch,  # list(len = B): dict, {link_name: (N_link, 3)}
                'robot_pc_initial': robot_pc_initial_batch,
                'robot_pc_target': robot_pc_target_batch,
                'robot_pc_wrist_target': robot_pc_wrist_target_batch,
                'cross_pc_target': cross_pc_target_batch,
                'cross_object_pc': cross_object_batch,
                'language_embedding': language_embedding_batch,
                'object_pc': object_pc_batch,
                'object_pc_ddpm': object_pc_ddpm_batch,
                'dro_gt': dro_gt_batch,
                'wrist_dro_gt': wrist_dro_gt_batch,
                'initial_q': initial_q_batch,
                'target_q': target_q_batch,
                'wrist_target_q': wrist_target_q_batch,
                'intent': intent_batch,
                'complex_language_embedding': complex_language_embedding_batch,
                'complex_language_embedding_clip_512': complex_language_embedding_clip_512_batch,
                'complex_language_embedding_clip_768': complex_language_embedding_clip_768_batch,
                'complex_language_embedding_openai_256': complex_language_embedding_openai_256_batch,
                'complex_language_sentence': complex_language_sentence_batch
            }
        else:  # validate
            robot_name, object_name = self.combination[index]
            hand = self.hands[robot_name]

            initial_q_batch = torch.zeros([self.batch_size, hand.dof], dtype=torch.float32)
            robot_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)
            object_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)

            for batch_idx in range(self.batch_size):

                if self.use_fixed_initial_q:
                    initial_q = hand.get_fixed_initial_q()
                else:
                    initial_q = hand.get_initial_q()
            
                robot_pc = hand.get_transformed_links_pc(initial_q, only_palm=self.only_palm)[:, :3]

                if self.object_pc_type == 'partial':
                    indices = torch.randperm(65536)[:self.num_points * 2]
                    object_pc = self.object_pcs[object_name][indices]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    _, indices = torch.sort(proj)
                    indices = indices[self.num_points:]
                    object_pc = object_pc[indices]
                else:
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
                    object_pc = torch.load(object_path)[:, :3]

                initial_q_batch[batch_idx] = initial_q
                robot_pc_batch[batch_idx] = robot_pc
                object_pc_batch[batch_idx] = object_pc

            # B, N, DOF = self.batch_size, self.num_points, len(hand.pk_chain.get_joint_parameter_names())
            # assert initial_q_batch.shape == (B, DOF), \
            #     f"Expected: {(B, DOF)}, Actual: {initial_q_batch.shape}"
            # assert robot_pc_batch.shape == (B, N, 3), \
            #     f"Expected: {(B, N, 3)}, Actual: {robot_pc_batch.shape}"
            # assert object_pc_batch.shape == (B, N, 3), \
            #     f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"

            return {
                'robot_name': robot_name,  # str
                'object_name': object_name,  # str
                'initial_q': initial_q_batch,
                'robot_pc': robot_pc_batch,
                'object_pc': object_pc_batch
            }

    def __len__(self):
        if self.is_train:
            return math.ceil((len(self.cmap_metadata)+len(self.oakink_metadata)) / self.batch_size)
        else:
            return len(self.combination)
        
    def _get_object_pc(self, object_name, object_id, robot_name, num_points=None):
        if num_points is None:
            num_points = self.num_points
        if self.object_pc_type == 'fixed':
            name = object_name.split('+')
            object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt') if (robot_name == 'mano' or robot_name == 'retarget_shadowhand') \
            else os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
            object_pc = torch.load(object_path)[:, :3]
        elif self.object_pc_type == 'random':
            indices = torch.randperm(65536)[:num_points]
            object_pc = self.object_pcs[object_id][indices] if (robot_name == 'mano' or robot_name == 'retarget_shadowhand') \
            else self.object_pcs[object_name][indices]
            object_pc += torch.randn(object_pc.shape) * 0.002
        else:  # 'partial', remove 50% points
            indices = torch.randperm(65536)[:num_points * 2]
            object_pc = self.object_pcs[object_id][indices] if (robot_name == 'mano' or robot_name == 'retarget_shadowhand') \
            else self.object_pcs[object_name][indices]
            direction = torch.randn(3)
            direction = direction / torch.norm(direction)
            proj = object_pc @ direction
            _, indices = torch.sort(proj)
            indices = indices[num_points:]
            object_pc = object_pc[indices]

        return object_pc


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

    # print(f"Expected Robot Counts: {expected_counts}")
    # print(f"Allocated Robot Counts: {counts}")
    # print(f"Final Robot Counts: {robots_num}")
    
    return robots_num

def custom_collate_fn(batch):
    return batch[0]


def create_dataloader(cfg, is_train, fix_sample=False, fixed_initial_q=False):

    print(f"Creating dataloader: complex_language_type={cfg.complex_language_type}")
    dataset = CombineDataset(
        batch_size=cfg.batch_size,
        robot_names=cfg.robot_names,
        is_train=is_train,
        debug_object_names=cfg.debug_object_names,
        object_pc_type=cfg.object_pc_type,
        data_ratio=cfg.ratio,
        fix_sample=fix_sample,
        fixed_initial_q=fixed_initial_q,
        complex_language_type=cfg.complex_language_type,
        use_valid_data=cfg.use_valid_data
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=cfg.num_workers,
        shuffle=is_train
    )
    return dataloader
