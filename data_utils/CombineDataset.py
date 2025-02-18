import os
import sys
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

RATIO_MAP = {
    'shadowhand': 5,
    'mano': 5,
    'allegro': 4,
    'barrett': 3
}

VALIDATE_RATIO_MAP = {
    'shadowhand': 5,
    'allegro': 4,
    'barrett': 3
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
        use_fixed_initial_q: bool = True
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names if robot_names is not None else ['barrett', 'allegro', 'shadowhand']
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.use_fixed_initial_q = use_fixed_initial_q

        # Load Hand (Both CMapDataset and OakInkDataset)
        self.hands = {}
        self.dofs = []
        self.robot_ratio ={}
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(math.sqrt(self.hands[robot_name].dof))
            self.robot_ratio[robot_name] = RATIO_MAP[robot_name]


        # Load CMapDataset
        print("Loading CMapDataset...")
        cmap_split_json_path = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered/split_train_validate_objects.json')
        cmap_dataset_split = json.load(open(cmap_split_json_path))
        self.cmap_object_names = cmap_dataset_split['train'] if is_train else cmap_dataset_split['validate']
        if debug_object_names is not None:
            print("!!! Using debug objects for CMapDataset !!!")
            self.cmap_object_names = debug_object_names

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
            # Load OakInkDataset metadata
            oakink_split_json_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/split_train_validate_objects.json')
            oakink_dataset_split = json.load(open(oakink_split_json_path))
            self.oakink_object_names = oakink_dataset_split['train'] if is_train else oakink_dataset_split['validate']
            if debug_object_names is not None:
                print("!!! Using debug objects for OakInkDataset !!!")
                self.oakink_object_names = debug_object_names

            oakink_dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all.pt')
            oakink_metadata = torch.load(oakink_dataset_path)['metadata']
            self.oakink_metadata = [m for m in oakink_metadata if m[5] in self.oakink_object_names and m[6] in self.robot_names]

            # Load OakInkDataset object_pcs
            print("Loading OakInkShape pcs...")
            oakink_object_pcs_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_object_pcs.pt')
            self.oakink_object_pcs = torch.load(oakink_object_pcs_path)

        # Metadata Robot
        self.metadata_robots = {}
        for robot_name in self.robot_names:
            if robot_name == 'mano':
                metadata_robot = [m for m in self.oakink_metadata if m[6] == robot_name]
            else:
                metadata_robot = [(m[0], m[1]) for m in self.cmap_metadata if m[2] == robot_name]
            self.metadata_robots[robot_name] = metadata_robot

        # Combine object_pcs
        self.object_pcs = {}
        self.object_pcs.update(self.cmap_object_pcs)
        if "mano" in self.robot_names:
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
            robot_pc_target_batch = []
            object_pc_batch = []
            dro_gt_batch = []
            initial_q_batch = []
            target_q_batch = []
            robot_names = calculate_robot_counts(self.batch_size, self.robot_ratio)

            for idx, robot_name in enumerate(robot_names):

                # >>> Debug Only >>>
                robot_name = random.choice(self.robot_names)
                # robot_name = "shadowhand" 
                # <<< End Debug <<<

                robot_name_batch.append(robot_name)
                hand: HandModel = self.hands[robot_name]
                metadata_robot = self.metadata_robots[robot_name]

                if robot_name == 'mano':
                    # metadata_robot = [m for m in self.oakink_metadata if m[6] == robot_name]
                    hand_pose, hand_shape, tsl, target_q, intent, object_name, robot_name, object_id, real_object_id, hand_verts = random.choice(metadata_robot)
                else:
                    # metadata_robot = [(m[0], m[1]) for m in self.cmap_metadata if m[2] == robot_name]
                    # >>> Debug Only >>>
                    target_q, object_name = random.choice(metadata_robot)
                    # target_q, object_name = metadata_robot[0]
                    # <<< End Debug <<<  

                target_q_batch.append(target_q)
                object_name_batch.append(object_name)

                if robot_name == 'mano':
                    robot_links_pc_batch.append(None)
                    object_id_batch.append(object_id)
                else:
                    robot_links_pc_batch.append(hand.links_pc)
                    object_id_batch.append(None)

                if self.object_pc_type == 'fixed':
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_id}.pt') if robot_name == 'mano' \
                    else os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
                    object_pc = torch.load(object_path)[:, :3]
                elif self.object_pc_type == 'random':
                    indices = torch.randperm(65536)[:self.num_points]
                    object_pc = self.object_pcs[object_id][indices] if robot_name == 'mano' \
                    else self.object_pcs[object_name][indices]
                    object_pc += torch.randn(object_pc.shape) * 0.002
                else:  # 'partial', remove 50% points
                    indices = torch.randperm(65536)[:self.num_points * 2]
                    object_pc = self.object_pcs[object_id][indices] if robot_name == 'mano' \
                    else self.object_pcs[object_name][indices]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    _, indices = torch.sort(proj)
                    indices = indices[self.num_points:]
                    object_pc = object_pc[indices]

                object_pc_batch.append(object_pc)

                if robot_name == 'mano':
                    robot_pc_target, _ = hand.get_mano_pc_from_verts(torch.tensor(hand_verts))
                    robot_pc_target_batch.append(robot_pc_target)
                    # target_q_batch.append(target_q)
                    target_q_batch.append(None)

                    if self.use_fixed_initial_q:
                        initial_q = hand.get_fixed_initial_q()
                    else:
                        initial_q = hand.get_initial_q(target_q)

                    # initial_q_batch.append(initial_q)
                    initial_q_batch.append(None)
                    robot_pc_initial, _ = hand.get_mano_pc(initial_q, tsl, hand_pose.unsqueeze(0), hand_shape.unsqueeze(0))
                    robot_pc_initial_batch.append(robot_pc_initial)
                else:
                    robot_pc_target = hand.get_transformed_links_pc(target_q)[:, :3]
                    robot_pc_target_batch.append(robot_pc_target)

                    if self.use_fixed_initial_q:
                        initial_q = hand.get_fixed_initial_q()
                    else:
                        # >>> Debug Only >>>
                        initial_q = hand.get_initial_q(target_q)
                        # initial_q = hand.get_initial_q()
                        # <<< End Debug <<<  

                    initial_q_batch.append(initial_q)
                    robot_pc_initial = hand.get_transformed_links_pc(initial_q)[:, :3]
                    robot_pc_initial_batch.append(robot_pc_initial)

                dro = torch.cdist(robot_pc_target, object_pc, p=2)
                dro_gt_batch.append(dro)

            robot_pc_initial_batch = torch.stack(robot_pc_initial_batch)
            robot_pc_target_batch = torch.stack(robot_pc_target_batch)
            object_pc_batch = torch.stack(object_pc_batch)
            dro_gt_batch = torch.stack(dro_gt_batch)

            B, N = self.batch_size, self.num_points
            assert robot_pc_initial_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_initial_batch.shape}"
            assert robot_pc_target_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_target_batch.shape}"
            assert object_pc_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
            assert dro_gt_batch.shape == (B, N, N),\
                f"Expected: {(B, N, N)}, Actual: {dro_gt_batch.shape}"

            return {
                'robot_name': robot_name_batch,  # list(len = B): str
                'object_name': object_name_batch,  # list(len = B): str
                'object_id': object_id_batch,  # list(len = B): str
                'robot_links_pc': robot_links_pc_batch,  # list(len = B): dict, {link_name: (N_link, 3)}
                'robot_pc_initial': robot_pc_initial_batch,
                'robot_pc_target': robot_pc_target_batch,
                'object_pc': object_pc_batch,
                'dro_gt': dro_gt_batch,
                'initial_q': initial_q_batch,
                'target_q': target_q_batch
            }
        else:  # validate
            robot_name, object_name = self.combination[index]
            hand = self.hands[robot_name]

            initial_q_batch = torch.zeros([self.batch_size, hand.dof], dtype=torch.float32)
            robot_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)
            object_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)

            for batch_idx in range(self.batch_size):
                # initial_q = hand.get_initial_q()
                initial_q = hand.get_fixed_initial_q()
                robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3]

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

            B, N, DOF = self.batch_size, self.num_points, len(hand.pk_chain.get_joint_parameter_names())
            assert initial_q_batch.shape == (B, DOF), \
                f"Expected: {(B, DOF)}, Actual: {initial_q_batch.shape}"
            assert robot_pc_batch.shape == (B, N, 3), \
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_batch.shape}"
            assert object_pc_batch.shape == (B, N, 3), \
                f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"

            return {
                'robot_name': robot_name,  # str
                'object_name': object_name,  # str
                'initial_q': initial_q_batch,
                'robot_pc': robot_pc_batch,
                'object_pc': object_pc_batch
            }

    def __len__(self):
        if self.is_train:
            return math.ceil(len(self.cmap_metadata) / self.batch_size)
        else:
            return len(self.combination)


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


def create_dataloader(cfg, is_train):
    dataset = CombineDataset(
        batch_size=cfg.batch_size,
        robot_names=cfg.robot_names,
        is_train=is_train,
        debug_object_names=cfg.debug_object_names,
        object_pc_type=cfg.object_pc_type
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=cfg.num_workers,
        shuffle=is_train
    )
    return dataloader
