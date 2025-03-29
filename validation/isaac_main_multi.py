import os
import sys
import json
import argparse
import warnings
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from validation.isaac_validator import IsaacValidator  # IsaacGym must be imported before PyTorch
from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler

import torch

DATASET_MAP = {
    "cmap": "data/CMapDataset/cmap_dataset.pt",
    "oakink": "data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand_coacd.pt"
}


def isaac_main(
    mode: str,
    robot_name: str,
    object_name: str = None,
    batch_size: int = 1,
    object_id: str = None,
    q_batch: torch.Tensor = None,
    object_file: str = None,
    gpu: int = 0,
    use_gui: bool = False,
    dataset_name: str = 'cmap'
):
    """
    For filtering dataset and validating grasps.
    Modified to support batch processing of multiple objects.

    :param mode: str, 'filter', 'validation', or 'batch_validation'
    :param robot_name: str
    :param object_name: str, used for single object validation
    :param batch_size: int, number of grasps in Isaac Gym simultaneously
    :param object_id: str, used for single object validation
    :param q_batch: torch.Tensor (validation only)
    :param object_file: str, path to file containing object data for batch validation
    :param gpu: int, specify the GPU device used by Isaac Gym
    :param use_gui: bool, whether to visualize Isaac Gym simulation process
    :param dataset_name: str, dataset name
    :return: success: (batch_size,), bool, whether each grasp is successful in Isaac Gym;
             q_isaac: (success_num, DOF), torch.float32, successful joint values after the grasp phase
    """
    if mode == 'filter' and batch_size == 0:  # special judge for num_per_object = 0 in dataset
        return 0, None
    if use_gui:  # for unknown reason otherwise will segmentation fault :(
        gpu = 0

    data_urdf_path = os.path.join(ROOT_DIR, 'data/data_urdf')
    urdf_assets_meta = json.load(open(os.path.join(data_urdf_path, 'robot/urdf_assets_meta.json')))
    robot_urdf_path = urdf_assets_meta['urdf_path'][robot_name]
    
    # Handle single object validation
    if mode in ['filter', 'validation']:
        object_name_split = object_name.split('+') if object_name is not None else None
        if dataset_name == "cmap":
            object_urdf_path = f'{object_name_split[0]}/{object_name_split[1]}/coacd_decomposed_object_one_link.urdf'
        else:
            object_urdf_path = f'{object_name_split[0]}/{object_name_split[1]}/coacd_decomposed_object_one_link_{object_id}.urdf'
            if not os.path.exists(os.path.join(data_urdf_path, 'object', object_urdf_path)):
                return 0, None
            print(f"[Isaac] Using specific object urdf: {object_id}")
    
    # Handle batch validation with multiple objects
    elif mode == 'batch_validation':
        object_data = torch.load(object_file)
        object_names = object_data['object_names']
        object_ids = object_data['object_ids']
        
        # Prepare object URDF paths for each object in the batch
        object_urdf_paths = []
        for i, obj_name in enumerate(object_names):
            obj_name_split = obj_name.split('+') if obj_name is not None else None
            obj_id = object_ids[i] if object_ids is not None else None
            
            if dataset_name == "cmap":
                obj_urdf_path = f'{obj_name_split[0]}/{obj_name_split[1]}/coacd_decomposed_object_one_link.urdf'
            else:
                obj_urdf_path = f'{obj_name_split[0]}/{obj_name_split[1]}/coacd_decomposed_object_one_link_{obj_id}.urdf'
                if not os.path.exists(os.path.join(data_urdf_path, 'object', obj_urdf_path)):
                    print(f"[Isaac] Object URDF not found for {obj_name} with ID {obj_id}")
                    # Skip this object or use a placeholder - depends on requirements
                    obj_urdf_path = None  # Mark as not found
            
            object_urdf_paths.append(obj_urdf_path)

    hand = create_hand_model(robot_name)
    joint_orders = hand.get_joint_orders()

    simulator = IsaacValidator(
        robot_name=robot_name, 
        joint_orders=joint_orders, 
        batch_size=batch_size,
        gpu=gpu, 
        is_filter=(mode == 'filter'),
        use_gui=use_gui
    )
    print("[Isaac] IsaacValidator is created.")

    # Set assets based on the validation mode
    if mode in ['filter', 'validation']:
        simulator.set_asset(
            robot_path=os.path.join(data_urdf_path, 'robot'),
            robot_file=robot_urdf_path[21:],  # ignore 'data/data_urdf/robot/'
            object_path=os.path.join(data_urdf_path, 'object'),
            object_file=object_urdf_path
        )
        simulator.create_envs()

    elif mode == 'batch_validation':
        # Filter out None paths (objects not found)
        valid_indices = [i for i, path in enumerate(object_urdf_paths) if path is not None]
        valid_paths = [object_urdf_paths[i] for i in valid_indices]
        
        # Create a mask to track which objects were processed
        success_mask = torch.zeros(batch_size, dtype=torch.bool)
        
        # Only process objects with valid URDFs
        if len(valid_paths) > 0:
            object_path_list = [os.path.join(data_urdf_path, 'object')] * len(valid_paths)
            simulator.set_asset_list(
                robot_path=os.path.join(data_urdf_path, 'robot'),
                robot_file=robot_urdf_path[21:],  # ignore 'data/data_urdf/robot/'
                object_path_list=object_path_list,
                object_file_list=valid_paths
            )
        
        simulator.create_envs_list()
    
    print("[Isaac] IsaacValidator preparation is done.")

    if mode == 'filter':
        dataset_path = os.path.join(ROOT_DIR, DATASET_MAP[dataset_name])
        metadata = torch.load(dataset_path)['metadata']
        if dataset_name == "oakink":
            q_batch = [m[3] for m in metadata if m[7] == object_id and m[6] == robot_name]
        else:        
            q_batch = [m[1] for m in metadata if m[2] == object_name and m[3] == robot_name]
        q_batch = torch.stack(q_batch, dim=0).to(torch.device('cpu'))
    
    if q_batch.shape[-1] != len(joint_orders):
        q_batch = q_rot6d_to_q_euler(q_batch)

    # For batch validation, only process valid objects
    if mode == 'batch_validation':
        valid_q_batch = q_batch[valid_indices].to(torch.device('cpu'))
        simulator.set_actor_pose_dof(valid_q_batch)
        valid_success, valid_q_isaac = simulator.run_sim_list()
        
        # Initialize full result tensors
        success = torch.zeros(batch_size, dtype=torch.bool)
        q_isaac = torch.zeros((batch_size, valid_q_isaac.shape[1] if valid_q_isaac is not None else 0))
        
        # Populate results for valid objects
        for i, idx in enumerate(valid_indices):
            success[idx] = valid_success[i]
            if valid_q_isaac is not None and i < valid_q_isaac.shape[0]:
                q_isaac[idx] = valid_q_isaac[i]
    else:
        simulator.set_actor_pose_dof(q_batch.to(torch.device('cpu')))
        success, q_isaac = simulator.run_sim()
    
    simulator.destroy()
    return success, q_isaac


# for Python scripts subprocess call to avoid Isaac Gym GPU memory leak problem
if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--robot_name', type=str, required=True)
    parser.add_argument('--object_name', type=str, required=False)
    parser.add_argument('--object_id', type=str, required=False)
    parser.add_argument('--object_file', type=str, required=False, default=None)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--dataset_name',type=str, required=True)
    parser.add_argument('--q_file', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--use_gui', action='store_true')
    args = parser.parse_args()

    print(f'GPU: {args.gpu}')
    assert args.mode in ['filter', 'validation', 'batch_validation'], f"Unknown mode: {args.mode}!"
    assert ((args.object_id is not None or args.object_file is not None) or (args.dataset_name == "cmap")), "Object id is required for OakInk dataset!"

    q_batch = torch.load(args.q_file, map_location=f'cpu') if args.q_file is not None else None
    success, q_isaac = isaac_main(
        mode=args.mode,
        robot_name=args.robot_name,
        object_name=args.object_name,
        object_id=args.object_id,
        batch_size=args.batch_size,
        q_batch=q_batch,
        object_file=args.object_file,
        gpu=args.gpu,
        use_gui=args.use_gui,
        dataset_name=args.dataset_name
    )

    success_num = success.sum().item()
    print(f"Success: {success_num}/{args.batch_size}")

    if args.object_name is None:
        object_name_type = f"{args.object_name}_{args.object_id}" if args.dataset_name == "oakink" else args.object_name
    else:
        object_name_type = "batch"

    if args.mode == 'filter':
        print(f"<{args.robot_name}/{object_name_type}> before: {args.batch_size}, after: {success_num}")
        if success_num > 0:
            q_filtered = q_isaac[success]
            if args.dataset_name == "oakink":
                save_dir = str(os.path.join(ROOT_DIR, 'data/OakinkDataset_filtered', args.robot_name, args.object_name))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(q_filtered, os.path.join(save_dir, f'{args.object_id}_{success_num}.pt'))
            else:
                save_dir = str(os.path.join(ROOT_DIR, 'data/CMapDataset_filtered', args.robot_name))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(q_filtered, os.path.join(save_dir, f'{args.object_name}_{success_num}.pt'))
    elif args.mode in ['validation', 'batch_validation']:
        cprint(f"[{args.robot_name}/{object_name_type}] Result: {success_num}/{args.batch_size}", 'green')
        save_data = {
            'success': success,
            'q_isaac': q_isaac
        }
        os.makedirs(os.path.join(ROOT_DIR, 'tmp'), exist_ok=True)
        torch.save(save_data, os.path.join(ROOT_DIR, f'tmp/isaac_main_ret_{args.gpu}.pt'))
