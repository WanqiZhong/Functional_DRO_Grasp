import array
from logging import config
import os
from glob import glob
import logging
import multiprocessing

import numpy as np
from transforms3d import quaternions as tq
import torch
import pickle

from DRO_Grasp.utils.rotation_format_utils import shadowhand_from_oakink_to_bodex, leaphand_from_oakink_to_bodex, leaphand_from_bodex_to_oakink
from DRO_Grasp.utils.rotation import matrix_to_euler, euler_to_matrix, euler_to_quaternion
from copy import deepcopy
from tqdm import tqdm
import argparse 

def convert_dataset_to_bodex(input_path):
    """
    Determine the type of dataset and convert accordingly.
    
    Args:
        input_path: Path to either the OakInk dataset or combined grasp results
    """
    # Load the dataset
    print(f"Loading data from {input_path}")
    data = torch.load(input_path)
    
    # # Determine if this is a combined results file or original dataset
    # if isinstance(data, dict) and "results" in data and "metadata" in data:
    #     # This is a combined results file from our grasp solver
    #     return convert_combined_results(input_path, data)
    # elif isinstance(data, dict) and "metadata" in data:
    # This is likely the original OakInk dataset
    return convert_oakink_dataset(input_path, data)
    

def convert_oakink_dataset(input_path, data):
    """
    Convert the original OakInk dataset.
    
    Args:
        input_path: Path to the dataset file
        data: The loaded dataset
    """
    # Determine output path
    filename, ext = os.path.splitext(input_path)
    output_path = f"{filename}_mujoco{ext}"
    
    # Get the metadata
    metadata = data['metadata']
    print(f"Found {len(metadata)} items in metadata")
    
    # Process each item in metadata
    for i, item in enumerate(tqdm(metadata, desc="Converting robot values")):
        # Extract values for conversion
        robot_value = item[3]
        robot_name = item[6]

        # print only first time
        if i == 0:
            print(f"robot_name: {robot_name}")

        if robot_name == "shadowhand":     
            # Convert robot value to BoDex format
            converted_value = shadowhand_from_oakink_to_bodex(robot_value, robot_value[7], robot_value[6])
            converted_value = np.concatenate([converted_value[:7], converted_value[-5:], converted_value[9:-5]])
            assert converted_value.shape == (29,), f"converted_value shape: {converted_value.shape}"
        elif robot_name == "allegro":
            euler = robot_value[3:6]
            quat = euler_to_quaternion(torch.tensor(euler).float()).cpu().numpy()
            converted_value = np.concatenate([robot_value[:3], quat[-1:], quat[:-1], robot_value[6:]])  # change from x,y,z,w to w,x,y,z
            assert converted_value.shape == (23,), f"converted_value shape: {converted_value.shape}"
        elif robot_name == "leaphand":
            # Convert robot value to BoDex format
            robot_value = robot_value.numpy()
            converted_value = leaphand_from_oakink_to_bodex(robot_value)
            # converted_value[7:] = leaphand_order_from_oakink_to_bodex(converted_value[7:])
            assert converted_value.shape == (23,), f"converted_value shape: {converted_value.shape}"
        else:
            raise ValueError(f"Unsupported robot name: {robot_name}")
        
        # Create a new item with the converted value
        item = list(item)  # Convert to list if it's a tuple
        item[3] = torch.tensor(converted_value)
        item[8] = float(item[8])  # Ensure the grasp score is a float
        metadata[i] = tuple(item)
    
    # Save the converted dataset
    print(f"Saving converted dataset to {output_path}")
    torch.save(data, output_path)
    print(f"Conversion complete!")
    
    return output_path

# def convert_combined_results(input_path, data):
#     """
#     Convert the combined grasp results from our solver.
    
#     Args:
#         input_path: Path to the combined results file
#         data: The loaded data
#     """
#     # Determine output path
#     filename, ext = os.path.splitext(input_path)
#     output_path = f"{filename}_mujoco{ext}"
    
#     # Get the results and metadata
#     results = data.get("results", [])
#     metadata = data.get("metadata", [])
    
#     print(f"Found {len(results)} result entries")
    
#     # Create a new list to store the converted results
#     converted_results = []
    
#     # Process each result pair
#     for i, (item, result) in enumerate(tqdm(results, desc="Converting robot values")):
#         # Extract values for conversion
#         robot_value = item[3]
#         object_id = item[7]
#         robot_name = item[6]
        
#         # Convert robot value to Mujoco format
#         try:
#             converted_value = shadowhand_from_oakink_to_bodex(robot_value, object_id, robot_name)
            
#             # Create a new item with the converted value
#             new_item = list(item)  # Convert to list if it's a tuple
#             new_item[3] = converted_value
            
#             # Add to converted results
#             converted_results.append([new_item, result])
#         except Exception as e:
#             print(f"Error converting item {i}: {e}")
#             # Keep the original item if conversion fails
#             converted_results.append([item, result])
    
#     # Create new data with converted results
#     new_data = {
#         "results": converted_results,
#         "metadata": metadata  # Keep original metadata
#     }
    
#     # Save the converted results
#     print(f"Saving converted results to {output_path}")
#     torch.save(new_data, output_path)
#     print(f"Conversion complete!")
    
#     return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OakInk dataset or grasp results to Mujoco format")
    parser.add_argument(
        "--i",
        default="/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_leaphand.pt",
        type=str,
        help="Path to the OakInk dataset or combined grasp results PT file",
    )
    
    args = parser.parse_args()
    
    # Convert the dataset
    convert_dataset_to_bodex(args.i)