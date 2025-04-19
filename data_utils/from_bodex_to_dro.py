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

from DRO_Grasp.utils.rotation_format_utils import shadowhand_from_oakink_to_bodex, leaphand_from_oakink_to_bodex, leaphand_from_bodex_to_oakink, correct_shadowhand_wrist
from DRO_Grasp.utils.rotation import matrix_to_euler, euler_to_matrix, euler_to_quaternion, quaternion_to_euler
from copy import deepcopy
from tqdm import tqdm
import argparse 

def convert_bodex_to_dro(input_path):
    # Load the dataset
    print(f"Loading data from {input_path}")
    data = torch.load(input_path)
    return convert_dataset(input_path, data)
    

def convert_dataset(input_path, data):

    output_path = input_path.replace("_valid_bodex", "_valid_dro")
    
    metadata = data['metadata']
    new_metadata = []
    
    # Process each item in metadata
    for i, item in enumerate(tqdm(metadata, desc="Converting robot values")):
        # Extract values for conversion
        dataset_item = item[0]
        robot_value = item[1][0, 1, :]
        robot_name = dataset_item[6]

        # print only first time
        if i == 0:
            print(f"robot_name: {robot_name}")

        if robot_name == "shadowhand":     
            robot_value = torch.tensor(correct_shadowhand_wrist(robot_value))
            euler = quaternion_to_euler(torch.cat([robot_value[4:7], robot_value[3:4]]))
            converted_value = torch.cat([
                robot_value[:3],            # First 3 values
                euler,                      # Euler angles
                torch.tensor([0.0, 0.0]),   # Two zeros
                robot_value[12:],           # From index 12 to end
                robot_value[7:12]           # From index 7 to 12
            ], axis=-1).float()
            assert converted_value.shape == (30,), f"converted_value shape: {converted_value.shape}"
        elif robot_name == "allegro":
            converted_value = torch.cat([robot_value[:3], quaternion_to_euler(torch.cat([robot_value[4:7], robot_value[3:4]])), robot_value[7:]])
            assert converted_value.shape == (22,), f"converted_value shape: {converted_value.shape}"
        elif robot_name == "leaphand":
            # Convert robot value to BoDex format
            converted_value = leaphand_from_bodex_to_oakink(robot_value)
            converted_value = torch.from_numpy(converted_value)
            assert converted_value.shape == (22,), f"converted_value shape: {converted_value.shape}"
        else:
            raise ValueError(f"Unsupported robot name: {robot_name}")
        
        # Create a new item with the converted value
        dataset_item = list(dataset_item)  # Convert to list if it's a tuple
        dataset_item[3] = converted_value
        dataset_item[8] = float(dataset_item[8])
        new_metadata.append(tuple(dataset_item))

    # Save the converted dataset
    data['metadata'] = new_metadata
    print(f"Saving {len(new_metadata)} dro items to {output_path}")
    torch.save(data, output_path)
    if os.path.exists(input_path):
        os.remove(input_path)
        print(f"Removed original file: {input_path}")
    else:
        print(f"Original file not found: {input_path}")
    print(f"Conversion complete!")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BoDex dataset to DRO format")
    parser.add_argument(
        "--i",
        required=True,
        type=str,
        help="Path to the OakInk dataset or combined grasp results PT file",
    )
    
    args = parser.parse_args()    
    convert_bodex_to_dro(args.i)