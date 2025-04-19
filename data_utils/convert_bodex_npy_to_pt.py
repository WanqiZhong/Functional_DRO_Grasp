import os
import numpy as np
from collections import defaultdict, Counter
import os
import sys 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch

# Base directory path
base_dir = "/data/zwq/code/DexGraspBench/output/oakink_all_leap_bodex/succgrasp"
hand_name = "leaphand"
# hand_name = "allegro"
output_path = os.path.join(ROOT_DIR, f"data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{hand_name}_valid_bodex.pt")


def traverse_and_extract():
    """
    Traverse all .npy files and extract oakink_item and robot_pose
    Returns a dictionary with the required structure
    """
    metadata = []
    object_stats = defaultdict(lambda: defaultdict(Counter))
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        print("Processing file: ", root)
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                # print(file_path)
                
                try:
                    # Load the .npy file
                    data = np.load(file_path, allow_pickle=True).item()
                    
                    # Extract the required fields
                    if 'oakink_item' in data and 'robot_pose' in data:
                        oakink_item = data['oakink_item']
                        robot_pose = data['robot_pose'].cpu()
                        # print(f"robot_pose: {robot_pose.shape}")

                        new_oakink_item = tuple()
                        for i in range(len(oakink_item)):
                            if isinstance(oakink_item[i], torch.Tensor):
                                new_oakink_item += (oakink_item[i].clone(),)
                            else:
                                new_oakink_item += (oakink_item[i],)
                        
                        # Add to metadata list
                        metadata_entry = [new_oakink_item, robot_pose, file_path]
                        metadata.append(metadata_entry)
                        
                        # Extract object information for statistics
                        object_key = oakink_item[5]
                        object_id = oakink_item[7]
                        hand_id = oakink_item[6]

                        # Update statistics
                        object_stats[hand_id][object_key]['all'] += 1
                        object_stats[hand_id][object_key][object_id] += 1
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Create the final dictionary structure
    result = {
        'metadata': metadata,
        'info': create_info_dict(object_stats),
        'version': {
            'metadata': '3.0.0',
            'format': 'bodex',
            'description': 'Metadata with valid entries and COACD compatibility.'
        }
    }
    
    return result

def create_info_dict(object_stats):
    """
    Create the info dictionary based on collected statistics
    """
    info = {}
    
    for robot_name, robot_data in object_stats.items():
        robot_info = {
            'robot_name': 'shadowhand',
            'num_total': sum(data['all'] for data in robot_data.values()),
            'num_per_object': {}
        }
        
        # Calculate max entries for any object
        max_entries = 0
        for object_key, object_data in robot_data.items():
            object_dict = {
                'all': object_data['all']
            }
            
            # Add individual object IDs
            for obj_id, count in object_data.items():
                if obj_id != 'all':
                    object_dict[obj_id] = count
            
            robot_info['num_per_object'][object_key] = object_dict
            max_entries = max(max_entries, object_data['all'])
        
        robot_info['num_upper_object'] = max_entries
        info[robot_name] = robot_info
    
    return info

# Process and save the result
result = traverse_and_extract()
# Save the result to a file
torch.save(result, output_path)
print(f"Processed data saved to {output_path}")

# Print summary statistics
print("Processing complete.")
print(f"Total metadata entries: {len(result['metadata'])}")
# for robot, robot_info in result['info'].items():
#     print(f"Robot: {robot}")
#     print(f"  Total entries: {robot_info['num_total']}")
#     print(f"  Max entries per object: {robot_info['num_upper_object']}")
#     print("  Objects:")
#     for obj_key, obj_data in robot_info['num_per_object'].items():
#         print(f"    {obj_key}: {obj_data['all']} entries")