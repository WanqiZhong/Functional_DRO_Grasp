import os
import json
import pickle
import torch

# Define directory paths
OAKINK_DIR = '/data/zwq/data/oakink'  
DATASET_SAVE_PATH = '/data/zwq/code/DRO-Grasp/data/OakInkDataset/oakink_dataset.pt'

# Paths to metadata JSON files
REAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'object_id.json')
VIRTUAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'virtual_object_id.json')

# Load metadata
def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

real_meta = load_json(REAL_META_FILE)
virtual_meta = load_json(VIRTUAL_META_FILE)

# Load intent mappings
INTENT_MAP = {
    '0001': 'use',
    '0002': 'hold',
    '0003': 'liftup',
    '0004': 'handover',
}

# Initialize dataset
dataset = {
    'info': {},
    'metadata': []
}

# Start traversing
INTERACTION_DIR = os.path.join(OAKINK_DIR, 'shape', 'oakink_shape_v2')
robot_name = 'mano'
dataset['info'][robot_name] = {
    'robot_name': robot_name,
    'num_total': 0,
    'num_upper_object': 0,
    'num_per_object': {}
}

if not os.path.isdir(INTERACTION_DIR):
    print(f"Interaction set directory does not exist: {INTERACTION_DIR}")
    exit(1)

for object_type in os.listdir(INTERACTION_DIR):
    object_type_path = os.path.join(INTERACTION_DIR, object_type)
    if not os.path.isdir(object_type_path):
        continue

    object_key = f"oakink+{object_type}"
    dataset['info'][robot_name]['num_per_object'][object_key] = 0

    for real_object_id in os.listdir(object_type_path):
        real_object_path = os.path.join(object_type_path, real_object_id)
        if not os.path.isdir(real_object_path):
            continue

        # Check if real_object_id is in the metadata
        if real_object_id not in real_meta:
            print(f"Real object ID {real_object_id} not found in metadata, skipping.")
            continue

        # Process grasping for the real object
        for grasp_id in os.listdir(real_object_path):
            grasp_path = os.path.join(real_object_path, grasp_id)
            if not os.path.isdir(grasp_path):
                continue

            hand_param_path = os.path.join(grasp_path, 'hand_param.pkl')
            source_txt_path = os.path.join(grasp_path, 'source.txt')

            # Check if hand_param.pkl and source.txt exist
            if not os.path.isfile(hand_param_path) or not os.path.isfile(source_txt_path):
                print(f"Missing hand_param.pkl or source.txt in grasp path {grasp_path}, skipping this grasp.")
                continue

            # Read hand_param.pkl
            try:
                with open(hand_param_path, 'rb') as f:
                    hand_param = pickle.load(f)
            except Exception as e:
                print(f"Failed to load hand_param.pkl: {hand_param_path}, error: {e}")
                continue

            # Check if hand_param contains necessary keys
            if 'pose' not in hand_param or 'shape' not in hand_param or 'tsl' not in hand_param:
                print(f"Missing pose, shape, or tsl in {hand_param_path}, skipping this grasp.")
                continue

            # Read pose, shape, tsl
            pose = hand_param['pose']  # numpy array
            shape = hand_param['shape']  # numpy array
            tsl = hand_param['tsl']  # numpy array

            # Read intent
            with open(source_txt_path, 'r') as f:
                source_line = f.readline().strip()
                # Parse source_line to extract intent_id
                # Example format: pass1/A16012_0001_0005/2021-10-03-16-34-40/dom_alt.pkl
                parts = source_line.split('/')
                if len(parts) >= 2:
                    obj_intent_subject = parts[1]
                    intent_parts = obj_intent_subject.split('_')
                    if len(intent_parts) >= 2:
                        intent_id = intent_parts[1]
                        intent = INTENT_MAP.get(intent_id, 'unknown')
                    else:
                        print(f"Failed to parse intent_id from file {source_txt_path}, using 'unknown'.")
                        intent = 'unknown'
                else:
                    print(f"Failed to parse intent_id from file {source_txt_path}, using 'unknown'.")
                    intent = 'unknown'

            # Update metadata
            dataset['metadata'].append((
                torch.tensor(pose),
                torch.tensor(shape),
                torch.tensor(tsl),
                intent,
                object_key,
                robot_name,
                real_object_id,
                -1  # parent_object_id is -1 for real objects
            ))

            # Update statistics
            dataset['info'][robot_name]['num_total'] += 1
            dataset['info'][robot_name]['num_per_object'][object_key] += 1

            # Check for virtual objects
            for item in os.listdir(grasp_path):
                virtual_object_path = os.path.join(grasp_path, item)
                if os.path.isdir(virtual_object_path) and item.startswith('s'):
                    virtual_object_id = item

                    # Check if virtual_object_id is in the metadata
                    if virtual_object_id not in virtual_meta:
                        print(f"Virtual object ID {virtual_object_id} not found in metadata, skipping.")
                        continue

                    virtual_hand_param_path = os.path.join(virtual_object_path, 'hand_param.pkl')

                    if not os.path.isfile(virtual_hand_param_path):
                        print(f"Missing hand_param.pkl in virtual object path {virtual_object_path}, skipping this virtual object.")
                        continue

                    # Read virtual object's hand_param.pkl
                    try:
                        with open(virtual_hand_param_path, 'rb') as f:
                            virtual_hand_param = pickle.load(f)
                    except Exception as e:
                        print(f"Failed to load hand_param.pkl: {virtual_hand_param_path}, error: {e}")
                        continue

                    # Check if virtual hand_param contains necessary keys
                    if 'pose' not in virtual_hand_param or 'shape' not in virtual_hand_param or 'tsl' not in virtual_hand_param:
                        print(f"Missing pose, shape, or tsl in {virtual_hand_param_path}, skipping this virtual object.")
                        continue

                    # Read pose, shape, tsl
                    v_pose = virtual_hand_param['pose']
                    v_shape = virtual_hand_param['shape']
                    v_tsl = virtual_hand_param['tsl']

                    # Update metadata
                    dataset['metadata'].append((
                        torch.tensor(v_pose),
                        torch.tensor(v_shape),
                        torch.tensor(v_tsl),
                        intent,  # Virtual object's intent is the same as the real object's
                        object_key,
                        robot_name,
                        virtual_object_id,  # Virtual object ID
                        real_object_id  # parent_object_id is the real object's ID
                    ))

                    # Update statistics
                    dataset['info'][robot_name]['num_total'] += 1
                    dataset['info'][robot_name]['num_per_object'][object_key] += 1

    # Update num_upper_object
    dataset['info'][robot_name]['num_upper_object'] = max(dataset['info'][robot_name]['num_upper_object'], dataset['info'][robot_name]['num_per_object'][object_key])

# Save the dataset
try:
    os.makedirs(os.path.dirname(DATASET_SAVE_PATH), exist_ok=True)
    torch.save(dataset, DATASET_SAVE_PATH)
except Exception as e:
    print(f"Failed to save dataset: {DATASET_SAVE_PATH}, error: {e}")
    exit(1)
