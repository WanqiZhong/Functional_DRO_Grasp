import os
import sys
import time
import warnings
from termcolor import cprint
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import spacy
import random
import open3d as o3d
import glob
from utils.hand_model import farthest_point_sampling

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import trimesh
from model.network import create_network
from data_utils.CMapDataset import create_dataloader
from utils.hand_model import create_hand_model
import viser
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from validation.validate_utils import validate_isaac
import math

nlp = spacy.load('en_core_web_sm')
metadata = torch.load('/data/zwq/code/DRO-Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand.pt')['metadata']

server = viser.ViserServer(host='127.0.0.1', port=8080)
print("Loading OakInkShape pcs...")
object_pcs = torch.load(os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_object_pcs.pt'))

all_robot_names = [
    "shadowhand",
    "allegro",
]
train_objects = [
    "oakink+gamecontroller",
    "oakink+toothbrush",
    "oakink+wineglass",
    "oakink+cup",
    "oakink+mouse",
    "oakink+binoculars",
    "oakink+lightbulb",
    "oakink+lotion_pump",
    "oakink+squeezable",
    "oakink+hammer",
    "oakink+pen",
    "oakink+pincer",
    "oakink+mug",
    "oakink+screwdriver",
    "oakink+banana",
    "oakink+stapler",
    "oakink+fryingpan",
    "oakink+bowl",
    "oakink+phone",
    "oakink+scissors",
    "oakink+flashlight",
    "oakink+eyeglasses",
    "oakink+teapot",
    "oakink+power_drill",
    "oakink+wrench",
    "oakink+trigger_sprayer",
    "oakink+donut",
    "oakink+cylinder_bottle"
]

validate_objects = [
    "oakink+apple",
    "oakink+bottle",
    "oakink+cameras",
    "oakink+knife",
    "oakink+headphones"
]

train_object_names = [obj.split('+')[1] for obj in train_objects]
validate_object_names = [obj.split('+')[1] for obj in validate_objects]

def preprocess_metadata(metadata):
    category_intent_to_entries = {}
    for entry in metadata:
        object_key = entry[5]  # object_key
        intent = entry[4]      # intent
        object_name = object_key.split('+')[1] 
        
        if object_name not in category_intent_to_entries:
            category_intent_to_entries[object_name] = {}
        
        if intent not in category_intent_to_entries[object_name]:
            category_intent_to_entries[object_name][intent] = []
        
        category_intent_to_entries[object_name][intent].append(entry)
    
    return category_intent_to_entries

category_intent_to_entries = preprocess_metadata(metadata)

def recognize_object(object_pc):
    recognized_object_name = "apple"  
    return recognized_object_name

def find_similar_objects(recognized_object_name, category_intent_to_entries, intent, top_k=10):
    recognized_doc = nlp(recognized_object_name)
    similarities = []
    
    for obj_name in train_object_names:
        obj_doc = nlp(obj_name)
        sim = recognized_doc.similarity(obj_doc)
        similarities.append((sim, obj_name))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    for sim, obj_name in similarities[:top_k]:
        print(f"Similarity: {sim}, Object: {obj_name}")
        if intent in category_intent_to_entries.get(obj_name, {}):
            entries = category_intent_to_entries[obj_name][intent]
            selected_entry = random.choice(entries)  
            print(f"Found similar object: {obj_name} with intent {intent}")
            return selected_entry 
    
    raise ValueError(f"No matching object found for {recognized_object_name} with intent {intent}")
    return None

def compute_hand_vertices(hand, hand_pose, hand_shape, hand_tsl):
    hand_output = hand(hand_pose, hand_shape)
    hand_vertices = hand_output.verts
    return hand_vertices

@hydra.main(version_base="1.2", config_path="configs", config_name="validate_w_mano")
def main(cfg):
    print("******************************** [Config] ********************************")
    
    os.makedirs(os.path.join(ROOT_DIR, 'validate_output', "reltrival_validate"), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/reltrival_validate/{cfg.name}.log')
    print('Log file:', log_file_name)

    device = torch.device(f'cuda:{0}')
    network = create_network(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{40}.pth", map_location=device))
    network.eval()

    val_metadata = [entry for entry in metadata if entry[5] in validate_objects]
    global_robot_name = None
    hand = None

    hands = {}
    dofs = []

    for robot_name in all_robot_names:
        hands[robot_name] = create_hand_model(robot_name, device)
        dofs.append(math.sqrt(hands[robot_name].dof))

    for i, data in enumerate(val_metadata):

        robot_name = np.random.choice(all_robot_names)
        intent = data[4]         
        recognized_object_name = data[5]
        object_id = data[7]

        hand = hands[robot_name]
        initial_q = hand.get_initial_q().unsqueeze(0).to(device)
        robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3]

        indices = torch.randperm(65536)[:512]
        object_pc = object_pcs[object_id][indices]
        object_pc += torch.randn(object_pc.shape) * 0.002

        print(f"Input recognized object: {recognized_object_name}")
        print(f"Input intent: {intent}")

        similar_entries = find_similar_objects(recognized_object_name, category_intent_to_entries, intent)

        if not similar_entries:
            print(f"No matching object found for {recognized_object_name} with intent {intent}")
            continue

        target_pc, _ = farthest_point_sampling(torch.from_numpy(similar_entries[-1]), 512)

        robot_pc = robot_pc.unsqueeze(0).to(device)
        object_pc = object_pc.unsqueeze(0).to(device)
        target_pc = target_pc.unsqueeze(0).to(device)

        with torch.no_grad():
            dro = network(
                robot_pc,
                object_pc,
                target_pc
            )['dro'].detach()

        mlat_pc = multilateration(dro, object_pc)
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
        optim_transform = process_transform(hand.pk_chain, transform)

        layer = create_problem(hand.pk_chain, optim_transform.keys())
        start_time = time.time()
        predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
        end_time = time.time()
        print("Time: ", end_time - start_time)

        name = data[5].split('+')
        obj_mesh_path = list(
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{data[7]}.obj')) +
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{data[7]}.ply')))
        assert len(obj_mesh_path) == 1
        obj_path = obj_mesh_path[0]
        object_mesh = o3d.io.read_triangle_mesh(obj_path)
        vertices = np.asarray(object_mesh.vertices)
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
        object_mesh.vertices = o3d.utility.Vector3dVector(vertices - bbox_center)
        robot_trimesh = hand.get_trimesh_q(predict_q)["visual"]

        server.scene.add_mesh_simple(
            'object',
            object_mesh.vertices,
            np.asarray(object_mesh.triangles),
            color=(239, 132, 167),
            opacity=1
        )

        server.scene.add_mesh_simple(
            'robot',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=0.8
        )

        server.render()  

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()
    while True:
        time.sleep(1)