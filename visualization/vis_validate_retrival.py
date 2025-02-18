import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
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
import trimesh
from model.network import create_network
from data_utils.CombineDataset import create_dataloader
from utils.hand_model import create_hand_model, HandModel
import viser
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import *
from validation.validate_utils import validate_isaac
import math
from manotorch.manolayer import ManoLayer
import en_core_web_lg
from transformers import AutoTokenizer, AutoModel
from simcse import SimCSE
from torch.nn.functional import cosine_similarity


tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
print("Loading metadata...")
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

all_intent = [
    "use",
    "hold",
    "liftup",
    "handover"
]



def preprocess_metadata(metadata):

    category_intent_to_entries = {}
    category_object_to_entries = {}

    for entry in metadata:
        object_key = entry[5]  # object_key
        intent = entry[4]      # intent
        object_name = object_key.split('+')[1] 
        
        if object_name not in category_intent_to_entries:
            category_intent_to_entries[object_name] = {}
            category_object_to_entries[object_name] = []
        
        if intent not in category_intent_to_entries[object_name]:
            category_intent_to_entries[object_name][intent] = []
        
        category_intent_to_entries[object_name][intent].append(entry)
        category_object_to_entries[object_name].append(entry)
    
    return category_intent_to_entries, category_object_to_entries

def recognize_object(object_pc):
    recognized_object_name = "apple"  
    return recognized_object_name

def find_similar_objects(topk_similar_names, recognized_object_name, category_intent_to_entries, intent, top_k=3):

    recognized_object_name = recognized_object_name.split('+')[1]
    similarities = topk_similar_names[recognized_object_name]
    random.seed()
    random.shuffle(similarities)
    
    for obj_name in similarities:
        if intent in category_intent_to_entries.get(obj_name, {}):
            entries = category_intent_to_entries[obj_name][intent]
            selected_entry = random.choice(entries)  
            print(f"Found similar object: {obj_name} with intent {intent}")
            return selected_entry 
    
    raise ValueError(f"No matching object found for {recognized_object_name} with intent {intent}")

def compute_hand_vertices(hand, hand_pose, hand_shape, hand_tsl):
    hand_output = hand(hand_pose, hand_shape)
    hand_vertices = hand_output.verts
    return hand_vertices

def get_object_vertices(object_name, object_id):

    name = object_name.split('+')
    obj_mesh_path = list(
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.obj')) +
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.ply'))
    )
    if len(obj_mesh_path) == 0:
        print(f"No mesh file found for object ID {object_id}")
        return
    assert len(obj_mesh_path) == 1, f"Multiple mesh files found for object ID {object_id}"
    obj_path = obj_mesh_path[0]
    object_mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(object_mesh.vertices)

    return vertices, np.asarray(object_mesh.triangles)

    # mesh = trimesh.load(obj_path, force='mesh')
    # bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    # mesh.vertices -= bbox_center

    # return mesh.vertices, mesh.faces


mano_layer = ManoLayer( rot_mode="axisang",
                        mano_assets_root="assets/mano",
                        use_pca=False,
                        flat_hand_mean=True)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
train_object_names = [obj.split('+')[1] for obj in train_objects]
validate_object_names = [obj.split('+')[1] for obj in validate_objects]
all_object_names = train_object_names + validate_object_names
category_intent_to_entries, category_object_to_entries = preprocess_metadata(metadata)


def get_similarity_matrix(top_k = 1):
    all_object_inputs = tokenizer(all_object_names, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**all_object_inputs, output_hidden_states=True, return_dict=True).pooler_output
    num_train = len(train_object_names)
    train_embeddings = embeddings[:num_train]  
    validate_embeddings = embeddings[num_train:]
    similarity_matrix = cosine_similarity(
        embeddings.unsqueeze(1), 
        train_embeddings.unsqueeze(0), 
        dim=2
    )  
    topk_values, topk_indices = torch.topk(similarity_matrix, top_k, dim=1, largest=True, sorted=True)  
    topk_similar_names = {all_object_names[i]: [train_object_names[idx] for idx in indices] for i, indices in enumerate(topk_indices)}
    return topk_similar_names

@hydra.main(version_base="1.2", config_path="../configs", config_name="validate_mix_mano")
def main(cfg):
    print("******************************** [Config] ********************************")
    
    os.makedirs(os.path.join(ROOT_DIR, 'validate_output', "reltrival_validate"), exist_ok=True)
    log_file_name = os.path.join(ROOT_DIR, f'validate_output/reltrival_validate/{cfg.name}.log')
    print('Log file:', log_file_name)

    device = torch.device(f'cuda:{cfg.gpu}')
    network = create_network(cfg.model, mode='validate').to(device)
    print(f"Load from output/{cfg.name}/state_dict/epoch_{cfg.validate_epochs[0]}.pth")
    network.load_state_dict(torch.load(f"output/{cfg.name}/state_dict/epoch_{cfg.validate_epochs[0]}.pth", map_location=device))
    network.eval()

    hands = {}
    dofs = []
    topk_similar_names = get_similarity_matrix()

    for robot_name in all_robot_names:
        hands[robot_name] = create_hand_model(robot_name, device)
        dofs.append(math.sqrt(hands[robot_name].dof))

    def on_update(robot_id, intent_id, object_id, grasp_id):
        if grasp_id < 0 or grasp_id >= len(metadata):
            print(f"grasp_id {grasp_id} out of range.")
            return
        
        data = category_object_to_entries[all_object_names[object_id]][grasp_id]
        robot_name = all_robot_names[robot_id]
        intent = all_intent[intent_id] 
        recognized_object_name = data[5]
        object_id_data = data[7]

        hand: HandModel = hands[robot_name]
        initial_q = hand.get_initial_q().unsqueeze(0).to(device)
        robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3]

        indices = torch.randperm(65536)[:512]
        object_pc = object_pcs[object_id_data][indices]
        object_pc += torch.randn(object_pc.shape) * 0.002

        print(f"Input {grasp_id} recognized object: {recognized_object_name} with intent {intent}")

        # if recognized_object_name in train_objects:
        print("Using original object")
        similar_entries = data
        # else:
        #     similar_entries = find_similar_objects(topk_similar_names, recognized_object_name, category_intent_to_entries, intent)
        #     if not similar_entries:
        #         print(f"No matching object found for {recognized_object_name} with intent {intent}")
        #         return

        target_q = similar_entries[3]
        origin_hand_verts = similar_entries[9]
        target_pc, _ = farthest_point_sampling(torch.from_numpy(origin_hand_verts), 512)

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
        robot_trimesh = hand.get_trimesh_q(predict_q)["visual"]

        retrival_object_vertices, retrival_object_faces = get_object_vertices(similar_entries[5], similar_entries[7])
        object_vertices, object_faces = get_object_vertices(recognized_object_name, object_id_data)

        target_hand = hands[all_robot_names[0]]
        target_robot_trimesh = target_hand.get_trimesh_q(target_q)["visual"]

        server.scene.add_mesh_simple(
            'object',
            object_vertices,
            object_faces,
            color=(239, 132, 167),
            opacity=1
        )

        server.scene.add_mesh_simple(
            'origin_object',
            retrival_object_vertices,
            retrival_object_faces,
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

        server.scene.add_mesh_simple(
            'human',
            origin_hand_verts,
            mano_layer.th_faces.cpu().numpy(),
            color=(192, 102, 255),
            opacity=0.8
        )

        server.scene.add_mesh_simple(
            'target_robot',
            target_robot_trimesh.vertices,
            target_robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=0.8
        )



    robot_slider = server.gui.add_slider(
        label='Robot',
        min=0,
        max=len(all_robot_names)-1,
        step=1,
        initial_value=0
    )

    object_slider = server.gui.add_slider(
        label='Object',
        min=0,
        max=len(all_object_names)-1,
        step=1,
        initial_value=0
    )

    intent_slider = server.gui.add_slider(
        label='Intent',
        min=0,
        max=len(all_intent)-1,
        step=1,
        initial_value=0
    )

    grasp_slider = server.gui.add_slider(
        label='Grasp',
        min=0,
        max=199,
        step=1,
        initial_value=0
    )


    def slider_update_callback(_):
        robot_id = int(robot_slider.value)
        intent_id = int(intent_slider.value)
        object_id = int(object_slider.value)
        grasp_id = int(grasp_slider.value)
        on_update(robot_id, intent_id, object_id, grasp_id)

    robot_slider.on_update(slider_update_callback)
    intent_slider.on_update(slider_update_callback)
    object_slider.on_update(slider_update_callback)
    grasp_slider.on_update(slider_update_callback)
    

    print("GUI sliders initialized. Ready for interaction.")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()