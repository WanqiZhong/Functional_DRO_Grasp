import os
import json
import torch
from typing import Dict, List
from tqdm import tqdm
import clip  # 确保已经安装了 CLIP：pip install git+https://github.com/openai/CLIP.git

INTENT_MAP = {
    '0001': 'use',
    '0002': 'hold',
    '0003': 'liftup',
    '0004': 'handover',
}

data_contactdb = {
    "train": [
        "contactdb+alarm_clock", "contactdb+banana", "contactdb+binoculars",
        "contactdb+cell_phone", "contactdb+cube_large", "contactdb+cube_medium",
        "contactdb+cube_small", "contactdb+cylinder_large", "contactdb+cylinder_small",
        "contactdb+elephant", "contactdb+flashlight", "contactdb+hammer",
        "contactdb+light_bulb", "contactdb+mouse", "contactdb+piggy_bank", "contactdb+ps_controller",
        "contactdb+pyramid_large", "contactdb+pyramid_medium", "contactdb+pyramid_small",
        "contactdb+stanford_bunny", "contactdb+stapler", "contactdb+toothpaste", "contactdb+torus_large",
        "contactdb+torus_medium", "contactdb+torus_small", "contactdb+train",
        "ycb+bleach_cleanser", "ycb+cracker_box", "ycb+foam_brick", "ycb+gelatin_box", "ycb+hammer",
        "ycb+lemon", "ycb+master_chef_can", "ycb+mini_soccer_ball", "ycb+mustard_bottle", "ycb+orange", "ycb+peach",
        "ycb+pitcher_base", "ycb+plum", "ycb+power_drill", "ycb+pudding_box",
        "ycb+rubiks_cube", "ycb+sponge", "ycb+strawberry", "ycb+sugar_box", "ycb+toy_airplane",
        "ycb+tuna_fish_can", "ycb+wood_block"
    ],
    "validate": [
        "contactdb+apple", "contactdb+camera", "contactdb+cylinder_medium", "contactdb+rubber_duck",
        "contactdb+door_knob",  "contactdb+water_bottle", "ycb+baseball", "ycb+pear", "ycb+potted_meat_can",
        "ycb+tomato_soup_can"
    ]
}

data_oakink = {
    "train": [
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
    ],
    "validate": [
        "oakink+apple",
        "oakink+bottle",
        "oakink+cameras",
        "oakink+knife",
        "oakink+headphones"
    ]
}

def process_objects(objects: List[str], intent_map: Dict[str, str], model, preprocess, device) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Processes a list of object keys to generate embeddings for each intent using CLIP.
    
    Args:
        objects (List[str]): List of object keys in the format 'category+object_name'.
        intent_map (Dict[str, str]): Mapping from intent codes to descriptions.
        model: CLIP model.
        preprocess: CLIP preprocess (unused here, kept for compatibility).
        device: Device to run the model on.
    
    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Nested dictionary with embeddings.
    """
    embedding_dict = {}
    
    for object_key in tqdm(objects, desc="Processing objects"):
        try:
            _, category = object_key.split('+', 1)
        except ValueError:
            print(f"Invalid object key format: {object_key}")
            continue
        
        category_clean = category.replace('_', ' ').strip()
        if not category_clean:
            print(f"Empty category after cleaning for object key: {object_key}")
            continue
        
        embedding_dict[object_key] = {}
        
        article = 'an' if category_clean[0].lower() in 'aeiou' else 'a'
        
        for _, intent_desc in intent_map.items():
            sentence = f"{intent_desc.capitalize()} {article} {category_clean}."
            
            text_inputs = clip.tokenize([sentence]).to(device) 
            
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            text_embedding = text_features[0].cpu()            
            embedding_dict[object_key][intent_desc] = text_embedding
    
    return embedding_dict

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-L/14@336px", device=device, download_root="/data/zwq/code/CLIP")
    model = model.float()
    model.eval() 
    
    combined_objects = data_contactdb["train"] + data_oakink["train"] + data_contactdb["validate"] + data_oakink["validate"]
    print(f"Total objects to process: {len(combined_objects)}")
    
    embeddings = process_objects(combined_objects, INTENT_MAP, model, preprocess, device)
    
    save_path = os.path.join("/data/zwq/code/DRO-Grasp/data/OakInkDataset", 'clip_object_intent_embeddings.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    torch.save(embeddings, save_path)
    print(f"Embeddings have been successfully saved to {save_path}")

if __name__ == "__main__":
    main()