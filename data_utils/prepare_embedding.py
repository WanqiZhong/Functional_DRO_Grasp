import json
import torch
from openai import OpenAI
from typing import Dict, List
from tqdm import tqdm

# Initialize OpenAI API client
client = OpenAI(api_key="your-key")

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

def get_embedding(text, client, model="text-embedding-3-small", dimensions=64):
    try:
        response = client.embeddings.create(
            input=text,
            model=model,
            dimensions=dimensions
        )
        return response.data[0]
    except Exception as e:
        print(f"Get embedding error: {e}")
        return None

def process_objects(objects: List[str], intent_map: Dict[str, str]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Processes a list of object keys to generate embeddings for each intent.
    
    Args:
        objects (List[str]): List of object keys in the format 'category+object_name'.
        intent_map (Dict[str, str]): Mapping from intent codes to descriptions.
    
    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Nested dictionary with embeddings.
    """
    embedding_dict = {}
    
    for object_key in tqdm(objects):
        try:
            _, category = object_key.split('+', 1)
        except ValueError:
            print(f"Invalid object key format: {object_key}")
            continue
        
        embedding_dict[object_key] = {}
        
        article = 'an' if category[0].lower() in 'aeiou' else 'a'
        category = category.replace('_', ' ')
        
        for _, intent_desc in intent_map.items():
            sentence = f"{intent_desc.capitalize()} {article} {category}."
            print(sentence)
            embedding = get_embedding(sentence, client=client, dimensions=256)
            
            if embedding:
                embedding_tensor = torch.tensor(embedding.embedding, dtype=torch.float32)
                embedding_dict[object_key][intent_desc] = embedding_tensor
            else:
                print(f"Skipping embedding for {category} with intent {intent_desc} due to retrieval error.")
    
    return embedding_dict

def main():

    combined_objects = data_contactdb["train"] + data_oakink["train"] + data_contactdb["validate"] + data_oakink["validate"]        
    embeddings = process_objects(combined_objects, INTENT_MAP)

    save_path = "/data/zwq/code/DRO-Grasp/data/OakInkDataset/object_intent_embeddings_256.pt"
    torch.save(embeddings, save_path)
    print(f"Embeddings have been successfully saved to {save_path}")

if __name__ == "__main__":
    main()