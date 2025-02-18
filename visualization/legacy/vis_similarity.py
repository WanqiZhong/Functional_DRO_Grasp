from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 数据集 A 和 B
a_objects = [
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
]

b_objects = [
    "oakink+gamecontroller", "oakink+toothbrush", "oakink+wineglass", "oakink+cup",
    "oakink+mouse", "oakink+binoculars", "oakink+lightbulb", "oakink+lotion_pump",
    "oakink+squeezable", "oakink+hammer", "oakink+pen", "oakink+pincer", "oakink+mug",
    "oakink+screwdriver", "oakink+banana", "oakink+stapler", "oakink+fryingpan", "oakink+bowl",
    "oakink+phone", "oakink+scissors", "oakink+flashlight", "oakink+eyeglasses", "oakink+teapot",
    "oakink+power_drill", "oakink+wrench", "oakink+trigger_sprayer", "oakink+donut", "oakink+cylinder_bottle"
]

validate_object_names = [obj.split("+")[1] for obj in a_objects]
train_object_names = [obj.split("+")[1] for obj in b_objects]

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

def get_similarity_matrix(train_object_names, validate_object_names, top_k=3):
    all_object_names = train_object_names + validate_object_names
    inputs = tokenizer(all_object_names, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    train_embeddings = embeddings[:len(train_object_names)]
    validate_embeddings = embeddings[len(train_object_names):]
    similarity_matrix = F.cosine_similarity(
        validate_embeddings.unsqueeze(1), train_embeddings.unsqueeze(0), dim=2
    )
    topk_values, topk_indices = torch.topk(similarity_matrix, top_k, dim=1, largest=True, sorted=True)
    topk_similar_names = {
        validate_object_names[i]: [
            (train_object_names[idx], round(value.item(), 4))  
            for value, idx in zip(topk_values[i], indices)
        ]
        for i, indices in enumerate(topk_indices)
    }
    
    return topk_similar_names

result = get_similarity_matrix(train_object_names, validate_object_names)

cnt = 0
for key, value in result.items():
    # filter value < 0.5
    value = [(k, v) for k, v in value if v > 0.5]
    if value == []:
        pass
        # print(f"{key}: No similar objects found.")
    else:
        print(f"{key}: {value}")
        cnt += 1

print(f"Total {cnt}/{len(result)} objects have similar objects found.")