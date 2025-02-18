import os
import torch
import clip
from tqdm import tqdm

def preprocess_clip_embeddings_oakink(all_data, output_file, device='cuda'):

    unique_labels = set()
    unique_names = set()
    for record in tqdm(all_data):
        labels = record['label']
        unique_labels.update(labels)
        unique_names.add(record['name'])

    sorted_labels = sorted(unique_labels)

    print(f"Found {len(sorted_labels)} unique labels.")

    clip_model, _ = clip.load("ViT-B/32", device=device)
    text_inputs = clip.tokenize(sorted_labels).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs).cpu()

    label_to_clip = {label: feat for label, feat in zip(sorted_labels, text_features)}

    torch.save(label_to_clip, output_file)
    print(f"Saved CLIP embeddings for {len(sorted_labels)} labels to {output_file}")

    torch.save(unique_names, output_file.replace("_embeddings", "_names"))


def preprocess_clip_embeddings_shapenet(output_file, device='cuda'):

    cat2part = {'airplane': ['body','wing','tail','engine or frame'], 'bag': ['handle','body'], 'cap': ['panels or crown','visor or peak'], 
            'car': ['roof','hood','wheel or tire','body'],
            'chair': ['back','seat pad','leg','armrest'], 'earphone': ['earcup','headband','data wire'], 
            'guitar': ['head or tuners','neck','body'], 
            'knife': ['blade', 'handle'], 'lamp': ['leg or wire','lampshade'], 
            'laptop': ['keyboard','screen or monitor'], 
            'motorbike': ['gas tank','seat','wheel','handles or handlebars','light','engine or frame'], 'mug': ['handle', 'cup'], 
            'pistol': ['barrel', 'handle', 'trigger and guard'], 
            'rocket': ['body','fin','nose cone'], 'skateboard': ['wheel','deck','belt for foot'], 'table': ['desktop','leg or support','drawer']}
    
    # flatten the list of part names (use the category + part name as the key)
    all_parts = [f"{category} {part}" for category, parts in cat2part.items() for part in parts]
    print(all_parts)

    clip_model, _ = clip.load("ViT-B/32", device=device)
    text_inputs = clip.tokenize(all_parts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs).cpu()

    part_to_clip = {part: feat for part, feat in zip(all_parts, text_features)}
    torch.save({
        'part_to_clip': part_to_clip,
        'all_label': all_parts
    }, output_file)
    print(f"Saved CLIP embeddings for {len(all_parts)} parts to {output_file}")



if __name__ == "__main__":
    
    DIR_NAME = '/data/zwq/code/DRO_Grasp/data/OakInkDataset'
    # dataset_file = os.path.join(DIR_NAME, 'oakink_full_segmentation.pt')
    # all_data = torch.load(dataset_file)

    # output_file = os.path.join(DIR_NAME, 'label_clip_512_embeddings.pt')
    # preprocess_clip_embeddings_oakink(all_data, output_file)

    output_file = os.path.join(DIR_NAME, 'label_clip_512_embeddings_shapenet.pt')
    preprocess_clip_embeddings_shapenet(output_file)
