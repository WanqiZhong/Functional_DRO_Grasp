import torch
import os

FILTER_ROOT = "/data/zwq/code/BODex/src/curobo/content/assets/object/oakink_obj/processed_data"

def has_valid_simplified_json(object_key, object_id):
    if '+' not in object_key:
        return False
    prefix, object_name = object_key.split('+')  # e.g., 'oakink', 'bowl'
    folder_name = f"{object_name}_{object_id}"  # e.g., 'bowl_s12107'
    simplified_path = os.path.join(FILTER_ROOT, folder_name, "info", "simplified.json")
    return os.path.exists(simplified_path)

def filter_by_object_file(dataset, name=""):
    original_len = len(dataset['metadata'])
    filtered_metadata = []

    for i, entry in enumerate(dataset['metadata']):
        object_key = entry[5]  # e.g., 'oakink+bowl'
        object_id = entry[7]  # e.g., 's12107'

        if has_valid_simplified_json(object_key, object_id):
            filtered_metadata.append(entry)
        else:
            print(f"[{name}] Cannot find simplified.json for {object_key} with ID {object_id}. Entry {i} will be removed.")

    dataset['metadata'] = filtered_metadata
    print(f"[{name}] Filtered {original_len} entries to {len(filtered_metadata)} entries.")
    return dataset

shadow_path = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand.pt'
leap_path = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_leaphand.pt'
allegro_path = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_allegro.pt'

shadow = torch.load(shadow_path)
leap = torch.load(leap_path)
allegro = torch.load(allegro_path)

shadow = filter_by_object_file(shadow, "ShadowHand")
leap = filter_by_object_file(leap, "LeapHand")
allegro = filter_by_object_file(allegro, "AllegroHand")

torch.save(shadow, shadow_path)
torch.save(leap, leap_path)
torch.save(allegro, allegro_path)
