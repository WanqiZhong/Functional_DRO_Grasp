import os
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_name = "teapot"
robot_name = "shadowhand"
# target_objects = ["oakink+teapot", "oakink+lotion_pump"]
target_objects = ["oakink+teapot"]

dataset_path = os.path.join(
    ROOT_DIR, f'data/OakInkDataset/oakink_dataset_standard_all_retarget_to_{robot_name}_valid_dro_contact_map.pt'
)
pcs_path = os.path.join(
    ROOT_DIR, 'data/OakInkDataset/oakink_object_pcs_with_normals.pt'
)

data = torch.load(dataset_path, map_location='cpu')
metadata = data['metadata']
pcs_all = torch.load(pcs_path, map_location='cpu')

# 筛选 metadata + 对应 object ids
all_filtered = []
filtered_object_ids = set()

for target in target_objects:
    filtered = [m for m in metadata if m[5] == target]
    all_filtered.extend(filtered)
    filtered_object_ids.update([m[7] for m in filtered])
    print(f"Found {len(filtered)} samples for object '{target}' in robot '{robot_name}'.")

if len(all_filtered) > 0:
    sample = all_filtered[0]
    q = sample[3]
    object_id = sample[7]
    scale = sample[8]
    sentence = sample[10]
    print(f"  target_q      : {q}")
    print(f"  object_id     : {object_id}")
    print(f"  scale_factor  : {scale}")
    print(f"  sentence      : {sentence}")

filtered_pcs = {obj_id: pcs_all[obj_id] for obj_id in filtered_object_ids if obj_id in pcs_all}
print(f"Collected {len(filtered_pcs)} object point clouds.")

save_metadata_path = os.path.join(
    ROOT_DIR,
    f"data/OakInkDataset/oakink_{save_name}_dataset_standard_all_retarget_to_{robot_name}_valid_dro_contact_map.pt"
)

data['metadata'] = all_filtered
torch.save(data, save_metadata_path)
print(f"Saved filtered dataset with metadata and point clouds to: {save_metadata_path}")

save_pcs_path = os.path.join(
    ROOT_DIR,
    f"data/OakInkDataset/oakink_{save_name}_object_pcs_with_normals.pt"
)
torch.save(filtered_pcs, save_pcs_path)