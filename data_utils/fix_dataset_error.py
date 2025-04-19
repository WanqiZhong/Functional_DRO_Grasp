import torch
from tqdm import tqdm

filename = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_leaphand_valid_dro.pt'
data = torch.load(filename)
metadata = data['metadata']
fixed_metadata = []

for entry in tqdm(metadata):
    entry = list(entry)  

    if isinstance(entry[3], torch.Tensor):
        entry[3] = entry[3].float()

    entry[8] = float(entry[8])

    fixed_metadata.append(tuple(entry))  

data['metadata'] = fixed_metadata
torch.save(data, filename)
print(f"Fixed {len(fixed_metadata)} entries and saved to {filename}")