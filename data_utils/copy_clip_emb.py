import torch

# 路径
shadow_path = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand.pt'
leap_path = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_leaphand.pt'
allegro_path = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_allegro.pt'

# 加载
shadow = torch.load(shadow_path)
leap = torch.load(leap_path)
allegro = torch.load(allegro_path)

new_shadow_metadata = []
for i, entry in enumerate(shadow['metadata']):
    modified = list(entry)
    modified[8] = 1.0
    new_shadow_metadata.append(tuple(modified))
shadow['metadata'] = new_shadow_metadata

def get_extra_fields(entry):
    return entry[10:] 

def align_metadata(base, target, name=''):
    new_target_metadata = []
    for i, (base_item, tgt_item) in enumerate(zip(base['metadata'], target['metadata'])):
        for j in range(9):
            if j in [3, 6, 8]:
                continue
            b, t = base_item[j], tgt_item[j]
            if isinstance(b, torch.Tensor):
                if not torch.equal(b, t):
                    raise ValueError(f"[{name}] Entry {i}, Field {j} mismatch in tensor value.")
            else:
                if b != t:
                    raise ValueError(f"[{name}] Entry {i}, Field {j} mismatch: {b} != {t}")

        new_entry = tgt_item[:10] + base_item[10:]
        new_target_metadata.append(new_entry)

    target['metadata'] = new_target_metadata
    print(f"[{name}]  {len(new_target_metadata)} entries aligned.")
    return target

leap = align_metadata(shadow, leap, 'LeapHand')
allegro = align_metadata(shadow, allegro, 'AllegroHand')

torch.save(shadow, shadow_path)
torch.save(leap, leap_path)
torch.save(allegro, allegro_path)

