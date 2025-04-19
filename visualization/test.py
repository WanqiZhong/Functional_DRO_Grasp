import numpy as np
import torch
n1 = np.load("/data/zwq/code/DexGraspBench/output/oakink_teapot_step020_max_leap_bodex/succgrasp/teapot/o13104/1.npy", allow_pickle=True).item()
t1 = torch.load("/data/zwq/code/DRO_Grasp/data/OakInkDataset/teapot_oakink_dataset_standard_all_retarget_to_allegro_bodex_valid_bodex_version.pt")

print(n1['robot_pose'][0][0])
print(t1['metadata'][0][-1])

print(1+1)
