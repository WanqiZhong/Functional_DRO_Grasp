import os
import sys
import time
from typing import List, Dict

import numpy as np
import torch
import open3d as o3d
import viser  

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

DATA_FILE = '/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_full_segmentation_avg_5000.pt'

server = viser.ViserServer(host='127.0.0.1', port=8080)
scene = server.scene

def create_color_map(parts_names: List[str]) -> Dict[str, List[float]]:
    np.random.seed(42)  
    color_map = {}
    for part in parts_names:
        color_map[part] = [float(c) for c in np.random.rand(3)]
    return color_map

def visualize_object(record: dict, scene, color_map: Dict[str, List[float]]):
    points = record['points']
    labels = record['label']
    
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    random_indices =  np.random.choice(len(points), size=512, replace=False)
    points = points[random_indices]
    labels = labels[random_indices]


    scene.reset()

    for lab in set(labels):
        color = color_map.get(lab, [0.5, 0.5, 0.5])
        indices = labels == lab
        point = points[indices]
        print("Label: ", lab, " Num: ", len(point))

        scene.add_point_cloud(
            name=lab,
            points=point,
            point_size=0.002,
            point_shape="circle",
            colors=color
        )

def main():

    all_results = torch.load(DATA_FILE)

    all_labels = set()
    for record in all_results:
        labels = record['label']
        for lab in labels:
            all_labels.add(lab)
    color_map = create_color_map(list(all_labels))

    index_slider = server.gui.add_slider(
        label='Index',
        min=0,
        max=len(all_results) - 1,
        step=1,
        initial_value=0
    )

    def update_visualization(_):
        idx = index_slider.value
        record = all_results[idx]
        visualize_object(record, scene, color_map)

    index_slider.on_update(update_visualization)
    print("GUI is ready...")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()