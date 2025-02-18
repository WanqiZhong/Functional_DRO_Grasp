import argparse
import os
import sys
from typing import List
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.func_utils import farthest_point_sampling
import numpy as np
import json
import torch
import open3d as o3d
from oikit.oak_base import OakBase
from oikit.oak_base import ObjectAffordanceKnowledge as OAK
from tqdm import tqdm


OAKINK_DIR = '/data/zwq/data/oakink'
TARGET_DIR = '/data/zwq/code/DRO_Grasp/data/OakInkDataset'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'object_id.json')
VIRTUAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'virtual_object_id.json')

def get_segmentation_for_object(object_id, object_name, obj_metadata, obj_nameid_metadata, max_points=20000, use_fps=False):
    """
        Use farthest point sampling to limit the number of points of the object.
    """
    object_nameid = obj_metadata.get(object_id, {}).get("name", "")
    object_name = object_name.split("+")[1]
    obj_meta: OAK = obj_nameid_metadata.get(object_nameid, None)
    if obj_meta is None:
        return None, None

    points_list = []
    labels_list = []
    for part in obj_meta.part_names:
        part_seg = obj_meta.part_name_to_segs.get(part, None)
        if part_seg is None or not os.path.isfile(part_seg):
            continue
        pcd = o3d.io.read_point_cloud(part_seg)
        if not pcd.has_points():
            continue
        part_points = np.asarray(pcd.points)
        points_list.append(part_points)
        labels_list.extend([part.replace("_", " ")] * len(part_points))

    if not points_list:
        print(f"Object {object_name} has no valid parts.")
        return None, None

    points = np.concatenate(points_list, axis=0)
    labels = np.array(labels_list)

    assert len(points) == len(labels), f"Length mismatch: {len(points)} != {len(labels)}"
    
    if len(points) > max_points:
        points_tensor = torch.from_numpy(points).float()  # shape: [N, 3]
        if use_fps:
            s_p, indices = farthest_point_sampling(points_tensor, max_points)  # shape: [max_points, 3], [max_points]
            indices_np = np.array(indices)
        else:
            indices_np = np.random.choice(len(points), size=max_points, replace=False)

        points = points[indices_np]
        labels = labels[indices_np]

    return points, labels


def get_segmentation_for_object_avg(object_id, object_name, obj_metadata, obj_nameid_metadata, max_points=20000, use_fps=False):
    """
        Use farthest point sampling to limit the number of points in each part of the object,
        and each part has a similar number of points, and the final concatenated total size does not exceed max_points.
    """
    object_nameid = obj_metadata.get(object_id, {}).get("name", "")
    object_name = object_name.split("+")[1]
    obj_meta: OAK = obj_nameid_metadata.get(object_nameid, None)
    if obj_meta is None:
        return None, None

    num_parts = len(obj_meta.part_names)
    if num_parts == 0:
        return None, None
    max_points_per_part = max_points // num_parts

    sampled_points_list = []
    sampled_labels_list = []

    for part in obj_meta.part_names:
        part_seg = obj_meta.part_name_to_segs.get(part, None)
        if part_seg is None or not os.path.isfile(part_seg):
            continue

        pcd = o3d.io.read_point_cloud(part_seg)
        if not pcd.has_points():
            continue

        part_points = np.asarray(pcd.points)
        part_labels = np.array([part.replace("_", " ")] * len(part_points))

        if len(part_points) > max_points_per_part:
            points_tensor = torch.from_numpy(part_points).float()  # shape: [N, 3]
            if use_fps:
                s_p, indices = farthest_point_sampling(points_tensor, max_points_per_part)  # shape: [max_points, 3], [max_points]
                indices_np = np.array(indices)
            else:
                indices_np = np.random.choice(len(part_points), size=max_points_per_part, replace=False)

            part_points = part_points[indices_np]
            part_labels = part_labels[indices_np]

        sampled_points_list.append(part_points)
        sampled_labels_list.append(part_labels)

    if not sampled_points_list:
        print(f"Object {object_name} has no valid parts after sampling.")
        return None, None

    points = np.concatenate(sampled_points_list, axis=0)
    labels = np.concatenate(sampled_labels_list, axis=0)

    assert len(points) == len(labels), f"Length mismatch: {len(points)} != {len(labels)}"

    if len(points) > max_points:
        indices = np.random.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        labels = labels[indices]

    return points, labels


# def get_segmentation_for_object(object_id, object_name, obj_metadata, obj_nameid_metadata):
#     """
#     对给定的 object_id，从元数据中读取其各部分点云，并生成完整的点云和对应的标签数组。
#     """
#     object_nameid = obj_metadata.get(object_id, {}).get("name", "")
#     object_name = object_name.split("+")[1]
#     obj_meta: OAK = obj_nameid_metadata.get(object_nameid, None)
#     if obj_meta is None:
#         return None, None

#     points_list = []
#     labels_list = []
#     for part in obj_meta.part_names:
#         part_seg = obj_meta.part_name_to_segs.get(part, None)
#         if part_seg is None or not os.path.isfile(part_seg):
#             continue
#         pcd = o3d.io.read_point_cloud(part_seg)
#         if not pcd.has_points():
#             continue
#         part_points = np.asarray(pcd.points)
#         points_list.append(part_points)
#         labels_list.extend([part.replace("_", " ")] * len(part_points))

#     if not points_list:
#         print(f"Object {object_name} has no valid parts.")
#         return None, None

#     points = np.concatenate(points_list, axis=0)
#     labels = np.array(labels_list)

#     assert len(points) == len(labels), f"Length mismatch: {len(points)} != {len(labels)}"
    
#     return points, labels

def main(args):
    # 读取 Meta 数据
    with open(REAL_META_FILE, 'r', encoding='utf-8') as f:
        real_meta_file = json.load(f)
    with open(VIRTUAL_META_FILE, 'r', encoding='utf-8') as f:
        virtual_meta_file = json.load(f)

    oakink_dataset_path = os.path.join(ROOT_DIR, 'data/OakInkDataset/oakink_dataset_standard_all.pt')
    oakink_data = torch.load(oakink_dataset_path)
    oakink_metadata = oakink_data.get('metadata', [])

    # 合并 real 和 virtual meta 数据
    obj_metadata = {**real_meta_file, **virtual_meta_file}

    oakbase = OakBase()
    all_cates: List[str] = list(oakbase.categories.keys())
    obj_nameid_metadata = {}
    for cate in all_cates:
        for obj in oakbase.get_objs_by_category(cate):
            obj_nameid_metadata[obj.obj_id] = obj

    all_results = []
    added_ids = set()
    all_lables = set()

    for idx, meta in enumerate(tqdm(oakink_metadata)):
        object_id = meta[7]    # object_id 在 meta 元组的第 8 项（索引 7）
        object_name = meta[5]  # object_name 在 meta 元组的第 6 项（索引 5）

        if object_id in added_ids:
            continue

        points, labels = get_segmentation_for_object_avg(object_id, object_name, obj_metadata, obj_nameid_metadata, max_points=5000, use_fps=False)
        if points is None or labels is None:
            print(f"Skipping object_name: {object_name}, segmentation failed.")
            continue

        result_dict = {
            'id': object_id,
            'name': object_name,
            'points': torch.from_numpy(points),  # 转换为 PyTorch tensor
            'label': labels                      
        }
        all_results.append(result_dict)
        added_ids.add(object_id)

        all_lables.update(np.unique(labels))

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(oakink_metadata)} objects.")

    os.makedirs(TARGET_DIR, exist_ok=True)
    output_file = os.path.join(TARGET_DIR, 'oakink_full_segmentation_avg_5000.pt')
    all_lables = sorted(all_lables)
    torch.save({
        "data": all_results,
        "label": all_lables
    }, output_file)
    print(f"Saved segmentation results for all objects to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple objects and save segmented point clouds with labels as a .pt file.")
    args = parser.parse_args()
    main(args)