import argparse
import os
from typing import List, Tuple, Dict

import viser
import numpy as np
import json
import torch
import open3d as o3d
from oikit.oak_base import OakBase
from oikit.oak_base import ObjectAffordanceKnowledge as OAK
import time
from scipy.spatial import KDTree

# 配置路径
OAKINK_DIR = '/data/zwq/data/oakink'  
TARGET_DIR = '/data/zwq/code/DRO-Grasp/data/OakInkDataset'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Meta 文件路径
REAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'object_id.json')
VIRTUAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'virtual_object_id.json')

# 初始化 Viser 服务器
server = viser.ViserServer(host='127.0.0.1', port=8080)
scene = server.scene

def find_closest_part_conditional(tip: np.ndarray, parts_pcs: List[np.ndarray], parts_names: List[str], threshold: float = 0.05) -> Tuple[str, np.ndarray]:
    """
    条件性地找到与给定手指tip最近的部分名称及其最近点坐标。
    如果与某部分的最小距离小于阈值，则使用最近点分配；
    否则，使用基于质心的分配。

    Args:
        tip (np.ndarray): 手指tip的坐标。
        parts_pcs (List[np.ndarray]): 对象各部分的点云数组列表。
        parts_names (List[str]): 对象各部分的名称列表。
        threshold (float): 距离阈值，用于决定使用哪种分配方法。

    Returns:
        Tuple[str, np.ndarray]: 最近的部分名称及最近点的坐标。
    """
    min_dist = float('inf')
    closest_part_candidate = None
    closest_point_candidate = None

    # 第一阶段：查找所有部分中最小的点距离
    for part_pc, part_name in zip(parts_pcs, parts_names):
        if part_pc.size == 0:
            continue
        # 使用KD-Tree加速最近点搜索
        tree = KDTree(part_pc)
        distance, idx = tree.query(tip)
        if distance < min_dist:
            min_dist = distance
            closest_part_candidate = part_name
            closest_point_candidate = part_pc[idx]

    if min_dist < threshold:
        # 如果最小距离小于阈值，返回最近点分配结果
        return closest_part_candidate, closest_point_candidate
    else:
        # 第二阶段：基于质心的分配
        centroids = [pc.mean(axis=0) for pc in parts_pcs if pc.size > 0]
        valid_parts = [part for part, pc in zip(parts_names, parts_pcs) if pc.size > 0]

        if not centroids:
            return None, None

        centroids = np.array(centroids)
        distances_to_centroids = np.linalg.norm(centroids - tip, axis=1)
        min_centroid_idx = distances_to_centroids.argmin()
        assigned_part = valid_parts[min_centroid_idx]
        assigned_pc = parts_pcs[min_centroid_idx]

        # 在分配的部分中找到最近点
        tree = KDTree(assigned_pc)
        distance, idx = tree.query(tip)
        closest_point = assigned_pc[idx] if distance < float('inf') else None

        return assigned_part, closest_point

def process_meta(meta: Tuple, obj_metadata: Dict, obj_nameid_metadata: dict, all_cates: List[str]) -> dict:
    """
    处理单个元数据项，找到每个手指tip最近的对象部分名称及其最近点坐标，基于部分质心。
    
    Args:
        meta (Tuple): 元数据项。
        obj_metadata (Dict): 对象元数据字典。
        obj_nameid_metadata (dict): 对象名称ID元数据字典。
        all_cates (List[str]): 所有类别列表。
    
    Returns:
        dict: 处理结果，包括对象ID、对象名称、部分名称以及每个tip对应的最近部分名称和最近点坐标。
    """
    hand_pose, hand_shape, tsl, target_q, intent, object_name, robot_name, object_id, real_object_id, hand_verts = meta

    tip_ids = [745, 317, 444, 556, 673]
    tip_verts = hand_verts[tip_ids]

    object_nameid = obj_metadata.get(object_id, {}).get("name", "")
    obj_meta: OAK = obj_nameid_metadata.get(object_nameid, None)
    if obj_meta is None:
        closest_parts = {tip_id: object_name.split('+')[1] for tip_id in tip_ids}
        return {
            "object_id": object_id, 
            "object_name": object_name,
            "object_parts": object_name, 
            "closest_parts": closest_parts, 
            "closest_points": {tip_id: None for tip_id in tip_ids},
            "hand_tips": tip_verts,
            "hands": hand_verts
        }

    parts_names = obj_meta.part_names
    parts_pcs = []
    valid_parts = []
    for part in parts_names:
        part_seg = obj_meta.part_name_to_segs.get(part, None)
        if part_seg is None or not os.path.isfile(part_seg):
            continue
        # 读取点云并转换为 numpy 数组
        pcd = o3d.io.read_point_cloud(part_seg)
        if not pcd.has_points():
            continue
        part_pc = np.asarray(pcd.points)
        parts_pcs.append(part_pc)
        valid_parts.append(part)

    if not parts_pcs:
        return {"object_id": object_id, "error": "No valid part point clouds found"}

    closest_parts = {}
    closest_points = {}
    for i, tip in enumerate(tip_verts):
        closest_part, closest_point = find_closest_part_conditional(tip, parts_pcs, valid_parts)
        closest_parts[tip_ids[i]] = closest_part
        closest_points[tip_ids[i]] = closest_point.tolist() if closest_point is not None else None

    return {
        "object_id": object_id, 
        "object_name": object_name,
        "object_parts": valid_parts, 
        "closest_parts": closest_parts,
        "closest_points": closest_points,
        "hand_tips": tip_verts,
        "hands": hand_verts
    }

def create_color_map(parts_names: List[str]) -> Dict[str, List[float]]:
    """
    为每个部分分配一个随机颜色。
    
    Args:
        parts_names (List[str]): 对象部分名称列表。
    
    Returns:
        Dict[str, List[float]]: 部分名称与颜色的映射。
    """
    np.random.seed(42)  # For reproducibility
    color_map = {}
    for part in parts_names:
        color_map[part] = [float(c) for c in np.random.rand(3)]
    return color_map

def visualize_result(result: dict, color_map: Dict[str, List[float]], obj_meta: OAK):
    """
    使用 Viser 可视化处理结果，包括对象部分、手指tip和最近点。
    
    Args:
        result (dict): 处理结果字典。
        color_map (Dict[str, List[float]]): 部分名称与颜色的映射。
        obj_meta (OAK): 对象的 OAK 元数据。
    """
    object_name = result["object_name"]
    parts_names = result.get("object_parts", [])
    hand_tips = np.array(result.get("hand_tips", []))
    closest_points = result.get("closest_points", {})

    scene.reset()

    # 可视化每个部分
    for part, color in color_map.items():
        part_id = f"part_{part}"
        part_seg = obj_meta.part_name_to_segs.get(part, None)
        if part_seg is None or not os.path.isfile(part_seg):
            continue
        pcd = o3d.io.read_point_cloud(part_seg)
        if not pcd.has_points():
            continue
        part_pc = np.asarray(pcd.points)
        scene.add_point_cloud(
            name=part_id,
            points=part_pc,
            point_size=0.002,
            point_shape="circle",
            colors=color,  # Assign color per point
        )

    # 可视化手指tip（红色）
    if hand_tips.size > 0:
        scene.add_point_cloud(
            name="hand_tips",
            points=hand_tips,
            point_size=0.004,
            point_shape="circle",
            colors=[255, 0, 0],  # Red color per tip
        )

    scene.add_point_cloud(
        name="hands",
        points=result["hands"],
        colors=[192, 102, 255],
        point_size=0.002,
        point_shape="circle",
    )

    # 可视化最近点（黄色）
    closest_pts = [point for point in closest_points.values() if point is not None]
    if closest_pts:
        closest_pts = np.array(closest_pts)
        scene.add_point_cloud(
            name="closest_points",
            points=closest_pts,
            point_size=0.004,
            point_shape="circle",
            colors=[[255, 255, 0]] * len(closest_pts),  # Yellow color per point
        )

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

    # 使用 OakBase OAK 对象创建一个字典
    for cate in all_cates:
        for obj in oakbase.get_objs_by_category(cate):
            obj_nameid_metadata[obj.obj_id] = obj

    def update(index):

        if index < 0 or index >= len(oakink_metadata):
            print(f"Invalid index {index}. Please provide an index between 0 and {len(oakink_metadata)-1}.")
            return

        meta = oakink_metadata[index]
        result = process_meta(meta, obj_metadata, obj_nameid_metadata, all_cates)

        if "error" in result:
            print(f"Error processing object_id {result['object_id']}: {result['error']}")
            return

        # 可视化
        object_id = result["object_id"]

        object_nameid = obj_metadata.get(object_id, {}).get("name", "")
        obj_meta: OAK = obj_nameid_metadata.get(object_nameid, None)
        if obj_meta is None:
            print(f"Object metadata not found for object_id: {object_id}")
            return

        # 为每个部分分配不同的颜色
        color_map = create_color_map(obj_meta.part_names)

        # 可视化结果
        visualize_result(result, color_map, obj_meta)

    # Add a slider to control the index
    index_slider = server.gui.add_slider(
        label='Index',
        min=0,
        max=len(oakink_metadata)-1,
        step=1,
        initial_value=0
    )

    def update_visualization(_):
        update(index_slider.value)  

    index_slider.on_update(update_visualization)
    print("GUI is ready...")

    while True:
        time.sleep(1)

    # # 保存结果
    # output_file = os.path.join(TARGET_DIR, f'closest_parts_result_{index}.json')
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(result, f, indent=4, ensure_ascii=False)

    # print(f"Process finished for index {index}, results saved to {output_file}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Find closest object parts to hand tips and visualize using Viser.")
#     args = parser.parse_args()
#     main(args)