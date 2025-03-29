import os
import torch
import argparse
from tqdm import tqdm

def filter_oakink_dataset(dataset_path, assets_base_path):
    """
    过滤OakInk数据集，移除没有必要文件的条目
    
    Args:
        dataset_path: OakInk数据集的路径
        assets_base_path: 资源文件的基础路径
    
    Returns:
        filtered_dataset: 过滤后的数据集
        filtered_indices: 保留的条目索引
        removed_indices: 移除的条目索引
    """
    print(f"加载数据集: {dataset_path}")
    dataset = torch.load(dataset_path)
    metadata = dataset['metadata']
    
    filtered_metadata = []
    filtered_indices = []
    removed_indices = []
    removed_reasons = {}
    
    # 统计信息
    total_items = len(metadata)
    object_counts = {}  # 每种对象类型的数量
    missing_file_counts = {"mesh": 0, "info": 0, "urdf": 0}
    
    print(f"检查{total_items}条数据的文件存在性...")
    for idx, item in enumerate(tqdm(metadata)):
        # 提取对象信息
        object_key = item[5]  # 例如 'oakink+apple'
        object_id = item[7]   # 例如 'C90001'
        
        # 获取对象代码
        if '+' in object_key:
            object_type = object_key.split('+')[1]
        else:
            object_type = object_key
        
        object_code = f"{object_type}_{object_id}"
        
        # 统计对象类型
        if object_type not in object_counts:
            object_counts[object_type] = 0
        object_counts[object_type] += 1
        
        # 检查文件是否存在
        mesh_path = os.path.join(assets_base_path, f"object/oakink_obj/processed_data/{object_code}/mesh/simplified.obj")
        info_path = os.path.join(assets_base_path, f"object/oakink_obj/processed_data/{object_code}/info/simplified.json")
        urdf_path = os.path.join(assets_base_path, f"object/oakink_obj/processed_data/{object_code}/urdf/coacd.urdf")
        
        # 检查每个文件
        missing_files = []
        if not os.path.exists(mesh_path):
            missing_files.append("mesh")
            missing_file_counts["mesh"] += 1
        
        if not os.path.exists(info_path):
            missing_files.append("info")
            missing_file_counts["info"] += 1
        
        if not os.path.exists(urdf_path):
            missing_files.append("urdf")
            missing_file_counts["urdf"] += 1
        
        # 如果所有文件都存在，则保留该条目
        if not missing_files:
            filtered_metadata.append(item)
            filtered_indices.append(idx)
        else:
            removed_indices.append(idx)
            removed_reasons[idx] = {
                "object_code": object_code,
                "missing_files": missing_files
            }
    
    # 创建过滤后的数据集
    filtered_dataset = dataset.copy()
    filtered_dataset['metadata'] = filtered_metadata
    
    # 打印统计信息
    print(f"\n过滤完成!")
    print(f"原始数据集: {total_items}条")
    print(f"过滤后数据集: {len(filtered_metadata)}条 ({len(filtered_metadata)/total_items*100:.2f}%)")
    print(f"移除的条目: {len(removed_indices)}条 ({len(removed_indices)/total_items*100:.2f}%)")
    
    print("\n各对象类型数量:")
    for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {obj_type}: {count}条")
    
    print("\n缺失文件情况:")
    for file_type, count in missing_file_counts.items():
        print(f"  缺失{file_type}文件: {count}条")
    
    # 输出前10个移除的条目信息
    if removed_indices:
        print("\n前10个移除的条目:")
        for i, idx in enumerate(removed_indices[:10]):
            reason = removed_reasons[idx]
            print(f"  {i+1}. 索引 {idx}: {reason['object_code']} - 缺失文件: {', '.join(reason['missing_files'])}")
    
    return filtered_dataset, filtered_indices, removed_indices, removed_reasons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="过滤OakInk数据集，移除没有必要文件的条目")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/zwq/code/DRO_Grasp/data/OakInkDataset/oakink_dataset_standard_all_retarget_to_shadowhand_mujoco.pt",
        help="OakInk数据集的路径"
    )
    
    parser.add_argument(
        "--assets_path",
        type=str,
        default="/data/zwq/code/BODex/src/curobo/content/assets/",
        help="资源文件的基础路径"
    )
    
    parser.add_argument(
        "--save_filtered_missing",
        action="store_true",
        help="是否保存缺失文件的详细信息"
    )
    
    args = parser.parse_args()
    
    # 过滤数据集
    filtered_dataset, filtered_indices, removed_indices, removed_reasons = filter_oakink_dataset(
        args.dataset_path, 
        args.assets_path
    )
    
    # 创建输出路径
    dataset_dir = os.path.dirname(args.dataset_path)
    dataset_filename = os.path.basename(args.dataset_path)
    dataset_name, dataset_ext = os.path.splitext(dataset_filename)
    
    output_path = os.path.join(dataset_dir, f"{dataset_name}_coacd{dataset_ext}")
    
    # 保存过滤后的数据集
    print(f"\n保存过滤后的数据集到: {output_path}")
    torch.save(filtered_dataset, output_path)
    
    # 可选: 保存移除条目的详细信息
    if args.save_filtered_missing and removed_indices:
        missing_info_path = os.path.join(dataset_dir, f"{dataset_name}_missing_files.pt")
        missing_info = {
            "removed_indices": removed_indices,
            "removed_reasons": removed_reasons
        }
        torch.save(missing_info, missing_info_path)
        print(f"保存缺失文件详细信息到: {missing_info_path}")
    
    print("\n处理完成!")