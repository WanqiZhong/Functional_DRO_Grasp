import os
import sys
import argparse
import time
import logging
import traceback
import signal
import torch
import trimesh
import open3d as o3d
import numpy as np
from tqdm import tqdm

# 自定义超时异常
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    """信号处理器，当闹钟信号触发时抛出 TimeoutException"""
    raise TimeoutException("Operation timed out")

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_format = '%(asctime)s %(levelname)s: %(message)s'
    
    # 超时日志
    timeout_logger = logging.getLogger('timeout_logger')
    timeout_logger.setLevel(logging.INFO)
    timeout_handler = logging.FileHandler(os.path.join(log_dir, 'timeout.log'))
    timeout_handler.setFormatter(logging.Formatter(log_format))
    timeout_logger.addHandler(timeout_handler)
    
    # 错误日志
    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(os.path.join(log_dir, 'error.log'))
    error_handler.setFormatter(logging.Formatter(log_format))
    error_logger.addHandler(error_handler)
    
    # 无效 Mesh 日志
    invalid_mesh_logger = logging.getLogger('invalid_mesh_logger')
    invalid_mesh_logger.setLevel(logging.WARNING)
    invalid_mesh_handler = logging.FileHandler(os.path.join(log_dir, 'invalid_mesh.log'))
    invalid_mesh_handler.setFormatter(logging.Formatter(log_format))
    invalid_mesh_logger.addHandler(invalid_mesh_handler)
    
    return timeout_logger, error_logger, invalid_mesh_logger

def sample_with_open3d(obj_path, num_points):
    """
    使用 Open3D 读取 mesh 并进行采样，返回点坐标和法向量。
    """
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if mesh.has_triangles():
        # 如果有三角形，进行均匀采样
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        return points, normals, None  # Trimesh的face_indices暂时为None
    elif mesh.has_vertices():
        # 如果没有三角形，但有顶点，直接使用顶点
        points = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else np.zeros_like(points)
        
        if len(points) == 0:
            raise ValueError("Mesh has no vertices.")

        if len(points) >= num_points:
            # 随机选择 num_points 个顶点
            indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[indices]
            sampled_normals = normals[indices]
        else:
            # 如果顶点不足，通过重复采样补齐
            repeat_factor = num_points // len(points) + 1
            sampled_points = np.tile(points, (repeat_factor, 1))[:num_points]
            sampled_normals = np.tile(normals, (repeat_factor, 1))[:num_points]
        
        return sampled_points, sampled_normals, None  # Trimesh的face_indices暂时为None
    else:
        raise ValueError("Mesh has neither triangles nor vertices.")

def sample_with_trimesh(obj_path, num_points):

    mesh = trimesh.load(obj_path, force='mesh')
    
    if mesh.is_empty:
        raise ValueError("Mesh is empty.")
    
    if mesh.faces.shape[0] > 0:
        sampled_points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        normals = mesh.face_normals[face_indices]
        return sampled_points, normals, face_indices
    elif mesh.vertices.shape[0] > 0:
        points = mesh.vertices
        normals = mesh.vertex_normals if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0 else np.zeros_like(points)
        
        if len(points) == 0:
            raise ValueError("Mesh has no vertices.")

        if len(points) >= num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[indices]
            sampled_normals = normals[indices] if normals.size > 0 else np.zeros_like(sampled_points)
            face_indices = None  # 没有面索引
        else:
            repeat_factor = num_points // len(points) + 1
            sampled_points = np.tile(points, (repeat_factor, 1))[:num_points]
            sampled_normals = np.tile(normals, (repeat_factor, 1))[:num_points] if normals.size > 0 else np.zeros_like(sampled_points)
            face_indices = None  # 没有面索引
        
        return sampled_points, sampled_normals, face_indices
    else:
        raise ValueError("Mesh has neither faces nor vertices.")

def process_with_open3d(obj_path, num_points, timeout):

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        sampled_points, sampled_normals, _ = sample_with_open3d(obj_path, num_points)
        signal.alarm(0)  # 取消闹钟
        return sampled_points, sampled_normals, None
    except TimeoutException:
        raise TimeoutException(f"Open3D processing timed out after {timeout} seconds.")
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)  # 确保闹钟被取消

def process_with_trimesh(obj_path, num_points, timeout):

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        sampled_points, sampled_normals, face_indices = sample_with_trimesh(obj_path, num_points)
        signal.alarm(0)  # 取消闹钟
        return sampled_points, sampled_normals, face_indices
    except TimeoutException:
        raise TimeoutException(f"Trimesh processing timed out after {timeout} seconds.")
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)  # 确保闹钟被取消

def process_obj(obj_path, output_path, num_points, timeout, error_logger, timeout_logger, invalid_mesh_logger):
    """
    尝试使用 Open3D 和 Trimesh 处理 .obj 文件，并保存采样结果。
    """
    # 尝试使用 Open3D 处理
    try:
        sampled_points, sampled_normals, _ = process_with_open3d(obj_path, num_points, timeout)
        # 保存点坐标和法向量
        object_pc_normals = torch.cat([
            torch.from_numpy(sampled_points).float(),
            torch.from_numpy(sampled_normals).float()
        ], dim=-1)  # (num_points, 6)
        torch.save(object_pc_normals, output_path)  # Save as .pt
        return
    except TimeoutException as te:
        timeout_logger.info(f"{os.path.relpath(obj_path)} - Open3D Timeout occurred.")
    except ValueError as ve:
        invalid_mesh_logger.warning(f"{os.path.relpath(obj_path)} - Open3D Invalid mesh: {str(ve)}")
    except Exception as e:
        error_logger.error(f"{os.path.relpath(obj_path)} - Open3D Error: {str(e)}")
    
    try:
        sampled_points, sampled_normals, _ = process_with_trimesh(obj_path, num_points, timeout)
        object_pc_normals = torch.cat([
            torch.from_numpy(sampled_points).float(),
            torch.from_numpy(sampled_normals).float()
        ], dim=-1)  # (num_points, 6)
        torch.save(object_pc_normals, output_path)  # Save as .pt
        return
    except TimeoutException as te:
        timeout_logger.info(f"{os.path.relpath(obj_path)} - Trimesh Timeout occurred.")
    except ValueError as ve:
        invalid_mesh_logger.warning(f"{os.path.relpath(obj_path)} - Trimesh Invalid mesh: {str(ve)}")
    except Exception as e:
        error_logger.error(f"{os.path.relpath(obj_path)} - Trimesh Error: {str(e)}")
    
    error_logger.error(f"{os.path.relpath(obj_path)} - Error: Failed to process with both Open3D and Trimesh.")

def parse_error_log(error_log_path):
    """
    从 error.log 提取所有出错的 .obj 文件路径。
    """
    obj_files = []
    with open(error_log_path, 'r') as log_file:
        for line in log_file:
            if "Error sampling mesh" in line:
                start_idx = line.find("/data/")  
                end_idx = line.find(": Mesh has neither triangles nor vertices.")
                if start_idx != -1 and end_idx != -1:
                    obj_files.append(line[start_idx:end_idx])
    return obj_files

def main(args):
    ROOT_DIR = os.getcwd()  # 根据需要设置根目录
    log_dir = os.path.join(ROOT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志
    timeout_logger, error_logger, invalid_mesh_logger = setup_logging(log_dir)

    input_dir = os.path.join(ROOT_DIR, args.object_source_path, 'oakink')
    output_dir = os.path.join(ROOT_DIR, args.save_path, 'object', 'oakink')
    os.makedirs(output_dir, exist_ok=True)

    # 解析 error.log 获取出错的 .obj 文件路径
    error_log_path = os.path.join(log_dir, 'error.log')
    if not os.path.exists(error_log_path):
        print(f"Error log {error_log_path} does not exist. Exiting.")
        return

    obj_files = parse_error_log(error_log_path)
    print(f"Found {len(obj_files)} files to process.")

    for obj_path in tqdm(obj_files, desc="Processing problematic .obj files"):
        obj_type, obj_name = os.path.split(obj_path)
        obj_base_name = os.path.splitext(obj_name)[0]
        type_output_dir = os.path.join(output_dir, obj_type)
        os.makedirs(type_output_dir, exist_ok=True)
        output_path = os.path.join(type_output_dir, f"{obj_base_name}.pt")
        
        if os.path.exists(output_path):
            print(f"{output_path} already exists, skipping.")
            continue

        try:
            process_obj(obj_path, output_path, args.num_points, args.timeout, error_logger, timeout_logger, invalid_mesh_logger)
        except Exception as e:
            error_details = traceback.format_exc()
            error_logger.error(f"{obj_type}/{obj_name} - Error: {str(e)}")
            error_logger.error(f"Error Details: {error_details}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process problematic .obj and .stl files with Open3D and Trimesh.")
    parser.add_argument('--object_name', type=str, default='oakink', help='Name of the object category.')
    parser.add_argument('--object_source_path',default='data/data_urdf/object', type=str)
    parser.add_argument('--save_path', default='data/PointCloud/', type=str)
    parser.add_argument('--num_points', type=int, default=65536, help='Number of points to sample.')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout for processing each file in seconds.')
    args = parser.parse_args()
    main(args)