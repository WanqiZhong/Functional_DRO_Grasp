import os
import sys
import argparse
import time
import viser
import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from DRO_Grasp.utils.hand_model import create_hand_model
import os
import torch
import trimesh
from tqdm import tqdm

import os
import torch
import trimesh
from tqdm import tqdm
from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool
import time
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import open3d as o3d
import numpy as np
import traceback
import argparse
import signal
import os
import torch
from tqdm import tqdm
import open3d as o3d
import numpy as np
import traceback
import argparse
import signal
import logging

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_format = '%(asctime)s %(levelname)s: %(message)s'
    
    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(os.path.join(log_dir, 'error.log'))
    error_handler.setFormatter(logging.Formatter(log_format))
    error_logger.addHandler(error_handler)
    
    info_logger = logging.getLogger('info_logger')
    info_logger.setLevel(logging.INFO)
    info_logger_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
    info_logger_handler.setFormatter(logging.Formatter(log_format))
    info_logger.addHandler(info_logger_handler)
    
    return error_logger, info_logger

def sample_with_open3d(obj_path, num_points, return_mesh):

    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    bbox_center = (vertices.min(0) + vertices.max(0)) / 2
    mesh.vertices = o3d.utility.Vector3dVector(vertices - bbox_center)

    if return_mesh:
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles), "open3d_vertices"

    if mesh.has_triangles():
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        return points, normals, "open3d_faces"  
    
    elif mesh.has_vertices():
        points = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else np.zeros_like(points)
        
        if len(points) == 0:
            raise ValueError("Mesh has no vertices.")

        if len(points) >= num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[indices]
            sampled_normals = normals[indices]
        else:
            repeat_factor = num_points // len(points) + 1
            sampled_points = np.tile(points, (repeat_factor, 1))[:num_points]
            sampled_normals = np.tile(normals, (repeat_factor, 1))[:num_points]
        
        return sampled_points, sampled_normals, "open3d_vertices"  
    else:
        raise ValueError("Mesh has neither triangles nor vertices.")

def sample_with_trimesh(obj_path, num_points):

    mesh = trimesh.load(obj_path, force='mesh')
    bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    mesh.vertices -= bbox_center
    
    if mesh.is_empty:
        raise ValueError("Mesh is empty.")
    
    if mesh.faces.shape[0] > 0:
        sampled_points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        normals = mesh.face_normals[face_indices]
        return sampled_points, normals, "trimesh_faces"
    
    elif mesh.vertices.shape[0] > 0:
        points = mesh.vertices
        normals = mesh.vertex_normals if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0 else np.zeros_like(points)
        
        if len(points) == 0:
            raise ValueError("Mesh has no vertices.")

        if len(points) >= num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[indices]
            sampled_normals = normals[indices] if normals.size > 0 else np.zeros_like(sampled_points)
        else:
            repeat_factor = num_points // len(points) + 1
            sampled_points = np.tile(points, (repeat_factor, 1))[:num_points]
            sampled_normals = np.tile(normals, (repeat_factor, 1))[:num_points] if normals.size > 0 else np.zeros_like(sampled_points)
        
        return sampled_points, sampled_normals, "trimesh_vertices"
    else:
        raise ValueError("Mesh has neither faces nor vertices.")

def process_with_open3d(obj_path, num_points, timeout):

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        sampled_points, sampled_normals, method = sample_with_open3d(obj_path, num_points)
        signal.alarm(0)  
        return sampled_points, sampled_normals, method
    except TimeoutException:
        raise TimeoutException(f"Open3D processing timed out after {timeout} seconds.")
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)  

def process_with_trimesh(obj_path, num_points, timeout):

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        sampled_points, sampled_normals, method = sample_with_trimesh(obj_path, num_points)
        signal.alarm(0)  
        return sampled_points, sampled_normals, method
    except TimeoutException:
        raise TimeoutException(f"Trimesh processing timed out after {timeout} seconds.")
    except Exception as e:
        raise e
    finally:
        signal.alarm(0) 

def process_obj(obj_path, output_path, num_points, timeout, error_logger, info_logger, package):
    open3d_success = False
    trimesh_success = False
    open3d_error = ""
    trimesh_error = ""

    try:
        sampled_points, sampled_normals, method = process_with_open3d(obj_path, num_points, timeout)
        if not package:
            object_pc_normals = torch.cat([
                torch.from_numpy(sampled_points).float(),
                torch.from_numpy(sampled_normals).float()
            ], dim=-1)  # (num_points, 6)
            torch.save(object_pc_normals, output_path) 
            open3d_success = True
        else:
            return torch.cat([
                torch.from_numpy(sampled_points).float(),
                torch.from_numpy(sampled_normals).float()
            ], dim=-1)
    except TimeoutException:
        open3d_error = "Open3D Timeout occurred."
    except ValueError as ve:
        open3d_error = f"Open3D Invalid mesh: {str(ve)}"
    except Exception as e:
        open3d_error = f"Open3D Error: {str(e)}"
        info_logger.error(f"{os.path.relpath(obj_path)} - Error: {str(e)}")

    if not open3d_success:
        try:
            sampled_points, sampled_normals, method = process_with_trimesh(obj_path, num_points, timeout)
            if not package:
                object_pc_normals = torch.cat([
                    torch.from_numpy(sampled_points).float(),
                    torch.from_numpy(sampled_normals).float()
                ], dim=-1)  # (num_points, 6)
                torch.save(object_pc_normals, output_path)  
                trimesh_success = True
            else:
                return torch.cat([
                    torch.from_numpy(sampled_points).float(),
                    torch.from_numpy(sampled_normals).float()
                ], dim=-1)
        except TimeoutException:
            trimesh_error = "Trimesh Timeout occurred."
        except ValueError as ve:
            trimesh_error = f"Trimesh Invalid mesh: {str(ve)}"
        except Exception as e:
            trimesh_error = f"Trimesh Error: {str(e)}"
            info_logger.error(f"{os.path.relpath(obj_path)} - Error: {str(e)}")

    if open3d_success or trimesh_success:
        info_logger.warning(f"{os.path.relpath(obj_path)} - Success: Processed with Open3D: {open3d_success}, Trimesh: {trimesh_success}, Final method: {method}")
    else:
        info_logger.error(f"{os.path.relpath(obj_path)} - Failed to process with Open3D: {open3d_error}, Trimesh: {trimesh_error}")
        
    if not open3d_success and not trimesh_success:
        combined_error = f"Open3D Errors: {open3d_error} | Trimesh Errors: {trimesh_error}"
        error_logger.error(f"{os.path.relpath(obj_path)} - Error: Failed to process with both Open3D and Trimesh. Details: {combined_error}")

    return None


def generate_object_pc(args):

    if args.object_name == 'oakink':

        input_dir = os.path.join(ROOT_DIR, args.object_source_path, 'oakink')
        output_dir = os.path.join(ROOT_DIR, args.save_path, 'object', 'oakink')
        log_dir = os.path.join(ROOT_DIR, 'log')
        os.makedirs(output_dir, exist_ok=True)
        
        error_logger, info_logger = setup_logging(log_dir)

        obj_types = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        obj_pcs = {}
        
        for obj_type in tqdm(obj_types, desc="Processing object types"):
            type_input_dir = os.path.join(input_dir, obj_type)
            type_output_dir = os.path.join(output_dir, obj_type)
            os.makedirs(type_output_dir, exist_ok=True)

            obj_files = [f for f in os.listdir(type_input_dir) if f.endswith('.obj') and not f.startswith('coacd_')]

            for obj_name in tqdm(obj_files, desc=f"Processing {obj_type} objects"):
                
                obj_path = os.path.join(type_input_dir, obj_name)
                obj_base_name = os.path.splitext(obj_name)[0]
                output_path = os.path.join(type_output_dir, f"{obj_base_name}.pt")

                if not args.package_obj:
                    if os.path.exists(output_path):
                        print(f"{output_path} already exists, skipping.")
                        continue
                    process_obj(obj_path, output_path, args.num_points, args.timeout, error_logger, info_logger, args.package_obj)
                else:
                    sampled_points = process_obj(obj_path, output_path, args.num_points, args.timeout, error_logger, info_logger, args.package_obj)
                    obj_pcs[obj_base_name] = sampled_points.float()

        if args.package_obj:
            output_dir = os.path.join(ROOT_DIR, args.package_save_path)
            torch.save(obj_pcs, output_dir)
            print(f"Object point clouds saved to {output_dir}")
                
    else:
        """
        object/{contactdb, ycb}/<object_name>.pt: (num_points, 6), point xyz + normal
        From data/data_urdf/object/{contactdb, ycb}/<object_name>/<object_name>.stl (mesh)
        To data/PointCloud/object/{contactdb, ycb}/<object_name>.pt (point cloud)
        """
        for dataset_type in ['contactdb', 'ycb']:
            input_dir = str(os.path.join(ROOT_DIR, args.object_source_path, dataset_type))
            output_dir = str(os.path.join(ROOT_DIR, args.save_path, 'object', dataset_type))
            os.makedirs(output_dir, exist_ok=True)

            obj_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
            for object_name in tqdm(obj_dirs, desc=f"Processing {dataset_type} objects"):
                mesh_path = os.path.join(input_dir, object_name, f'{object_name}.stl')
                try:
                    mesh = trimesh.load_mesh(mesh_path)
                    object_pc, face_indices = mesh.sample(args.num_points, return_index=True)  # Sample points
                    object_pc = torch.tensor(object_pc, dtype=torch.float32)  # Convert to tensor
                    normals = torch.tensor(mesh.face_normals[face_indices], dtype=torch.float32)  # Get normals
                    object_pc_normals = torch.cat([object_pc, normals], dim=-1)  # Concatenate xyz + normals
                    torch.save(object_pc_normals, os.path.join(output_dir, f'{object_name}.pt'))  # Save as .pt
                except Exception as e:
                    print(f"Failed to process {mesh_path}: {e}")

    print("\nGenerating object point cloud finished.")


def generate_robot_pc(args):
    output_dir = str(os.path.join(ROOT_DIR, args.save_path, 'robot'))
    output_path = str(os.path.join(output_dir, f'{args.robot_name}.pt'))
    os.makedirs(output_dir, exist_ok=True)

    hand = create_hand_model(args.robot_name, torch.device('cpu'), args.num_points)
    links_pc = hand.vertices
    sampled_pc, sampled_pc_index = hand.get_sampled_pc(num_points=args.num_points)
    print("Hand joint order:")

    filtered_links_pc = {}
    for link_index, (link_name, points) in enumerate(links_pc.items()):
        mask = [i % args.num_points for i in sampled_pc_index
                if link_index * args.num_points <= i < (link_index + 1) * args.num_points]
        links_pc[link_name] = torch.tensor(points, dtype=torch.float32)
        filtered_links_pc[link_name] = torch.tensor(points[mask], dtype=torch.float32)
        print(f"[{link_name}] original shape: {links_pc[link_name].shape}, filtered shape: {filtered_links_pc[link_name].shape}")

    data = {
        'original': links_pc,
        'filtered': filtered_links_pc
    }
    torch.save(data, output_path)
    print("\nGenerating robot point cloud finished.")

    server = viser.ViserServer(host='127.0.0.1', port=8080)
    server.scene.add_point_cloud(
        'point cloud',
        sampled_pc[:, :3].numpy(),
        point_size=0.001,
        point_shape="circle",
        colors=(0, 0, 200)
    )
    while True:
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='object', type=str)
    parser.add_argument('--save_path', default='data/PointCloud/', type=str)
    parser.add_argument('--num_points', default=512, type=int)
    # for object pc generation
    parser.add_argument('--object_source_path', default='data/data_urdf/object', type=str)
    # for robot pc generation
    parser.add_argument('--robot_name', default='mano', type=str)
    parser.add_argument('--object_name', default='oakink', type=str)
    parser.add_argument('--timeout', default=20, type=int)
    parser.add_argument('--package_obj', action='store_true', help='Output all object point cloud in a single file')
    parser.add_argument('--package_save_path', default='data/OakInkDataset/oakink_object_pcs_with_normals.pt', type=str)
    args = parser.parse_args()

    if args.type == 'robot':
        generate_robot_pc(args)
    elif args.type == 'object':
        # Security check
        # if not use package, the num points should be 512
        assert args.num_points == 512 or args.package_obj, "num_points should be 512 if not using package"
        # if use package, the num points should be 65536
        assert args.num_points == 65536 or not args.package_obj, "num_points should be 65536 if using package"
        print(f"num_points: {args.num_points}, package_obj: {args.package_obj}")
        # Get input to continue (y/n)
        check = input("Do you want to continue? (y/n): ")
        if check.lower() != 'y':
            print("Exiting...")
            exit(0)
        generate_object_pc(args)
    else:
        raise NotImplementedError
