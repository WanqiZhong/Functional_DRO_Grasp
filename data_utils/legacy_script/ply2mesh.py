import os
import trimesh
import open3d as o3d
import signal
from tqdm import tqdm


def trimesh_convert(ply_path, obj_path):
    """
    Convert .ply to .obj using trimesh
    """
    mesh = trimesh.load(ply_path)
    mesh.export(obj_path)
    print(f"Successfully converted {ply_path} to {obj_path} using trimesh.")


def open3d_convert(ply_path, obj_path):
    """
    Convert .ply to .obj using open3d
    """
    mesh = o3d.io.read_triangle_mesh(ply_path)
    o3d.io.write_triangle_mesh(obj_path, mesh)
    print(f"Successfully converted {ply_path} to {obj_path} using open3d.")


def timeout_handler(signum, frame):
    """
    Signal handler for timeout
    """
    raise TimeoutError("Conversion timed out.")


def convert_with_timeout(ply_path, obj_path, timeout):
    """
    Convert .ply to .obj with timeout handling
    """
    signal.signal(signal.SIGALRM, timeout_handler)  # Set up the timeout signal
    signal.alarm(timeout)  # Start the timer
    try:
        trimesh_convert(ply_path, obj_path)
        signal.alarm(0)  # Cancel the timer if successful
    except TimeoutError:
        print(f"Trimesh conversion timed out for {ply_path}. Switching to open3d...")
        open3d_convert(ply_path, obj_path)
    except Exception as e:
        print(f"Trimesh conversion failed for {ply_path} due to: {e}. Switching to open3d...")
        open3d_convert(ply_path, obj_path)
    
    # delete origin ply file
    os.remove(ply_path)


def batch_convert_ply_to_obj(input_dir, timeout=30):
    """
    Batch process all .ply files in the directory and convert them to .obj
    """
    # Gather all .ply files
    ply_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".ply"):
                ply_path = os.path.join(root, file)
                obj_path = os.path.splitext(ply_path)[0] + ".obj"
                # if not os.path.exists(obj_path):
                ply_files.append((ply_path, obj_path))

    # Use tqdm to display progress
    with tqdm(total=len(ply_files), desc="Converting .ply to .obj", unit="file") as pbar:
        for ply_path, obj_path in ply_files:
            try:
                convert_with_timeout(ply_path, obj_path, timeout)
            except Exception as e:
                print(f"Failed to convert {ply_path}. Error: {e}")
            pbar.update(1)


# Specify the target directory
input_directory = "/data/zwq/code/DRO-Grasp/data/data_urdf/object/oakink"
batch_convert_ply_to_obj(input_directory, timeout=30)