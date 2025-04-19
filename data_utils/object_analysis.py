import os
import glob
import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import pickle
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Project root directory
ROOT_DIR = '/data/zwq/code/DRO_Grasp'
OBJECT_DIR = os.path.join(ROOT_DIR, 'data/data_urdf/object/oakink')
CACHE_PATH = os.path.join(ROOT_DIR, 'data/OakInkDataset/object_sizes.pkl')

def get_object_files():
    """Get all object file paths"""
    obj_files = []
    ply_files = []
    
    # Walk through all subdirectories of oakink folder
    for root, dirs, files in os.walk(OBJECT_DIR):
        for file in files:
            if file.endswith('.obj'):
                obj_files.append(os.path.join(root, file))
            elif file.endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    
    logging.info(f"Found {len(obj_files)} .obj files and {len(ply_files)} .ply files")
    return obj_files + ply_files

def get_object_size(file_path):
    """Read object file and calculate its dimensions"""
    try:
        if file_path.endswith('.ply'):
            object_trimesh = o3d.io.read_triangle_mesh(file_path)
            vertices = np.asarray(object_trimesh.vertices)
            triangles = np.asarray(object_trimesh.triangles)
            bbox_center = (vertices.min(0) + vertices.max(0)) / 2
            vertices = np.array(vertices - bbox_center)
            object_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        else:  # .obj file
            object_trimesh = trimesh.load(file_path, process=False, force='mesh', skip_materials=True)
            bbox_center = (object_trimesh.vertices.min(0) + object_trimesh.vertices.max(0)) / 2
            object_trimesh.vertices -= bbox_center
        
        # Calculate bounding box dimensions
        extents = object_trimesh.bounding_box.extents
        # Maximum dimension as the characteristic size
        max_dimension = max(extents)
        # Volume and surface area (convert to cm³ and cm² from m³ and m²)
        volume = object_trimesh.volume * 1e6  # m³ to cm³
        surface_area = object_trimesh.area * 1e4  # m² to cm²
        
        return {
            'max_dimension': max_dimension,  # kept in meters
            'extents': extents,  # kept in meters
            'volume': volume,  # in cm³
            'surface_area': surface_area,  # in cm²
            'file_path': file_path
        }
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def load_or_create_size_data():
    """Load size data from cache or create if not exists"""
    # Check if cache file exists and is recent
    if os.path.exists(CACHE_PATH):
        cache_time = os.path.getmtime(CACHE_PATH)
        current_time = time.time()
        cache_age = (current_time - cache_time) / 86400  # days
        
        if cache_age < 7:  # Cache is less than 7 days old
            logging.info(f"Loading object size data from cache ({cache_age:.1f} days old)")
            try:
                with open(CACHE_PATH, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading cache file: {e}")
                # Continue to recalculate if loading fails
    
    # If cache doesn't exist or is too old, recalculate
    logging.info("Calculating object sizes (this may take a while)")
    files = get_object_files()
    object_sizes = []
    
    for file_path in tqdm(files):
        size_info = get_object_size(file_path)
        if size_info:
            object_sizes.append(size_info)
    
    # Save results to cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(object_sizes, f)
    logging.info(f"Saved {len(object_sizes)} object size records to {CACHE_PATH}")
    
    return object_sizes

def analyze_sizes(object_sizes):
    """Analyze object sizes and generate histograms"""
    if not object_sizes:
        print("No valid files were processed")
        return
    
    # Extract dimension data for histogram
    max_dimensions = [obj['max_dimension'] * 100 for obj in object_sizes]  # Convert to cm
    volumes = [obj['volume'] for obj in object_sizes]  # Already in cm³
    areas = [obj['surface_area'] for obj in object_sizes]  # Already in cm²
    
    # Calculate means
    mean_dim = np.mean(max_dimensions)
    mean_vol = np.mean(volumes)
    mean_area = np.mean(areas)
    
    # Create histograms
    plt.figure(figsize=(12, 8))
    
    # Main histogram - Maximum dimension distribution
    plt.subplot(2, 2, 1)
    plt.hist(max_dimensions, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(x=mean_dim, color='red', linestyle='--', 
                label=f'Mean: {mean_dim:.2f} cm')
    plt.title('Object Maximum Dimension Distribution')
    plt.xlabel('Maximum Dimension (cm)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Log scale histogram
    plt.subplot(2, 2, 2)
    plt.hist(max_dimensions, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(x=mean_dim, color='red', linestyle='--', 
                label=f'Mean: {mean_dim:.2f} cm')
    plt.title('Object Maximum Dimension Distribution (Log Scale)')
    plt.xlabel('Maximum Dimension (cm)')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Volume distribution
    plt.subplot(2, 2, 3)
    plt.hist(volumes, bins=30, color='salmon', edgecolor='black')
    plt.axvline(x=mean_vol, color='red', linestyle='--', 
                label=f'Mean: {mean_vol:.2f} cm³')
    plt.title('Object Volume Distribution')
    plt.xlabel('Volume (cm³)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Surface area distribution
    plt.subplot(2, 2, 4)
    plt.hist(areas, bins=30, color='lightblue', edgecolor='black')
    plt.axvline(x=mean_area, color='red', linestyle='--', 
                label=f'Mean: {mean_area:.2f} cm²')
    plt.title('Object Surface Area Distribution')
    plt.xlabel('Surface Area (cm²)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('object_size_distribution.png', dpi=300)
    # plt.show()
    
    # Statistical information
    avg_size = np.mean(max_dimensions) / 100  # Back to meters for comparison with robotic hands
    median_size = np.median(max_dimensions) / 100
    min_size = np.min(max_dimensions) / 100
    max_size = np.max(max_dimensions) / 100
    
    print(f"\nSize Statistics:")
    print(f"Average maximum size: {avg_size*100:.2f} cm")
    print(f"Median maximum size: {median_size*100:.2f} cm")
    print(f"Minimum size: {min_size*100:.2f} cm")
    print(f"Maximum size: {max_size*100:.2f} cm")
    
    # Convert back to meters for the objects (for hand comparison)
    for obj in object_sizes:
        obj['max_dimension_cm'] = obj['max_dimension'] * 100
    
    return object_sizes

def compare_with_robotic_hands(object_sizes):
    """Compare object sizes with robotic hand dimensions"""
    # Robotic hand dimension data (in meters)
    hand_info = {
        'ShadowHand': {
            'finger_length': 0.10,  # finger length ~10 cm
            'max_grasp_size': 0.12,  # maximum grasp size ~12 cm
            'min_grasp_size': 0.005  # minimum reliable grasp size ~5 mm
        },
        'LeapHand': {
            'finger_length': 0.095,  # finger length ~9.5 cm
            'max_grasp_size': 0.11,  # maximum grasp size ~11 cm
            'min_grasp_size': 0.008  # minimum reliable grasp size ~8 mm
        },
        'Allegro': {
            'finger_length': 0.11,  # finger length ~11 cm
            'max_grasp_size': 0.15,  # maximum grasp size ~15 cm
            'min_grasp_size': 0.01   # minimum reliable grasp size ~1 cm
        }
    }
    
    # Analyze for each robotic hand
    for hand_name, specs in hand_info.items():
        too_small_count = 0
        too_large_count = 0
        ideal_count = 0
        too_small_objects = []
        
        for obj in object_sizes:
            size = obj['max_dimension']
            if size < specs['min_grasp_size']:
                too_small_count += 1
                too_small_objects.append(obj)
            elif size > specs['max_grasp_size']:
                too_large_count += 1
            else:
                ideal_count += 1
        
        total = len(object_sizes)
        print(f"\n{hand_name} Grasp Size Analysis:")
        print(f"Finger length: {specs['finger_length']*100:.1f} cm")
        print(f"Maximum grasp size: {specs['max_grasp_size']*100:.1f} cm")
        print(f"Minimum reliable grasp size: {specs['min_grasp_size']*100:.1f} mm")
        print(f"Objects too small to grasp: {too_small_count} ({too_small_count/total*100:.1f}%)")
        print(f"Objects too large to grasp: {too_large_count} ({too_large_count/total*100:.1f}%)")
        print(f"Objects suitable for grasping: {ideal_count} ({ideal_count/total*100:.1f}%)")
        
        # Save the list of objects that are too small for this hand
        if too_small_objects:
            small_objects_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/too_small_for_{hand_name}.pkl')
            with open(small_objects_path, 'wb') as f:
                pickle.dump(too_small_objects, f)
            print(f"Saved list of objects too small for {hand_name} to {small_objects_path}")

if __name__ == "__main__":
    try:
        # Load or create object size data
        object_sizes = load_or_create_size_data()
        
        # Analyze all object sizes and generate histograms
        object_sizes = analyze_sizes(object_sizes)
        
        # Compare with robotic hand dimensions
        if object_sizes:
            compare_with_robotic_hands(object_sizes)
    except Exception as e:
        logging.error(f"Error during execution: {e}")