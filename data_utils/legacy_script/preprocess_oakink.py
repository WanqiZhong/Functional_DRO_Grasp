import os
import json
import shutil

OAKINK_DIR = '/data/zwq/data/oakink'  
TARGET_DIR = '/data/zwq/code/DRO-Grasp/data/data_urdf/object/oakink'      

# Paths to the meta files
REAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'object_id.json')
VIRTUAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'virtual_object_id.json')

# Paths to the source directories
INTERACTION_DIR = os.path.join(OAKINK_DIR, 'shape', 'oakink_shape_v2')
REAL_SOURCE_DIR = os.path.join(OAKINK_DIR, 'shape', 'OakInkObjectsV2')
VIRTUAL_SOURCE_DIR = os.path.join(OAKINK_DIR, 'shape', 'OakInkVirtualObjectsV2')

# Supported mesh file extensions
MESH_EXTENSIONS = ['.obj', '.ply', '.stl']

def ensure_dir(path):
    """Ensure the directory exists, create it if not."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_json(json_path):
    """Load a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_mesh_files(directory):
    """
    Find all supported mesh files in the specified directory.
    Returns a list of full file paths.
    """
    mesh_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in MESH_EXTENSIONS:
                mesh_files.append(os.path.join(root, file))
    return mesh_files

def process_object(object_id, is_virtual, meta_data, source_dir, category):
    """
    Process a single object (real or virtual).
    """
    obj_info = meta_data.get(object_id)
    if not obj_info:
        print(f"Object ID {object_id} not found in metadata, skipping.")
        return

    object_name = obj_info.get('name')
    if not object_name:
        print(f"Object ID {object_id} has no name, skipping.")
        return

    # Choose the source directory based on whether it is a virtual object
    suffix = ''

    if not os.path.isdir(mesh_source_dir):
        print(f"Mesh directory does not exist: {mesh_source_dir}, skipping object ID {object_id}.")
        return

    # Find all mesh files
    mesh_files = find_mesh_files(mesh_source_dir)
    if not mesh_files:
        print(f"No mesh files found in directory {mesh_source_dir}, skipping object ID {object_id}.")
        return

    # Target directory
    target_object_dir = os.path.join(TARGET_DIR, category)
    ensure_dir(target_object_dir)

    for mesh_file in mesh_files:
        extension = os.path.splitext(mesh_file)[1].lower()
        base_name = os.path.splitext(os.path.basename(mesh_file))[0]
        target_file_name = f"{object_id}_{suffix}{extension}"
        target_file_path = os.path.join(target_object_dir, target_file_name)

        # Copy the file
        shutil.copy2(mesh_file, target_file_path)
        

def main():
    ensure_dir(TARGET_DIR)

    # Load meta data
    real_meta = load_json(REAL_META_FILE)
    virtual_meta = load_json(VIRTUAL_META_FILE)

    # Traverse the interaction set
    if not os.path.isdir(INTERACTION_DIR):
        print(f"Interaction set directory does not exist: {INTERACTION_DIR}")
        return

    categories = [d for d in os.listdir(INTERACTION_DIR) if os.path.isdir(os.path.join(INTERACTION_DIR, d))]

    for category in categories:
        category_path = os.path.join(INTERACTION_DIR, category)
        # Traverse real objects under each category
        real_objects = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]

        for real_object_id in real_objects:
            # Process real object
            process_object(real_object_id, is_virtual=False, meta_data=real_meta, source_dir=REAL_SOURCE_DIR, category=category)

            # Check if there are virtual objects
            real_object_path = os.path.join(category_path, real_object_id)
            
            for grasp_item in os.listdir(real_object_path):
                
                real_object_grasp_path = os.path.join(real_object_path, grasp_item)
                    
                virtual_objects = [d for d in os.listdir(real_object_grasp_path) if os.path.isdir(os.path.join(real_object_grasp_path, d))]

                for virtual_object_id in virtual_objects:
                    # Process virtual object
                    process_object(virtual_object_id, is_virtual=True, meta_data=virtual_meta, source_dir=VIRTUAL_SOURCE_DIR, category=category)


if __name__ == "__main__":
    main()
