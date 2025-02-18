import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import viser
import torch
import numpy as np
import pickle
import time
import trimesh
from utils.hand_model import create_hand_model
import glob
import open3d as o3d


# Load the pickle file
def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Load your results
results = load_results('/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_03-22-31_6dof_oakink/eval/final/2025-01-08_17-43-07/res_diffuser.pkl')
# results = load_results('/data/zwq/code/Scene-Diffuser/outputs/2025-01-08_03-22-31_6dof_oakink/eval/final_validate_data/2025-01-08_17-04-19/res_diffuser.pkl')
num_samples = len(results['results'])

def get_object_mesh(object_name, object_id):
    name = object_name.split('+')
    obj_mesh_path = list(
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.obj')) +
        glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.ply')))
    assert len(obj_mesh_path) == 1
    object_path = obj_mesh_path[0]

    if object_path.endswith('.ply'):
        object_trimesh = o3d.io.read_triangle_mesh(object_path)
        vertices = np.asarray(object_trimesh.vertices)
        triangles = np.asarray(object_trimesh.triangles)
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
        vertices = vertices - bbox_center
        object_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:  
        object_trimesh = trimesh.load(object_path, process=False, force='mesh', skip_materials=True)
        bbox_center = (object_trimesh.vertices.min(0) + object_trimesh.vertices.max(0)) / 2
        object_trimesh.vertices -= bbox_center
    
    return object_trimesh

def on_update(sample_idx):
    # Clear previous visualizations
    server.scene.reset()
    
    data = results['results'][sample_idx]
    robot_name = data['robot_name'][0]
    object_name = data['object_name'][0]
    object_id = data['object_id'][0]
    
    # Get object mesh
    object_trimesh = get_object_mesh(object_name, object_id)
    
    # Create hand model
    hand = create_hand_model(robot_name)
    
    # Get predicted hand mesh
    initial_q = hand.get_fixed_initial_q()
    predict_q = torch.cat([data['ddpm_qpos'][0], initial_q[6:]])
    predict_robot_trimesh = hand.get_trimesh_q(predict_q)["visual"]
    
    # Get target hand mesh
    target_q = data['nofix_initial_q'][0]
    target_robot_trimesh = hand.get_trimesh_q(target_q)["visual"]
    
    # Add object mesh
    server.scene.add_mesh_simple(
        'object',
        object_trimesh.vertices,
        object_trimesh.faces,
        color=(239, 132, 167),
        opacity=1
    )
    
    # Add predicted hand mesh
    server.scene.add_mesh_simple(
        'predict_robot',
        predict_robot_trimesh.vertices,
        predict_robot_trimesh.faces,
        color=(102, 192, 255),  # Light blue for predicted
        opacity=0.8
    )
    
    # Add target hand mesh
    server.scene.add_mesh_simple(
        'target_robot',
        target_robot_trimesh.vertices,
        target_robot_trimesh.faces,
        color=(192, 102, 255),  # Purple for target
        opacity=0.8
    )

def update_visualization(_):
    on_update(sample_slider.value)

# Initialize viser server
server = viser.ViserServer(host='127.0.0.1', port=8080)

# Add slider for sample index
sample_slider = server.gui.add_slider(
    label='Sample Index',
    min=0,
    max=num_samples - 1,
    step=1,
    initial_value=0
)

# Register update callback
sample_slider.on_update(update_visualization)

# Initial visualization
update_visualization(None)

# Keep the server running
while True:
    time.sleep(1)