import os
import sys
import time
import numpy as np
import torch
import viser
from typing import Dict, List, Tuple, Optional

# Initialize viser server
server = viser.ViserServer(host='127.0.0.1', port=8080)

# Load the data
def load_numpy_data(file_path: str) -> Dict:
    """Load data from numpy file"""
    data = np.load(file_path, allow_pickle=True)
    return data.item() if data.dtype == np.dtype('O') else data

# Function to visualize robot pose
def visualize_robot_pose(pose_index: int, data: Dict) -> None:
    """Visualize a specific robot pose"""
    server.scene.reset()
    
    # Extract robot pose data for the selected index
    selected_pose = data['robot_pose'][pose_index]
    
    # Get contact points if available
    if 'contact_point' in data:
        contact_points = data['contact_point'][pose_index]
        
    # Get object information if available
    if 'obj_pose' in data and 'obj_path' in data:
        obj_pose = data['obj_pose'][pose_index] if data['obj_pose'].shape[0] > 1 else data['obj_pose'][0]
        obj_path = data['obj_path'][pose_index] if len(data['obj_path']) > 1 else data['obj_path'][0]
        
    # Display robot pose
    # For visualization purposes, let's represent the pose as points or a simple skeleton
    # This will depend on the exact format of your data
    
    # Example: if selected_pose contains joint positions
    for i in range(selected_pose.shape[0]):
        for j in range(selected_pose.shape[1]):
            points = selected_pose[i, j, :, :3]  # Assuming the first 3 elements are xyz coordinates
            
            server.scene.add_point_cloud(
                name=f"pose_{pose_index}_link_{i}_point_{j}",
                points=points,
                colors=(102, 192, 255),  # Light blue
                point_size=0.01,
                point_shape="circle",
            )
            
    # Visualize contact points if available
    if 'contact_point' in data:
        for i in range(contact_points.shape[0]):
            cp_points = contact_points[i, 0]  # Assuming this gives us the contact points
            
            server.scene.add_point_cloud(
                name=f"contact_points_{i}",
                points=cp_points,
                colors=(255, 50, 50),  # Red
                point_size=0.01,
                point_shape="circle",
            )
    
    # Add spheres at link positions
    link_spheres = None
    if 'link_spheres' in data:
        link_spheres = data.get('link_spheres', None)
        if link_spheres is not None:
            link_spheres_data = link_spheres[pose_index] if link_spheres.shape[0] > 1 else link_spheres[0]
            for i in range(link_spheres_data.shape[0]):
                for j in range(link_spheres_data.shape[2]):
                    sphere_data = link_spheres_data[i, 0, j]
                    if np.any(sphere_data):  # Check if any data is non-zero
                        position = sphere_data[:3]
                        radius = sphere_data[3]
                        
                        server.scene.add_sphere(
                            name=f"link_sphere_{i}_{j}",
                            center=position,
                            radius=radius,
                            color=(50, 200, 50),  # Green
                            opacity=0.7,
                        )
    
    # Display any additional information as labels
    server.scene.add_label(
        'pose_info',
        f'Pose Index: {pose_index}',
        position=(-0.2, 0.2, 0.2)
    )
    
    # Add joint names if available
    if 'joint_names' in data:
        joint_names = data['joint_names']
        server.scene.add_label(
            'joint_names',
            f'Joints: {", ".join(joint_names[:5])}...',
            position=(-0.2, 0.15, 0.2)
        )

def main():
    # Path to your numpy file
    # In a real application, you would want to make this configurable
    numpy_file_path = '/data/zwq/code/BODex/src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata/bowl_s12105/scale100_pose000_grasp.npy' 
    
    # Load data
    try:
        data = load_numpy_data(numpy_file_path)
        print(f"Data loaded successfully from {numpy_file_path}")
        print(f"robot_pose shape: {data['robot_pose'].shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
    
    # Number of available poses
    num_poses = data['robot_pose'].shape[0]
    
    # Create slider for selecting pose
    pose_slider = server.gui.add_slider(
        label='Robot Pose',
        min=0,
        max=num_poses-1,
        step=1,
        initial_value=0
    )
    
    # Set up callback for slider
    def slider_update_callback(_):
        pose_index = int(pose_slider.value)
        visualize_robot_pose(pose_index, data)
    
    pose_slider.on_update(slider_update_callback)
    
    # Visualize initial pose
    # visualize_robot_pose(0, data)
    
    print("Visualization server started. Open a web browser and navigate to http://localhost:8080")
    print("Use the slider to select different robot poses")
    
    # Keep the server running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Server stopped by user")

if __name__ == "__main__":
    main()