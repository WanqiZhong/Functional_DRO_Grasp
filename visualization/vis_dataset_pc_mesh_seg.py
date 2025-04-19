import os
import sys
import time
import torch
import viser
import numpy as np
import open3d as o3d
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import trimesh
import glob
import json
from utils.hand_model import create_hand_model, HandModel
from oikit.oak_base import OakBase
from oikit.oak_base import ObjectAffordanceKnowledge as OAK
from utils.func_utils import farthest_point_sampling

robot_names = ['leaphand']
object_names = [
    "oakink+lotion_pump",
    "oakink+cylinder_bottle",
    "oakink+mug",
    "oakink+teapot",
    "oakink+bowl",
    "oakink+cup",
    "oakink+knife",
    "oakink+pen",
    "oakink+bottle",
    "oakink+headphones"
]

# Paths
dataset_path = os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_dataset_standard_all_retarget_to_leaphand_valid_dro.pt')  
point_cloud_dataset = torch.load(os.path.join(ROOT_DIR, 'data', 'OakInkDataset', 'oakink_object_pcs.pt'), weights_only=False)
metadata = torch.load(dataset_path, map_location=torch.device('cpu'), weights_only=False)['metadata']

# Meta data files for segmentation
OAKINK_DIR = os.environ['OAKINK_DIR']
REAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'object_id.json')
VIRTUAL_META_FILE = os.path.join(OAKINK_DIR, 'shape', 'metaV2', 'virtual_object_id.json')

# Load metadata for part segmentation
try:
    with open(REAL_META_FILE, 'r', encoding='utf-8') as f:
        real_meta_file = json.load(f)
    with open(VIRTUAL_META_FILE, 'r', encoding='utf-8') as f:
        virtual_meta_file = json.load(f)
    obj_metadata = {**real_meta_file, **virtual_meta_file}
    
    oakbase = OakBase()
    all_cates = list(oakbase.categories.keys())
    obj_nameid_metadata = {}
    for cate in all_cates:
        for obj in oakbase.get_objs_by_category(cate):
            obj_nameid_metadata[obj.obj_id] = obj
    
    metadata_loaded = True
    print("Loaded object metadata for segmentation")
except (FileNotFoundError, json.JSONDecodeError) as e:
    metadata_loaded = False
    print(f"Failed to load metadata: {e}")

def get_segmentation_for_object_avg(object_id, object_name, max_points=512, use_fps=False):
    """
    Get segmented point cloud for an object with balanced points per part
    """
    if not metadata_loaded:
        return None, None
        
    object_name_clean = object_name.split("+")[1]
    object_nameid = obj_metadata.get(object_id, {}).get("name", "")
    obj_meta = obj_nameid_metadata.get(object_nameid, None)
    
    if obj_meta is None:
        print(f"No metadata found for object {object_name_clean} (ID: {object_id})")
        return None, None

    num_parts = len(obj_meta.part_names)
    if num_parts == 0:
        print(f"Object {object_name_clean} has no parts defined")
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
            points_tensor = torch.from_numpy(part_points).float()
            if use_fps:
                _, indices = farthest_point_sampling(points_tensor, max_points_per_part)
                indices_np = np.array(indices)
            else:
                indices_np = np.random.choice(len(part_points), size=max_points_per_part, replace=False)

            part_points = part_points[indices_np]
            part_labels = part_labels[indices_np]

        sampled_points_list.append(part_points)
        sampled_labels_list.append(part_labels)

    if not sampled_points_list:
        print(f"Object {object_name_clean} has no valid parts after sampling")
        return None, None

    points = np.concatenate(sampled_points_list, axis=0)
    labels = np.concatenate(sampled_labels_list, axis=0)

    assert len(points) == len(labels), f"Length mismatch: {len(points)} != {len(labels)}"

    if len(points) > max_points:
        indices = np.random.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        labels = labels[indices]

    return points, labels

def get_object_mesh(object_name, object_id, scale_factor=1.0):
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
        vertices = np.asarray(vertices - bbox_center)            
        object_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        
    else:  
        object_trimesh = trimesh.load(object_path, process=False, force='mesh', skip_materials=True)
        bbox_center = (object_trimesh.vertices.min(0) + object_trimesh.vertices.max(0)) / 2
        object_trimesh.vertices -= bbox_center
        object_trimesh= trimesh.Trimesh(vertices=object_trimesh.vertices, faces=object_trimesh.faces, process=False)

    return object_trimesh.apply_scale(float(scale_factor))

def get_hand_point_cloud(hand:HandModel, q, num_points=512):
    sampled_pc, _ = hand.get_sampled_pc(q, num_points=num_points)
    sampled_pc = sampled_pc.cpu().numpy()
    return sampled_pc[:, :3]

def get_object_point_cloud(object_id, object_pcs, num_points=512, random=False, scale_factor=1.0):
    if object_id not in object_pcs:
        print(f'Object {object_id} not found!')
        return None

    indices = torch.randperm(65536)[:num_points]
    object_pc = np.array(object_pcs[object_id])
    object_pc = object_pc[indices]
    object_pc = torch.tensor(object_pc)
    if random:
        object_pc += torch.randn(object_pc.shape) * 0.002
    object_pc = object_pc * float(scale_factor)
    object_pc = object_pc.numpy()

    return object_pc

# Color mapping for part labels
def get_color_for_label(label, unique_labels):
    # Create a deterministic color mapping
    if not hasattr(get_color_for_label, "color_map"):
        get_color_for_label.color_map = {}
        
    if label not in get_color_for_label.color_map:
        # Define a list of distinct colors (RGB tuples)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 255, 128),  # Mint
            (128, 255, 0),  # Lime
            (255, 0, 128),  # Pink
            (0, 128, 255),  # Sky Blue
            (255, 128, 128),# Light Pink
            (128, 255, 128),# Light Green
            (128, 128, 255),# Light Blue
            (192, 192, 192),# Silver
            (128, 64, 0),   # Brown
            (0, 128, 64),   # Forest Green
            (128, 0, 64),   # Burgundy
            (64, 0, 128)    # Indigo
        ]
        
        # Assign colors to unique labels
        for idx, unique_label in enumerate(unique_labels):
            color_idx = idx % len(colors)
            get_color_for_label.color_map[unique_label] = colors[color_idx]
    
    return get_color_for_label.color_map.get(label, (200, 200, 200))  # Default to gray if not found

def on_update(robot_idx, object_idx, grasp_idx, show_segmentation, num_points=512):
    server.scene.reset()
        
    robot_name = robot_names[robot_idx]
    object_name = object_names[object_idx]    
    metadata_curr = [m for m in metadata if m[5] == object_name and m[6] == robot_name]
    if len(metadata_curr) == 0:
        object_info.content = 'No metadata found!'
        return
    
    grasp_item = metadata_curr[grasp_idx % len(metadata_curr)]
    object_name = grasp_item[5]
    object_id = grasp_item[7]
    scale_factor = grasp_item[8]
    q = grasp_item[3].float()
    
    complex_sentence = grasp_item[10]

    hand = create_hand_model(robot_name, torch.device('cpu'), num_points)

    hand_pc = get_hand_point_cloud(hand, q)
    
    # Decide which point cloud to display based on the segmentation toggle
    if show_segmentation:

        object_pc, segment_labels = get_segmentation_for_object_avg(object_id, object_name)
            
        if object_pc is not None:
            # Scale the points
            object_pc = object_pc * float(scale_factor)
            
            # Get unique labels
            unique_labels = np.unique(segment_labels)
            
            # Add segments as separate point clouds
            for label in unique_labels:
                mask = segment_labels == label
                if np.sum(mask) > 0:
                    label_pc = object_pc[mask]
                    color = get_color_for_label(label, unique_labels)
                    
                    server.scene.add_point_cloud(
                        f'object_pc_{label}',
                        label_pc,
                        point_size=0.002,
                        point_shape="circle",
                        colors=color
                    )
            
            # Add a legend for the labels
            legend_text = "### Part Segmentation\n"
            for label in unique_labels:
                color = get_color_for_label(label, unique_labels)
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                legend_text += f"- <span style='color: {hex_color}'>■</span> {label}\n"
            
            segment_info.content = legend_text
        else:
            # Fallback to regular point cloud
            object_pc = get_object_point_cloud(object_id, point_cloud_dataset, scale_factor=scale_factor)
            server.scene.add_point_cloud(
                'object_pc',
                object_pc,
                point_size=0.002,
                point_shape="circle",
                colors=(239, 132, 167)
            )
            segment_info.content = "No segmentation data available for this object"
    else:
        # Display regular point cloud
        object_pc = get_object_point_cloud(object_id, point_cloud_dataset, scale_factor=scale_factor)
        server.scene.add_point_cloud(
            'object_pc',
            object_pc,
            point_size=0.002,
            point_shape="circle",
            colors=(239, 132, 167)
        )
        segment_info.content = ""

    # Continue with the rest of the visualization
    hand_mesh = hand.get_trimesh_q(q)['visual']
    object_mesh = get_object_mesh(object_name, object_id, scale_factor)

    object_size = object_mesh.bounding_box.extents

    if object_pc is None: 
        object_info.content = "Failed to load object point cloud"
        return
    
    server.scene.add_point_cloud(
        'hand_pc',
        hand_pc,  
        point_size=0.002,
        point_shape="circle",
        colors=(102, 192, 255)
    )

    server.scene.add_mesh_simple(
        'hand_mesh',
        hand_mesh.vertices,
        hand_mesh.faces,
        color=(239, 132, 167),
        opacity=0.7 if show_segmentation else 1.0,
    )

    server.scene.add_mesh_simple(
        'object_mesh',
        object_mesh.vertices,
        object_mesh.faces,
        color=(102, 192, 255),
        opacity=0.2 if show_segmentation else 0.7,
    )

    object_info.content = f"### Object Info\nFunctional Sentence: {complex_sentence}\n Size: x={object_size[0]:.3f}m, y={object_size[1]:.3f}m, z={object_size[2]:.3f}m"

def update_visualization(_):
    on_update(
        robot_slider.value, 
        object_slider.value, 
        grasp_slider.value, 
        show_segmentation_checkbox.value
    )

# Set up the visualization server
server = viser.ViserServer(host='127.0.0.1', port=8080)

# UI controls
robot_slider = server.gui.add_slider(
    label='Robot',
    min=0,
    max=len(robot_names) - 1,
    step=1,
    initial_value=0
)
object_slider = server.gui.add_slider(
    label='Object',
    min=0,
    max=len(object_names) - 1,
    step=1,
    initial_value=0
)
grasp_slider = server.gui.add_slider(
    label='Grasp',
    min=0,
    max=199,  
    step=1,
    initial_value=0
)
show_segmentation_checkbox = server.gui.add_checkbox(
    label='Show Part Segmentation',
    initial_value=False
)
object_info = server.gui.add_markdown(
    content=""
)
segment_info = server.gui.add_markdown(
    content=""
)

# Register update callbacks
robot_slider.on_update(update_visualization)
object_slider.on_update(update_visualization)
grasp_slider.on_update(update_visualization)
show_segmentation_checkbox.on_update(update_visualization)

# Initial visualization
update_visualization(None)

# Keep the server running
while True:
    time.sleep(1)