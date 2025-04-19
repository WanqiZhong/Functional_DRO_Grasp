import os
import sys
import time
import glob
import torch
import viser
import trimesh
import open3d as o3d
import numpy as np

from utils.hand_model import create_hand_model
from utils.rotation import quaternion_to_euler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

PT_PATHS = [
    "/data/zwq/code/DRO_Grasp/data/pt1.pt",
    "/data/zwq/code/DRO_Grasp/data/pt2.pt",
    "/data/zwq/code/DRO_Grasp/data/pt3.pt"
]

PT_NAMES = [
    "Baseline",
    "Diffusion",
    "Ours"
]

assert len(PT_PATHS) == len(PT_NAMES), "PT_PATHS and PT_NAMES must have the same length"

# Load all metadata from each .pt file
all_metadata = []
for pt_path in PT_PATHS:
    print(f"Loading from {pt_path}")
    data = torch.load(pt_path, map_location=torch.device("cpu"))
    all_metadata.append(data["metadata"])

# Sanity check: ensure all metadata are same length
lengths = [len(meta) for meta in all_metadata]
assert len(set(lengths)) == 1, "All pt files must contain same number of entries"

N = lengths[0]
print(f"Loaded {len(PT_PATHS)} datasets with {N} items each.")

# Hand model (same across all)
hand = create_hand_model('shadowhand')

# Color map for different sources
COLORS = [
    (255, 102, 102),   # Red
    (102, 192, 255),   # Blue
    (192, 255, 102),   # Green
    (255, 192, 102),   # Orange
    (192, 102, 255),   # Purple
    (255, 255, 102),   # Yellow
]

# Load object mesh by key and id
def load_object_mesh(object_key, object_id):
    name = object_key.split('+')
    mesh_paths = glob.glob(os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{object_id}.*'))

    if not mesh_paths:
        print(f"[!] Object not found: {object_key}/{object_id}")
        return None

    path = mesh_paths[0]
    if path.endswith('.ply'):
        mesh = o3d.io.read_triangle_mesh(path)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        center = (vertices.min(0) + vertices.max(0)) / 2
        vertices = o3d.utility.Vector3dVector(vertices - center)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        mesh = trimesh.load(path, process=False)
        mesh.vertices -= mesh.vertices.mean(0)
        return mesh

# Start Viser
server = viser.ViserServer(host="127.0.0.1", port=8080)
print("http://127.0.0.1:8080")

# Visualization update
def update_scene(index):
    server.scene.reset()

    # Get the base info from the first pt
    base_meta = all_metadata[0][index]
    object_key = base_meta[5]
    object_id = base_meta[7]

    print(f"Index {index} | Object: {object_key} | ID: {object_id}")

    # Load object mesh
    obj_mesh = load_object_mesh(object_key, object_id)
    if obj_mesh:
        server.scene.add_mesh_simple(
            name='object',
            vertices=obj_mesh.vertices,
            faces=obj_mesh.faces,
            color=(239, 132, 167),
            opacity=1.0
        )

    # Display all hands
    for i, meta in enumerate(all_metadata):
        item = meta[index]
        robot_q = item[3]
        hand_mesh = hand.get_trimesh_q(robot_q)["visual"]

        server.scene.add_mesh_simple(
            name=f"{PT_NAMES[i]}_hand",
            vertices=hand_mesh.vertices,
            faces=hand_mesh.faces,
            color=COLORS[i % len(COLORS)],
            opacity=0.75
        )

        # succ_flag = item[10] if len(item) > 10 else None
        # label_text = f"{PT_NAMES[i]}{' ✅' if succ_flag else ''}" if succ_flag is not None else PT_NAMES[i]

        # Slightly offset each label
        server.scene.add_label(
            f"label_{i}",
            label_text,
            wxyz=(1, 0, 0, 0),
            position=(1 + i * 0.1, 1, 1)
        )

# Add slider
index_slider = server.gui.add_slider(
    label="Index",
    min=0,
    max=N - 1,
    step=1,
    initial_value=0
)

index_slider.on_update(lambda _: update_scene(index_slider.value))

# Initial render
update_scene(0)

# Run server loop
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    server.close()