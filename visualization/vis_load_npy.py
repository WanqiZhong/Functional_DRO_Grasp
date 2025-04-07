import os
import glob
import viser
import trimesh
import numpy as np
import torch
from utils.hand_model import create_hand_model, HandModel
from utils.rotation import quaternion_to_euler, matrix_to_euler
# Forbidden torch warning & numpy warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

grasp_status = 'squeeze_qpos' # ['pregrasp_qpos', 'grasp_qpos', 'squeeze_qpos']

ROOT_DIR = "/data/zwq/code/Visualize/"

ITEMS = [
    "DRO_Early_Custom",
    "DRO_Early_Multiple",
    "DRO_Custom",
    "DRO_Multiple",
    "Diffusion_Custom",
    "Diffusion_Multiple"
]

ITEM_COLORS = {
    "DRO_Early_Custom": (192, 102, 255),
    "DRO_Early_Multiple": (192, 255, 102),
    "DRO_Custom": (102, 255, 192),
    "DRO_Multiple": (255, 192, 102),
    "Diffusion_Custom": (255, 102, 192),
    "Diffusion_Multiple": (102, 192, 255),
}

def torch_quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    quaternions = torch.as_tensor(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def rotation_matrix_z(angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

def rotation_matrix_y(angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

def rotation_matrix_x(angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])

def transform_matrix(translation, rotation_matrix):
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform

def correct_shadowhand_wrist_from_mujoco_to_urdf(robot_pose, wrj2_angle=0.0):

    position = robot_pose[:3]
    quat = robot_pose[3:7]
    
    R_palm = torch_quaternion_to_matrix(torch.tensor(quat)).cpu().numpy()

    R_wrj2_axis = rotation_matrix_y(wrj2_angle)
    t_wrj2 = np.array([0.00, -0.01, 0.21301])
    
    T_wrj2 = transform_matrix(t_wrj2, R_wrj2_axis)
    
    global_transform = np.eye(4)
    global_transform[:3, :3] = R_palm
    
    T_wrist_to_forearm = global_transform @ T_wrj2
    
    corrected_position = position - T_wrist_to_forearm[:3, 3]
    
    return np.concatenate([corrected_position, quat, robot_pose[7:]])


def load_grasp_data_and_mesh(hand, folder_path, item_name):
    """
    Load the grasp mesh (.obj) and corresponding .npy data for a given item in the folder.
    Returns: (mesh: trimesh.Trimesh, success_flag: bool)
    """
    # Match grasp mesh and data with wildcards
    npy_pattern = os.path.join(folder_path, f"{item_name}*.npy")
    npy_files = glob.glob(npy_pattern)
    
    try:
        grasp_data = np.load(npy_files[0], allow_pickle=True).item()
        q = grasp_data[grasp_status]  

        q = torch.tensor(q)
        quaternion = quaternion_to_euler(torch.cat([q[4:7], q[3:4]]))
        q = torch.tensor(correct_shadowhand_wrist_from_mujoco_to_urdf(q))
        reorder_q = torch.cat([q[:3], quaternion, torch.tensor([0.0, 0.0]), q[7:]], axis=-1)
        reorder_q = reorder_q.float()
        
        robot_trimesh = hand.get_trimesh_q(reorder_q)["visual"]
        succ = grasp_data.get('succ_flag', False)

        return robot_trimesh, succ
    except Exception as e:
        print(f"[{item_name}] Error loading files: {e}")
        return None, None


def main():
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    print("Server started at http://127.0.0.1:8080")

    # Get available object folders
    available_objects = sorted([
        f for f in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, f))
    ])

    current_object = available_objects[0] if available_objects else ""
    hand = create_hand_model("shadowhand")

    def update_scene():
        server.scene.reset()

        if not current_object:
            print("No object selected.")
            return

        folder_path = os.path.join(ROOT_DIR, current_object)
        success_text = ""

        # Load and visualize each grasp item
        for item in ITEMS:
            mesh, succ = load_grasp_data_and_mesh(hand, folder_path, item)
            if mesh is not None:
                server.scene.add_mesh_simple(
                    name=f"{item}_mesh",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    color=ITEM_COLORS[item],
                    opacity=0.7
                )
                success_text += f"{item}: {succ}\n"

        # Load and visualize the object mesh (e.g., object.obj)
        try:
            obj_path = os.path.join(folder_path, "object.obj")
            if os.path.exists(obj_path):
                obj_mesh = trimesh.load_mesh(obj_path)
                server.scene.add_mesh_simple(
                    name="object",
                    vertices=obj_mesh.vertices,
                    faces=obj_mesh.faces,
                    color=(239, 132, 167),
                    opacity=1.0
                )
        except Exception as e:
            print(f"Error loading object mesh: {e}")

        # Add label with success info
        server.scene.add_label(
            "Success",
            success_text.strip(),
            wxyz=(1, 0, 0, 0),
            position=(1, 1, 1)
        )

    with server.gui.add_folder("Object Selection"):
        object_dropdown = server.gui.add_dropdown(
            "Choose Object",
            options=available_objects,
            initial_value=current_object
        )

        @object_dropdown.on_update
        def on_object_change(event):
            nonlocal current_object
            current_object = event.target.value
            update_scene()

    update_scene()

    try:
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        server.close()


if __name__ == "__main__":
    main()