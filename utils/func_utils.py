import os
import sys
import torch
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from pytorch3d.ops.knn import knn_gather, knn_points


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"[Time] {func.__name__}: {time.perf_counter() - start:.4f}s")
        return result
    return wrapper


def calculate_depth(robot_pc, object_names, object_ids=None):
    """
    Calculate the average penetration depth of predicted pc into the object.

    :param robot_pc: (B, N, 3)
    :param object_name: list<str>, len = B
    :return: calculated depth, (B,)
    """
    object_pc_list = []
    normals_list = []

    if object_ids is not None:
        for idx, object_name in enumerate(object_ids):
            name = object_names[idx].split('+')
            object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}/{object_name}.pt')
            object_pc_normals = torch.load(object_path).to(robot_pc.device)
            object_pc_list.append(object_pc_normals[:, :3])
            normals_list.append(object_pc_normals[:, 3:])
    else:
        for object_name in object_names:
            name = object_name.split('+')
            object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
            object_pc_normals = torch.load(object_path).to(robot_pc.device)
            object_pc_list.append(object_pc_normals[:, :3])
            normals_list.append(object_pc_normals[:, 3:])

    object_pc = torch.stack(object_pc_list, dim=0)
    normals = torch.stack(normals_list, dim=0)

    distance = torch.cdist(robot_pc, object_pc)
    distance, index = torch.min(distance, dim=-1)
    index = index.unsqueeze(-1).repeat(1, 1, 3)
    object_pc_indexed = torch.gather(object_pc, dim=1, index=index)
    normals_indexed = torch.gather(normals, dim=1, index=index)
    get_sign = torch.vmap(torch.vmap(lambda x, y: torch.where(torch.dot(x, y) >= 0, 1, -1)))
    signed_distance = distance * get_sign(robot_pc - object_pc_indexed, normals_indexed)
    signed_distance[signed_distance > 0] = 0
    return -torch.mean(signed_distance)


def farthest_point_sampling(point_cloud, num_points=1024):
    """
    :param point_cloud: (N, 3) or (N, 4), point cloud (with link index)
    :param num_points: int, number of sampled points
    :return: ((N, 3) or (N, 4), list), sampled point cloud (numpy) & index
    """
    point_cloud_origin = point_cloud
    if point_cloud.shape[1] == 4:
        point_cloud = point_cloud[:, :3]

    selected_indices = [0]
    distances = torch.norm(point_cloud - point_cloud[selected_indices[-1]], dim=1)
    for _ in range(num_points - 1):
        farthest_point_idx = torch.argmax(distances)
        selected_indices.append(farthest_point_idx)
        new_distances = torch.norm(point_cloud - point_cloud[farthest_point_idx], dim=1)
        distances = torch.min(distances, new_distances)
    sampled_point_cloud = point_cloud_origin[selected_indices]

    return sampled_point_cloud, selected_indices

def get_euclidean_distance(src_xyz, trg_xyz):
    '''
    Calculate direct Euclidean distance affordance between source points and target points using kNN
    
    :param src_xyz: [B, N1, 3] source points (v_o)
    :param trg_xyz: [B, N2, 3] target points (H)
    :return: euclidean_distances: [B, N1] tensor for Euclidean distances
    :return: nearest_indices: [B, N1] tensor for indices of nearest target points
    '''
    B = src_xyz.size(0)
    
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each batch
    
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    
    # Use knn_points to find nearest neighbors (k=1)
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=1)  # [dists, idx]
    
    euclidean_distances = src_nn.dists[..., 0]  # [B, N1]
    nearest_indices = src_nn.idx[..., 0]  # [B, N1]
    
    euclidean_distances = torch.sqrt(euclidean_distances)
    
    euclidean_distances *= 100
    
    return euclidean_distances, nearest_indices

# @timed
def get_aligned_distance(src_xyz, trg_xyz, src_normals, gamma = 1.0):
    '''
    Calculate the aligned distance as per the formula:
    D(v_o, H) = min_{v_h∈H} e^(\gemma (1-<v_h-v_o,n_o>)) * sqrt(||v_o - v_h||_2)
    
    :param src_xyz: [B, N1, 3] source points (v_o)
    :param trg_xyz: [B, N2, 3] target points (H)
    :param src_normals: [B, N1, 3] surface normals at source points (n_o)
    :return: aligned_distances: [B, N1] tensor for aligned distance
    '''
    B = src_xyz.size(0)
    N1 = src_xyz.size(1)
    N2 = trg_xyz.size(1)
    
    # Calculate direction vectors from source to target points: (v_o - v_h)
    direction_vectors =  trg_xyz.unsqueeze(1) - src_xyz.unsqueeze(2)  # [B, N1, N2, 3]
    # direction_vectors =  src_xyz.unsqueeze(2) - trg_xyz.unsqueeze(1)  # [B, N1, N2, 3]
    euclidean_dists = torch.norm(direction_vectors, dim=-1)  # [B, N1, N2]
    
    # Normalize direction vectors
    normalized_dirs = direction_vectors / (torch.norm(direction_vectors, dim=-1, keepdim=True) + 1e-10)  # [B, N1, N2, 3]
    
    # Normalize surface normals
    normalized_normals = src_normals / (torch.norm(src_normals, dim=-1, keepdim=True) + 1e-10)  # [B, N1, 3]
    
    # Calculate inner product between normalized direction vectors and surface normals
    expanded_normals = normalized_normals.unsqueeze(2)

    inner_products = torch.sum(normalized_dirs * expanded_normals, dim=-1)  # [B, N1, N2]
    
    # Apply the aligned distance formula
    aligned_distances_all = torch.exp(gamma * (1 - inner_products)) * euclidean_dists  # [B, N1, N2]
    
    # Find the minimum aligned distance for each source point across all target points
    aligned_distances, aligned_idx = torch.min(aligned_distances_all, dim=2)  # [B, N1]

    aligned_distances *= 100
    
    return aligned_distances, aligned_idx

def get_contact_map(aligned_distances):
    '''
    Calculate contact map using aligned distances as per formula:
    C(v_o, H) = 1 - 2(Sigmoid(D(v_o, H)) - 0.5)
    
    :param aligned_distances: [B, N1] aligned distances D(v_o, H)
    :return: contact_map: [B, N1] tensor with values in range [0,1]
    '''
    sigmoid_result = torch.sigmoid(aligned_distances)
    contact_map = 1.0 - 2.0 * (sigmoid_result - 0.5)
    
    return contact_map
