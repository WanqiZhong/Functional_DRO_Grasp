import os
import glob
import trimesh
import torch
import pickle as pk
from tqdm import tqdm
from oikit.oi_shape import OakInkShape
import sys
import open3d as o3d
import numpy as np
import torch
import os
import glob
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

class ObjectPointCloudExtractor:
    def __init__(self, num_samples=65536, cache_path=None, grasp_cache_path=None):
        self.num_samples = num_samples
        self.cache_path = cache_path
        self.object_pcs = {}
        if grasp_cache_path and os.path.exists(grasp_cache_path):
            self.load_from_grasp_cache(grasp_cache_path)
        else:
            self.grasp_list = OakInkShape(data_split='train', category='mug', mano_assets_root="assets/mano").grasp_list

    def load_from_grasp_cache(self, grasp_cache_path):
        print(f"Loading grasp cache from {grasp_cache_path}")
        grasp_cache = pk.load(open(grasp_cache_path, 'rb'))
        self.grasp_list = grasp_cache.grasp_list

    def load_from_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading object point clouds from cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.object_pcs = pk.load(f)
            print("Object point clouds loaded from cache.")
        else:
            print("Cache not found. Point clouds need to be processed.")

    def save_to_cache(self):
        if self.cache_path:
            with open(self.cache_path, 'wb') as f:
                pk.dump(self.object_pcs, f)
            print(f"Object point clouds saved to cache: {self.cache_path}")


    def process_object(self, grasp_item):
        obj_id = grasp_item['obj_id']
        if obj_id in self.object_pcs:
            return self.object_pcs[obj_id]

        obj_mesh_path = list(
            glob.glob(os.path.join(ROOT_DIR, f"data/data_urdf/object/oakink/{grasp_item['cate_id']}/{obj_id}.obj")) +
            glob.glob(os.path.join(ROOT_DIR, f"data/data_urdf/object/oakink/{grasp_item['cate_id']}/{obj_id}.ply"))
        )
        assert len(obj_mesh_path) == 1, f"Object mesh not found or ambiguous for obj_id={obj_id}"
        obj_path = obj_mesh_path[0]

        if obj_path.endswith('.ply'):
            mesh = o3d.io.read_triangle_mesh(obj_path)
            vertices = np.asarray(mesh.vertices)
            bbox_center = (vertices.min(0) + vertices.max(0)) / 2
            mesh.vertices = o3d.utility.Vector3dVector(vertices - bbox_center)
            pcd = mesh.sample_points_uniformly(number_of_points=self.num_samples)
            object_pc = np.asarray(pcd.points)
        else:  
            obj_trimesh = trimesh.load(obj_path, process=False, force='mesh', skip_materials=True)
            bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
            obj_trimesh.vertices -= bbox_center
            object_pc, _ = obj_trimesh.sample(self.num_samples, return_index=True)

        object_pc_tensor = torch.tensor(object_pc, dtype=torch.float32)
        self.object_pcs[obj_id] = object_pc_tensor
        return object_pc_tensor

    def process_all_objects(self):
        for grasp_item in tqdm(self.grasp_list, desc="Processing object point clouds"):
            self.process_object(grasp_item)

    def get_object_pc(self, obj_id):
        return self.object_pcs.get(obj_id, None)

if __name__ == "__main__":
    extractor = ObjectPointCloudExtractor(cache_path=os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_all_object_pcs.pt'),
                                          grasp_cache_path = os.path.join(ROOT_DIR, f'data/OakInkDataset/oakink_dataset.pt'))
    extractor.process_all_objects()
    extractor.save_to_cache()