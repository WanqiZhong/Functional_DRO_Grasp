import viser
import trimesh
import time
import os
import glob
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

obj_dir = os.path.join(ROOT_DIR, 'real_object')
obj_files = sorted(glob.glob(os.path.join(obj_dir, '*.obj')))

server = viser.ViserServer(host='127.0.0.1', port=8081)

for idx, obj_path in enumerate(obj_files):
    name = os.path.basename(obj_path)
    mesh = trimesh.load(obj_path, force='mesh', process=False)

    # 计算位置偏移，让它们不重叠摆放
    x_offset = (idx % 5) * 0.3  # 每行 5 个
    y_offset = (idx // 5) * 0.4

    # 平移 mesh 顶点
    mesh.vertices += [x_offset, y_offset, 0]

    # 添加 mesh 到 viser
    server.scene.add_mesh_simple(
        name,
        mesh.vertices,
        mesh.faces,
        color=(150, 200, 255),
        opacity=0.8,
    )

    # 显示尺寸信息
    size = mesh.bounding_box.extents
    info = f"Size: x={size[0]:.3f}m y={size[1]:.3f}m z={size[2]:.3f}m"
    server.gui.add_markdown(f"### {name}\n{info}")

while True:
    time.sleep(1)