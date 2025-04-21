#!/usr/bin/env python3
"""
auto_viser_capture.py
———————————————
批量加载 functional_test_mix.pt 的每个样本，
在 viser 中重建场景 → 浏览器端渲染 → Python 端抓屏 → 保存 4 K PNG。
★ 纯粹脚本文件，无需手动 GUI 操作。
"""

import os, sys, glob, time, textwrap
import numpy as np
import torch, trimesh, open3d as o3d
import imageio.v3 as iio
import viser
import imageio.v3 as iio
import math
from PIL import Image, ImageDraw, ImageFont

# ─── 路径 & 常量 ─────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRO_PATH   = "/data/zwq/code"
RESULTS_PT = "/data/zwq/code/DRO_Grasp/functional_test/functional_test_mix.pt"
OUT_ROOT   = "/data/zwq/code/DRO_Grasp/functional_test/viser_output"

WIDTH, HEIGHT = 3840, 2160        # 截图分辨率 (4K 16:9)
OBJ_COLOR  = (239, 132, 167)      # RGB (0‑255)
HAND_COLOR = (102, 192, 255)
FONT_SIZE = 80

# ─── 将 DRO 仓库加入 import 路径 ────────────────────────────
sys.path.extend([ROOT_DIR, DRO_PATH])
from DRO_Grasp.utils.hand_model import create_hand_model          # noqa

# ─── 数据加载 ───────────────────────────────────────────────
results      = torch.load(RESULTS_PT)
num_samples  = len(results)
hand_cache   = {}
os.makedirs(OUT_ROOT, exist_ok=True)

# ─── Mesh 工具 ──────────────────────────────────────────────
def center_mesh(m: trimesh.Trimesh):
    m.vertices -= (m.vertices.min(0) + m.vertices.max(0)) / 2
    return m

def get_object_mesh(objname: str, objid: str) -> trimesh.Trimesh:
    cat, sub = objname.split('+')
    patterns = [
        os.path.join(ROOT_DIR, f"data/data_urdf/object/{cat}/{sub}/{objid}.obj"),
        os.path.join(ROOT_DIR, f"data/data_urdf/object/{cat}/{sub}/{objid}.ply"),
    ]
    paths = [p for pat in patterns for p in glob.glob(pat)]
    assert len(paths) == 1, f"Mesh not found for {objname}/{objid}"
    path = paths[0]

    if path.endswith(".ply"):
        m = o3d.io.read_triangle_mesh(path)
        mesh = trimesh.Trimesh(vertices=np.asarray(m.vertices),
                               faces=np.asarray(m.triangles),
                               process=False)
    else:
        mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    return center_mesh(mesh)

def add_mesh(scene: viser.ClientHandle, name: str,
             tm: trimesh.Trimesh, rgb, alpha: float):
    scene.add_mesh_simple(
        name,
        vertices=tm.vertices.astype(np.float32),
        faces=tm.faces.astype(np.int32),
        color=rgb,          # (R,G,B) 0‑255
        opacity=alpha       # 0‑1
    )

# ─── viser 服务器 ───────────────────────────────────────────
server = viser.ViserServer(host="127.0.0.1", port=8080, show_default_gui=False)
print("🟢 服务器已启动，请在浏览器打开 http://127.0.0.1:8080")

# 等待至少一个客户端连入
while len(server.get_clients().values()) == 0:
    time.sleep(0.1)
client = next(iter(server.get_clients().values()))
print("✅ 浏览器连接成功，开始批量截屏")
time.sleep(1.5) # Wait for the first image to load

def draw_caption(img_np, text):
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except OSError: font = ImageFont.load_default()
    margin = 20
    wrap   = textwrap.fill(text, 80)
    tb_h   = FONT_SIZE*(wrap.count('\n')+1)+margin*2
    bar    = Image.new("RGBA",(img.width,tb_h),(0,0,0,180))
    img.paste(bar,(0,img.height-tb_h),bar)
    draw = ImageDraw.Draw(img)
    draw.multiline_text((margin,img.height-tb_h+margin),wrap,
                        fill=(255,255,255),font=font)
    return np.asarray(img)

results.insert(0, results[0]) # padding the list to combat the loading delay
# ─── 主循环 ────────────────────────────────────────────────
for idx, sample in enumerate(results):

    # 0) 场景初始化
    server.scene.reset()
    # server.scene.set_background(np.ones(4, dtype=np.float32))  # 纯白 RGBA

    # 1) 物体 mesh
    obj_mesh = get_object_mesh(sample["object_name"], sample["object_id"])
    add_mesh(server.scene, "object", obj_mesh, OBJ_COLOR, alpha=1.0)

    # 2) 预测手 mesh
    robot_key = sample["robot_name"].split("_")[-1] if "retarget" in sample["robot_name"] \
                else sample["robot_name"]
    if robot_key not in hand_cache:
        hand_cache[robot_key] = create_hand_model(robot_key)
    hand_model = hand_cache[robot_key]

    pred_q   = torch.from_numpy(sample["predict_q"])
    hand_tm  = hand_model.get_trimesh_q(pred_q)["visual"]
    add_mesh(server.scene, "hand", hand_tm, HAND_COLOR, alpha=0.85)

    # 3) 手动相机对准
    verts = np.vstack([obj_mesh.vertices, hand_tm.vertices])
    center = verts.mean(0)
    diag   = np.linalg.norm(verts.ptp(0))
    radius = diag * 0.5
    # —— 相机对准 & 放大 —— #
    client.camera.look_at = center.astype(np.float32)
    # 2. 从一个偏上的前方方向看过去（球面坐标，仰角约 45 度）
    # theta = math.radians(45)   # elevation / pitch
    # phi   = math.radians(45)   # azimuth / yaw
    
    # x = radius * math.cos(theta) * math.cos(phi)
    # y = radius * math.cos(theta) * math.sin(phi)
    # z = radius * math.sin(theta)
    
    # camera_position = center + np.array([x, y, z]) * 1  # 缩近一些
    # client.camera.position = camera_position.astype(np.float32)
    # print("current camera position: ", camera_position)
    
    # camera_position_dir = np.array([1, 1, 1]) # top
    
    camera_pos_dict = {
        # 'top': [np.array([0,0,0.5]), np.array([0,1,0])],
        # 'bottom': [np.array([0,0,-0.5]), np.array([0,-1,0])],
        'right': [np.array([0,0.5,0]), np.array([0,0,1])],
        # 'left': [np.array([0,-0.5,0]), np.array([0,0,1])],
        # 'front': [np.array([0.5,0,0]), np.array([0,0,1])],
        # 'back': [np.array([-0.5,0,0]), np.array([0,0,1])],
    }
    for direction, v_dir in camera_pos_dict.items():
        client.camera.position = v_dir[0] # top
    
        client.camera.up       = v_dir[1]
        client.camera.fov      = 30.0          # float 不用改类型
        
        print(direction + " current position: ", client.camera.position)
        print("current up: ",client.camera.up)

        # 4) 等一小段时间让浏览器完成渲染
        time.sleep(1.5)

        # 5) 抓屏 —— 兼容新版 & 旧版 API
        if hasattr(client, "get_render"):
            frame = client.get_render(height=HEIGHT, width=WIDTH)
        else:
            frame = client.camera.get_render(height=HEIGHT, width=WIDTH)
        frame = frame[..., :3]        # 去掉 alpha
        frame = draw_caption(frame, sample["complex_language_sentence"])

        # 6) 保存
        cat_dir = os.path.join(OUT_ROOT, sample["object_name"])
        os.makedirs(cat_dir, exist_ok=True)
        model = "k2" if len(sample.keys()) == 29 else "k3"
        fname = f"{sample['vis_sample_idx']}_{model}_{direction}.png"
        iio.imwrite(os.path.join(cat_dir, fname), frame)
        
        time.sleep(0.2) #保存一下别立刻渲染下一个

    if idx % 50 == 0:
        print(f"[{idx:>5}/{num_samples}] saved {fname}")

print("🎉 完成！所有图片已保存至:", OUT_ROOT)
server.stop()
