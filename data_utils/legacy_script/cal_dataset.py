import torch
import pandas as pd

# 加载 .pt 文件
file_path = '/data/zwq/code/DRO-Grasp/data/OakInkDataset/oakink_dataset_w_q.pt'
data = torch.load(file_path, weights_only=True)

metadata = data['metadata']
object_counts = {}

# 遍历 metadata 中的所有样本
for sample in metadata:
    device = sample[6]          # 假设第7个元素是设备名称
    object_name = sample[5]     # 假设第6个元素是对象名称

    if object_name not in object_counts:
        object_counts[object_name] = {}

    if device not in object_counts[object_name]:
        object_counts[object_name][device] = 0

    object_counts[object_name][device] += 1

# 确定所有设备的名称以作为表格的列
devices = sorted({device for counts in object_counts.values() for device in counts})

# 添加 "Total" 列到设备列表的最前面
devices.insert(0, "Total")

# 初始化一个列表来存储所有行数据
rows = []

# 初始化总计字典
total_counts = {device: 0 for device in devices if device != "Total"}

# 填充行数据
for object_name, counts in object_counts.items():
    row = {"Object Name": object_name}
    total = 0
    for device in devices[1:]:
        count = counts.get(device, 0)
        row[device] = count
        total += count
        total_counts[device] += count
    row["Total"] = total
    rows.append(row)

# 创建 DataFrame
df = pd.DataFrame(rows, columns=["Object Name"] + devices)

# 创建总计行
total_row = {"Object Name": "**Total**"}
for device in devices[1:]:
    total_row[device] = f"**{total_counts[device]}**"
total_row["Total"] = f"**{sum(total_counts.values())}**"
df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# 将数据框转换为 Markdown 表格
markdown_table = df.to_markdown(index=False)

# 打印 Markdown 表格
print(markdown_table)

# 将表格保存到 README.md 文件中
with open('/data/zwq/code/DRO-Grasp/data/OakInkDataset/README.md', 'w') as f:
    f.write(markdown_table)