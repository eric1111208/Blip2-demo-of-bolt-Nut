import json
import os

# === 定义方位判断函数 ===
def get_position(x, y):
    if y < 0.33:
        vertical = "top"
    elif y > 0.66:
        vertical = "bottom"
    else:
        vertical = "center"

    if x < 0.33:
        horizontal = "left"
    elif x > 0.66:
        horizontal = "right"
    else:
        horizontal = "center"

    if vertical == "center" and horizontal == "center":
        return "in the center"
    return f"in the {vertical}-{horizontal} region"

# === 输入输出路径 ===
input_path = "/content/drive/MyDrive/project-ai/data3/blip2_training_clean_data3.jsonl"
output_path = "/content/drive/MyDrive/project-ai/data3/blip2_grounded_caption.jsonl"

# === 处理并写入新 JSONL ===
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line)
        boxes = item.get("bboxes", [])
        parts = []

        for b in boxes:
            obj_class = b.get("class", "unknown object").replace("_", " ")
            pos = get_position(b["x"], b["y"])
            parts.append(f"a {obj_class} located {pos}")

        # 合并描述
        grounded_caption = "In the image, there is " + ", and ".join(parts) + "."

        new_item = {
            "image": item["image"],
            "caption": grounded_caption,
            "bboxes": boxes  # 可保留 bbox 信息供未来使用
        }
        outfile.write(json.dumps(new_item) + "\n")

print("✅ Finished. Saved to:", output_path)