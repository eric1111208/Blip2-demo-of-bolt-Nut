import multiprocessing
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from peft import PeftModel, PeftConfig
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO

# ========== 配置 ==========

yolo_model_path = "/home/eric/PycharmProjects/PythonProject2/multimodal_project/runs/detect/bolt_yolo11_train12/weights/best.pt"  # Adjust to your YOLO model path
blip_model_path = "/home/eric/PycharmProjects/PythonProject2/multimodal_project/data3/blip2_lora_output/checkpoint-60"  # Adjusted path
camera_index = 2  # Adjust to your webcam index
conf_threshold = 0.5
speak_interval = 3  # Seconds between same caption
HAND_WINDOW_NAME = "Hand Detected"
max_new_tokens = 32  # Reduced for memory

# 状态缓存
last_spoken = {}
last_caption_time = 0


# ========== 语音播报子进程 ==========
def speaker_process(q):
    while True:
        text = q.get()
        if text is None:
            break
        os.system(f'espeak "{text}" >/dev/null 2>&1')


speak_q = multiprocessing.Queue()
speak_proc = multiprocessing.Process(target=speaker_process, args=(speak_q,))
speak_proc.start()

# ========== 模型加载 ==========
print("\U0001F4E6 加载 YOLOv11模型...")
yolo_model = YOLO(yolo_model_path)

print("\U0001F9E0 加载 BLIP-2 模型...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)

try:
    # Validate checkpoint directory
    if not os.path.exists(blip_model_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {blip_model_path}")
    if not os.path.exists(os.path.join(blip_model_path, "adapter_config.json")):
        raise FileNotFoundError(f"adapter_config.json not found in {blip_model_path}")

    peft_config = PeftConfig.from_pretrained(blip_model_path)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        # torch_dtype=torch.float32,  # Use float32 for CPU
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    model = PeftModel.from_pretrained(base_model, blip_model_path).to("cuda" if torch.cuda.is_available() else "cpu")

    print("✅ LoRA 模型加载成功")
except (FileNotFoundError, ValueError) as e:
    print(f"❌ 无法加载 LoRA 模型: {str(e)}")
    print("🔄 使用预训练 BLIP-2 模型作为后备...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,  # Use float32 for CPU
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 预训练模型加载成功")

# Memory management
torch.cuda.empty_cache()  # Clear any residual GPU memory
# device = torch.device("cpu")  # Explicitly set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

peft_config = PeftConfig.from_pretrained(blip_model_path)
base_model = Blip2ForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
).to(device)
model = PeftModel.from_pretrained(base_model, blip_model_path).to(device)

# ========== 摄像头初始化 ==========
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("\U0001F3A5 正在运行：按 'q' 键退出")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    results = yolo_model.predict(source=frame, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    # 绘制检测框
    if boxes is not None and len(boxes.cls) > 0:
        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i])
            class_name = yolo_model.names[cls_id]
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), font, 0.6, (255, 255, 0), 2)

            # 检测手并生成字幕
            # if class_name == "hand":
            if True:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    hand_crop = frame[y1:y2, x1:x2].copy()
                    gray_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray_crop)

                    if brightness > 10:
                        try:
                            pil_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                            inputs = processor(images=pil_image, return_tensors="pt").to(device)
                            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
                            caption = processor.decode(out[0], skip_special_tokens=True)
                        except Exception as e:
                            caption = "BLIP-2 推理失败"
                            print("推理失败：", str(e))

                        cv2.putText(frame, caption, (x1, y2 + 25), font, 0.6, (0, 255, 255), 2)

                        caption_img = np.ones((200, 1000, 3), dtype=np.uint8) * 255
                        cv2.putText(caption_img, caption, (30, 120), font, 2, (255, 0, 0), 5)
                        cv2.imshow("Caption Text", caption_img)

                        if now - last_spoken.get("caption", 0) > speak_interval:
                            speak_q.put(caption)
                            last_spoken["caption"] = now

                        resized = cv2.resize(hand_crop, (0, 0), fx=0.5, fy=0.5)
                        cv2.imshow(HAND_WINDOW_NAME, resized)
                        print("📢 Caption:", caption)

    cv2.imshow("YOLO + BLIP-2 Caption", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
speak_q.put(None)
speak_proc.join()
print("已退出程序")
