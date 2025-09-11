import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

# === COCO Category ID ? Class Name from arXiv paper ===
class_names = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
    47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

# === Load model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()
transform = transforms.Compose([transforms.ToTensor()])
base_dir = os.getcwd()
# === Root folder with STC train sequences ===
root_folder = f'{base_dir}/shanghaitech/train'
sequence_folders = sorted([f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))])

unique_classes = set()

for seq_folder in sequence_folders:
    image_folder = os.path.join(root_folder, seq_folder)
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
    if not image_files:
        print(f"?? Skipping {seq_folder}: No frames found.")
        continue

    print(f"?? Processing {seq_folder} ({len(image_files)} frames)...")
    records = []

    for fname in image_files:
        try:
            frame_id = int(os.path.splitext(fname)[0])
        except:
            continue
        img_path = os.path.join(image_folder, fname)

        with Image.open(img_path).convert("RGB") as img:
            img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)[0]

        for i in range(len(output['boxes'])):
            score = output['scores'][i].item()
            if score < 0.5:
                continue
            class_id = output['labels'][i].item()
            class_name = class_names.get(class_id)
            if class_name is None:
                continue

            x1, y1, x2, y2 = output['boxes'][i].cpu().numpy()
            records.append({
                'frame': frame_id,
                'track_id': -1,
                'class_id': class_id,
                'class': class_name,
                'score': round(score, 4),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2
            })
            unique_classes.add(class_name)

    if records:
        df = pd.DataFrame(records).sort_values(by='frame')
        df.to_csv(os.path.join(image_folder, 'tracks.csv'), index=False)
        print(f"? Saved: {os.path.join(image_folder, 'tracks.csv')}")
    else:
        print(f"?? No detections in {seq_folder}")

# === Save unique class list to TXT ===
sorted_classes = sorted(unique_classes)
txt_path = os.path.join(root_folder, 'unique_classes.txt')
with open(txt_path, 'w') as f:
    for c in sorted_classes:
        f.write(c + '\n')
print(f"\n?? Unique classes saved to {txt_path}")
print(f"?? Classes found: {sorted_classes}")
