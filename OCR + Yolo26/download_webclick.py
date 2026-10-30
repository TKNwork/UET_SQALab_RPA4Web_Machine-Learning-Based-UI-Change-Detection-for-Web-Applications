#!/usr/bin/env python3
import io
import os
import json
import pandas as pd
from PIL import Image

OUTPUT_DIR = "data/webclick"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

print("Downloading WebClick dataset via Parquet stream...")

parquet_urls = [
    "https://huggingface.co/datasets/Machivelli/WebClick/resolve/main/test-00000-of-00004.parquet",
    "https://huggingface.co/datasets/Machivelli/WebClick/resolve/main/test-00001-of-00004.parquet",
    "https://huggingface.co/datasets/Machivelli/WebClick/resolve/main/test-00002-of-00004.parquet",
    "https://huggingface.co/datasets/Machivelli/WebClick/resolve/main/test-00003-of-00004.parquet"
]

annotations = []
global_idx = 0

for p_idx, url in enumerate(parquet_urls, 1):
    print(f"\n[Part {p_idx}/{len(parquet_urls)}] Downloading parquet data...")
    df = pd.read_parquet(url, engine="fastparquet")
    
    for _, row in df.iterrows():
        img_bytes = row["image"]
        instruction = str(row["instruction"])
        raw_bbox = row["bbox"]
        bbox = raw_bbox.tolist() if hasattr(raw_bbox, "tolist") else list(raw_bbox)
        bucket = str(row["bucket"])

        image_name = f"{global_idx:04d}.png"
        image_path = os.path.join(IMAGE_DIR, image_name)

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(image_path)

        annotations.append({
            "id": global_idx,
            "image": f"images/{image_name}",
            "instruction": instruction,
            "bbox": bbox,
            "bucket": bucket
        })

        if global_idx % 100 == 0:
            print(f"Saved {global_idx} samples...")

        global_idx += 1

annotation_path = os.path.join(OUTPUT_DIR, "annotations.json")
with open(annotation_path, "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("SUCCESS: WebClick dataset downloaded successfully!")
print(f"Total samples: {len(annotations)}")
print(f"Images: {IMAGE_DIR}")
print(f"Annotations: {annotation_path}")
print("=" * 60)
