#!/usr/bin/env python3
"""
Standardize all WebClick perception JSON files to the unified compact schema:
- element_id: integer (0..N-1)
- parent: integer or None
- width, height: image dimensions
- bbox: [x1, y1, x2, y2]
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_webclick_perception import validate_sample_completion

def standardize_sample(output_dir: Path, images_dir: Path, stem: str) -> bool:
    sample_dir = output_dir / stem
    full_path = sample_dir / f"{stem}_ui.json"
    compact_path = sample_dir / f"{stem}_ui_compact.json"
    img_path = images_dir / f"{stem}.png"

    if not full_path.exists() or not compact_path.exists() or not img_path.exists():
        return False

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        with open(compact_path, "r", encoding="utf-8") as f:
            compact_data = json.load(f)

        # Dimension
        img_w = compact_data.get("width")
        img_h = compact_data.get("height")

        if not img_w or not img_h:
            if isinstance(full_data.get("image"), dict):
                img_w = full_data["image"].get("width")
                img_h = full_data["image"].get("height")

        if not img_w or not img_h:
            with Image.open(img_path) as img:
                img_w, img_h = img.size

        # Standardize compact elements
        raw_elements = compact_data.get("elements", [])
        new_compact_elements = []

        for idx, elem in enumerate(raw_elements):
            # element_id
            eid = elem.get("element_id")
            if eid is None and "id" in elem:
                id_str = str(elem["id"])
                if id_str.startswith("fused_"):
                    eid = int(id_str.replace("fused_", ""))
                elif id_str.isdigit():
                    eid = int(id_str)
                else:
                    eid = idx

            # parent
            parent = elem.get("parent")
            if isinstance(parent, str):
                if parent.startswith("fused_"):
                    parent = int(parent.replace("fused_", ""))
                elif parent.isdigit():
                    parent = int(parent)
                else:
                    parent = None
            elif not isinstance(parent, int):
                parent = None

            # bbox
            bbox = elem.get("bbox", [])

            new_compact_elements.append({
                "element_id": eid,
                "text": elem.get("text"),
                "class_label": elem.get("class_label"),
                "bbox": bbox,
                "parent": parent
            })

        # Ensure elements are sorted by element_id
        new_compact_elements.sort(key=lambda x: x["element_id"])

        # Re-index if not continuous 0..N-1
        eids = [e["element_id"] for e in new_compact_elements]
        if eids != list(range(len(new_compact_elements))):
            id_map = {old_id: new_id for new_id, old_id in enumerate(eids)}
            for e in new_compact_elements:
                e["element_id"] = id_map[e["element_id"]]
                if e["parent"] in id_map:
                    e["parent"] = id_map[e["parent"]]
                else:
                    e["parent"] = None

        new_compact_data = {
            "image": f"{stem}.png",
            "width": img_w,
            "height": img_h,
            "elements": new_compact_elements
        }

        # Update full_data element_id and parent as well
        if isinstance(full_data.get("elements"), list):
            for idx, elem in enumerate(full_data["elements"]):
                if "element_id" not in elem:
                    if "id" in elem and str(elem["id"]).startswith("fused_"):
                        elem["element_id"] = int(str(elem["id"]).replace("fused_", ""))
                    else:
                        elem["element_id"] = idx
                
                # primary_parent
                pp = elem.get("primary_parent")
                if isinstance(pp, str) and pp.startswith("fused_"):
                    elem["primary_parent"] = int(pp.replace("fused_", ""))

        # Save back JSON files
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)
        with open(compact_path, "w", encoding="utf-8") as f:
            json.dump(new_compact_data, f, ensure_ascii=False, indent=2)

        return validate_sample_completion(sample_dir, stem)
    except Exception as e:
        print(f"Error standardizing {stem}: {e}")
        return False

def main():
    output_dir = PROJECT_ROOT / "output_webclick_perception"
    images_dir = PROJECT_ROOT / "data/webclick/images"

    image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix == ".png"])
    total = len(image_files)

    print(f"Standardizing perception JSON schemas for {total} samples...")
    success_count = 0
    fail_count = 0
    failed_stems = []

    for f in image_files:
        stem = f.stem
        if standardize_sample(output_dir, images_dir, stem):
            success_count += 1
        else:
            fail_count += 1
            failed_stems.append(stem)

    print("\n" + "=" * 60)
    print("STANDARDIZATION & VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total samples : {total}")
    print(f"Complete      : {success_count}")
    print(f"Failed/Missing: {fail_count}")
    if failed_stems:
        print(f"Failed IDs    : {failed_stems[:20]}")
    print("=" * 60)

if __name__ == "__main__":
    main()
