#!/usr/bin/env python3
"""Build CSV files mapping class/image names to IDs for the THINGS dataset."""

import argparse
import csv
from pathlib import Path


def build_metadata(things_dir: Path) -> None:
    image_dir = things_dir / "object_images"
    if not image_dir.exists():
        raise FileNotFoundError(f"{image_dir} does not exist")

    class_names = sorted([p.name for p in image_dir.iterdir() if p.is_dir()])
    class_to_id = {cls: idx for idx, cls in enumerate(class_names)}

    class_csv = things_dir / "class_to_id.csv"
    with class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "class_id"])
        for cls, idx in class_to_id.items():
            writer.writerow([cls, idx])

    image_csv = things_dir / "image_to_id.csv"
    with image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "class_name", "image_id", "class_id"])
        image_id = 0
        for cls in class_names:
            class_id = class_to_id[cls]
            for img_path in sorted((image_dir / cls).iterdir()):
                if not img_path.is_file():
                    continue
                writer.writerow([img_path.stem, cls, image_id, class_id])
                image_id += 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CSV mappings of THINGS classes/images to IDs.")
    parser.add_argument(
        "--things-dir",
        type=Path,
        default=Path("/home/acg17270jl/projects/brain-multimodal-vae/data/things"),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_metadata(args.things_dir)
