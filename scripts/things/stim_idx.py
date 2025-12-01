#!/usr/bin/env python3
"""Build CSV files mapping concept/image names to IDs for the THINGS dataset."""

import argparse
import csv
from pathlib import Path


def build_metadata(things_dir: Path) -> None:
    image_dir = things_dir / "object_images"
    if not image_dir.exists():
        raise FileNotFoundError(f"{image_dir} does not exist")

    concept_names = sorted([p.name for p in image_dir.iterdir() if p.is_dir()])
    concept_to_idx = {cls: idx for idx, cls in enumerate(concept_names)}

    concept_csv = things_dir / "concept_to_idx.csv"
    with concept_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "concept_idx"])
        for cls, idx in concept_to_idx.items():
            writer.writerow([cls, idx])

    image_csv = things_dir / "image_to_idx.csv"
    with image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "concept_name", "image_idx", "concept_idx"])
        image_idx = 0
        for cls in concept_names:
            concept_idx = concept_to_idx[cls]
            for img_path in sorted((image_dir / cls).iterdir()):
                if not img_path.is_file():
                    continue
                writer.writerow([img_path.stem, cls, image_idx, concept_idx])
                image_idx += 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CSV mappings of THINGS conceptes/images to IDs.")
    parser.add_argument(
        "--things-dir",
        type=Path,
        default=Path("/home/acg17270jl/projects/brain-multimodal-vae/data/things"),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_metadata(args.things_dir)
