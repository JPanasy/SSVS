"""Precompute ARCADE signed distance maps for boundary loss.

Boundary loss should not calculate distance transforms inside the training loop.
This script builds one normalized signed distance map per image from the COCO
annotations and stores it as .npy files.

Output layout:
    <output-dir>/<phase>/vessel/<image_stem>.npy

Example:
    python precompute_arcade_distance_maps.py --dataroot ./datasets/arcade --phases train val
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None


def mask_from_polygon(segmentation, width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if not polygon:
            continue
        try:
            coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(coords, outline=1, fill=1)
        except (IndexError, TypeError, ValueError):
            continue
    return np.array(mask, dtype=bool)


def decode_annotation(annotation, width, height):
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, dict):
        if coco_mask is None:
            raise ImportError("RLE annotations require pycocotools.")
        return coco_mask.decode(segmentation).astype(bool)
    return mask_from_polygon(segmentation, width, height)


def vessel_category_ids(coco):
    categories = {category["id"]: category.get("name", "") for category in coco.get("categories", [])}
    stenosis_ids = {
        category_id
        for category_id, name in categories.items()
        if "stenosis" in str(name).lower()
    }
    return set(categories) - stenosis_ids


def signed_distance_map(mask, max_distance):
    mask = mask.astype(bool)
    if not np.any(mask):
        return np.ones(mask.shape, dtype=np.float32)
    if np.all(mask):
        return -np.ones(mask.shape, dtype=np.float32)

    outside_distance = distance_transform_edt(~mask)
    inside_distance = distance_transform_edt(mask)
    signed_distance = outside_distance - inside_distance
    signed_distance = np.clip(signed_distance, -max_distance, max_distance)
    signed_distance = signed_distance / float(max_distance)
    return signed_distance.astype(np.float32)


def precompute_phase(dataroot, output_dir, phase, max_distance, overwrite):
    annotation_path = dataroot / phase / "annotations" / f"{phase}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as file:
        coco = json.load(file)

    images = {image["id"]: image for image in coco.get("images", [])}
    selected_categories = vessel_category_ids(coco)
    masks = {
        image_id: np.zeros((image["height"], image["width"]), dtype=bool)
        for image_id, image in images.items()
    }

    for annotation in coco.get("annotations", []):
        if annotation.get("category_id") not in selected_categories:
            continue
        image = images.get(annotation.get("image_id"))
        if not image:
            continue
        masks[image["id"]] |= decode_annotation(annotation, image["width"], image["height"])

    phase_output = output_dir / phase / "vessel"
    phase_output.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for image_id, image in sorted(images.items()):
        output_path = phase_output / f"{Path(image['file_name']).stem}.npy"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        distance_map = signed_distance_map(masks[image_id], max_distance)
        np.save(output_path, distance_map)
        written += 1

    print(f"{phase}: wrote {written}, skipped {skipped}, output={phase_output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Precompute ARCADE signed distance maps")
    parser.add_argument("--dataroot", default="./datasets/arcade")
    parser.add_argument("--output-dir", default="", help="Defaults to <dataroot>/distance_maps")
    parser.add_argument("--phases", nargs="+", default=["train", "val"])
    parser.add_argument("--max-distance", type=float, default=32.0, help="Clip and normalize distances to +/- this many pixels")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    dataroot = Path(args.dataroot)
    output_dir = Path(args.output_dir) if args.output_dir else dataroot / "distance_maps"
    for phase in args.phases:
        precompute_phase(dataroot, output_dir, phase, args.max_distance, args.overwrite)


if __name__ == "__main__":
    main()
