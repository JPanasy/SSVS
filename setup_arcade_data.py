#!/usr/bin/env python
r"""Prepare ARCADE data for transfer learning.

This script creates a symlink from an ARCADE subset into ./datasets so the
training command can use --dataroot ./datasets/arcade.

Usage:
    python setup_arcade_data.py --arcade-root "C:\path\to\ARCADE"
    python setup_arcade_data.py --arcade-root "C:\path\to\ARCADE" --subset syntax
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


DEFAULT_ARCADE_ROOT = os.environ.get(
    "ARCADE_ROOT",
    r"C:\monai-projects\vascular_proto\data_raw\ARCADE",
)


def setup_arcade_data(arcade_root, subset, target_name="arcade", copy_data=False, force=False):
    """Create a symlink or copy for an ARCADE subset under ./datasets."""
    arcade_path = Path(arcade_root) / subset
    target_path = Path("./datasets") / target_name

    if not arcade_path.exists():
        print(f"ERROR: ARCADE source not found at {arcade_path}")
        print("       Set --arcade-root or the ARCADE_ROOT environment variable.")
        return False

    required_dirs = ["train/images", "train/annotations"]
    for required_dir in required_dirs:
        required_path = arcade_path / required_dir
        if not required_path.exists():
            print(f"ERROR: Expected directory not found: {required_path}")
            return False

    print(f"OK: Source verified: {arcade_path}")

    if target_path.exists():
        if target_path.is_symlink():
            print(f"Removing existing symlink: {target_path}")
            target_path.unlink()
        else:
            print(f"WARNING: Directory {target_path} already exists and is not a symlink.")
            response = "y" if force else input("Overwrite? (y/n): ").strip().lower()
            if response != "y":
                print("Aborted")
                return False
            shutil.rmtree(target_path)

    try:
        if copy_data:
            print("Copying ARCADE data. This may take a while...")
            shutil.copytree(arcade_path, target_path, dirs_exist_ok=True)
            print(f"OK: Copied ARCADE data to: {target_path}")
        else:
            target_path.symlink_to(arcade_path, target_is_directory=True)
            print(f"OK: Created symlink: {target_path} -> {arcade_path}")
    except Exception as exc:
        print(f"ERROR: Could not prepare ARCADE data: {exc}")
        if not copy_data:
            print("       Retry with --copy if symlink creation is unavailable.")
        return False

    count_images = len(list((target_path / "train" / "images").glob("*.png")))
    print(f"OK: Verified {count_images} training images found")

    print("\nSetup complete.")
    print("\nYou can now train with:")
    print("  python train.py \\")
    print(f"    --dataroot ./datasets/{target_name} \\")
    print("    --model arcade_supervision \\")
    print("    --dataset_mode arcade")

    return True


def main():
    parser = argparse.ArgumentParser(description="Setup ARCADE data for transfer learning")
    parser.add_argument(
        "--arcade-root",
        type=str,
        default=DEFAULT_ARCADE_ROOT,
        help="Root path to ARCADE dataset, or set ARCADE_ROOT",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="syntax",
        choices=["syntax", "stenosis"],
        help="Which ARCADE subset to use",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="arcade",
        help="Name for symlink in ./datasets/",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy ARCADE files instead of creating a symlink",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing non-symlink target without prompting",
    )

    args = parser.parse_args()
    Path("./datasets").mkdir(exist_ok=True)

    success = setup_arcade_data(
        args.arcade_root,
        args.subset,
        args.target_name,
        copy_data=args.copy,
        force=args.force,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
