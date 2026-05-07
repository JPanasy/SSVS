#!/usr/bin/env python
"""
Setup script for ARCADE transfer learning data.

This script creates a symbolic link from the ARCADE data source to the local
datasets folder, making it easy to train with arcade_supervision model.

Usage:
    python setup_arcade_data.py
    python setup_arcade_data.py --arcade-root "C:\path\to\ARCADE" --subset syntax
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

def setup_arcade_symlink(arcade_root, subset, target_name='arcade'):
    """Create symlink to ARCADE data."""
    
    arcade_path = Path(arcade_root) / subset
    target_path = Path('./datasets') / target_name
    
    # Verify source exists
    if not arcade_path.exists():
        print(f"❌ Error: ARCADE source not found at {arcade_path}")
        print(f"   Please verify the path exists and try again")
        return False
    
    # Check required structure
    required_dirs = ['train/images', 'train/annotations', 'val', 'test']
    for req_dir in required_dirs[:2]:  # Check at least train structure
        if not (arcade_path / req_dir).exists():
            print(f"❌ Error: Expected directory not found: {arcade_path / req_dir}")
            return False
    
    print(f"✓ Source verified: {arcade_path}")
    
    # Remove existing symlink/directory
    if target_path.exists():
        if target_path.is_symlink():
            print(f"Removing existing symlink: {target_path}")
            target_path.unlink()
        else:
            print(f"⚠  Warning: Directory {target_path} already exists (not symlink)")
            response = input(f"Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted")
                return False
            shutil.rmtree(target_path)
    
    # Create symlink
    try:
        if sys.platform == 'win32':
            # Windows requires different handling
            import ctypes
            kernel32 = ctypes.windll.kernel32
            
            # Try to create junction (works without admin on Windows 10+)
            result = kernel32.CreateSymbolicLinkW(
                str(target_path), 
                str(arcade_path), 
                1  # 1 = directory, 0 = file
            )
            
            if result:
                print(f"✓ Created symlink: {target_path} -> {arcade_path}")
            else:
                # Fall back to copying (slower but always works)
                print(f"⚠  Symlink failed (admin required). Copying files instead...")
                print(f"   This may take a minute...")
                shutil.copytree(arcade_path, target_path, dirs_exist_ok=True)
                print(f"✓ Copied ARCADE data to: {target_path}")
        else:
            # Linux/Mac
            os.symlink(arcade_path, target_path)
            print(f"✓ Created symlink: {target_path} -> {arcade_path}")
    
    except Exception as e:
        print(f"❌ Error creating symlink: {e}")
        return False
    
    # Verify
    count_images = len(list((target_path / 'train' / 'images').glob('*.png')))
    print(f"✓ Verified: {count_images} training images found")
    
    print(f"\n✅ Setup complete!")
    print(f"\nYou can now train with:")
    print(f"  python train.py \\")
    print(f"    --dataroot ./datasets/{target_name} \\")
    print(f"    --model arcade_supervision \\")
    print(f"    --dataset_mode arcade")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Setup ARCADE data for transfer learning')
    parser.add_argument('--arcade-root', type=str, 
                        default=r'C:\monai-projects\vascular_proto\data_raw\ARCADE',
                        help='Root path to ARCADE dataset')
    parser.add_argument('--subset', type=str, default='syntax', choices=['syntax', 'stenosis'],
                        help='Which ARCADE subset to use')
    parser.add_argument('--target-name', type=str, default='arcade',
                        help='Name for symlink in ./datasets/')
    
    args = parser.parse_args()
    
    # Ensure datasets directory exists
    Path('./datasets').mkdir(exist_ok=True)
    
    success = setup_arcade_symlink(
        args.arcade_root,
        args.subset,
        args.target_name
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
