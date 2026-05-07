import os
import json
import numpy as np
from PIL import Image
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

class ARCADEDataset(BaseDataset):
    """
    ARCADE dataset loader for supervised pretraining.
    
    Loads COCO-format annotations and converts masks to binary segmentation maps.
    Supports both vessel (SYNTAX) and stenosis subsets.
    
    Directory structure expected:
    - dataroot/
        - images/  (all image files)
        - annotations/
            - train.json / val.json / test.json
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--arcade_subset', type=str, default='syntax',
                            choices=['syntax', 'stenosis'],
                            help='Which ARCADE subset to use: syntax (vessels) or stenosis')
        parser.add_argument('--arcade_mask_type', type=str, default='vessel',
                            choices=['vessel', 'stenosis', 'both'],
                            help='Which masks to generate: vessel or stenosis or both')
        return parser

    def __init__(self, opt):
        """Initialize ARCADE dataset."""
        BaseDataset.__init__(self, opt)
        
        # Build paths
        self.image_dir = os.path.join(opt.dataroot, opt.phase, 'images')
        annotations_dir = os.path.join(opt.dataroot, opt.phase, 'annotations')
        self.annotation_file = os.path.join(annotations_dir, f'{opt.phase}.json')
        
        # Load COCO annotations
        self.coco = COCO(self.annotation_file)
        self.image_ids = sorted(self.coco.getImgIds())
        
        # Filter images by dataset size if needed
        if opt.max_dataset_size > 0:
            self.image_ids = self.image_ids[:opt.max_dataset_size]
        
        # Setup transforms
        self.transform = get_transform(opt, grayscale=True)
        
        # Category mapping for vessel/stenosis
        self.category_map = {cat['id']: cat['name'] for cat in self.coco.dataset['categories']}
        self.stenosis_cat_id = None
        for cat_id, cat_name in self.category_map.items():
            if 'stenosis' in str(cat_name).lower():
                self.stenosis_cat_id = cat_id
                break

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Return paired data: (image, vessel_mask) or (image, stenosis_mask) or both.
        
        Returns:
            dict with keys:
                - 'image': Input image tensor
                - 'vessel_mask': Binary vessel segmentation mask (if mask_type includes vessel)
                - 'stenosis_mask': Binary stenosis mask (if mask_type includes stenosis)
                - 'paths': Image file path
        """
        image_id = self.image_ids[index]
        img_info = self.coco.loadImgs(image_id)[0]
        
        # Load image
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert('L')  # Grayscale
        
        # Get image dimensions
        height, width = img_info['height'], img_info['width']
        
        # Initialize mask arrays
        vessel_mask = np.zeros((height, width), dtype=np.uint8)
        stenosis_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get all annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Rasterize masks
        for ann in annotations:
            segmentation = ann['segmentation']
            category_id = ann['category_id']
            
            # Decode RLE mask
            if isinstance(segmentation, dict):
                # RLE encoded
                mask = coco_mask.decode(segmentation).astype(np.uint8)
            else:
                # Polygon format (shouldn't happen in this dataset, but handle anyway)
                mask = self._mask_from_polygon(segmentation, height, width)
            
            # Add to appropriate mask based on category
            if category_id == self.stenosis_cat_id:
                stenosis_mask = np.maximum(stenosis_mask, mask)
            else:
                # All other categories are vessel segments
                vessel_mask = np.maximum(vessel_mask, mask)
        
        # Convert to PIL Images for transforms
        image_pil = image
        vessel_mask_pil = Image.fromarray(vessel_mask, mode='L')
        stenosis_mask_pil = Image.fromarray(stenosis_mask, mode='L')
        
        # Apply transforms
        image = self.transform(image_pil)
        vessel_mask = self.transform(vessel_mask_pil)
        stenosis_mask = self.transform(stenosis_mask_pil)
        
        # Build return dict based on mask_type option
        result = {'image': image, 'paths': image_path}
        
        if self.opt.arcade_mask_type in ['vessel', 'both']:
            result['vessel_mask'] = vessel_mask
        
        if self.opt.arcade_mask_type in ['stenosis', 'both']:
            result['stenosis_mask'] = stenosis_mask
        
        return result

    @staticmethod
    def _mask_from_polygon(polygon, height, width):
        """Convert polygon coordinates to binary mask."""
        from PIL import Image, ImageDraw
        mask = Image.new('L', (width, height), 0)
        if polygon and len(polygon) > 0:
            # Assume first polygon in list
            coords = polygon[0]
            # Flatten coordinate list
            if len(coords) > 0:
                try:
                    coords_list = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    ImageDraw.Draw(mask).polygon(coords_list, outline=1, fill=1)
                except:
                    pass
        return np.array(mask, dtype=np.uint8)
