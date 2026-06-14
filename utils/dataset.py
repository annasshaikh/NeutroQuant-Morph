import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json
import cv2
import numpy as np
from pathlib import Path

class NeutroQuantDataset(Dataset):
    def __init__(self, data_root, split_file, mode="normal"):
        self.data_root = Path(data_root)
        self.mode = mode
        
        if self.mode not in ["normal", "filtered"]:
            raise ValueError("Mode must be either 'normal' or 'filtered'.")
            
        with open(split_file, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        img_path = self.data_root / rel_path
        
        # Handle potential .tif vs .tiff extension mismatches
        if not img_path.exists():
            if img_path.suffix.lower() == '.tif':
                img_path = img_path.with_suffix('.tiff')
            elif img_path.suffix.lower() == '.tiff':
                img_path = img_path.with_suffix('.tif')
                
        ann_path = img_path.parent.parent / "annotations" / img_path.with_suffix(".json").name
        drug_name = Path(rel_path).parts[0] 
        
        # 1. Load Image in True Color (BGR)
        image_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_raw is None:
            print(f"Error: {rel_path} - Image load failed")
            return None
            
        # Convert BGR (OpenCV default) to RGB (Matplotlib/PyTorch default)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_raw.shape[:2] # Get dimensions
            
        # Resize to 256x256
        image = cv2.resize(image_raw, (256, 256))
        
        # Normalize and convert to PyTorch Shape [Channels, Height, Width] -> [3, 256, 256]
        image = image.astype(np.float32) / 255.0
        img_tensor = torch.tensor(image).permute(2, 0, 1)
        
        # 2. Load JSON
        if not ann_path.exists():
            print(f"Error: {rel_path} - Annotation file missing")
            return None
            
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_cls = data.get("classifications", {})
            
            # --- FILTERED MODE LOGIC ---
            if self.mode == "filtered":
                if raw_cls.get("NUMBER_OF_LOBS") == "CONDENSED":
                    return None
            
            # 3. Create Segmentation Mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            for ann in data.get("annotations", []):
                pts_list = ann.get("coordinates", {}).get("points", [])
                if not pts_list:
                    continue
                
                scale_x, scale_y = 256 / orig_w, 256 / orig_h
                pts = np.array(pts_list, np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = np.int32(pts)
                
                val = 1 if "cell" in ann.get("annotationType", "").lower() else 2
                cv2.fillPoly(mask, [pts], val)
            
            # 4. Classification Mapping
            labels = {}
            for k, v in raw_cls.items():
                if self.mode == "filtered" and k == "NUCLEAR_SHAPE":
                    continue 
                    
                if k == "NUMBER_OF_LOBS":
                    lobe_map = {"2": 0, "3": 1, "4": 2, "5": 3, "5+": 4, "CONDENSED": 5}
                    labels[k] = lobe_map.get(str(v), 0)
                else:
                    labels[k] = 1 if v in ["YES", "IRREGULAR", "LOBED"] else 0
            
            return {
                "image": img_tensor, 
                "mask": torch.tensor(mask).long(), 
                "labels": labels, 
                "drug": drug_name,
                "path": rel_path
            }

        except Exception as e:
            print(f"Error: {rel_path} - Unexpected error: {str(e)}")
            return None

def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return default_collate(batch)