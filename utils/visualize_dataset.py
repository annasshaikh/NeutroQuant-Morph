import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import your custom dataloader
from dataset import NeutroQuantDataset, safe_collate
from dataset_config import cfg

splits_dir = cfg.SPLITS_DIR
output_dir = cfg.OUTPUT_DIR
def visualize_random_samples(data_root, split_file, num_samples=3, mode="normal"):
    """
    Loads random samples from the dataset and plots:
    1. The raw image
    2. The ground truth segmentation mask (Color coded)
    3. The translucent overlay blended on top of the image
    """
    # Initialize your dataset
    dataset = NeutroQuantDataset(data_root, split_file, mode=mode)
    
    # We use batch_size=1 and shuffle=True to fetch completely random single images
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=safe_collate)
    
    count = 0
    print(f"Opening visualization window for {num_samples} random samples...")
    
    for batch in loader:
        if batch is None:
            continue
            
        # Extract tensors (remove batch dimension [0])
        img_tensor = batch["image"][0]   # Shape: [3, 256, 256]
        mask_tensor = batch["mask"][0]   # Shape: [256, 256]
        labels = batch["labels"]
        drug_name = batch["drug"][0]
        file_path = batch["path"][0]
        
        # 1. Process Image: convert PyTorch [3, 256, 256] -> NumPy [256, 256, 3]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        # 2. Process Mask
        mask_np = mask_tensor.cpu().numpy()
        
        # Create a Color Overlay: Green for Cell Wall (1), Red for Nucleus Wall (2)
        color_overlay = np.zeros_like(img_uint8)
        color_overlay[mask_np == 1] = [0, 255, 0]   # Green (BGR/RGB compatible)
        color_overlay[mask_np == 2] = [255, 0, 0]   # Red
        
        # 3. Create Blended Image (70% original image, 30% colored mask)
        blended = cv2.addWeighted(img_uint8, 0.7, color_overlay, 0.3, 0)
        
        # --- Matplotlib Plotting ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # Subplot 1: Raw Image
        axes[0].imshow(img_uint8)
        axes[0].set_title(f"Original Image\nCondition: {drug_name}", fontsize=11)
        axes[0].axis("off")
        
        # Subplot 2: Color Coded Mask Only
        axes[1].imshow(color_overlay)
        axes[1].set_title("Segmentation Mask\n(Green = Cell Wall | Red = Nucleus)", fontsize=11)
        axes[1].axis("off")
        
        # Subplot 3: Translucent Overlay Blend
        axes[2].imshow(blended)
        
        # Decode classifications back to text to display in subtitle
        cls_text = []
        for k, v in labels.items():
            val = v.item()
            if k == "NUMBER_OF_LOBS":
                lobe_inv = {0: "2", 1: "3", 2: "4", 3: "5", 4: "5+", 5: "CONDENSED"}
                val_str = lobe_inv.get(val, str(val))
            else:
                val_str = "YES" if val == 1 else "NO"
            cls_text.append(f"{k}: {val_str}")
            
        # Split text into two lines for cleaner subtitle formatting
        sub_title = ", ".join(cls_text[:3]) + "\n" + ", ".join(cls_text[3:])
        axes[2].set_title(f"Visual Overlay Blend\n{sub_title}", fontsize=9, color="darkblue")
        axes[2].axis("off")
        
        plt.suptitle(f"Sample File: {file_path}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        count += 1
        if count >= num_samples:
            break

if __name__ == "__main__":
    # --- Paths Configuration ---
    DATA_ROOT = cfg.DATA_ROOT
    SPLIT_FILE = cfg.SPLITS_DIR / "train.txt"
    
    # Run visualization (Set mode="filtered" to test your filtered dataset mode)
    visualize_random_samples(DATA_ROOT, SPLIT_FILE, num_samples=3, mode="normal")