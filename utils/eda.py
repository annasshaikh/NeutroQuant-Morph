import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

# Ensure you are importing from your local dataset.py
from dataset import NeutroQuantDataset, safe_collate

def run_eda():
    data_root = Path(r"D:\IBA\FYP\Dataset\QA Dataset Final\FINAL V1_Cleaned\Consolidated_Dataset_By_Drug")
    splits_dir = Path(r"D:\IBA\FYP\Dataset\QA Dataset Final\FINAL V1_Cleaned\Consolidated_Dataset_By_Drug\Dataset_Splits")
    output_dir = Path(r"D:\IBA\FYP\Dataset\QA Dataset Final\FINAL V1_Cleaned")
    
    # Create output dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"Warning: {split_file} not found.")
            continue
            
        ds = NeutroQuantDataset(data_root, split_file,mode="filtered")
        loader = DataLoader(ds, batch_size=1, collate_fn=safe_collate)
        
        print(f"Processing {split} split...")
        for batch in loader:
            if batch is None: continue
            
            # Record data
            record = {"split": split, "drug": batch["drug"][0]}
            for key, val in batch["labels"].items():
                record[key] = val.item()
            all_records.append(record)
            
    if not all_records:
        print("No data loaded. Check paths and logs.")
        return

    df = pd.DataFrame(all_records)
    
    # --- VISUALIZATIONS ---
    cols_to_plot = ["drug"] + [c for c in df.columns if c not in ["split", "drug"]]
    num_plots = len(cols_to_plot)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))
    if num_plots == 1: axes = [axes]
    
    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        sns.countplot(data=df, x=col, hue="split", ax=ax, palette="viridis")
        ax.set_title(f"Distribution of {col.upper()} by Split")
        
        if col == "drug":
            ax.tick_params(axis='x', rotation=45)
            
        # --- NEW: Write numbers on top of each bar ---
        for p in ax.patches:
            height = p.get_height()
            # Seaborn sometimes creates empty patches (NaN) for missing hue categories
            if pd.notnull(height) and height > 0:
                ax.annotate(f'{int(height)}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', 
                            fontsize=9, color='black', 
                            xytext=(0, 3), # 3 points vertical offset
                            textcoords='offset points')

    # tight_layout ensures labels don't get cut off when saving
    plt.tight_layout()
            
    # Save to the specific output directory
    save_path = output_dir / "dataset_eda_distributions.png"
    plt.savefig(save_path, bbox_inches='tight')
    
    print(f"\nEDA Complete. Saved plot to: {save_path}")
    
    # Show the plot window (Note: Script will pause here until you close the window)
    plt.show()

if __name__ == "__main__":
    run_eda()