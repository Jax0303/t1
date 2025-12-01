"""
Script to download and prepare TableBank and DocLayNet datasets.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def download_doclaynet():
    """Download DocLayNet using HuggingFace datasets."""
    try:
        from datasets import load_dataset
        print("Downloading DocLayNet dataset (this may take a while)...")
        
        # Download only train split with a small subset for testing
        dataset = load_dataset(
            "data/raw/DocLayNet/DocLayNet.py",
            split="train[:100]",  # Only first 100 samples for testing
            trust_remote_code=True
        )
        
        print(f"Downloaded {len(dataset)} samples")
        print(f"Sample keys: {dataset[0].keys()}")
        
        # Save to disk
        output_dir = Path("data/raw/DocLayNet/downloaded")
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir / "train_subset"))
        
        print(f"Saved to {output_dir / 'train_subset'}")
        return dataset
        
    except Exception as e:
        print(f"Error downloading DocLayNet: {e}")
        return None

def download_tablebank():
    """Download TableBank using HuggingFace datasets."""
    try:
        from datasets import load_dataset
        print("Downloading TableBank dataset (this may take a while)...")
        
        # TableBank is very large, download a small subset
        dataset = load_dataset(
            "liminghao1630/TableBank",
            split="train[:100]",  # Only first 100 samples
            trust_remote_code=True
        )
        
        print(f"Downloaded {len(dataset)} samples")
        print(f"Sample keys: {dataset[0].keys()}")
        
        # Save to disk
        output_dir = Path("data/raw/TableBank/downloaded")
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir / "train_subset"))
        
        print(f"Saved to {output_dir / 'train_subset'}")
        return dataset
        
    except Exception as e:
        print(f"Error downloading TableBank: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Downloading datasets...")
    print("=" * 60)
    
    doclaynet = download_doclaynet()
    print()
    tablebank = download_tablebank()
    
    print("\nDownload complete!")
