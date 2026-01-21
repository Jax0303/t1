import os
import tarfile
from pathlib import Path

DATA_ROOT = Path("/home/user/t1-4/data/external")

def extract_tar(tar_path, extract_path):
    print(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Extracted to {extract_path}")
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")

def process_fintabnet():
    src = DATA_ROOT / "fintabnet_c" / "FinTabNet.c-Structure.tar.gz"
    if src.exists():
        extract_tar(src, DATA_ROOT / "fintabnet_c" / "extracted")

def process_pubtables():
    for split in ["Train", "Val", "Test"]:
        src = DATA_ROOT / "pubtables-1m" / f"PubTables-1M-Structure_Annotations_{split}.tar.gz"
        if src.exists():
            extract_tar(src, DATA_ROOT / "pubtables-1m" / "extracted" / split.lower())

def process_scitsr():
    src = DATA_ROOT / "scitsr" / "SciTsr_Logical.tar.gz"
    if src.exists():
        extract_tar(src, DATA_ROOT / "scitsr" / "extracted")

if __name__ == "__main__":
    process_fintabnet()
    process_pubtables()
    process_scitsr()
    print("\nBenchmark extraction complete.")
