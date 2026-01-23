import os
import subprocess
from pathlib import Path

DATA_ROOT = Path("/root/t1-9/data/external")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {cmd}: {e}")

def download_hf_dataset(repo_id, folder_name, include_glob=None):
    print(f"Downloading {repo_id} to {folder_name}...")
    target_dir = DATA_ROOT / folder_name
    target_dir.mkdir(exist_ok=True)
    cmd = f"huggingface-cli download {repo_id} --local-dir {target_dir} --repo-type dataset"
    if include_glob:
        cmd += f" --include '{include_glob}'"
    run_cmd(cmd)

def download_pubtables_1m():
    # Only structural annotations to minimize initial payload (Fixed Pattern)
    download_hf_dataset("bsmock/pubtables-1m", "pubtables-1m", "PubTables-1M-Structure_Annotations_*.tar.gz")

def download_fintabnet_c():
    # Already successfully downloaded (~3.2GB)
    pass

def download_wtw():
    # Mirror found: johannssky/wtw_dataset
    download_hf_dataset("johannssky/wtw_dataset", "wtw")

def download_scitsr():
    # Mirror found: saeed11b95/SciTsr-Logical
    download_hf_dataset("saeed11b95/SciTsr-Logical", "scitsr")

def download_ctdar2019():
    # Mirror found: byh711/ICDAR2019_cTDaR_TRACKA
    download_hf_dataset("byh711/ICDAR2019_cTDaR_TRACKA", "ctdar2019")

if __name__ == "__main__":
    download_pubtables_1m()
    download_wtw()
    download_scitsr()
    download_ctdar2019()
    print("\nAll benchmark downloads initiated with corrected paths.")
