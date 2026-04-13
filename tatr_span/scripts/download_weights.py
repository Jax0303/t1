"""
Download TATR pretrained weights from HuggingFace.
====================================================
bsmock/tatr-pubtables1m-v1.0 → weights/pubtables1m_structure_detr_r18.pth

실행:
  python tatr_span/scripts/download_weights.py
  python tatr_span/scripts/download_weights.py --out-dir /mnt/d/weights
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


HF_REPO_ID  = "bsmock/tatr-pubtables1m-v1.0"
WEIGHT_FILE = "pubtables1m_structure_detr_r18.pth"
DEFAULT_OUT = Path(__file__).parents[2] / "weights"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / WEIGHT_FILE

    if dest.exists():
        print(f"[weights] already exists: {dest}")
        return dest

    print(f"[weights] downloading {HF_REPO_ID}/{WEIGHT_FILE} …")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=WEIGHT_FILE,
            local_dir=str(out_dir),
        )
        print(f"[weights] saved → {path}")
        print(f"[weights] sha256: {sha256(Path(path))}")
        return Path(path)
    except ImportError:
        raise SystemExit(
            "huggingface_hub 미설치: pip install huggingface_hub"
        )
    except Exception as e:
        raise SystemExit(f"[weights] 다운로드 실패: {e}")


def main():
    parser = argparse.ArgumentParser("Download TATR pretrained weights")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT),
                        help=f"저장 디렉토리 (기본: {DEFAULT_OUT})")
    args = parser.parse_args()
    download(Path(args.out_dir))


if __name__ == "__main__":
    main()
