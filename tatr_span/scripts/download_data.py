"""
Dataset Download Helper
=========================
지원 데이터셋:
  1. PubTables-1M (structure recognition, ~120GB)
     - Microsoft Research 공식: https://msropendata.com/datasets/
     - 스크립트: azcopy 또는 wget
  2. FinTabNet.c (HuggingFace, bsmock/FinTabNet.c)
     - HF datasets 라이브러리로 자동 다운로드
  3. SciTSR (GitHub tarball)

실행:
  python tatr_span/scripts/download_data.py --dataset pubtables1m --out-dir /mnt/d/pubtables1m
  python tatr_span/scripts/download_data.py --dataset fintabnet_c --cache-dir /mnt/d/hf_cache
  python tatr_span/scripts/download_data.py --dataset scitsr --out-dir /data/SciTSR
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# PubTables-1M
# ─────────────────────────────────────────────────────────────

PUBTABLES1M_README = """
PubTables-1M (~120GB) 수동 다운로드 안내
=========================================
Microsoft Research Open Data에서 제공하며 직접 API 다운로드가 제한될 수 있습니다.

방법 1) Hugging Face Mirror (권장):
  pip install huggingface_hub
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download(
      repo_id='bsmock/pubtables-1m',
      repo_type='dataset',
      local_dir='{out_dir}',
  )
  "

방법 2) Microsoft Research Open Data (AzCopy):
  https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3
  → AzCopy URL 획득 후:
  azcopy copy '<sas_url>' '{out_dir}' --recursive

예상 구조:
  {out_dir}/
    images/    *.png (또는 분할 tarball)
    xml/       *.xml (PASCAL VOC)
    splits/
      train.txt  val.txt  test.txt
"""

FINTABNET_C_README = """
FinTabNet.c (HuggingFace bsmock/FinTabNet.c) 다운로드
======================================================
datasets 라이브러리로 자동 다운로드됩니다.

  from datasets import load_dataset
  ds = load_dataset('bsmock/FinTabNet.c', cache_dir='{cache_dir}')

또는 이 스크립트:
  python tatr_span/scripts/download_data.py --dataset fintabnet_c --cache-dir {cache_dir}
"""

SCITSR_URL = "https://github.com/Academic-Hammer/SciTSR/archive/refs/heads/master.zip"
SCITSR_README = """
SciTSR GitHub 다운로드
========================
  wget {url} -O scitsr.zip
  unzip scitsr.zip
  mv SciTSR-master {out_dir}

SciTSR-COMP.list 위치: {out_dir}/SciTSR-COMP.list
이미지:     {out_dir}/images/
어노테이션: {out_dir}/structure/
"""


# ─────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────

def download_pubtables1m(out_dir: Path):
    print(PUBTABLES1M_README.format(out_dir=out_dir))

    # HuggingFace snapshot_download 시도
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[error] huggingface_hub 미설치: pip install huggingface_hub")
        return

    print(f"[pubtables1m] HuggingFace snapshot_download → {out_dir}")
    print("[pubtables1m] 약 120GB — 충분한 공간 확보 후 실행하세요.")
    confirm = input("계속? (y/N): ").strip().lower()
    if confirm != "y":
        print("[pubtables1m] 취소.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="bsmock/pubtables-1m",
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    print(f"[pubtables1m] 완료 → {out_dir}")


def download_fintabnet_c(cache_dir: Path):
    print(FINTABNET_C_README.format(cache_dir=cache_dir))

    try:
        from datasets import load_dataset
    except ImportError:
        print("[error] datasets 미설치: pip install datasets")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fintabnet_c] 다운로드 → {cache_dir}")
    ds = load_dataset("bsmock/FinTabNet.c", cache_dir=str(cache_dir))
    print(f"[fintabnet_c] 완료: {ds}")


def download_scitsr(out_dir: Path):
    print(SCITSR_README.format(url=SCITSR_URL, out_dir=out_dir))

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[scitsr] 이미 존재: {out_dir}")
        return

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_dir.parent / "scitsr_tmp.zip"

    print(f"[scitsr] 다운로드: {SCITSR_URL}")
    try:
        subprocess.run(["wget", "-q", SCITSR_URL, "-O", str(tmp_zip)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[scitsr] wget 실패. 수동으로 다운로드하세요:")
        print(f"  {SCITSR_URL}")
        return

    print(f"[scitsr] 압축 해제 → {out_dir.parent}")
    subprocess.run(["unzip", "-q", str(tmp_zip), "-d", str(out_dir.parent)], check=True)
    extracted = out_dir.parent / "SciTSR-master"
    if extracted.exists():
        extracted.rename(out_dir)
    tmp_zip.unlink(missing_ok=True)
    print(f"[scitsr] 완료 → {out_dir}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Dataset download helper")
    parser.add_argument("--dataset", required=True,
                        choices=["pubtables1m", "fintabnet_c", "scitsr"],
                        help="다운로드할 데이터셋")
    parser.add_argument("--out-dir",   default=None,
                        help="저장 경로 (pubtables1m, scitsr)")
    parser.add_argument("--cache-dir", default=None,
                        help="HF 캐시 경로 (fintabnet_c)")
    args = parser.parse_args()

    if args.dataset == "pubtables1m":
        out = Path(args.out_dir or "/mnt/d/pubtables1m")
        download_pubtables1m(out)

    elif args.dataset == "fintabnet_c":
        cache = Path(args.cache_dir or os.path.expanduser("~/.cache/huggingface"))
        download_fintabnet_c(cache)

    elif args.dataset == "scitsr":
        out = Path(args.out_dir or "/data/SciTSR")
        download_scitsr(out)


if __name__ == "__main__":
    main()
