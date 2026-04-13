"""
Deformable-DETR Baseline Setup
================================
fundamentalvision/Deformable-DETR를 PubTables-1M 6-class 구조 인식에 파인튜닝.

사전 준비:
  1. 저장소 클론
     git clone https://github.com/fundamentalvision/Deformable-DETR.git baselines/deformable_detr/src
  2. CUDA extension 빌드
     cd baselines/deformable_detr/src
     pip install -r requirements.txt
     cd models/ops && python setup.py build install && cd ../..
  3. COCO pretrained weight 다운로드 (선택)
     wget https://huggingface.co/fundamentalvision/deformable-detr/resolve/main/r50_deformable_detr-checkpoint.pth

이 스크립트 실행:
  python baselines/deformable_detr/setup.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC_DIR  = Path(__file__).parent / "src"
REPO_URL = "https://github.com/fundamentalvision/Deformable-DETR.git"


def clone():
    if SRC_DIR.exists() and any(SRC_DIR.iterdir()):
        print(f"[deformable_detr] 이미 존재: {SRC_DIR}")
        return
    print(f"[deformable_detr] 클론: {REPO_URL}")
    subprocess.run(
        ["git", "clone", "--depth=1", REPO_URL, str(SRC_DIR)],
        check=True,
    )
    print(f"[deformable_detr] 클론 완료 → {SRC_DIR}")


def build_ops():
    ops_dir = SRC_DIR / "models" / "ops"
    if not ops_dir.exists():
        print("[deformable_detr] ops 디렉토리 없음. 먼저 clone() 실행.")
        return
    print("[deformable_detr] CUDA MultiScaleDeformableAttention 빌드...")
    subprocess.run(
        [sys.executable, "setup.py", "build", "install"],
        cwd=str(ops_dir),
        check=True,
    )
    print("[deformable_detr] 빌드 완료.")


if __name__ == "__main__":
    clone()
    build_ops()
    print("[deformable_detr] 설정 완료.")
    print("  다음 단계: python baselines/deformable_detr/finetune.py --config baselines/deformable_detr/config.yaml")
