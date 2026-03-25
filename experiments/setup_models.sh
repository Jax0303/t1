#!/bin/bash
set -e

echo "=================================================="
echo " Setting up environment for Legacy TSR Models"
echo " (Cycle-CenterNet & LORE-TSR)"
echo "=================================================="

WORK_DIR="/home/user/t1-8/experiments/models"
mkdir -p $WORK_DIR
cd $WORK_DIR

# 1. Create Conda environment
ENV_NAME="tsr_legacy"
echo "[1/4] Creating conda environment: $ENV_NAME"
source $HOME/miniconda3/etc/profile.d/conda.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# We already have the env created and pytorch installed, but we can reuse it
conda activate $ENV_NAME

# Install specific CUDA toolkit
conda install -c conda-forge cudatoolkit=11.1 -y
conda install -c conda-forge cxx-compiler -y

# 2. Clone Repositories
echo "[3/4] Cloning repositories"
if [ ! -d "Cycle-CenterNet" ]; then
    git clone https://github.com/ArchieAlexArkhipov/Cycle-CenterNet.git
fi

if [ ! -d "AdvancedLiterateMachinery" ]; then
    git clone https://github.com/AlibabaResearch/AdvancedLiterateMachinery.git
fi

# 3. Compile DCNv2 for Cycle-CenterNet and LORE-TSR
echo "[4/4] Compiling DCNv2 for Cycle-CenterNet & LORE-TSR"
# Cycle-CenterNet Uses MMDetection which has its own DCNv2 or MMCV dependecy
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html

# Fix issue where old C++ standards fail on recent GCC versions for DCNv2
# LORE-TSR DCNv2
cd AdvancedLiterateMachinery/DocumentUnderstanding/LORE-TSR/src/lib/models/networks/DCNv2
# Clean old build
rm -rf build/
# Set correct arch for RTX 3060 Ti (sm_86)
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA="1"
export MAX_JOBS=4
python setup.py build develop

echo "=================================================="
echo " Setup complete!"
echo "=================================================="
