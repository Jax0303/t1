
import json
import os
import sys
import time
import torch
import numpy as np
import logging
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 절대 경로 설정 (환경에 따라 다를 수 있는 상대 경로 대신 확실한 절대 경로 사용)
# 현재 노트북이 scripts/ 폴더 안에 있다고 가정하거나, 프로젝트 루트를 명시합니다.
PROJECT_ROOT = '/root/t1-9'
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

print(f"Adding paths:\n - {PROJECT_ROOT}\n - {SRC_PATH}")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    
# GNN-CSP 모듈 임포트
# try-except를 제거하여 에러 발생 시 원인을 바로 확인할 수 있게 합니다.
from hiertable_rag.gnn_csp.pipeline import GNNCSPPipeline
from hiertable_rag.evaluation.metrics import calculate_teds
print("Successfully imported GNN-CSP modules")
