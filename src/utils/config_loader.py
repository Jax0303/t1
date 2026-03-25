"""
설정 파일 로더
"""
import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # 기본 설정
        return {
            'api': {
                'provider': 'ollama',
                'model': 'llama3.2',
                'api_key': 'dummy'
            }
        }



