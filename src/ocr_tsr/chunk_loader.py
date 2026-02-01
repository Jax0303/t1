"""
SciTSR Chunk Loader - Perfect OCR Substitute

Loads pre-extracted chunks from SciTSR dataset as a "perfect OCR" baseline.
"""

import json
from pathlib import Path
from typing import List, Dict


def load_scitsr_chunk(table_id: str, chunk_dir: str = 'data/SciTSR/test/chunk') -> List[Dict]:
    """
    Load SciTSR chunk file as OCR substitute
    
    Args:
        table_id: Table identifier (e.g., "1504.01806v1.4")
        chunk_dir: Directory containing chunk files
    
    Returns:
        List of OCR-like detections:
        [
          {
            "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            "text": "cell content",
            "confidence": 1.0
          }
        ]
    """
    chunk_path = Path(chunk_dir) / f'{table_id}.chunk'
    
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
    
    with open(chunk_path, 'r') as f:
        data = json.load(f)
    
    ocr_results = []
    
    for chunk in data.get('chunks', []):
        pos = chunk.get('pos', [])
        text = chunk.get('text', '')
        
        # Convert pos [x1, x2, y1, y2] to bbox [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        if len(pos) == 4:
            x1, x2, y1, y2 = pos
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            # Fallback
            bbox = [[0, 0], [10, 0], [10, 10], [0, 10]]
        
        ocr_results.append({
            'bbox': bbox,
            'text': text,
            'confidence': 1.0  # Perfect OCR
        })
    
    return ocr_results


if __name__ == '__main__':
    # Test
    table_id = '1504.01806v1.4'
    chunks = load_scitsr_chunk(table_id)
    
    print(f"Loaded {len(chunks)} chunks for {table_id}")
    print(f"Sample: {chunks[0]}")
