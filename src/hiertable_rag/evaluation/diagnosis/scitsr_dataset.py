"""
SciTSR Dataset Loader

Robust loader with schema detection and graceful fallback for SciTSR dataset.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np

from .utils import normalize_bbox, extract_text_content

logger = logging.getLogger(__name__)


class SciTSRDataset:
    """
    SciTSR dataset loader with schema auto-detection
    
    Handles various JSON schema formats and provides unified interface.
    """
    
    def __init__(self, data_root: str):
        """
        Initialize SciTSR dataset loader
        
        Args:
            data_root: Root directory of SciTSR dataset
        """
        self.data_root = Path(data_root)
        self.img_dir = self.data_root / 'test' / 'img'
        self.struct_dir = self.data_root / 'test' / 'structure'
        self.chunk_dir = self.data_root / 'test' / 'chunk'
        
        # Validate directories
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.struct_dir.exists():
            raise FileNotFoundError(f"Structure directory not found: {self.struct_dir}")
        
        # Chunk directory is optional
        self.has_chunk = self.chunk_dir.exists()
        if not self.has_chunk:
            logger.warning(f"Chunk directory not found: {self.chunk_dir}. S2/S4 scenarios will be skipped.")
    
    def get_available_tables(self) -> List[str]:
        """
        Get list of all available table IDs
        
        Returns:
            List of table IDs (without extension)
        """
        struct_files = list(self.struct_dir.glob('*.json'))
        table_ids = [f.stem for f in struct_files]
        return sorted(table_ids)
    
    def load_image(self, table_id: str) -> Optional[np.ndarray]:
        """
        Load table image
        
        Args:
            table_id: Table identifier
        
        Returns:
            Image as numpy array (H, W, 3) or None if not found
        """
        # Try PNG first, then JPG
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = self.img_dir / f'{table_id}{ext}'
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    return np.array(img)
                except Exception as e:
                    logger.error(f"Failed to load image {img_path}: {e}")
                    return None
        
        logger.warning(f"Image not found for table: {table_id}")
        return None
    
    def load_structure(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Load ground truth structure with schema auto-detection
        
        Args:
            table_id: Table identifier
        
        Returns:
            Normalized structure dict with 'cells' field or None if loading fails
        """
        struct_path = self.struct_dir / f'{table_id}.json'
        
        if not struct_path.exists():
            logger.warning(f"Structure file not found: {struct_path}")
            return None
        
        try:
            with open(struct_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Schema detection and normalization
            return self._normalize_structure(data, table_id)
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {struct_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load structure {struct_path}: {e}")
            return None
    
    def _normalize_structure(self, data: Any, table_id: str) -> Dict[str, Any]:
        """
        Normalize structure data to unified format
        
        Handles multiple schema variations:
        - Direct list of cells
        - Dict with 'cells' key
        - Nested structures
        
        Args:
            data: Raw JSON data
            table_id: Table ID for logging
        
        Returns:
            Normalized dict with 'cells' field
        """
        cells = []
        
        # Case 1: Direct list of cells
        if isinstance(data, list):
            cells = data
        
        # Case 2: Dict with 'cells' key
        elif isinstance(data, dict):
            if 'cells' in data:
                cells = data['cells']
            # Case 3: Nested structure (fallback)
            else:
                logger.warning(f"Unknown structure schema for {table_id}, attempting fallback")
                cells = data.get('data', data.get('table', []))
        
        else:
            logger.error(f"Unexpected data type for {table_id}: {type(data)}")
            return {'cells': []}
        
        # Normalize each cell
        normalized_cells = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            
            # Extract coordinates (multiple field name variations)
            start_row = cell.get('start_row', cell.get('row', 0))
            end_row = cell.get('end_row', cell.get('row', start_row))
            start_col = cell.get('start_col', cell.get('col', 0))
            end_col = cell.get('end_col', cell.get('col', start_col))
            
            # Extract content
            content = extract_text_content(cell)
            
            # Extract bbox
            bbox = normalize_bbox(cell.get('bbox', cell.get('box', [0, 0, 10, 10])))
            
            normalized_cells.append({
                'start_row': start_row,
                'end_row': end_row,
                'start_col': start_col,
                'end_col': end_col,
                'content': content,
                'box': bbox
            })
        
        return {'cells': normalized_cells}
    
    def load_chunk(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Load chunk data (for perfect OCR scenario)
        
        Args:
            table_id: Table identifier
        
        Returns:
            Chunk data dict or None if not available
        """
        if not self.has_chunk:
            return None
        
        chunk_path = self.chunk_dir / f'{table_id}.chunk'
        
        if not chunk_path.exists():
            # Try .json extension as fallback
            chunk_path = self.chunk_dir / f'{table_id}.json'
            if not chunk_path.exists():
                return None
        
        try:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_path}: {e}")
            return None
    
    def load_table(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Load complete table data (image + structure + optional chunk)
        
        Args:
            table_id: Table identifier
        
        Returns:
            Dict with 'table_id', 'image', 'structure', 'chunk' (optional), 'paths'
            Returns None if essential data (image or structure) is missing
        """
        image = self.load_image(table_id)
        structure = self.load_structure(table_id)
        
        # Essential data must exist
        if image is None or structure is None:
            logger.warning(f"Skipping table {table_id}: missing essential data")
            return None
        
        # Chunk is optional
        chunk = self.load_chunk(table_id)
        
        # Build paths
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = self.img_dir / f'{table_id}{ext}'
            if p.exists():
                img_path = str(p)
                break
        
        struct_path = str(self.struct_dir / f'{table_id}.json')
        
        chunk_path = None
        if chunk is not None:
            for ext in ['.chunk', '.json']:
                p = self.chunk_dir / f'{table_id}{ext}'
                if p.exists():
                    chunk_path = str(p)
                    break
        
        return {
            'table_id': table_id,
            'image': image,
            'structure': structure,
            'chunk': chunk,
            'paths': {
                'image': img_path,
                'structure': struct_path,
                'chunk': chunk_path
            }
        }
    
    def load_batch(self, table_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Load batch of tables
        
        Args:
            table_ids: List of table IDs to load
        
        Returns:
            List of successfully loaded table dicts (may be shorter than input)
        """
        tables = []
        
        for table_id in table_ids:
            table = self.load_table(table_id)
            if table is not None:
                tables.append(table)
        
        logger.info(f"Loaded {len(tables)}/{len(table_ids)} tables successfully")
        return tables
