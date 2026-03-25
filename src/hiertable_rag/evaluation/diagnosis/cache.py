"""
Caching System for OCR and TSR Results

Avoids expensive recomputation by caching results to disk.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Cache for OCR and TSR results
    """
    
    def __init__(self, cache_dir: str = 'outputs/cache'):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Result cache initialized at: {self.cache_dir}")
    
    def _get_cache_key(self, *args) -> str:
        """
        Generate cache key from arguments
        
        Args:
            *args: Arguments to hash
        
        Returns:
            Cache key string
        """
        # Join all arguments and hash
        key_str = '_'.join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_ocr_cache_path(self, ocr_source: str, table_id: str) -> Path:
        """
        Get cache file path for OCR results
        
        Args:
            ocr_source: OCR source identifier ('noisy', 'perfect', 'gt')
            table_id: Table identifier
        
        Returns:
            Cache file path
        """
        filename = f"ocr_{ocr_source}_{table_id}.json"
        return self.cache_dir / filename
    
    def get_tsr_cache_path(self, tsr_engine: str, ocr_source: str, table_id: str) -> Path:
        """
        Get cache file path for TSR results
        
        Args:
            tsr_engine: TSR engine identifier ('baseline', 'gnn_csp', 'gt')
            ocr_source: OCR source identifier
            table_id: Table identifier
        
        Returns:
            Cache file path
        """
        filename = f"tsr_{tsr_engine}_{ocr_source}_{table_id}.json"
        return self.cache_dir / filename
    
    def load_ocr(self, ocr_source: str, table_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load cached OCR results
        
        Args:
            ocr_source: OCR source identifier
            table_id: Table identifier
        
        Returns:
            OCR results or None if not cached
        """
        cache_path = self.get_ocr_cache_path(ocr_source, table_id)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded OCR from cache: {cache_path.name}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load OCR cache {cache_path}: {e}")
            return None
    
    def save_ocr(self, ocr_source: str, table_id: str, ocr_results: List[Dict[str, Any]]):
        """
        Save OCR results to cache
        
        Args:
            ocr_source: OCR source identifier
            table_id: Table identifier
            ocr_results: OCR results to cache
        """
        cache_path = self.get_ocr_cache_path(ocr_source, table_id)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_results, f, indent=2)
            logger.debug(f"Saved OCR to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save OCR cache {cache_path}: {e}")
    
    def load_tsr(self, tsr_engine: str, ocr_source: str, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Load cached TSR results
        
        Args:
            tsr_engine: TSR engine identifier
            ocr_source: OCR source identifier
            table_id: Table identifier
        
        Returns:
            TSR structure or None if not cached
        """
        cache_path = self.get_tsr_cache_path(tsr_engine, ocr_source, table_id)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded TSR from cache: {cache_path.name}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load TSR cache {cache_path}: {e}")
            return None
    
    def save_tsr(self, tsr_engine: str, ocr_source: str, table_id: str, structure: Dict[str, Any]):
        """
        Save TSR results to cache
        
        Args:
            tsr_engine: TSR engine identifier
            ocr_source: OCR source identifier
            table_id: Table identifier
            structure: TSR structure to cache
        """
        cache_path = self.get_tsr_cache_path(tsr_engine, ocr_source, table_id)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(structure, f, indent=2)
            logger.debug(f"Saved TSR to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save TSR cache {cache_path}: {e}")
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cache files
        
        Args:
            pattern: Optional glob pattern to match specific files
                    If None, clears all cache files
        """
        if pattern:
            files = list(self.cache_dir.glob(pattern))
        else:
            files = list(self.cache_dir.glob('*.json'))
        
        for file in files:
            try:
                file.unlink()
                logger.debug(f"Deleted cache file: {file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete cache file {file}: {e}")
        
        logger.info(f"Cleared {len(files)} cache files")
