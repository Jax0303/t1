"""
OCR Source Interfaces

Provides unified interface for noisy (real) and perfect (chunk-based) OCR.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from .utils import normalize_bbox, extract_text_content

logger = logging.getLogger(__name__)


class BaseOCRSource(ABC):
    """
    Abstract base class for OCR sources
    """
    
    @abstractmethod
    def get_ocr(self, image: np.ndarray, table_id: str) -> List[Dict[str, Any]]:
        """
        Get OCR results for image
        
        Args:
            image: Image array (H, W, 3)
            table_id: Table identifier for logging/caching
        
        Returns:
            List of OCR results with format:
            [{
                'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or [x1,y1,x2,y2],
                'text': str,
                'confidence': float
            }]
        """
        pass


class NoisyOCRSource(BaseOCRSource):
    """
    Noisy (real) OCR source
    
    Uses PaddleOCR if available, otherwise falls back to mock OCR.
    """
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize noisy OCR source
        
        Args:
            use_mock: If True, use mock OCR instead of real engine
        """
        self.use_mock = use_mock
        self.ocr_engine = None
        
        if not use_mock:
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False
                )
                logger.info("PaddleOCR initialized successfully")
            except ImportError:
                logger.warning("PaddleOCR not available, falling back to mock OCR")
                self.use_mock = True
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}. Using mock OCR.")
                self.use_mock = True
    
    def get_ocr(self, image: np.ndarray, table_id: str) -> List[Dict[str, Any]]:
        """
        Get OCR results from real or mock engine
        
        Args:
            image: Image array
            table_id: Table ID
        
        Returns:
            List of OCR results
        """
        if self.use_mock:
            return self._mock_ocr(image, table_id)
        
        try:
            result = self.ocr_engine.ocr(image, cls=True)
            
            ocr_results = []
            if result and result[0]:
                for line in result[0]:
                    if not line:
                        continue
                    
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]  # (text, confidence)
                    text = text_info[0] if isinstance(text_info, tuple) else str(text_info)
                    confidence = text_info[1] if isinstance(text_info, tuple) and len(text_info) > 1 else 0.9
                    
                    ocr_results.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': float(confidence)
                    })
            
            return ocr_results
        
        except Exception as e:
            logger.error(f"OCR failed for {table_id}: {e}")
            return []
    
    def _mock_ocr(self, image: np.ndarray, table_id: str) -> List[Dict[str, Any]]:
        """
        Mock OCR for testing when real engine is unavailable
        
        Returns empty list - will be populated from GT when needed
        """
        logger.debug(f"Using mock OCR for {table_id}")
        return []


class PerfectOCRSource(BaseOCRSource):
    """
    Perfect OCR source from chunk data (ground truth text with bboxes)
    """
    
    def __init__(self, chunk_data_map: Optional[Dict[str, Any]] = None):
        """
        Initialize perfect OCR source
        
        Args:
            chunk_data_map: Optional mapping of table_id to chunk data
        """
        self.chunk_data_map = chunk_data_map or {}
    
    def set_chunk_data(self, table_id: str, chunk_data: Dict[str, Any]):
        """
        Set chunk data for a specific table
        
        Args:
            table_id: Table identifier
            chunk_data: Chunk JSON data
        """
        self.chunk_data_map[table_id] = chunk_data
    
    def get_ocr(self, image: np.ndarray, table_id: str) -> List[Dict[str, Any]]:
        """
        Get perfect OCR from chunk data
        
        Args:
            image: Image array (not used, but kept for interface compatibility)
            table_id: Table identifier
        
        Returns:
            List of OCR results extracted from chunk
        """
        chunk_data = self.chunk_data_map.get(table_id)
        
        if chunk_data is None:
            logger.warning(f"No chunk data available for {table_id}")
            return []
        
        return self._extract_ocr_from_chunk(chunk_data)
    
    def _extract_ocr_from_chunk(self, chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract OCR results from chunk JSON
        
        Chunk format varies but typically contains 'chunks' or 'cells' with text and bbox.
        
        Args:
            chunk_data: Chunk JSON data
        
        Returns:
            List of OCR results
        """
        ocr_results = []
        
        # Try different field names
        chunks = chunk_data.get('chunks', chunk_data.get('cells', chunk_data.get('data', [])))
        
        if not isinstance(chunks, list):
            logger.warning("Chunk data is not a list, attempting fallback")
            chunks = []
        
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            
            # Extract text
            text = extract_text_content(chunk)
            if not text:
                continue
            
            # Extract bbox
            bbox = normalize_bbox(chunk.get('bbox', chunk.get('box', chunk.get('pos', [0, 0, 10, 10]))))
            
            # Convert to points format for consistency
            x1, y1, x2, y2 = bbox
            bbox_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            ocr_results.append({
                'bbox': bbox_points,
                'text': text,
                'confidence': 1.0  # Perfect OCR has 100% confidence
            })
        
        return ocr_results


class GTOCRSource(BaseOCRSource):
    """
    Ground truth OCR source from structure data
    
    Used as fallback when chunk data is not available.
    """
    
    def __init__(self, structure_map: Optional[Dict[str, Dict]] = None):
        """
        Initialize GT OCR source
        
        Args:
            structure_map: Mapping of table_id to structure data
        """
        self.structure_map = structure_map or {}
    
    def set_structure(self, table_id: str, structure: Dict[str, Any]):
        """
        Set structure data for a table
        
        Args:
            table_id: Table identifier
            structure: Structure dict with 'cells' field
        """
        self.structure_map[table_id] = structure
    
    def get_ocr(self, image: np.ndarray, table_id: str) -> List[Dict[str, Any]]:
        """
        Get OCR from ground truth structure
        
        Args:
            image: Image array (not used)
            table_id: Table identifier
        
        Returns:
            List of OCR results
        """
        structure = self.structure_map.get(table_id)
        
        if structure is None:
            logger.warning(f"No structure data available for {table_id}")
            return []
        
        cells = structure.get('cells', [])
        ocr_results = []
        
        for cell in cells:
            text = extract_text_content(cell)
            if not text:
                continue
            
            bbox = normalize_bbox(cell.get('box', cell.get('bbox', [0, 0, 10, 10])))
            
            # Convert to points format
            x1, y1, x2, y2 = bbox
            bbox_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            ocr_results.append({
                'bbox': bbox_points,
                'text': text,
                'confidence': 1.0
            })
        
        return ocr_results
