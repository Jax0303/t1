"""
Semantic Encoder Module for SARTP
Encodes cell text into semantic embeddings using transformer models.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticEncoder:
    """
    Encodes table cell text into semantic embeddings.
    
    Uses pretrained language models (RoBERTa, BERT, etc.) to create
    meaningful representations of cell content for semantic alignment.
    """
    
    def __init__(
        self, 
        model_name: str = "roberta-base",
        device: Optional[str] = None,
        max_length: int = 128
    ):
        """
        Initialize the semantic encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_length: Maximum token length for text encoding
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        logger.info(f"Loading semantic encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Encoder initialized (dim={self.embedding_dim}, device={self.device})")
    
    def encode_cells(
        self, 
        cells: List[Dict[str, Any]], 
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode a list of cells into semantic embeddings.
        
        Args:
            cells: List of cell dicts with 'content' key
            batch_size: Batch size for processing
            
        Returns:
            embeddings: Tensor of shape [N_cells, embedding_dim]
        """
        texts = [str(cell.get('content', '')).strip() for cell in cells]
        
        # Handle empty cells
        texts = [t if t else "[EMPTY]" for t in texts]
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings (use [CLS] token)
                outputs = self.model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, dim]
                
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode a single text string.
        
        Args:
            text: Input text
            
        Returns:
            embedding: Tensor of shape [embedding_dim]
        """
        return self.encode_cells([{'content': text}])[0]
    
    def compute_similarity(
        self, 
        embeddings1: torch.Tensor, 
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: [N, dim]
            embeddings2: [M, dim]
            
        Returns:
            similarity: [N, M] matrix
        """
        # Normalize
        emb1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        
        # Cosine similarity
        return torch.mm(emb1_norm, emb2_norm.t())


class HeaderDetector:
    """
    Classifies cells as header or data based on position and content.
    
    Uses heuristic rules combined with semantic features to identify
    header cells, including multi-level hierarchical headers.
    """
    
    def __init__(self, semantic_encoder: Optional[SemanticEncoder] = None):
        """
        Initialize header detector.
        
        Args:
            semantic_encoder: Optional pre-initialized encoder
        """
        self.encoder = semantic_encoder
    
    def classify_cells(
        self, 
        cells: List[Dict[str, Any]],
        use_semantics: bool = True
    ) -> List[bool]:
        """
        Classify each cell as header (True) or data (False).
        
        Args:
            cells: List of cell dicts with 'row_idx', 'col_idx', 'content'
            use_semantics: Whether to use semantic features
            
        Returns:
            is_header: Boolean mask [N_cells]
        """
        is_header = []
        
        # Get row groups
        row_groups = {}
        for i, cell in enumerate(cells):
            row_idx = cell.get('row_idx', 0)
            if row_idx not in row_groups:
                row_groups[row_idx] = []
            row_groups[row_idx].append((i, cell))
        
        # Heuristic: First N rows are likely headers
        max_row = max(row_groups.keys()) if row_groups else 0
        header_threshold_row = min(2, max_row // 3)  # First 2 rows or top 1/3
        
        for i, cell in enumerate(cells):
            row_idx = cell.get('row_idx', 0)
            content = str(cell.get('content', '')).strip()
            
            # Rule 1: Position-based (first rows)
            if row_idx <= header_threshold_row:
                is_header.append(True)
                continue
            
            # Rule 2: Content-based heuristics
            # Headers often contain: short text, uppercase, no numbers
            is_short = len(content) < 30
            has_uppercase = content.isupper() if content else False
            has_no_digits = not any(c.isdigit() for c in content)
            
            if has_uppercase or (is_short and has_no_digits and row_idx < max_row // 2):
                is_header.append(True)
            else:
                is_header.append(False)
        
        return is_header
    
    def detect_header_levels(self, cells: List[Dict[str, Any]]) -> Dict[int, int]:
        """
        Detect hierarchical header levels.
        
        Args:
            cells: List of cell dicts
            
        Returns:
            row_to_level: Mapping from row_idx to header level (0 = top)
        """
        is_header = self.classify_cells(cells, use_semantics=False)
        
        row_to_level = {}
        current_level = 0
        
        row_groups = {}
        for i, cell in enumerate(cells):
            if is_header[i]:
                row_idx = cell.get('row_idx', 0)
                if row_idx not in row_groups:
                    row_groups[row_idx] = True
        
        for row_idx in sorted(row_groups.keys()):
            row_to_level[row_idx] = current_level
            current_level += 1
        
        return row_to_level
