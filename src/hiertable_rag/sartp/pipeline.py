"""
SARTP End-to-End Pipeline
Integrates all SARTP components for semantic-aware table parsing.
"""

import torch
from typing import Dict, Any, List, Optional
import logging

from src.hiertable_rag.core.rca_model import RCAModel
from src.hiertable_rag.core.grid_restorer import IntersectionRestorer
from src.hiertable_rag.sartp.semantic_encoder import SemanticEncoder, HeaderDetector
from src.hiertable_rag.sartp.alignment_scorer import HeaderDataAligner
from src.hiertable_rag.sartp.graph_optimizer import GraphOptimizer

logger = logging.getLogger(__name__)


class SARTPPipeline:
    """
    Complete SARTP (Semantics-Aware Table Parsing) pipeline.
    
    Combines visual structure prediction (RCA) with semantic alignment
    to produce more accurate table structures, especially for complex tables
    with ambiguous boundaries.
    """
    
    def __init__(
        self,
        rca_checkpoint: Optional[str] = None,
        semantic_model: str = "roberta-base",
        alpha: float = 0.5,  # Visual weight
        beta: float = 0.3,   # Semantic weight  
        gamma: float = 0.2,  # Layout weight
        device: Optional[str] = None
    ):
        """
        Initialize SARTP pipeline.
        
        Args:
            rca_checkpoint: Path to pretrained RCA model weights
            semantic_model: HuggingFace model for semantic encoding
            alpha: Weight for visual features
            beta: Weight for semantic features
            gamma: Weight for layout features
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing SARTP Pipeline on {self.device}")
        
        # Phase 1: Visual structure prediction
        self.rca_model = RCAModel().to(self.device)
        if rca_checkpoint:
            logger.info(f"Loading RCA weights from {rca_checkpoint}")
            self.rca_model.load_state_dict(torch.load(rca_checkpoint, map_location=self.device))
        self.rca_model.eval()
        
        self.grid_restorer = IntersectionRestorer()
        
        # Phase 2: Semantic encoding
        self.semantic_encoder = SemanticEncoder(
            model_name=semantic_model,
            device=self.device
        )
        self.header_detector = HeaderDetector(semantic_encoder=self.semantic_encoder)
        
        # Phase 3: Alignment scoring
        self.aligner = HeaderDataAligner(similarity_threshold=0.5)
        
        # Phase 4: Structure optimization
        self.optimizer = GraphOptimizer(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            method='hungarian'
        )
        
        logger.info("SARTP Pipeline initialized successfully")
    
    def parse(
        self,
        image: torch.Tensor,
        ocr_boxes: List[Dict[str, Any]],
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Parse table image into structured cells using SARTP.
        
        Args:
            image: Input image tensor [B, 3, H, W]
            ocr_boxes: List of OCR boxes with 'box' and 'content'
            return_intermediate: If True, return intermediate results for debugging
            
        Returns:
            Dictionary containing:
                - 'cells': Final refined cell structure
                - 'initial_cells': Before refinement (if return_intermediate=True)
                - 'alignment_scores': Semantic scores (if return_intermediate=True)
                - 'visual_pred': RCA output (if return_intermediate=True)
        """
        logger.info(f"Parsing table with {len(ocr_boxes)} OCR boxes")
        
        # Step 1: Visual structure prediction (RCA)
        logger.info("[1/5] Running visual structure prediction (RCA)...")
        with torch.no_grad():
            visual_pred = self.rca_model.predict_boundaries(
                image,
                threshold=0.5,
                return_confidence=True
            )
        
        # Step 2: Initial cell restoration
        logger.info("[2/5] Restoring initial cell structure...")
        initial_cells = self.grid_restorer.restore_cells(visual_pred, ocr_boxes)
        logger.info(f"  Restored {len(initial_cells)} initial cells")
        
        if len(initial_cells) == 0:
            logger.warning("No cells restored, returning empty result")
            return {'cells': [], 'initial_cells': []}
        
        # Step 3: Semantic encoding
        logger.info("[3/5] Encoding cell semantics...")
        cell_embeddings = self.semantic_encoder.encode_cells(initial_cells)
        header_mask = self.header_detector.classify_cells(initial_cells, use_semantics=False)
        logger.info(f"  Detected {sum(header_mask)} header cells")
        
        # Step 4: Alignment scoring
        logger.info("[4/5] Computing semantic alignment scores...")
        alignment_scores = self.aligner.score(
            initial_cells,
            cell_embeddings,
            header_mask
        )
        logger.info(f"  Global alignment score: {alignment_scores['global_score']:.3f}")
        
        # Step 5: Structure refinement
        logger.info("[5/5] Refining structure with graph optimization...")
        refined_cells = self.optimizer.refine(
            visual_pred,
            alignment_scores,
            initial_cells
        )
        logger.info(f"  Final cell count: {len(refined_cells)}")
        
        result = {'cells': refined_cells}
        
        if return_intermediate:
            result['initial_cells'] = initial_cells
            result['alignment_scores'] = alignment_scores
            result['visual_pred'] = visual_pred
            result['header_mask'] = header_mask
            result['embeddings'] = cell_embeddings
        
        return result
    
    def parse_batch(
        self,
        images: List[torch.Tensor],
        ocr_boxes_list: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Parse multiple tables in batch.
        
        Args:
            images: List of image tensors
            ocr_boxes_list: List of OCR boxes for each image
            
        Returns:
            List of parse results (one per image)
        """
        results = []
        for img, ocr_boxes in zip(images, ocr_boxes_list):
            result = self.parse(img.unsqueeze(0), ocr_boxes, return_intermediate=False)
            results.append(result)
        return results
    
    def set_weights(self, alpha: float, beta: float, gamma: float):
        """
        Update the visual/semantic/layout weights.
        
        Useful for ablation studies or adaptive weighting.
        """
        self.optimizer.alpha = alpha
        self.optimizer.beta = beta
        self.optimizer.gamma = gamma
        
        # Normalize
        total = alpha + beta + gamma
        self.optimizer.alpha /= total
        self.optimizer.beta /= total
        self.optimizer.gamma /= total
        
        logger.info(f"Updated weights: α={self.optimizer.alpha:.2f}, β={self.optimizer.beta:.2f}, γ={self.optimizer.gamma:.2f}")
