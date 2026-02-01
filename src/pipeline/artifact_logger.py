"""
Artifact Logger for Error Atlas Framework

Centralizes logging of intermediate pipeline outputs:
- Input images and metadata
- OCR tokens and overlays
- TSR inputs and outputs
- Final results and evaluations
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ArtifactLogger:
    """Centralized artifact logging for reproducible error analysis"""
    
    def __init__(self, table_id: str, base_dir: str = 'artifacts'):
        self.table_id = table_id
        self.base_dir = Path(base_dir)
        self.artifact_dir = self.base_dir / table_id
        
        # Create subdirectories
        self.dirs = {
            'input': self.artifact_dir / 'A_input',
            'ocr': self.artifact_dir / 'B_ocr',
            'tsr_input': self.artifact_dir / 'C_tsr_input',
            'tsr': self.artifact_dir / 'D_tsr',
            'result': self.artifact_dir / 'E_result',
            'eval': self.artifact_dir / 'F_eval'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_input(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Save input image and metadata
        
        Args:
            image: Input table image (H, W, 3)
            metadata: Dict with table_id, split, size, etc.
        """
        # Save image
        img_pil = Image.fromarray(image.astype(np.uint8))
        img_pil.save(self.dirs['input'] / 'input.png')
        
        # Save metadata
        with open(self.dirs['input'] / 'meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_ocr(self, 
                 tokens: List[Dict[str, Any]], 
                 image: np.ndarray,
                 draw_overlay: bool = True):
        """
        Save OCR outputs
        
        Args:
            tokens: List of dicts with keys: text, bbox, confidence, line_id, block_id
            image: Original image for overlay
            draw_overlay: Whether to generate overlay visualization
        """
        # Save OCR tokens
        with open(self.dirs['ocr'] / 'ocr_tokens.json', 'w') as f:
            json.dump(tokens, f, indent=2)
        
        if draw_overlay:
            self._draw_ocr_overlay(tokens, image)
    
    def _draw_ocr_overlay(self, tokens: List[Dict], image: np.ndarray):
        """Draw OCR bounding boxes and text on image"""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = None
        
        for token in tokens:
            bbox = token['bbox']  # [x1, y1, x2, y2]
            text = token.get('text', '')
            conf = token.get('confidence', 1.0)
            
            # Draw bounding box
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                linewidth=1, 
                edgecolor='red' if conf < 0.8 else 'green',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw text
            ax.text(bbox[0], bbox[1] - 2, text, 
                   fontsize=8, color='red', 
                   bbox=dict(boxstyle='round,pad=0.2', 
                            facecolor='white', alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.dirs['ocr'] / 'ocr_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_tsr_input(self, tsr_input: Dict[str, Any]):
        """
        Save TSR input (after OCR→TSR transformation)
        
        Args:
            tsr_input: TSR module's actual input
        """
        with open(self.dirs['tsr_input'] / 'tsr_input.json', 'w') as f:
            json.dump(tsr_input, f, indent=2)
    
    def save_tsr(self, 
                 structure: Dict[str, Any], 
                 image: np.ndarray,
                 log: Optional[Dict] = None,
                 draw_overlay: bool = True):
        """
        Save TSR outputs
        
        Args:
            structure: Dict with keys: rows, cols, cells, adjacency, etc.
            image: Original image for overlay
            log: Processing log
            draw_overlay: Whether to generate overlay
        """
        # Save structure
        with open(self.dirs['tsr'] / 'pred_structure.json', 'w') as f:
            json.dump(structure, f, indent=2)
        
        # Save log
        if log:
            with open(self.dirs['tsr'] / 'tsr_log.json', 'w') as f:
                json.dump(log, f, indent=2)
        
        if draw_overlay:
            self._draw_tsr_overlay(structure, image)
    
    def _draw_tsr_overlay(self, structure: Dict, image: np.ndarray):
        """Draw predicted table structure on image"""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        cells = structure.get('cells', [])
        
        for i, cell in enumerate(cells):
            bbox = cell.get('box', cell.get('bbox', [0, 0, 10, 10]))
            
            # Draw cell bbox
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor='blue',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw cell index
            ax.text(bbox[0] + 2, bbox[1] + 10, str(i),
                   fontsize=10, color='blue', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='yellow', alpha=0.7))
            
            # Draw cell position (row, col)
            if 'row' in cell and 'col' in cell:
                pos_text = f"({cell['row']},{cell['col']})"
                ax.text(bbox[0] + 2, bbox[3] - 2, pos_text,
                       fontsize=8, color='green',
                       bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='white', alpha=0.6))
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.dirs['tsr'] / 'pred_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_result(self, 
                    table_data: Any,
                    postprocess_log: Optional[Dict] = None):
        """
        Save final result
        
        Args:
            table_data: Final table (CSV, HTML, or dict)
            postprocess_log: Log of postprocessing rules applied
        """
        # Save table
        if isinstance(table_data, str):
            # HTML or CSV string
            ext = '.html' if '<table>' in table_data else '.csv'
            with open(self.dirs['result'] / f'pred_table{ext}', 'w') as f:
                f.write(table_data)
        else:
            # JSON
            with open(self.dirs['result'] / 'pred_table.json', 'w') as f:
                json.dump(table_data, f, indent=2)
        
        # Save postprocess log
        if postprocess_log:
            with open(self.dirs['result'] / 'postprocess_log.json', 'w') as f:
                json.dump(postprocess_log, f, indent=2)
    
    def save_eval(self,
                  metrics: Dict[str, Any],
                  gt_structure: Optional[Dict] = None,
                  pred_structure: Optional[Dict] = None,
                  image: Optional[np.ndarray] = None):
        """
        Save evaluation results and visualizations
        
        Args:
            metrics: Evaluation metrics
            gt_structure: Ground truth structure for overlay
            pred_structure: Predicted structure for diff
            image: Original image for overlays
        """
        # Save metrics
        with open(self.dirs['eval'] / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Draw GT overlay
        if gt_structure and image is not None:
            self._draw_gt_overlay(gt_structure, image)
        
        # Draw diff overlay
        if gt_structure and pred_structure and image is not None:
            self._draw_diff_overlay(gt_structure, pred_structure, image)
    
    def _draw_gt_overlay(self, gt_structure: Dict, image: np.ndarray):
        """Draw ground truth structure on image"""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        cells = gt_structure.get('cells', [])
        
        for i, cell in enumerate(cells):
            bbox = cell.get('box', cell.get('bbox', [0, 0, 10, 10]))
            
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor='green',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
        
        ax.set_title('Ground Truth Structure', fontsize=14, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.dirs['eval'] / 'gt_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _draw_diff_overlay(self, gt_structure: Dict, pred_structure: Dict, image: np.ndarray):
        """Draw prediction vs ground truth diff"""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        # Draw GT in green dashed
        for cell in gt_structure.get('cells', []):
            bbox = cell.get('box', cell.get('bbox', [0, 0, 10, 10]))
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor='green',
                facecolor='none',
                linestyle='--',
                alpha=0.6
            )
            ax.add_patch(rect)
        
        # Draw Pred in red solid
        for cell in pred_structure.get('cells', []):
            bbox = cell.get('box', cell.get('bbox', [0, 0, 10, 10]))
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='green', linestyle='--', label='GT'),
            Patch(facecolor='none', edgecolor='red', label='Pred')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        ax.set_title('Prediction vs Ground Truth', fontsize=14, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.dirs['eval'] / 'diff_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def get_artifact_path(self, stage: str, filename: str) -> Path:
        """Get path to specific artifact file"""
        return self.dirs.get(stage, self.artifact_dir) / filename
    
    def list_artifacts(self) -> Dict[str, List[str]]:
        """List all saved artifacts"""
        artifacts = {}
        for stage, dir_path in self.dirs.items():
            if dir_path.exists():
                artifacts[stage] = [f.name for f in dir_path.iterdir()]
        return artifacts
