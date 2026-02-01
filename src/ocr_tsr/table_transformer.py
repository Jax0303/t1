"""
Table Transformer (TATR) Wrapper for Structure Recognition

Uses Microsoft's Table Transformer model to recognize table structure.
"""

import torch
from transformers import AutoModelForObjectDetection
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Tuple
import torch.nn.functional as F

class TableTransformerTSR:
    """
    Table Transformer (TATR) wrapper for structure recognition
    """
    
    def __init__(self, model_name: str = "microsoft/table-transformer-structure-recognition", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing Table Transformer on {self.device}...")
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Structure recognition labels
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        from torchvision import transforms
        
        # Simple transform as per TATR requirement
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        pixel_values = transform(image).unsqueeze(0).to(self.device)
        return pixel_values

    def process_structure(self, image_input: Any) -> Dict[str, Any]:
        """
        Run structure recognition on image
        
        Args:
            image_input: PIL Image or path to image
            
        Returns:
            Dict containing detected structure elements (rows, columns, headers, etc.)
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
            
        width, height = image.size
        pixel_values = self._preprocess(image)
        
        with torch.no_grad():
            outputs = self.model(pixel_values)
            
        # Post-process detections
        # Using a fixed threshold for now
        threshold = 0.3
        
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        
        # Target sizes for original image dimensions
        target_sizes = torch.tensor([[height, width]]).to(self.device)
        
        # For simplicity, using a basic post-processor if available or manual implementation
        # Here we manually convert boxes
        boxes = outputs.pred_boxes[0, keep]
        scores = probas[keep].max(-1).values
        labels = probas[keep].argmax(-1)
        
        # Convert from 0-1 to pixel coordinates
        boxes = self._box_cxcywh_to_xyxy(boxes)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        
        results = []
        for score, label, box in zip(scores, labels, boxes):
            results.append({
                'label': self.id2label[label.item()],
                'score': score.item(),
                'bbox': box.tolist() # [x1, y1, x2, y2]
            })
            
        return {'detections': results, 'image_size': (width, height)}

    def _box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def align_ocr_to_structure(self, ocr_results: List[Dict], structure_res: Dict) -> Dict[str, Any]:
        """
        Map OCR tokens to the detected grid structure (rows/cols) with span estimation.
        """
        detections = structure_res['detections']
        # Filter and sort rows/cols by center coordinates
        rows = sorted([d for d in detections if d['label'] == 'table row'], key=lambda x: (x['bbox'][1] + x['bbox'][3]) / 2)
        cols = sorted([d for d in detections if d['label'] == 'table column'], key=lambda x: (x['bbox'][0] + x['bbox'][2]) / 2)
        
        num_rows = len(rows)
        num_cols = len(cols)
        
        # 1. Identify Spanning Cells from detections
        spanning_cells = [d for d in detections if d['label'] == 'table spanning cell']
        
        # --- ROW RECOVERY STEP ---
        # TATR sometimes misses header rows. We recover them from OCR tokens 
        # that don't overlap with any detected row.
        
        # 1. Identify unmatched tokens
        def overlaps_any_row(token_bbox, row_list):
            ty1, ty2 = token_bbox[1], token_bbox[3]
            for r in row_list:
                ry1, ry2 = r['bbox'][1], r['bbox'][3]
                # Check vertical intersection
                if min(ty2, ry2) > max(ty1, ry1):
                    return True
            return False
            
        unmatched_tokens = []
        for token in ocr_results:
            # Normalize bbox to [x1, y1, x2, y2]
            bbox = token['bbox']
            if isinstance(bbox[0], list):
                tx1, ty1 = min(p[0] for p in bbox), min(p[1] for p in bbox)
                tx2, ty2 = max(p[0] for p in bbox), max(p[1] for p in bbox)
                flat_bbox = [tx1, ty1, tx2, ty2]
            else:
                flat_bbox = bbox
            
            if not overlaps_any_row(flat_bbox, rows):
                unmatched_tokens.append({'bbox': flat_bbox, 'text': token['text']})
        
        # 2. Cluster unmatched tokens into new rows
        if unmatched_tokens:
            unmatched_tokens.sort(key=lambda x: (x['bbox'][1] + x['bbox'][3])/2)
            new_rows = []
            current_row_tokens = [unmatched_tokens[0]]
            
            for t in unmatched_tokens[1:]:
                # Check strict Y overlap with current cluster center
                last_t = current_row_tokens[-1]
                # Simple IoU-like check or center distance
                # Use center Y distance < 8 pixels (heuristic) to split tight headers
                cy1 = (last_t['bbox'][1] + last_t['bbox'][3]) / 2
                cy2 = (t['bbox'][1] + t['bbox'][3]) / 2
                
                if abs(cy2 - cy1) < 8: # Threshold for same row
                    current_row_tokens.append(t)
                else:
                    # Finalize current row
                    # Create a bbox that spans all tokens
                    ry1 = min(tok['bbox'][1] for tok in current_row_tokens)
                    ry2 = max(tok['bbox'][3] for tok in current_row_tokens)
                    # Use image width for x (0 to img_w)? Or tighter?
                    # Use tight X for now, but rows usually span full width.
                    # TATR rows usually imply full width. 
                    # We'll use the union of token Xs, but maybe extend?
                    # Let's just use union of token Xs.
                    rx1 = min(tok['bbox'][0] for tok in current_row_tokens)
                    rx2 = max(tok['bbox'][2] for tok in current_row_tokens)
                    
                    new_rows.append({
                        'bbox': [rx1, ry1, rx2, ry2],
                        'label': 'table row (recovered)',
                        'score': 1.0
                    })
                    current_row_tokens = [t]
            
            # Finalize last cluster
            if current_row_tokens:
                ry1 = min(tok['bbox'][1] for tok in current_row_tokens)
                ry2 = max(tok['bbox'][3] for tok in current_row_tokens)
                rx1 = min(tok['bbox'][0] for tok in current_row_tokens)
                rx2 = max(tok['bbox'][2] for tok in current_row_tokens)
                new_rows.append({
                    'bbox': [rx1, ry1, rx2, ry2],
                    'label': 'table row (recovered)',
                    'score': 1.0
                })
            
            if new_rows:
                print(f"Recovered {len(new_rows)} rows from unmatched OCR tokens.")
                rows.extend(new_rows)
                # Re-sort and re-index
                rows.sort(key=lambda x: (x['bbox'][1] + x['bbox'][3]) / 2)
        
        num_rows = len(rows)
        # ---------------------
        
        # 2. Aggregate OCR tokens by (row, col)
        # We first map each token to ALL rows and cols it overlaps with significantly
        token_to_cells = []
        for token in ocr_results:
            bbox = token['bbox']
            if isinstance(bbox[0], list):
                tx1, ty1 = min(p[0] for p in bbox), min(p[1] for p in bbox)
                tx2, ty2 = max(p[0] for p in bbox), max(p[1] for p in bbox)
            else:
                tx1, ty1, tx2, ty2 = bbox
            
            tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            
            # Find best row/col based on overlap (text line segment vs row/col box)
            # Row assignment: maximize vertical overlap (or containment in Y)
            # Col assignment: maximize horizontal overlap (or containment in X)
            
            # Simple intersection logic
            def get_intersection(range1, range2):
                start = max(range1[0], range2[0])
                end = min(range1[1], range2[1])
                return max(0, end - start)

            # DEBUG: Print first few tokens overlap check
            if len(token_to_cells) < 3:
                 print(f"Token: '{token['text']}' Box: {tx1:.1f},{ty1:.1f},{tx2:.1f},{ty2:.1f}")
                 # Print nearest row
                 r_dists = [(abs((r['bbox'][1]+r['bbox'][3])/2 - tcy), r['bbox']) for r in rows]
                 if r_dists:
                     nearest_r = min(r_dists, key=lambda x: x[0])
                     print(f"  Nearest Row: {nearest_r[1]} DistY: {nearest_r[0]:.1f}")

            # Assign Row
            best_ridx = -1
            max_r_overlap = 0
            token_h = ty2 - ty1
            
            for i, r in enumerate(rows):
                # Calculate Y-overlap
                overlap = get_intersection((ty1, ty2), (r['bbox'][1], r['bbox'][3]))
                if overlap > 0:
                    # Score can be overlap length or IoA
                    score = overlap
                    if score > max_r_overlap:
                        max_r_overlap = score
                        best_ridx = i
            
            # Assign Col
            best_cidx = -1
            max_c_overlap = 0
            token_w = tx2 - tx1
            
            for j, c in enumerate(cols):
                # Calculate X-overlap
                overlap = get_intersection((tx1, tx2), (c['bbox'][0], c['bbox'][2]))
                if overlap > 0:
                    score = overlap
                    if score > max_c_overlap:
                        max_c_overlap = score
                        best_cidx = j
            
            if best_ridx != -1 and best_cidx != -1:
                token_to_cells.append({'ridx': best_ridx, 'cidx': best_cidx, 'text': token['text'], 'bbox': [tx1, ty1, tx2, ty2]})

        # 3. Create grid of cells (Handling spanning cells later if detected, for now basic grid)
        # Note: TATR row/col detections are surprisingly good.
        # We merge tokens into logical cells.
        cell_agg = {} # (r, c) -> list of texts
        for tc in token_to_cells:
            key = (tc['ridx'], tc['cidx'])
            cell_agg.setdefault(key, []).append(tc['text'])
            
        final_cells = []
        for (r, c), texts in cell_agg.items():
            # Check if this (r, c) falls into a detected spanning cell
            # For simplicity in this version, we assume 1x1 if not found in spanning_cells
            # In a full impl, we'd check if any spanning_cell covers multiple rows/cols
            # and map this (r,c) to that spanning cell's ID.
            
            final_cells.append({
                'start_row': r,
                'end_row': r,
                'start_col': c,
                'end_col': c,
                'row': r, 'col': c, 'row_span': 1, 'col_span': 1, # Compatibility
                'text': " ".join(texts),
                'bbox': [rows[r]['bbox'][1], cols[c]['bbox'][0], rows[r]['bbox'][3], cols[c]['bbox'][2]] # [y1, x1, y2, x2]
            })
            
        return {
            'cells': final_cells,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'detections': detections
        }

if __name__ == '__main__':
    # Usage test (not running actually but providing as reference)
    try:
        tsr = TableTransformerTSR()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
