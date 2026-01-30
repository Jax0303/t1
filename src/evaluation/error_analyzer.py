import numpy as np
import Levenshtein
import re
from typing import List, Dict, Any, Tuple, Set
from src.evaluation.metrics import TEDS, calculate_adjacency_relation_metrics

class ErrorAnalyzer:
    """
    Analyzes table recognition errors by comparing OCR and TSR outputs against Ground Truth.
    Categorizes errors into OCR, TSR, and Combined types using SciTSR-specific logic.
    """
    def __init__(self, teds_structure_only: bool = False):
        self.teds_evaluator = TEDS(structure_only=teds_structure_only)
        self.teds_all_evaluator = TEDS(structure_only=False)

    def calculate_cer(self, pred_text: str, gt_text: str) -> float:
        """Character Error Rate."""
        if not gt_text:
            return 1.0 if pred_text else 0.0
        return Levenshtein.distance(pred_text, gt_text) / len(gt_text)

    def calculate_wer(self, pred_text: str, gt_text: str) -> float:
        """Word Error Rate."""
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        if not gt_words:
            return 1.0 if pred_words else 0.0
        
        all_words = list(set(pred_words + gt_words))
        word2char = {word: chr(i) for i, word in enumerate(all_words)}
        
        pred_seq = "".join([word2char[w] for w in pred_words if w in word2char])
        gt_seq = "".join([word2char[w] for w in gt_words if w in word2char])
        
        return Levenshtein.distance(pred_seq, gt_seq) / len(gt_words)

    def cells_to_relations(self, cells: List[Dict]) -> Set[Tuple[int, int, str]]:
        """
        Convert cell list with row/col info to a set of adjacency relations.
        Format: (cell_i_idx, cell_j_idx, relation_type)
        """
        relations = set()
        if not cells: return relations
        
        grid = {}
        for i, cell in enumerate(cells):
            r = cell.get('row', -1)
            c = cell.get('col', -1)
            if r == -1 or c == -1:
                lc = cell.get('logical_coords', [0,0,0,0])
                r, c = lc[0], lc[2]
            grid[(r, c)] = i
            
        for (r, c), idx in grid.items():
            if (r, c + 1) in grid:
                neighbor_idx = grid[(r, c + 1)]
                relations.add(tuple(sorted((idx, neighbor_idx))) + ('horz',))
            if (r + 1, c) in grid:
                neighbor_idx = grid[(r + 1, c)]
                relations.add(tuple(sorted((idx, neighbor_idx))) + ('vert',))
                
        return relations

    def analyze_categorical_ocr(self, pred_cells: List[Dict], gt_cells: List[Dict]) -> Dict[str, float]:
        """
        Analyze CER by category: Numeric, Alpha, Mixed.
        """
        categories = {
            "numeric": {"pred": "", "gt": ""},
            "alpha": {"pred": "", "gt": ""},
            "mixed": {"pred": "", "gt": ""}
        }
        
        for i in range(min(len(pred_cells), len(gt_cells))):
            gt_text = str(gt_cells[i].get('content', gt_cells[i].get('text', '')))
            pred_text = str(pred_cells[i].get('content', pred_cells[i].get('text', '')))
            
            if re.match(r'^[0-9\.,\-\+ ]+$', gt_text.strip()):
                cat = "numeric"
            elif re.match(r'^[a-zA-Z ]+$', gt_text.strip()):
                cat = "alpha"
            else:
                cat = "mixed"
                
            categories[cat]["gt"] += gt_text + " "
            categories[cat]["pred"] += pred_text + " "
            
        results = {}
        for cat, data in categories.items():
            results[f"cer_{cat}"] = self.calculate_cer(data["pred"], data["gt"])
            
        return results

    def analyze_sample(self, 
                       pred_html: str, 
                       gt_html: str, 
                       pred_cells: List[Dict], 
                       gt_cells: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single table sample.
        """
        teds_s = self.teds_evaluator.evaluate(pred_html, gt_html)
        teds_all = self.teds_all_evaluator.evaluate(pred_html, gt_html)
        cat_ocr = self.analyze_categorical_ocr(pred_cells, gt_cells)
        
        pred_full_text = " ".join([str(c.get('content', c.get('text', ''))) for c in pred_cells])
        gt_full_text = " ".join([str(c.get('content', c.get('text', ''))) for c in gt_cells])
        cer = self.calculate_cer(pred_full_text, gt_full_text)
        wer = self.calculate_wer(pred_full_text, gt_full_text)
        
        try:
            pred_rel = self.cells_to_relations(pred_cells)
            gt_rel = self.cells_to_relations(gt_cells)
            prec, rec, f1 = calculate_adjacency_relation_metrics(pred_rel, gt_rel)
        except:
            f1 = 0.0
            
        error_type = self.categorize_error(teds_s, cer, len(pred_cells), len(gt_cells))
        
        result = {
            "teds_s": teds_s,
            "teds_all": teds_all,
            "cer": cer,
            "wer": wer,
            "adj_f1": f1,
            "error_type": error_type
        }
        result.update(cat_ocr)
        return result

    def categorize_error(self, teds_s: float, cer: float, n_pred: int, n_gt: int) -> str:
        if teds_s > 0.98 and cer < 0.02:
            return "Success"
        if (cer < 0.05) and (teds_s < 0.9 or n_pred != n_gt):
            return "TSR Error"
        if (teds_s > 0.95) and cer >= 0.05:
            return "OCR Error"
        if teds_s < 0.8 and cer > 0.1:
            return "Combined Error"
        return "Minor Noise"

    def get_summary_stats(self, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {"mean_teds_s": 0.0, "mean_cer": 0.0, "error_distribution": {}}
            
        stats = {
            "mean_teds_s": float(np.mean([r["teds_s"] for r in results])),
            "mean_teds_all": float(np.mean([r["teds_all"] for r in results])),
            "mean_cer": float(np.mean([r["cer"] for r in results])),
            "mean_adj_f1": float(np.mean([r.get("adj_f1", 0.0) for r in results])),
            "cer_numeric": float(np.nanmean([r.get("cer_numeric", np.nan) for r in results])),
            "cer_alpha": float(np.nanmean([r.get("cer_alpha", np.nan) for r in results])),
            "error_distribution": {}
        }
        for r in results:
            err = r["error_type"]
            stats["error_distribution"][err] = stats["error_distribution"].get(err, 0) + 1
        return stats
