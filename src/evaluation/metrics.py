import numpy as np
from apted import APTED, Config
from lxml import etree, html
from collections import namedtuple

class TableTreeConfig(Config):
    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.text != node2.text):
            return 1
        return 0

    def children(self, node):
        """Get children of a node"""
        return list(node)

class TEDS:
    """
    Tree Edit Distance Similarity (TEDS) for Table Structure Recognition evaluation.
    Based on the implementation in PubTabNet / ICDAR 2019.
    """
    def __init__(self, structure_only=False, n_jobs=1):
        self.structure_only = structure_only
        self.n_jobs = n_jobs

    def tokenize(self, node):
        """ Tokenizes the node for tree construction """
        return node

    def load_html_tree(self, html_content):
        """Parses HTML string into an lxml tree"""
        try:
            parser = etree.HTMLParser(remove_blank_text=True)
            tree = etree.fromstring(html_content, parser)
            return tree
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return None

    def evaluate(self, pred_html, true_html):
        """
        Computes TEDS score between prediction and ground truth.
        """
        pred_tree = self.load_html_tree(pred_html)
        true_tree = self.load_html_tree(true_html)

        if pred_tree is None or true_tree is None:
            return 0.0

        if self.structure_only:
             # Remove text content for structure-only TEDS (TEDS-Struct)
            self._remove_text(pred_tree)
            self._remove_text(true_tree)

        apted = APTED(true_tree, pred_tree, TableTreeConfig())
        ted = apted.compute_edit_distance()
        
        # TEDS = 1 - TED / max(nodes in true, nodes in pred)
        # Note: This is a simplified calculation; full TEDS usually considers number of nodes directly.
        # However, for APTED comparison, we can use the number of nodes in the trees.
        
        n_nodes_pred = len(list(pred_tree.iter()))
        n_nodes_true = len(list(true_tree.iter()))
        
        if n_nodes_pred == 0 and n_nodes_true == 0:
            return 1.0
        if n_nodes_pred == 0 or n_nodes_true == 0:
            return 0.0

        teds_score = 1.0 - (ted / max(n_nodes_pred, n_nodes_true))
        return max(0.0, teds_score)

    def _remove_text(self, tree):
        for element in tree.iter():
            element.text = None
            element.tail = None

def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_adjacency_relation_metrics(pred_adj, true_adj):
    """
    Calculate Precision, Recall, F1 for adjacency relations.
    Args:
        pred_adj: Set of tuples representing predicted adjacency ((r1, c1), (r2, c2))
        true_adj: Set of tuples representing true adjacency
    """
    start_time = 0 # Placeholder if needed
    
    true_positives = len(pred_adj.intersection(true_adj))
    precision = true_positives / len(pred_adj) if len(pred_adj) > 0 else 0.0
    recall = true_positives / len(true_adj) if len(true_adj) > 0 else 0.0
    f1 = compute_f1(precision, recall)
    
    return precision, recall, f1

def calculate_physical_overlap_metrics(pred_bboxes, true_bboxes, iou_threshold=0.5):
    """
    Calculate Precision, Recall, F1 for physical coordinates based on IoU.
    """
    # Simply using a placeholder logic for now. 
    # Real implementation would require bbox matching (e.g., Hangarian algorithm or greedy match).
    # Assuming pred_bboxes and true_bboxes are lists of [x1, y1, x2, y2]
    
    matches = 0
    # This is a naive O(N*M) implementation for the placeholder
    matched_true_indices = set()
    
    for p_box in pred_bboxes:
        best_iou = 0
        best_t_idx = -1
        for t_idx, t_box in enumerate(true_bboxes):
            if t_idx in matched_true_indices:
                continue
            iou = compute_iou(p_box, t_box)
            if iou > best_iou:
                best_iou = iou
                best_t_idx = t_idx
        
        if best_iou >= iou_threshold:
            matches += 1
            if best_t_idx != -1:
                matched_true_indices.add(best_t_idx)

    precision = matches / len(pred_bboxes) if len(pred_bboxes) > 0 else 0.0
    recall = matches / len(true_bboxes) if len(true_bboxes) > 0 else 0.0
    f1 = compute_f1(precision, recall)
    
    return precision, recall, f1

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

