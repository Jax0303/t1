"""
Evaluation Metrics for Table Parsing
표 파싱을 위한 평가 메트릭 구현
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed")

from models.base_model import BoundingBox, TableStructure, Cell


def compute_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """두 바운딩 박스의 IoU 계산"""
    return bbox1.iou(bbox2)


def compute_precision_recall_f1(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Precision, Recall, F1 계산
    
    Args:
        predictions: 예측된 바운딩 박스 리스트
        ground_truths: 정답 바운딩 박스 리스트
        iou_threshold: IoU 임계값
        
    Returns:
        Dict with 'precision', 'recall', 'f1'
    """
    if not predictions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not ground_truths:
        return {'precision': 0.0, 'recall': 0.0 if predictions else 1.0, 'f1': 0.0}
    
    # 각 예측에 대해 최대 IoU를 가진 정답 찾기
    matched_gt = set()
    true_positives = 0
    
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(ground_truths) if ground_truths else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': len(predictions) - true_positives,
        'false_negatives': len(ground_truths) - true_positives
    }


def tree_edit_distance(tree1: Any, tree2: Any) -> int:
    """
    간단한 트리 편집 거리 (TEDS를 위한 기본 구현)
    실제로는 더 정교한 알고리즘이 필요하지만, 여기서는 간단한 버전 사용
    """
    if not BS4_AVAILABLE:
        raise ImportError("beautifulsoup4 is required for TEDS computation")
    
    # HTML을 파싱하여 트리 구조 비교
    # 이것은 간단한 구현이며, 실제 TEDS는 더 복잡함
    
    # 트리를 문자열로 변환하여 Levenshtein 거리 계산
    str1 = str(tree1)
    str2 = str(tree2)
    
    return levenshtein_distance(str1, str2)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein 편집 거리 계산"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    """
    TEDS (Tree Edit Distance-based Similarity) 계산
    
    Args:
        pred_html: 예측된 표 HTML
        gt_html: 정답 표 HTML
        structure_only: True면 S-TEDS (구조만), False면 TEDS (구조+내용)
        
    Returns:
        TEDS score (0-1, 1이 완벽한 매칭)
    """
    if not BS4_AVAILABLE:
        raise ImportError("beautifulsoup4 is required for TEDS computation")
    
    # HTML 파싱
    pred_soup = BeautifulSoup(pred_html, 'lxml')
    gt_soup = BeautifulSoup(gt_html, 'lxml')
    
    if structure_only:
        # 구조만 비교: 텍스트 내용 제거
        for tag in pred_soup.find_all(text=True):
            tag.replace_with('')
        for tag in gt_soup.find_all(text=True):
            tag.replace_with('')
    
    # 트리 편집 거리 계산
    distance = tree_edit_distance(pred_soup, gt_soup)
    
    # 정규화: max(len(pred), len(gt))로 나눔
    max_len = max(len(str(pred_soup)), len(str(gt_soup)))
    
    if max_len == 0:
        return 1.0
    
    teds = 1.0 - (distance / max_len)
    return max(0.0, min(1.0, teds))  # 0-1 범위로 클리핑


def compute_cell_adjacency_relation(structure: TableStructure) -> Dict[str, List[Tuple[int, int]]]:
    """
    Cell Adjacency Relation (CAR) 계산
    각 셀의 이웃 관계를 추출
    
    Args:
        structure: TableStructure object
        
    Returns:
        Dict mapping (row, col) to list of adjacent cells
    """
    adjacency = defaultdict(list)
    
    # 셀을 그리드로 매핑
    grid = {}
    for cell in structure.cells:
        grid[(cell.row_idx, cell.col_idx)] = cell
    
    # 각 셀에 대해 인접한 셀 찾기
    for cell in structure.cells:
        r, c = cell.row_idx, cell.col_idx
        
        # 오른쪽
        if (r, c + 1) in grid:
            adjacency[(r, c)].append(('right', r, c + 1))
        
        # 아래
        if (r + 1, c) in grid:
            adjacency[(r, c)].append(('down', r + 1, c))
        
        # 왼쪽
        if (r, c - 1) in grid:
            adjacency[(r, c)].append(('left', r, c - 1))
        
        # 위
        if (r - 1, c) in grid:
            adjacency[(r, c)].append(('up', r - 1, c))
    
    return dict(adjacency)


def compute_car_similarity(pred_structure: TableStructure, gt_structure: TableStructure) -> float:
    """
    CAR 기반 구조 유사도 계산
    
    Args:
        pred_structure: 예측된 표 구조
        gt_structure: 정답 표 구조
        
    Returns:
        Similarity score (0-1)
    """
    pred_car = compute_cell_adjacency_relation(pred_structure)
    gt_car = compute_cell_adjacency_relation(gt_structure)
    
    # 공통 셀 위치
    pred_cells = set((c.row_idx, c.col_idx) for c in pred_structure.cells)
    gt_cells = set((c.row_idx, c.col_idx) for c in gt_structure.cells)
    common_cells = pred_cells & gt_cells
    
    if not common_cells:
        return 0.0
    
    # 각 공통 셀에 대해 인접 관계 일치도 계산
    total_matches = 0
    total_adjacencies = 0
    
    for cell_pos in common_cells:
        pred_adj = set(pred_car.get(cell_pos, []))
        gt_adj = set(gt_car.get(cell_pos, []))
        
        matches = len(pred_adj & gt_adj)
        total = len(pred_adj | gt_adj)
        
        total_matches += matches
        total_adjacencies += total
    
    if total_adjacencies == 0:
        return 1.0
    
    return total_matches / total_adjacencies


class MetricsCalculator:
    """메트릭 계산을 위한 통합 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compute_teds_flag = config.get('compute_teds', True)
        self.compute_s_teds_flag = config.get('compute_s_teds', True)
        self.compute_precision_recall_flag = config.get('compute_precision_recall', True)
        self.compute_iou_flag = config.get('compute_iou', True)
        self.iou_threshold = config.get('iou_threshold', 0.5)
    
    def calculate_all_metrics(
        self,
        pred_structure: TableStructure,
        gt_structure: TableStructure,
        pred_tables: List[BoundingBox],
        gt_tables: List[BoundingBox]
    ) -> Dict[str, float]:
        """모든 메트릭 계산"""
        metrics = {}
        
        # Table Detection 메트릭
        if self.compute_precision_recall_flag and pred_tables and gt_tables:
            td_metrics = compute_precision_recall_f1(pred_tables, gt_tables, self.iou_threshold)
            metrics['td_precision'] = td_metrics['precision']
            metrics['td_recall'] = td_metrics['recall']
            metrics['td_f1'] = td_metrics['f1']
        
        # Table Structure 메트릭
        if pred_structure and gt_structure:
            # TEDS
            if self.compute_teds_flag:
                pred_html = pred_structure.to_html()
                gt_html = gt_structure.to_html()
                try:
                    metrics['teds'] = compute_teds(pred_html, gt_html, structure_only=False)
                except Exception as e:
                    print(f"Error computing TEDS: {e}")
                    metrics['teds'] = 0.0
            
            # S-TEDS
            if self.compute_s_teds_flag:
                pred_html = pred_structure.to_html()
                gt_html = gt_structure.to_html()
                try:
                    metrics['s_teds'] = compute_teds(pred_html, gt_html, structure_only=True)
                except Exception as e:
                    print(f"Error computing S-TEDS: {e}")
                    metrics['s_teds'] = 0.0
            
            # CAR
            try:
                metrics['car_similarity'] = compute_car_similarity(pred_structure, gt_structure)
            except Exception as e:
                print(f"Error computing CAR: {e}")
                metrics['car_similarity'] = 0.0
        
        return metrics
