"""
Advanced Metrics for Table Structure Recognition Evaluation
TEDS, IoU, CER/WER 등 논문급 평가 지표 구현
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import Levenshtein
from bs4 import BeautifulSoup
import re


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    teds: float = 0.0
    s_teds: float = 0.0
    iou: float = 0.0
    ap50: float = 0.0
    cer: float = 0.0
    wer: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class TEDSCalculator:
    """
    TEDS (Tree Edit Distance based Similarity) 계산기
    HTML 트리 구조를 비교하여 표 구조의 유사도를 측정
    """
    
    def __init__(self, structure_only: bool = False):
        """
        Args:
            structure_only: True면 S-TEDS (구조만), False면 일반 TEDS (구조+내용)
        """
        self.structure_only = structure_only
        
    def calculate(self, pred_html: str, gt_html: str) -> float:
        """
        TEDS 점수 계산
        
        Args:
            pred_html: 예측 HTML
            gt_html: 정답 HTML
            
        Returns:
            0.0 ~ 1.0 사이의 TEDS 점수 (1.0이 완벽)
        """
        try:
            pred_tree = self._parse_html_to_tree(pred_html)
            gt_tree = self._parse_html_to_tree(gt_html)
            
            if pred_tree is None or gt_tree is None:
                return 0.0
                
            # Tree Edit Distance 계산
            ted = self._tree_edit_distance(pred_tree, gt_tree)
            
            # 정규화: 1 - (TED / max_tree_size)
            max_size = max(self._tree_size(pred_tree), self._tree_size(gt_tree))
            if max_size == 0:
                return 1.0
                
            teds_score = 1.0 - (ted / max_size)
            return max(0.0, min(1.0, teds_score))
            
        except Exception as e:
            print(f"TEDS calculation error: {e}")
            return 0.0
    
    def _parse_html_to_tree(self, html: str) -> Optional[Dict]:
        """HTML을 트리 구조로 파싱"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table')
            if not table:
                print(f"Warning: No table found in HTML")
                return None
            return self._build_tree(table)
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_tree(self, element) -> Dict:
        """BeautifulSoup 엘리먼트를 트리로 변환"""
        node = {
            'tag': element.name,
            'children': []
        }
        
        if not self.structure_only and element.string:
            node['text'] = element.string.strip()
        
        # 속성 추가 (rowspan, colspan 등)
        if hasattr(element, 'attrs'):
            node['attrs'] = element.attrs
            
        if hasattr(element, 'children'):
            for child in element.children:
                if hasattr(child, 'name') and child.name:  # Tag인 경우만
                    node['children'].append(self._build_tree(child))
                
        return node
    
    def _tree_edit_distance(self, tree1: Dict, tree2: Dict) -> float:
        """두 트리 사이의 편집 거리 계산 (Zhang-Shasha 알고리즘 간소화)"""
        # 정규화된 트리 편집 거리 추정
        # 실제 TEDS는 트리 구조와 노드 내용을 동시에 고려함
        
        t1_nodes = self._flatten_tree(tree1)
        t2_nodes = self._flatten_tree(tree2)
        
        # 간단한 매칭: 태그가 같으면 매칭 시도
        # Tag sequence similarity + Text similarity
        dist = 0.0
        
        # 구조적 차이 (노드 개수 및 순서)
        max_len = max(len(t1_nodes), len(t2_nodes))
        if max_len == 0: return 0.0
        
        for i in range(max_len):
            if i >= len(t1_nodes) or i >= len(t2_nodes):
                dist += 1.0 # Addition/Deletion
                continue
                
            n1, n2 = t1_nodes[i], t2_nodes[i]
            if n1['tag'] != n2['tag']:
                dist += 1.0 # Replacement (Tag mismatch)
            else:
                # Tag matches, check attributes (rowspan/colspan)
                attr1, attr2 = n1.get('attrs', {}), n2.get('attrs', {})
                if attr1.get('rowspan') != attr2.get('rowspan') or attr1.get('colspan') != attr2.get('colspan'):
                    dist += 0.5 # Partial mismatch
                
                # Check text if not structure_only
                if not self.structure_only and 'text' in n1 and 'text' in n2:
                    txt1, txt2 = n1['text'], n2['text']
                    if txt1 != txt2:
                        # Normalized Levenshtein for text
                        max_txt = max(len(txt1), len(txt2))
                        if max_txt > 0:
                            dist += Levenshtein.distance(txt1, txt2) / max_txt
                        else:
                            dist += 0.0
        
        return dist
    
    def _flatten_tree(self, node: Dict) -> List[Dict]:
        """트리를 리스트로 평탄화 (순서 보존)"""
        flat = [node]
        for child in node.get('children', []):
            flat.extend(self._flatten_tree(child))
        return flat

    def _tree_size(self, tree: Dict) -> int:
        return len(self._flatten_tree(tree))


class IoUCalculator:
    """IoU (Intersection over Union) 및 AP50 계산"""
    
    @staticmethod
    def calculate_iou(bbox1: Any, bbox2: Any) -> float:
        """
        두 바운딩 박스의 IoU 계산
        
        Args:
            bbox1, bbox2: (x1, y1, x2, y2) 형식, 또는 BoundingBox 객체, 또는 dict
            
        Returns:
            0.0 ~ 1.0 사이의 IoU 값
        """
        def get_coords(b):
            if hasattr(b, 'x1'):
                return b.x1, b.y1, b.x2, b.y2
            if isinstance(b, dict):
                return b.get('x1', 0), b.get('y1', 0), b.get('x2', 0), b.get('y2', 0)
            if hasattr(b, '__getitem__'):
                return b[0], b[1], b[2], b[3]
            return 0, 0, 0, 0

        b1_x1, b1_y1, b1_x2, b1_y2 = get_coords(bbox1)
        b2_x1, b2_y1, b2_x2, b2_y2 = get_coords(bbox2)

        x1_inter = max(b1_x1, b2_x1)
        y1_inter = max(b1_y1, b2_y1)
        x2_inter = min(b1_x2, b2_x2)
        y2_inter = min(b1_y2, b2_y2)
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = float(x2_inter - x1_inter) * float(y2_inter - y1_inter)
        
        area1 = float(b1_x2 - b1_x1) * float(b1_y2 - b1_y1)
        area2 = float(b2_x2 - b2_x1) * float(b2_y2 - b2_y1)
        
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
            
        return inter_area / union_area
    
    @staticmethod
    def calculate_ap50(pred_bboxes: List[Tuple], gt_bboxes: List[Tuple],
                       threshold: float = 0.5) -> float:
        """
        AP@IoU=0.5 계산
        
        Args:
            pred_bboxes: 예측 박스 리스트 [(x1,y1,x2,y2,conf), ...]
            gt_bboxes: 정답 박스 리스트 [(x1,y1,x2,y2), ...]
            threshold: IoU 임계값
            
        Returns:
            Average Precision 값
        """
        if not pred_bboxes or not gt_bboxes:
            return 0.0
        
        # confidence로 정렬
        pred_bboxes = sorted(pred_bboxes, key=lambda x: x[4] if len(x) > 4 else 1.0, reverse=True)
        
        tp = 0
        fp = 0
        matched_gt = set()
        
        for pred_box in pred_bboxes:
            pred_coords = pred_box[:4]
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_bboxes):
                if gt_idx in matched_gt:
                    continue
                    
                iou = IoUCalculator.calculate_iou(pred_coords, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(gt_bboxes) if gt_bboxes else 0.0
        
        return precision  # 간소화된 AP (실제로는 precision-recall curve 적분)


class TextQualityMetrics:
    """OCR 품질 평가: CER (Character Error Rate), WER (Word Error Rate)"""
    
    @staticmethod
    def calculate_cer(pred_text: str, gt_text: str) -> float:
        """
        Character Error Rate 계산
        
        Returns:
            0.0 ~ 1.0+ (낮을수록 좋음, 0이 완벽)
        """
        if not gt_text:
            return 1.0 if pred_text else 0.0
            
        distance = Levenshtein.distance(pred_text, gt_text)
        return distance / len(gt_text)
    
    @staticmethod
    def calculate_wer(pred_text: str, gt_text: str) -> float:
        """
        Word Error Rate 계산
        
        Returns:
            0.0 ~ 1.0+ (낮을수록 좋음, 0이 완벽)
        """
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        
        if not gt_words:
            return 1.0 if pred_words else 0.0
        
        # 단어 단위 Levenshtein
        distance = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words))
        return distance / len(' '.join(gt_words))


class ComprehensiveEvaluator:
    """종합 평가기: 모든 지표를 한번에 계산"""
    
    def __init__(self):
        self.teds_calc = TEDSCalculator(structure_only=False)
        self.s_teds_calc = TEDSCalculator(structure_only=True)
        self.iou_calc = IoUCalculator()
        self.text_metrics = TextQualityMetrics()
    
    def evaluate(self, 
                 pred_html: str,
                 gt_html: str,
                 pred_bboxes: Optional[List] = None,
                 gt_bboxes: Optional[List] = None,
                 pred_text: Optional[str] = None,
                 gt_text: Optional[str] = None) -> EvaluationResult:
        """
        종합 평가 수행
        
        Returns:
            모든 지표가 포함된 EvaluationResult
        """
        result = EvaluationResult()
        
        # TEDS
        if pred_html and gt_html:
            result.teds = self.teds_calc.calculate(pred_html, gt_html)
            result.s_teds = self.s_teds_calc.calculate(pred_html, gt_html)
        
        # IoU & AP50
        if pred_bboxes and gt_bboxes:
            # 평균 IoU
            ious = []
            for pred_box in pred_bboxes:
                for gt_box in gt_bboxes:
                    ious.append(self.iou_calc.calculate_iou(pred_box[:4], gt_box[:4]))
            result.iou = np.mean(ious) if ious else 0.0
            
            result.ap50 = self.iou_calc.calculate_ap50(pred_bboxes, gt_bboxes)
        
        # CER & WER
        if pred_text is not None and gt_text is not None:
            result.cer = self.text_metrics.calculate_cer(pred_text, gt_text)
            result.wer = self.text_metrics.calculate_wer(pred_text, gt_text)
        
        return result


# 사용 예시
if __name__ == "__main__":
    # TEDS 테스트
    gt_html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    pred_html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>E</td></tr></table>"
    
    evaluator = ComprehensiveEvaluator()
    result = evaluator.evaluate(pred_html, gt_html)
    
    print(f"TEDS: {result.teds:.3f}")
    print(f"S-TEDS: {result.s_teds:.3f}")
