"""
평가 메트릭 구현
- 표 추출: F1-score, Precision, Recall, 처리 속도
- RAG 성능: EM score, Hit@K, BLEU score
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import re


class TableExtractionMetrics:
    """표 추출 성능 평가 메트릭"""
    
    @staticmethod
    def calculate_f1_score(predicted_tables: List[pd.DataFrame], 
                          ground_truth_tables: List[pd.DataFrame]) -> float:
        """
        표 추출 F1-score 계산
        
        Args:
            predicted_tables: 추출된 표 리스트
            ground_truth_tables: 정답 표 리스트
            
        Returns:
            F1-score
        """
        if len(predicted_tables) == 0 and len(ground_truth_tables) == 0:
            return 1.0
        
        if len(predicted_tables) == 0 or len(ground_truth_tables) == 0:
            return 0.0
        
        # 표 개수 기반 정밀도와 재현율
        precision = len(predicted_tables) / max(len(predicted_tables), len(ground_truth_tables))
        recall = len(predicted_tables) / max(len(ground_truth_tables), len(predicted_tables))
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # 각 표의 내용 일치도도 고려
        content_similarities = []
        min_len = min(len(predicted_tables), len(ground_truth_tables))
        
        for i in range(min_len):
            pred_df = predicted_tables[i]
            gt_df = ground_truth_tables[i]
            
            # DataFrame 내용 비교
            similarity = TableExtractionMetrics._dataframe_similarity(pred_df, gt_df)
            content_similarities.append(similarity)
        
        if content_similarities:
            content_f1 = np.mean(content_similarities)
            # 구조 F1과 내용 F1의 가중 평균
            f1 = 0.5 * f1 + 0.5 * content_f1
        
        return f1
    
    @staticmethod
    def _dataframe_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """두 DataFrame의 유사도 계산"""
        try:
            # 형태 비교
            shape_match = 1.0 if df1.shape == df2.shape else 0.5
            
            # 내용 비교 (문자열 정규화 후 비교)
            if df1.shape == df2.shape:
                df1_str = df1.astype(str).values.flatten()
                df2_str = df2.astype(str).values.flatten()
                
                matches = sum(1 for a, b in zip(df1_str, df2_str) 
                            if TableExtractionMetrics._normalize_text(a) == 
                               TableExtractionMetrics._normalize_text(b))
                content_match = matches / len(df1_str) if len(df1_str) > 0 else 0.0
            else:
                content_match = 0.0
            
            return 0.3 * shape_match + 0.7 * content_match
        
        except Exception:
            return 0.0
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """텍스트 정규화"""
        if pd.isna(text):
            return ""
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def calculate_extraction_time(extraction_times: List[float]) -> Dict[str, float]:
        """
        추출 시간 통계 계산
        
        Returns:
            평균, 중앙값, 총 시간
        """
        if not extraction_times:
            return {'mean': 0.0, 'median': 0.0, 'total': 0.0}
        
        return {
            'mean': np.mean(extraction_times),
            'median': np.median(extraction_times),
            'total': sum(extraction_times)
        }


class RAGMetrics:
    """RAG 시스템 성능 평가 메트릭"""
    
    @staticmethod
    def calculate_em_score(predicted_answer: str, ground_truth_answer: str) -> float:
        """
        Exact Match (EM) Score 계산
        
        Args:
            predicted_answer: 모델이 생성한 답변
            ground_truth_answer: 정답
            
        Returns:
            EM score (0 또는 1)
        """
        pred_normalized = RAGMetrics._normalize_answer(predicted_answer)
        gt_normalized = RAGMetrics._normalize_answer(ground_truth_answer)
        
        return 1.0 if pred_normalized == gt_normalized else 0.0
    
    @staticmethod
    def calculate_hit_at_k(retrieved_docs: List[Any], 
                          relevant_doc_ids: List[str], 
                          k: int) -> float:
        """
        Hit@K 계산
        
        Args:
            retrieved_docs: 검색된 문서 리스트
            relevant_doc_ids: 관련 문서 ID 리스트
            k: 상위 K개
            
        Returns:
            Hit@K score (0 또는 1)
        """
        if not retrieved_docs or not relevant_doc_ids:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        retrieved_ids = [doc.metadata.get('table_id', '') for doc in top_k_docs]
        
        # 관련 문서 중 하나라도 상위 K개에 포함되면 1
        for rel_id in relevant_doc_ids:
            if rel_id in retrieved_ids:
                return 1.0
        
        return 0.0
    
    @staticmethod
    def calculate_bleu_score(predicted_answer: str, ground_truth_answer: str) -> float:
        """
        BLEU Score 계산 (간단한 버전)
        
        실제로는 nltk의 BLEU를 사용하는 것이 좋지만, 
        여기서는 단순화된 버전 사용
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            pred_tokens = RAGMetrics._tokenize(predicted_answer)
            gt_tokens = RAGMetrics._tokenize(ground_truth_answer)
            
            if len(gt_tokens) == 0:
                return 0.0
            
            score = sentence_bleu([gt_tokens], pred_tokens)
            return score
        
        except Exception:
            # 간단한 n-gram 기반 유사도
            pred_tokens = RAGMetrics._tokenize(predicted_answer)
            gt_tokens = RAGMetrics._tokenize(ground_truth_answer)
            
            if len(gt_tokens) == 0:
                return 0.0
            
            # 1-gram precision
            pred_set = set(pred_tokens)
            gt_set = set(gt_tokens)
            
            if len(pred_set) == 0:
                return 0.0
            
            matches = len(pred_set & gt_set)
            precision = matches / len(pred_set)
            
            return precision
    
    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """답변 정규화"""
        answer = str(answer).strip().lower()
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        return answer
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """텍스트 토큰화"""
        text = RAGMetrics._normalize_answer(text)
        return text.split()

