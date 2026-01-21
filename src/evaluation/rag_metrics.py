import re
import string
from collections import Counter

class RAGEvaluator:
    """
    Evaluator for RAG systems, focusing on retrieval and generation quality.
    """
    def __init__(self, llm_bridge=None):
        self.llm_bridge = llm_bridge

    def normalize_answer(self, s):
        """Lower case, remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(truth))

    def compute_f1(self, prediction, truth):
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(truth).split()
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_recall_at_k(self, retrieved_ids, ground_truth_ids, k):
        """
        Calculate Recall@k for retrieval.
        Args:
            retrieved_ids: List of retrieved document/table IDs.
            ground_truth_ids: List of relevant document/table IDs.
            k: The rank threshold.
        """
        top_k = retrieved_ids[:k]
        hits = len(set(top_k).intersection(set(ground_truth_ids)))
        return hits / len(ground_truth_ids) if len(ground_truth_ids) > 0 else 0.0

    def llm_as_judge(self, question, prediction, context, ground_truth=None):
        """
        Judge the quality of a response using an LLM.
        This is a placeholder for actual LLM API calls.
        """
        if self.llm_bridge is None:
            raise ValueError("LLM bridge must be provided for llm_as_judge.")
        
        prompt = f"""
        당신은 RAG 시스템의 답변 공정성을 평가하는 심판입니다.
        질문과 제공된 컨텍스트, 그리고 시스템의 답변을 바탕으로 정답 여부를 판단하세요.

        질문: {question}
        컨텍스트: {context}
        Ground Truth (참고): {ground_truth if ground_truth else "N/A"}
        시스템 답변: {prediction}

        아래 기준에 따라 0 또는 1로 점수를 매기세요:
        1: 답변이 정확하고 컨텍스트에 기반하며 질문에 적절히 답함.
        0: 답변이 틀렸거나, 컨텍스트와 무관한 내용을 포함하거나, 질문에 답하지 못함.

        점수만 출력하세요 (0 또는 1):
        """
        
        try:
            response = self.llm_bridge.generate(prompt)
            # Simple parsing of 0/1
            score_match = re.search(r'\b(0|1)\b', response)
            if score_match:
                return int(score_match.group(1))
            return 0
        except Exception as e:
            print(f"Error in LLM-as-Judge: {e}")
            return 0

class SimpleLLMBridge:
    """A simple placeholder or wrapper for an LLM (e.g. OpenAI, LangChain)"""
    def __init__(self, model_callable):
        self.model_callable = model_callable

    def generate(self, prompt):
        return self.model_callable(prompt)

