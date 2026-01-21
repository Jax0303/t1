import unittest
from src.evaluation.metrics import TEDS, calculate_adjacency_relation_metrics, calculate_physical_overlap_metrics
from src.evaluation.rag_metrics import RAGEvaluator

class TestMetrics(unittest.TestCase):
    def test_teds_identical(self):
        teds = TEDS(structure_only=True)
        html_content = "<table><tr><td>cell1</td><td>cell2</td></tr></table>"
        score = teds.evaluate(html_content, html_content)
        self.assertAlmostEqual(score, 1.0)

    def test_teds_different(self):
        teds = TEDS(structure_only=True)
        html_true = "<table><tr><td>1</td><td>2</td></tr></table>"
        html_pred = "<table><tr><td>1</td></tr><tr><td>2</td></tr></table>"
        score = teds.evaluate(html_pred, html_true)
        self.assertTrue(score < 1.0)
        self.assertTrue(score > 0.0)

    def test_adjacency_metrics(self):
        true_adj = {((0,0), (0,1)), ((0,1), (0,2))}
        pred_adj = {((0,0), (0,1))}
        p, r, f1 = calculate_adjacency_relation_metrics(pred_adj, true_adj)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 0.5)
        self.assertAlmostEqual(f1, 2/3)

    def test_physical_f1(self):
        true_boxes = [[0, 0, 10, 10], [10, 0, 20, 10]]
        pred_boxes = [[0, 0, 10, 10]] # Missing one
        p, r, f1 = calculate_physical_overlap_metrics(pred_boxes, true_boxes)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 0.5)

    def test_rag_em(self):
        evaluator = RAGEvaluator()
        self.assertEqual(evaluator.compute_exact_match("Hello World", "hello world"), 1)
        self.assertEqual(evaluator.compute_exact_match("The apple", "apple"), 1)
        self.assertEqual(evaluator.compute_exact_match("Wrong", "Right"), 0)

    def test_recall_at_k(self):
        evaluator = RAGEvaluator()
        retrieved = [1, 2, 3, 4, 5]
        ground_truth = [3, 6]
        score = evaluator.compute_recall_at_k(retrieved, ground_truth, k=3)
        self.assertEqual(score, 0.5) # Only '3' is in top 3

if __name__ == '__main__':
    unittest.main()
