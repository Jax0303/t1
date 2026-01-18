# import pytest
import numpy as np
from src.hiertable_rag.core.rag import HierTableRAG
from src.hiertable_rag.core.uncertainty import TTAUncertaintyEstimator

def test_adaptive_ocr_logic():
    """
    Verifies that HierTableRAG uses different strategies and adjusts confidence.
    """
    rag = HierTableRAG()
    
    # 1. Simple Table (High Confidence expected)
    simple_table = {
        "id": "simple",
        "rows": [["Header 1", "Header 2"], ["Data A", "Data B"]]
    }
    
    # 2. Complex Table (Lower Confidence expected)
    complex_table = {
        "id": "complex",
        "rows": [
            [{"content": "Nested High", "row_span": 2}, "Col 1", "Col 2"],
            ["Data 1", "Data 2"],
            ["Data 3", "Data 4", "Data 5"] # Irregular row
        ],
        "headers": [{"id": "h1", "text": "H1"}]
    }
    
    # Check Uncertainty Estimator directly
    estimator = TTAUncertaintyEstimator()
    conf_simple = estimator.estimate_confidence(simple_table)
    conf_complex = estimator.estimate_confidence(complex_table)
    
    print(f"Simple Table Confidence: {conf_simple:.3f}")
    print(f"Complex Table Confidence: {conf_complex:.3f}")
    
    assert conf_simple >= conf_complex, "Simple table should have >= confidence than complex"
    
    # Ingest both
    rag.ingest_table(simple_table)
    rag.ingest_table(complex_table)
    
    # Search and check if results from complex table have lower weights (relative to similarity)
    # Since we use random embeddings here, we just check if it runs without error
    response = rag.query("Test query")
    assert "response" in response
    assert response["evidence_count"] > 0

if __name__ == "__main__":
    test_adaptive_ocr_logic()
