import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.hiertable_rag.core.grid_restorer import IntersectionRestorer

def test_tagr_refinement():
    restorer = IntersectionRestorer()
    
    # Synthetic case:
    # Row boundaries at 0.1, 0.49, 0.9 (0.49 is slightly shifted, should be 0.5)
    # OCR boxes: Cell 1 is from 0.15 to 0.45. Cell 2 is from 0.55 to 0.85
    # The gap is between 0.45 and 0.55. The center of the gap is 0.5.
    
    initial_boundaries = [0.1, 0.49, 0.9]
    ocr_boxes = [
        {'box': [0.1, 0.15, 0.9, 0.45], 'content': 'Row 1 content'},
        {'box': [0.1, 0.55, 0.9, 0.85], 'content': 'Row 2 content'}
    ]
    
    print(f"Initial boundaries: {initial_boundaries}")
    
    # Refine rows
    refined_boundaries = restorer._refine_grid_with_ocr(initial_boundaries, ocr_boxes, axis='y')
    print(f"Refined boundaries: {refined_boundaries}")
    
    # Check if 0.49 snapped to 0.45 or 0.55 (nearest edge)
    # In my implemented logic, it snaps to the nearest edge of the intersecting box.
    # 0.49 does not intersect [0.15, 0.45] or [0.55, 0.85] because 0.45 < 0.49 < 0.55 (it is in the gap).
    # Wait, the logic was: "If intersecting, find the nearest gap".
    # If 0.49 is in the gap [0.45, 0.55], it stays as is.
    
    # Case 2: Boundary intersects a box
    # Boundary at 0.3. Intersects Box 1 [0.15, 0.45].
    # Nearest edges are 0.15 and 0.45. 0.3 is equidistant. 
    initial_boundaries_2 = [0.1, 0.3, 0.9]
    refined_boundaries_2 = restorer._refine_grid_with_ocr(initial_boundaries_2, ocr_boxes, axis='y')
    print(f"Case 2 (0.3 intersects [0.15, 0.45]): {refined_boundaries_2}")
    
    assert 0.15 in refined_boundaries_2 or 0.45 in refined_boundaries_2
    print("TAGR Logic Verified: Snapped intersecting line to box edge.")

if __name__ == "__main__":
    test_tagr_refinement()
