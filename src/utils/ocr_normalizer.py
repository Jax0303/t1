from typing import List, Tuple, Dict, Any

class RobustNormalizer:
    """
    Normalizes PDF coordinate chunks to match Image coordinates.
    Used for aligning ground truth OCR chunks (from SciTSR PDF) with the rendered table image.
    """
    def __init__(self):
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.invert_y = False
        self.pdf_height = 0 

    def learn_scale_offset(self, chunks: List[Dict], img_size: Tuple[int, int]):
        """
        Estimate transform from simple heuristics.
        SciTSR chunks are usually in PDF points.
        Images are cropped renderings.
        """
        img_w, img_h = img_size
        if not chunks: return

        # Extract all coordinates
        xs = []
        ys = []
        for c in chunks:
            p = c['pos']
            # S2 chunk format seems to be [x1, x2, y1, y2] based on inspection
            # x1, x2 are horizontal boundaries
            # y1, y2 are vertical boundaries (PDF coordinates, likely bottom-up)
            xs.extend([p[0], p[1]])
            ys.extend([p[2], p[3]])
            
        if not xs or not ys:
            return

        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        
        # Scale factors
        span_x = self.max_x - self.min_x + 1e-5
        span_y = self.max_y - self.min_y + 1e-5
        
        self.scale_x = img_w / span_x
        self.scale_y = img_h / span_y
        
        # Invert Y logic: PDF max_y is "Top", Image 0 is "Top"
        # We want to map max_y -> 0 and min_y -> img_h
        self.invert_y = True
        
    def transform(self, box: List[float]) -> List[float]:
        """
        Transform a PDF box [x1, x2, y1, y2] to Image box [x1, y1, x2, y2]
        """
        x1, x2, y1, y2 = box
        
        # X mapping (linear left-to-right)
        nx1 = (x1 - self.min_x) * self.scale_x
        nx2 = (x2 - self.min_x) * self.scale_x
        
        # Y mapping (inverted)
        if self.invert_y:
            # Map top (y2) to top (0), bottom (y1) to bottom (img_h)
            # Y in PDF: y2 > y1
            # Y in Image: ny2 > ny1 (but larger value is lower on screen)
            
            # Distance from "Top" (max_y)
            ny1 = (self.max_y - y2) * self.scale_y
            ny2 = (self.max_y - y1) * self.scale_y
        else:
            ny1 = (y1 - self.min_y) * self.scale_y
            ny2 = (y2 - self.min_y) * self.scale_y
            
        return [min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)]
