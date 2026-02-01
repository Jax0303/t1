"""
Diagnostic Script for S2 (Perfect OCR) Anomaly
"""
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
import csv
import tqdm
from typing import List, Dict, Tuple, Any

# Minimal dependencies from user request
# No fancy pipeline imports to keep it reproducible and isolated

def load_json(path):
    with open(path, 'r') as f: return json.load(f)

def load_chunk(path):
    with open(path, 'r') as f: return json.load(f)['chunks']

def get_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box1[1])
    unionArea = box1Area + box2Area - interArea
    if unionArea == 0: return 0
    return interArea / unionArea

class RobustNormalizer:
    def __init__(self):
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.invert_y = False
        self.pdf_height = 0 # Need this for inversion

    def learn_scale_offset(self, chunks, img_size):
        """
        Estimate transform from simple heuristics.
        SciTSR chunks are usually in PDF points.
        Images are cropped renderings.
        We can try to find the bounding box of chunks and map it to the image size, 
        but SciTSR images are sometimes cropped tightly or loosely.
        
        Strategy:
        1. Calculate bounding box of all chunks.
        2. SciTSR PDF coords: origin usually bottom-left or top-left.
           Chunk format: [x1, y1, x2, y2]? OR [x1, y2, x2, y1]?
           Let's look at the data:
           "pos": [169.03 (x1), 206.64 (y1/y2?), 490.13 (x2), 495.12 (y2/y1?)]
           Since 206 < 495, likely y1 < y2.
           
        3. Check correlation with image size (W, H).
           Chunk BB width = max_x - min_x
           Chunk BB height = max_y - min_y
           Image width, height.
           
           Scale ~ Image_W / Chunk_W (roughly)
        """
        img_w, img_h = img_size
        if not chunks: return

        # Extract all coordinates
        xs = []
        ys = []
        for c in chunks:
            p = c['pos'] # [x1, y1, x2, y2] usually
            xs.extend([p[0], p[2]])
            ys.extend([p[1], p[3]])
            
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        chunk_w = max_x - min_x
        chunk_h = max_y - min_y
        
        # Heuristic 1: PDF to Image scaling
        # SciTSR images are often 72dpi or similar.
        # But crucially, SciTSR images are CROPPED from the PDF page.
        # So (0,0) of image != (0,0) of PDF.
        # However, the relative positions should scale.
        
        # Assumption: The chunks cover the visible table content in the image.
        # We map (min_x, min_y, max_x, max_y) of chunks to (0, 0, img_w, img_h) 
        # BUT leaving some margin? No, usually crop is tight or has random pad.
        # Actually SciTSR images usually include the table region.
        
        # Let's try to infer scale.
        # If we assume global scale factor (e.g. PDF pt -> 72dpi pixels is 1.0, or 300dpi is 4.16)
        
        # AUTO-DETECT Y-AXIS:
        # In PDF (bottom-left origin), larger y means higher up physically? No, usually higher y is upper in content stream? 
        # Actually PDF standard is bottom-left origin. So higher y is 'higher' on page.
        # But text reading order is top-to-bottom.
        # Let's check text order in chunks.
        # If chunks are sorted by reading order, look at y coordinates.
        y_trend = 0
        for i in range(len(chunks)-1):
            curr_y = chunks[i]['pos'][1]
            next_y = chunks[i+1]['pos'][1]
            if next_y < curr_y: y_trend += 1 # Decreasing Y
            if next_y > curr_y: y_trend -= 1 # Increasing Y
            
        # If y_trend is positive (decreasing Y mostly), then PDF y is likely Top-Down (standard image)
        # If y_trend is negative (increasing Y mostly), then PDF y is likely Bottom-Up (standard PDF)
        # However, looking at sample 1705.04916v1.2.chunk:
        # y1=206, y2=495 (huge span? No wait box is [x1, y1, x2, y2]?)
        # Let's look at 1705...chunk:
        # First chunk: "Datasets", pos: [169, 206, 490, 495]. (x1, y1, x2, y2)
        # Width=321, Height=289?? That's a huge box for "Datasets".
        # Ah, maybe [x1, y1, x2, y2] is NOT box but [x1, y1, x2, y2] quad?
        # Or [xmin, ymin, xmax, ymax]?
        # "Datasets" text.
        # Next chunk "BMM": [234, 259, 490, 495]. 
        # y range (259-495)?
        # 490, 495 is remarkably constant in these examples.
        # Wait, chunks 1,2,3,4,5,6 ALL have [..., ..., 490, 495].
        # Is 490, 495 the Page Height/Width? Or Table Boundary?
        # It looks like [x1, y1, TABLE_X2, TABLE_Y2]??? No.
        
        # Let's look closely at 1705...chunk again.
        # Chunk 1: Datasets, [169, 206, 490, 495]
        # Chunk 2: BMM, [234, 259, 490, 495]
        # This looks wrong. The last two numbers are constant for the first row?
        # Row 2 (Dance+Yoga): [159, 216, 477, 482]
        # It seems the last two coordinates are y_bottom or something shared?
        
        # RE-EVALUATION of SciTSR Chunk Format:
        # SciTSR structure json has "cells": [{"start_row":..., "end_row":..., "start_col":..., "end_col":..., "content": [...]}]
        # The Chunk file is distinct.
        
        # Let's assume standard [x1, y1, x2, y2] is WRONG for this specific file/dataset version?
        # Or maybe it is [x_left, y_bottom, x_right, y_top]?
        # 1705..chunk:
        # "Datasets": [169, 206, 490, 495] -> dx=321 ??
        # "BMM"     : [234, 259, 490, 495] -> dx=256 ??
        
        # HYPOTHESIS: SciTSR chunks might be [x1, y1, x2, y2] BUT y is PDF coords (inverted).
        # And maybe they are effectively row-based chunks?
        
        # Let's use the Image Size (665, 239) and the Chunk extremes to simple fitting.
        # Min X in chunks: ~153. Max X in chunks: ~490. Range: 337.
        # Image W: 665. Scale ~ 2.0?
        
        # Min Y in chunks: 206. Max Y in chunks: 495. Range: 289.
        # Image H: 239.
        # This range (289) is LARGER than image height (239).
        # This implies we are capturing full page coords, but image is cropped.
        
        # SCALE STRATEGY:
        # We need to map [min_x, max_x] -> [0 + margin, img_w - margin]
        # and [min_y, max_y] -> [0 + margin, img_h - margin]
        
        # Better: Center the content.
        # scale_x = (img_w * 0.95) / (max_x - min_x)
        # scale_y = (img_h * 0.95) / (max_y - min_y)
        # scale = min(scale_x, scale_y) # aspect ratio preserve?
        
        # Let's implement a 'safe' normalizer that stretches to fit image with slight padding.
        # This is strictly for S2/S4 (Correctness of content injection).
        # If we stretch too much, we deviate from ground truth structure geometry?
        # But S2 uses "Real Predicted Structure" which comes from the IMAGE.
        # So we absolutely MUST map chunks to the IMAGE coordinate system, otherwise
        # the tokens will land outside the structure boxes predicted by TATR/CenterNet.
        
        self.scale_x = img_w / (max_x - min_x + 1e-5)
        self.scale_y = img_h / (max_y - min_y + 1e-5)
        self.offset_x = -min_x * self.scale_x
        self.offset_y = -min_y * self.scale_y
        
        # Check inverted Y
        # If 'Datasets' (Header) has LARGER Y than data, we need inversion (Bottom-Up).
        # Chunk 1 (Datasets): y1=206. Chunk 7 (Dance+Yoga, row 2): y1=216.
        # Y increased. So Top-Down seems plausible (smaller y is top... wait)
        # 206 < 216. If 206 is top row, and 216 is row below, then Y INCREASES downwards.
        # So it is standard Top-Down Image coordinates.
        # But wait, earlier I saw 490, 495.
        # "Datasets" y=206. "Ours" y=455.
        # If "Ours" is last column header...
        # Let's assume standard Top-Down logic first.
        
    def transform(self, box):
        # box: [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        nx1 = x1 * self.scale_x + self.offset_x
        ny1 = y1 * self.scale_y + self.offset_y
        nx2 = x2 * self.scale_x + self.offset_x
        ny2 = y2 * self.scale_y + self.offset_y
        return [min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)]

def build_relations_from_structure(cells: List[Dict]) -> List[Tuple]:
    """
    Generate adjacency relations purely from structural cell definitions.
    cells: list of dicts with 'row', 'col', 'row_span', 'col_span'
    Returns: list of (id1, id2, direction)
    """
    # 1. Map (r, c) to cell indices
    grid = {} # (r, c) -> cell_index
    for idx, cell in enumerate(cells):
        r_start = cell.get('row', cell.get('start_row'))
        c_start = cell.get('col', cell.get('start_col'))
        r_span = cell.get('row_span', 1)
        c_span = cell.get('col_span', 1)
        
        for r in range(r_start, r_start + r_span):
            for c in range(c_start, c_start + c_span):
                grid[(r, c)] = idx
                
    relations = set()
    
    # 2. Scan for adjacencies
    sorted_locs = sorted(grid.keys())
    for r, c in sorted_locs:
        curr_idx = grid[(r, c)]
        
        # Horizontal neighbor (r, c+1)
        if (r, c+1) in grid:
            neighbor_idx = grid[(r, c+1)]
            if curr_idx != neighbor_idx:
                relations.add(tuple(sorted((curr_idx, neighbor_idx)) + ['horizontal']))
                
        # Vertical neighbor (r+1, c)
        if (r+1, c) in grid:
            neighbor_idx = grid[(r+1, c)]
            if curr_idx != neighbor_idx:
                relations.add(tuple(sorted((curr_idx, neighbor_idx)) + ['vertical']))
                
    return list(relations)

def diagnos_s2(table_ids):
    results = []
    
    for tid in table_ids:
        chunk_path = f"data/SciTSR/test/chunk/{tid}.chunk"
        img_path = f"data/SciTSR/test/img/{tid}.png"
        gt_path = f"data/SciTSR/test/structure/{tid}.json"
        
        if not os.path.exists(chunk_path): continue
        
        # Load Data
        img = Image.open(img_path)
        chunks = load_chunk(chunk_path)
        gt_struct = load_json(gt_path)
        
        # 1. Coordinate Normalization
        norm = RobustNormalizer()
        norm.learn_scale_offset(chunks, img.size)
        
        # Transform chunks
        norm_chunks = []
        for c in chunks:
            box = norm.transform(c['pos'])
            norm_chunks.append({'bbox': box, 'text': c['text']})
            
        # Draw Overlays
        debug_dir = f"results/debug/{tid}"
        os.makedirs(debug_dir, exist_ok=True)
        
        draw = ImageDraw.Draw(img.copy())
        for nc in norm_chunks:
            draw.rectangle(nc['bbox'], outline='blue')
        draw._image.save(f"{debug_dir}/chunk_overlay.png")
        
        # 2. Simulation S1 vs S2 using GT structure (To isolate normalization issue)
        # S2: Chunk tokens -> GT Structure
        # We need to map normalized chunks to GT structure cells
        # S2 Logic: Assign chunk tokens to cells based on IOU/Center
        
        # Ideally, we should use the pipeline, but let's emulate the core assignment here
        # to see if chunks land in cells.
        
        # GT Cells to Box (Need bbox for GT cells... prediction gives bbox, GT usually doesn't have spatial bbox in scitsr structure json?)
        # Correct: SciTSR structure json usually only has logical indices.
        # BUT for S2/S4, we map tokens to logical structure.
        # The key is: HOW did we get tokens assigned to cells in S2?
        # EnhancedPipeline uses: `struct = self.tsr.align_ocr_to_structure(ocr_res, s_res)`
        
        # If we emulate S4 (Chunk + GT), we have logical GT but we need SPATIAL GT to assign tokens.
        # If SciTSR structure json lacks bboxes, we CANNOT assign tokens spatially to GT.
        # WE MUST USE TSR PREDICTION for spatial reference.
        
        # So focusing on S2 (Chunk + Predicted TSR) vs S1 (Real OCR + Predicted TSR).
        # We will check if norm_chunks fall into predicted TSR boxes.
        
        # ... For this diagnostic script, we will just output S2 Normalization visual check.
        # If the visual overlay fits, then S2 should work IF assignment logic is sound.
        
        results.append({
            'table_id': tid,
            'status': 'Check visualization',
            'saved': debug_dir
        })
        
    return results

if __name__ == "__main__":
    tids = ['1705.04916v1.2', '0001020v1.12', '0003079v1.1', '0005074v1.1', '0006012v1.14']
    res = diagnos_s2(tids)
    print(json.dumps(res, indent=2))
