"""
Visualization Utility for OCR and TSR Evidence
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any

def draw_ocr_overlay(image: Image.Image, tokens: List[Dict]) -> Image.Image:
    """Draw OCR bounding boxes and text on image"""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    for t in tokens:
        bbox = t['bbox']
        # Convert [[x1,y1], ...] to flat list
        points = [tuple(p) for p in bbox]
        draw.polygon(points, outline="red", width=2)
        # Draw text small
        draw.text(points[0], t['text'][:10], fill="blue")
        
    return draw_img

def draw_tsr_overlay(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """Draw TSR bounding boxes (rows, cols) on image"""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    colors = {
        'table row': 'green',
        'table column': 'blue',
        'table spanning cell': 'orange',
        'table': 'red',
        'table column header': 'cyan',
        'table projected row header': 'magenta'
    }
    
    for d in detections:
        label = d['label']
        if label not in colors: continue
        
        bbox = d['bbox'] # [x1, y1, x2, y2]
        draw.rectangle(bbox, outline=colors[label], width=3)
        # Label
        draw.text((bbox[0], bbox[1]-10), label, fill=colors[label])
        
    return draw_img
