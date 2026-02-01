"""
OlmOCR-2-7B-1025 Wrapper for Grounded OCR Extraction
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import os
import re
from typing import List, Dict, Any

class OlmOCROCR:
    def __init__(self, model_id="allenai/olmOCR-2-7B-1025", device="cuda"):
        self.device = device
        print(f"Loading {model_id} in 4-bit quantization...")
        
        # Load model with 4-bit quantization to fit in 8GB VRAM
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def extract_tokens(self, image_path: str) -> List[Dict]:
        """
        Extract text tokens with bounding boxes using grounded prompting.
        Returns format: [{'text': str, 'bbox': [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], 'confidence': 1.0}]
        """
        # Load and resize image for processing
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        # Specific prompt for table grounding
        prompt = "Detect all text segments in this table. For each segment, output: <|box_start|>(y1,x1,y2,x2)<|box_end|> followed by the text content. Coordinates should be 0-1000 normalized."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generation
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        print(f"OlmOCR Output: {output_text[:200]}...")
        return self._parse_grounded_output(output_text, w, h)

    def _parse_grounded_output(self, text: str, img_w: int, img_h: int) -> List[Dict]:
        """Parse grounding tags like <|box_start|>(y1,x1,y2,x2)<|box_end|> content"""
        tokens = []
        # Pattern for <|box_start|>(y1,x1,y2,x2)<|box_end|> content
        pattern = r"<\|box_start\|>\((\d+),(\d+),(\d+),(\d+)\)<\|box_end\|>([^<|\n]+)"
        matches = re.finditer(pattern, text)
        
        for match in matches:
            y1, x1, y2, x2 = map(int, match.groups()[:4])
            content = match.group(5).strip()
            
            px1, py1 = (x1 * img_w / 1000), (y1 * img_h / 1000)
            px2, py2 = (x2 * img_w / 1000), (y2 * img_h / 1000)
            
            tokens.append({
                'text': content,
                'bbox': [[px1, py1], [px2, py1], [px2, py2], [px1, py2]],
                'confidence': 0.95
            })
            
        return tokens
