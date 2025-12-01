"""
Gemini API Key Manager and Client Wrapper.

Handles API key rotation and failover for Google Gemini API.
"""

import os
import time
import logging
import random
from typing import List, Optional, Any, Dict
from threading import Lock

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    from google.api_core import exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

class GeminiKeyManager:
    """
    Manages multiple Gemini API keys with rotation and failover.
    """
    
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_key_index = 0
        self.lock = Lock()
        self.key_status: Dict[str, bool] = {k: True for k in keys}
        
        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai library not found. Install with: pip install google-generativeai")
            
    def get_current_key(self) -> str:
        """Get the currently active key."""
        with self.lock:
            return self.keys[self.current_key_index]
            
    def rotate_key(self) -> str:
        """Switch to the next available key."""
        with self.lock:
            start_index = self.current_key_index
            while True:
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                current_key = self.keys[self.current_key_index]
                
                if self.key_status[current_key]:
                    logger.info(f"Rotated to Gemini API key index {self.current_key_index}")
                    return current_key
                
                if self.current_key_index == start_index:
                    # All keys marked as bad? Reset all and try again or raise error
                    logger.warning("All keys marked as exhausted. Resetting status.")
                    self.key_status = {k: True for k in self.keys}
                    return current_key

    def mark_key_exhausted(self, key: str):
        """Mark a key as exhausted (hit rate limit or quota)."""
        with self.lock:
            self.key_status[key] = False
            logger.warning(f"Marked Gemini key ending in ...{key[-4:]} as exhausted")

    def generate_content(self, model_name: str, prompt: str, temperature: float = 0.0) -> str:
        """
        Generate content with automatic key rotation on failure.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai library is required")

        max_attempts = len(self.keys) * 2
        
        for attempt in range(max_attempts):
            current_key = self.get_current_key()
            
            try:
                genai.configure(api_key=current_key)
                model = genai.GenerativeModel(model_name)
                
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature
                    )
                )
                
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                is_quota_error = "429" in error_str or "quota" in error_str or "resourceexhausted" in error_str
                
                if is_quota_error:
                    logger.warning(f"Quota exceeded for key ...{current_key[-4:]}. Rotating...")
                    self.mark_key_exhausted(current_key)
                    self.rotate_key()
                else:
                    logger.error(f"Gemini API error: {e}")
                    # For non-quota errors, maybe retry or just fail? 
                    # If it's a model error, rotation won't help.
                    # But let's rotate once just in case it's key specific?
                    # Actually, if it's not quota, it might be invalid key.
                    if "invalid" in error_str or "permission" in error_str:
                         self.mark_key_exhausted(current_key)
                         self.rotate_key()
                    else:
                        raise e
                        
        raise Exception("All Gemini API keys exhausted or failed.")
