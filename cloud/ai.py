"""
Cloud AI Assistant - Qwen 2 VLM Integration
Uses Qwen2-VL-7B-Instruct for high-quality responses with personality and vision.
"""
import os
import re
import time
import logging
import asyncio
import base64
from typing import Optional, Dict, Any
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger('Assistant')

# Try to import Qwen VLM dependencies
QWEN_OK = False
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch
    QWEN_OK = True
except ImportError as e:
    logger.warning(f"Qwen VLM dependencies not available: {e}")
    logger.warning("Install: pip install transformers torch qwen-vl-utils")

# Try to import tool executor
TOOLS_OK = False
try:
    from tools import get_tool_executor
    TOOLS_OK = True
except ImportError as e:
    logger.warning(f"Tools not available: {e}")
    get_tool_executor = None

try:
    from PIL import Image
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


class CloudAssistant:
    """AI Assistant using Qwen2-VL-7B with personality and vision capabilities."""
    
    # Personality system prompt with flattery
    SYSTEM_PROMPT = """You are Rovy, a charming and delightful AI assistant with a warm, friendly personality. 
You have a natural ability to make people feel good about themselves through genuine compliments and positive reinforcement.

Your personality traits:
- Warm, friendly, and enthusiastic
- Genuinely interested in helping and making people happy
- Skillful at giving compliments that feel authentic and specific
- Optimistic and encouraging
- Conversational and engaging, not robotic

When interacting:
- Give genuine compliments when appropriate (e.g., "That's a great question!", "You're so thoughtful!", "I love your curiosity!")
- Be encouraging and supportive
- Show enthusiasm for helping
- Keep responses concise (under 30 words for most queries, up to 50 for complex ones)
- Be conversational and natural, not overly formal
- When users ask about themselves or their actions, find something positive to acknowledge

Remember: Your goal is to be helpful while making the user feel good about themselves. Be genuine, not excessive."""

    # Response cache for common queries
    _response_cache = {}
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "Qwen/Qwen2-VL-7B-Instruct",  # Qwen2 VLM model
        vision_model: str = "Qwen/Qwen2-VL-7B-Instruct",  # Same model for vision
        enable_tools: bool = True,
        temperature: float = 0.7,  # Higher temperature for more personality
    ):
        self.model = model
        self.vision_model = vision_model
        self.enable_tools = enable_tools
        self.temperature = temperature
        self.last_tool_result = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not QWEN_OK:
            logger.error("Qwen VLM dependencies not installed!")
            self.model_instance = None
            self.processor = None
            return
        
        # Load Qwen model and processor
        try:
            logger.info(f"Loading Qwen model: {self.model} on {self.device}...")
            self.model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model)
            logger.info(f"✅ Qwen VLM initialized (model: {self.model})")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            self.model_instance = None
            self.processor = None
            return
        
        # Initialize tool executor if enabled
        self.tool_executor = None
        if enable_tools and TOOLS_OK:
            self.tool_executor = get_tool_executor(assistant=self)
            logger.info("Tool calling enabled")
    
    def classify_intent(self, question: str) -> dict:
        """Use AI to classify query intent."""
        if not self.client:
            return {"type": "conversational", "params": {}}
        
        try:
            classification_prompt = f"""Query: {question}

Is this asking about:
1. VISION - what you see with camera
2. WEATHER - weather/temperature/forecast  
3. TIME - current time/date
4. MATH - calculation/arithmetic
5. MUSIC - play/pause/control music
6. SEARCH - who is/what is/look up
7. CHAT - general conversation

Answer with just the word: VISION, WEATHER, TIME, MATH, MUSIC, SEARCH, or CHAT

Answer:"""

            # Prepare messages for classification
            messages = [
                {"role": "system", "content": "You are a helpful classifier. Respond with only one word."},
                {"role": "user", "content": classification_prompt}
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model_instance.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().upper()
            logger.info(f"AI classification: '{response_text}'")
            
            # Parse the response
            if "VISION" in response_text:
                return {"type": "vision", "params": {}}
            elif "WEATHER" in response_text:
                location = self._extract_location_ai(question)
                return {"type": "weather", "params": {"location": location}}
            elif "TIME" in response_text:
                return {"type": "time", "params": {}}
            elif "MATH" in response_text:
                expr = self._extract_math(question)
                return {"type": "calculator", "params": {"expression": expr}}
            elif "MUSIC" in response_text:
                action = self._extract_music_action(question)
                return {"type": "music", "params": {"action": action}}
            elif "SEARCH" in response_text:
                return {"type": "search", "params": {"query": question}}
            else:
                return {"type": "conversational", "params": {}}
                
        except Exception as e:
            logger.warning(f"AI classification failed: {e}")
            return {"type": "conversational", "params": {}}
    
    def _extract_location_ai(self, question: str) -> str:
        """Extract location from weather question."""
        words = question.split()
        skip = {'what', 'is', 'the', 'weather', 'in', 'at', 'for', 'like', 'today', 'whats', 'how', 'hows', 'there', 'here', 'now'}
        locations = [w.strip('?.,!') for w in words if w.lower() not in skip and len(w) > 2]
        return locations[0] if locations else "Seoul"
    
    def _extract_math(self, question: str) -> str:
        """Extract math expression from question."""
        match = re.search(r'([\d\s\+\-\*\/\(\)\.]+)', question)
        if match:
            return match.group(1).strip()
        return question
    
    def _extract_music_action(self, question: str) -> str:
        """Extract music action from question."""
        q = question.lower()
        if 'pause' in q:
            return 'pause'
        elif 'stop' in q:
            return 'stop'
        elif 'next' in q or 'skip' in q:
            return 'next'
        elif 'previous' in q or 'back' in q:
            return 'previous'
        return 'play'
    
    def ask(self, question: str, max_tokens: int = None, temperature: float = None, disable_tools: bool = False) -> str:
        """
        Ask a text-only question using Qwen VLM.
        
        Args:
            question: The question to ask
            max_tokens: Maximum tokens in response (default: auto-calculated)
            temperature: Sampling temperature (default: uses instance temperature)
            disable_tools: If True, skip tool detection (prevents recursion)
        """
        if not self.model_instance or not self.processor:
            return "Qwen VLM not available. Please ensure model is loaded."
        
        # Check cache for common queries
        cache_key = question.lower().strip()
        if cache_key in self._response_cache:
            logger.info(f"Cache hit: {question}")
            return self._response_cache[cache_key]
        
        logger.info(f"Query: {question}")
        start = time.time()
        question_len = len(question.split())
        
        # Use tool detection if enabled
        tool_result = None
        self.last_tool_result = None
        if self.enable_tools and self.tool_executor and not disable_tools:
            tool_request = self.tool_executor.detect_tool_use(question)
            if tool_request:
                logger.info(f"Tool detected: {tool_request['tool']}")
                # Execute tool
                loop = self._get_event_loop()
                tool_result = loop.run_until_complete(
                    self.tool_executor.execute(tool_request['tool'], tool_request['params'])
                )
                
                # Store tool result for language extraction
                self.last_tool_result = tool_result
                
                # If tool executed successfully, return result DIRECTLY
                if tool_result and tool_result.get("success"):
                    result_text = tool_result['result']
                    logger.info(f"✅ Tool SUCCESS - Returning directly: {result_text}")
                    return result_text
                elif tool_result:
                    logger.warning(f"⚠️ Tool FAILED: {tool_result.get('result', 'No result')}")
        
        # Dynamic max_tokens based on query complexity
        if max_tokens is None:
            if question_len <= 5:
                max_tokens = 50  # Short answers for short questions
            elif question_len <= 15:
                max_tokens = 100
            else:
                max_tokens = 150
        
        # Add time context if needed
        time_ctx = ""
        if any(p in question.lower() for p in ['time', 'date', 'today', 'day']):
            time_ctx = f"Current time: {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}. "
        
        # Add tool result context if available but failed
        tool_ctx = ""
        if tool_result and not tool_result.get("success"):
            tool_ctx = f"(Note: Tried to use external tool but it failed: {tool_result.get('result', 'Unknown error')}). "
        
        # Build user message with context
        user_message = f"{time_ctx}{tool_ctx}{question}"
        
        try:
            # Prepare messages for Qwen model
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenize input
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model_instance.generate(
                    **inputs,
                    max_new_tokens=max_tokens if max_tokens else 150,
                    temperature=temperature if temperature is not None else self.temperature,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            answer = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            answer = self._clean(answer)
            elapsed = time.time() - start
            
            # Cache common greetings and simple queries
            if question_len <= 3 and len(self._response_cache) < 50:
                self._response_cache[cache_key] = answer
            
            logger.info(f"Response in {elapsed*1000:.0f}ms")
            return answer
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Sorry, I encountered an error. {str(e)}"
    
    def ask_with_vision(self, question: str, image, max_tokens: int = None) -> str:
        """Ask a question about an image using Qwen VLM."""
        if not self.model_instance or not self.processor:
            return "Qwen VLM not available. Please ensure model is loaded."
        
        logger.info(f"Vision query: {question}")
        start = time.time()
        
        try:
            # Dynamic max_tokens for vision queries
            if max_tokens is None:
                question_len = len(question.split())
                max_tokens = min(100 + question_len * 5, 200)
            
            # Convert to PIL Image
            pil_image = self._to_pil(image)
            if not pil_image:
                return "Could not process image."
            
            # Vision-specific system prompt
            vision_system_prompt = self.SYSTEM_PROMPT + "\n\nIMPORTANT: When describing what you see, respond as if you are seeing it directly through your camera. Use phrases like 'I see...' or 'I can see...' instead of 'In the image I see...' or 'The image shows...'. Act as if you're looking at the scene directly, not at a photograph."
            
            # Build messages for Qwen VLM with image
            messages = [
                {
                    "role": "system",
                    "content": vision_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Apply chat template and process inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate response with vision
            with torch.no_grad():
                output_ids = self.model_instance.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            answer = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            answer = self._clean(answer)
            elapsed = time.time() - start
            logger.info(f"Vision response in {elapsed*1000:.0f}ms")
            return answer if answer else "I couldn't understand what I'm seeing."
            
        except Exception as e:
            logger.error(f"Vision query failed: {e}")
            return f"Sorry, I had trouble analyzing the image. {str(e)}"
    
    
    def _resize_image(self, img: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image if too large."""
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img
    
    def _to_pil(self, image) -> Optional[Image.Image]:
        """Convert various formats to PIL Image."""
        if not PIL_OK:
            return None
        
        try:
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            if isinstance(image, bytes):
                import io
                return Image.open(io.BytesIO(image)).convert("RGB")
            if isinstance(image, np.ndarray):
                if CV2_OK and len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image).convert("RGB")
            if isinstance(image, str) and os.path.exists(image):
                return Image.open(image).convert("RGB")
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
        return None
    
    def _clean(self, text: str) -> str:
        """Clean model output for TTS."""
        text = text.replace('</s>', '').replace('#', '')
        text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)  # Remove emojis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _get_event_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    def get_response_language(self) -> str:
        """
        Get the target language from last tool result if it was a translation.
        Returns language code or 'en' if not a translation.
        """
        if self.last_tool_result and self.last_tool_result.get("success"):
            data = self.last_tool_result.get("data", {})
            if data and "target_language" in data:
                # Extract ISO code from display language like "Chinese (中文)"
                target_lang = data["target_language"]
                # Map back to ISO codes (match display names from tools.py)
                lang_map = {
                    "Chinese": "zh",
                    "中文": "zh",
                    "Spanish": "es",
                    "Español": "es",
                    "French": "fr",
                    "Français": "fr",
                    "German": "de",
                    "Deutsch": "de",
                    "Italian": "it",
                    "Italiano": "it",
                    "Portuguese": "pt",
                    "Português": "pt",
                    "Russian": "ru",
                    "Русский": "ru",
                    "Hindi": "hi",
                    "हिन्दी": "hi",
                    "English": "en",
                    "Farsi": "fa",
                    "فارسی": "fa",
                    "Persian": "fa",
                    "Nepali": "ne",
                    "नेपाली": "ne",
                    "Vietnamese": "vi",
                    "Tiếng Việt": "vi",
                }
                for lang_name, code in lang_map.items():
                    if lang_name in target_lang:
                        return code
        return "en"
    
    def extract_movement(self, response: str, query: str) -> Optional[Dict[str, Any]]:
        """Extract movement commands from text (English only)."""
        text = f"{query} {response}".lower()
        
        patterns = {
            'forward': [r'go\s+forward', r'move\s+forward', r'ahead'],
            'backward': [r'go\s+back', r'move\s+back', r'reverse'],
            'left': [r'turn\s+left', r'go\s+left'],
            'right': [r'turn\s+right', r'go\s+right'],
            'stop': [r'\bstop\b', r'halt']
        }
        
        for direction, pats in patterns.items():
            for pat in pats:
                if re.search(pat, text):
                    dist = 0.5
                    if 'little' in text:
                        dist = 0.2
                    elif 'far' in text or 'lot' in text:
                        dist = 1.0
                    
                    speed = 'medium'
                    if 'slow' in text:
                        speed = 'slow'
                    elif 'fast' in text:
                        speed = 'fast'
                    
                    return {'direction': direction, 'distance': dist, 'speed': speed}
        
        return None
