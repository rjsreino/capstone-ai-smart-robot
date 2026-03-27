"""
Dual Model Assistant - Optimized for Jetson
Uses separate models for text and vision tasks for better performance
"""
import os
import time
import numpy as np
import cv2


class DualModelAssistant:
    """
    Efficient dual-model assistant:
    - Lightweight text model for fast chat/Q&A (Gemma-2-2B)
    - Vision model loaded on-demand for image tasks (LLaVA)
    """
    
    def __init__(
        self,
        text_model_path=None,
        vision_model_path=None,
        vision_mmproj_path=None,
        lazy_load_vision=True
    ):
        """
        Initialize dual model assistant.
        
        Args:
            text_model_path: Path to text-only LLM (default: Gemma-2-2B)
            vision_model_path: Path to VLM model (default: LLaVA 7B)
            vision_mmproj_path: Path to vision projector
            lazy_load_vision: If True, only load vision model when needed
        """
        # Find text model
        if text_model_path is None:
            possible_paths = [
                "/home/jetson/llama.cpp/models/gemma-2-2b-it-Q4_K_S.gguf",
                "/home/jetson/.cache/gemma-2-2b-it-Q4_K_S.gguf",
                "/home/jetson/models/gemma-2-2b-it-Q4_K_S.gguf",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    text_model_path = path
                    break
        
        # Find vision model - prioritize LLaVA-Phi-3-Mini (smaller, faster)
        if vision_model_path is None:
            possible_paths = [
                "/home/jetson/.cache/llava-phi-3-mini-int4.gguf",  # Phi-3 Mini (faster)
                "/home/jetson/.cache/llava-v1.5-7b-q4.gguf",  # Fallback to 7B
                "/home/jetson/models/llava-v1.5-7b-Q4_K_M.gguf",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    vision_model_path = path
                    break
        
        # Find vision projector - match it to the model
        if vision_mmproj_path is None:
            # If using Phi-3-Mini, use its specific mmproj
            if vision_model_path and "phi-3-mini" in vision_model_path:
                possible_paths = [
                    "/home/jetson/.cache/llava-phi-3-mini-mmproj-f16.gguf",
                ]
            else:
                # For LLaVA 1.5, use standard mmproj
                possible_paths = [
                    "/home/jetson/.cache/llava-mmproj-fixed.gguf",
                    "/home/jetson/.cache/llava-mmproj-f16.gguf",
                    "/home/jetson/models/mmproj-model-f16.gguf",
                ]
            
            for path in possible_paths:
                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    vision_mmproj_path = path
                    break
        
        self.text_model_path = text_model_path
        self.vision_model_path = vision_model_path
        self.vision_mmproj_path = vision_mmproj_path
        self.lazy_load_vision = lazy_load_vision
        
        print(f"[DualModel] Text model: {text_model_path}")
        print(f"[DualModel] Vision model: {vision_model_path}")
        print(f"[DualModel] Vision projector: {vision_mmproj_path}")
        
        # Initialize models
        self.text_llm = None
        self.vision_llm = None
        self.vision_chat_handler = None
        
        # Import llama-cpp-python
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            from PIL import Image
            self.Llama = Llama
            self.Llava15ChatHandler = Llava15ChatHandler
            self.PIL_Image = Image
            print("[DualModel] llama-cpp-python available")
        except ImportError as e:
            raise RuntimeError(f"llama-cpp-python required: {e}")
        
        # Free GPU memory before loading models
        self._free_gpu_memory()
        
        # Load text model immediately
        self._load_text_model()
        
        # Load vision model now or later
        if not lazy_load_vision:
            self._load_vision_model()
    
    def _free_gpu_memory(self):
        """Force garbage collection and free GPU memory."""
        import gc
        import torch
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[DualModel] 🧹 Cleared GPU memory")
        except Exception as e:
            # torch might not be available, that's ok
            pass
    
    def _load_text_model(self):
        """Load the lightweight text-only model."""
        if self.text_llm is not None:
            return
        
        if not self.text_model_path or not os.path.exists(self.text_model_path):
            raise RuntimeError(f"Text model not found: {self.text_model_path}")
        
        print("[DualModel] Loading text model...")
        start = time.time()
        
        try:
            self.text_llm = self.Llama(
                model_path=self.text_model_path,
                n_gpu_layers=35,  # Most layers on GPU for speed
                n_ctx=2048,  # Sufficient for chat
                n_threads=6,
                n_batch=512,
                verbose=False,
            )
            elapsed = time.time() - start
            print(f"[DualModel] ✅ Text model loaded in {elapsed:.2f}s")
        except Exception as e:
            print(f"[DualModel] ❌ Failed to load text model: {e}")
            raise
    
    def _load_vision_model(self):
        """Load vision model using llama.cpp (works for both LLaVA and llava-phi-3-mini)."""
        if self.vision_llm is not None:
            return
        
        if not self.vision_model_path or not os.path.exists(self.vision_model_path):
            raise RuntimeError(f"Vision model not found: {self.vision_model_path}")
        
        if not self.vision_mmproj_path or not os.path.exists(self.vision_mmproj_path):
            raise RuntimeError(f"Vision projector not found: {self.vision_mmproj_path}")
        
        # Detect model type for logging
        model_type = "Phi-3-Mini Vision" if "phi" in self.vision_model_path.lower() else "LLaVA 1.5"
        print(f"[DualModel] Loading {model_type} with llama.cpp...")
        start = time.time()
        
        try:
            # Both LLaVA 1.5 and llava-phi-3-mini use the same Llava15ChatHandler
            self.vision_chat_handler = self.Llava15ChatHandler(
                clip_model_path=self.vision_mmproj_path
            )
            
            self.vision_llm = self.Llama(
                model_path=self.vision_model_path,
                chat_handler=self.vision_chat_handler,
                n_gpu_layers=35,
                n_ctx=2048,
                n_threads=6,
                n_batch=512,
                logits_all=True,
                verbose=False,
            )
            
            elapsed = time.time() - start
            print(f"[DualModel] ✅ {model_type} loaded in {elapsed:.2f}s")
        except Exception as e:
            print(f"[DualModel] ❌ Failed to load vision model: {e}")
            raise
    
    def ask_text(self, question, max_tokens=100, temperature=0.7):
        """
        Ask a text-only question (fast path using lightweight model).
        
        Args:
            question: Question text
            max_tokens: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            str: Model's response
        """
        if self.text_llm is None:
            self._load_text_model()
        
        print(f"[DualModel] Text query: {question}")
        start = time.time()
        
        try:
            # Gemma doesn't support system role, use user message with instruction
            response = self.text_llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": f"{question}"
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            
            elapsed = time.time() - start
            answer = response['choices'][0]['message']['content'].strip()
            
            # Clean up for robot assistant speech
            import re
            # Remove emojis (don't work well with TTS)
            answer = re.sub(r'[\U0001F300-\U0001F9FF]', '', answer)  # Emoticons
            answer = re.sub(r'[\U0001F600-\U0001F64F]', '', answer)  # Faces
            answer = re.sub(r'[\U00002600-\U000027BF]', '', answer)  # Misc symbols
            # Remove markdown bold/italic
            answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)  # **text** -> text
            answer = re.sub(r'\*([^*]+)\*', r'\1', answer)      # *text* -> text
            # Remove bullet points
            answer = re.sub(r'^\s*[\*\-•]\s+', '', answer, flags=re.MULTILINE)
            # Remove numbered lists prefix
            answer = re.sub(r'^\s*\d+\.\s+', '', answer, flags=re.MULTILINE)
            # Collapse multiple newlines and spaces
            answer = re.sub(r'\n{3,}', '\n\n', answer)
            answer = re.sub(r'\s+', ' ', answer)  # Collapse multiple spaces
            answer = answer.strip()
            
            print(f"[DualModel] ⚡ Text response in {elapsed:.2f}s")
            return answer
            
        except Exception as e:
            print(f"[DualModel] ❌ Text query failed: {e}")
            return f"Error: {str(e)}"
    
    def ask_vision(self, question, image, max_tokens=150, temperature=0.7):
        """
        Ask a question about an image (vision model).
        
        Args:
            question: Question about the image
            image: OpenCV image (numpy array)
            max_tokens: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            str: Model's response
        """
        # Load vision model once and keep it loaded (stable approach)
        if self.vision_llm is None:
            print(f"[DualModel] Loading vision model...")
            load_start = time.time()
            
            self.vision_chat_handler = self.Llava15ChatHandler(
                clip_model_path=self.vision_mmproj_path, 
                verbose=False
            )
            
            self.vision_llm = self.Llama(
                model_path=self.vision_model_path,
                chat_handler=self.vision_chat_handler,
                n_gpu_layers=28,  # Optimized for Jetson - balance speed/memory
                n_ctx=1024,  # Reduced context for faster inference
                logits_all=True,
                verbose=False,
            )
            print(f"[DualModel] ✅ Vision model loaded in {time.time()-load_start:.2f}s")
        
        print(f"[DualModel] Vision query: {question}")
        start = time.time()
        
        # Reset KV cache
        self.vision_llm.reset()
        
        vision_model = self.vision_llm
        
        try:
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = self.PIL_Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # Use llama.cpp for all vision models (LLaVA and llava-phi-3-mini)
            import io, base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=70)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data_uri = f"data:image/jpeg;base64,{img_str}"
            
            # Use question directly - simple prompts work better
            detailed_question = question
            
            response = vision_model.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": detailed_question}
                    ]
                }],
                max_tokens=max_tokens,
                temperature=max(0.1, temperature),  # Minimum 0.1 to avoid degenerate output
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.2,  # Strong penalty to avoid repetition
                frequency_penalty=0.2,  # Additional repetition control
            )
            
            # Extract response content
            answer = ""
            if isinstance(response, dict) and 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    answer = choice['message']['content']
                elif 'text' in choice:
                    answer = choice['text']
            
            answer = answer.strip() if answer else ""
            
            # Filter out junk responses (common with vision models on cache pollution)
            # If response is only repeated characters (like #####...), treat as empty
            if answer and len(set(answer.replace('\n', '').replace(' ', ''))) <= 2:
                print(f"[DualModel] ⚠️ Filtered junk output (repeated chars)")
                answer = ""
            
            elapsed = time.time() - start
            
            # Clean up response
            answer = answer.replace('#', '').replace('</s>', '').strip()
            answer = " ".join(answer.split())
            
            print(f"[DualModel] ⚡ Vision response in {elapsed:.2f}s")
            return answer
            
        except Exception as e:
            print(f"[DualModel] ❌ Vision query failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def ask(self, question, image=None, max_tokens=512, temperature=0.7, realtime_context=None):
        """
        Ask a question, automatically routing to text or vision model.
        Compatible interface with LLaVaAssistant for drop-in replacement.
        
        Args:
            question: Question text
            image: OpenCV image (numpy array), or None for text-only
            max_tokens: Maximum response length
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            realtime_context: Optional string with real-time info (time, weather, etc.)
            
        Returns:
            str: Model's response
        """
        # Add realtime context to the question if provided
        if realtime_context:
            question = f"{realtime_context}\n\n{question}"
        
        # Route to appropriate model based on presence of image
        if image is not None:
            # Vision task - use Phi vision model
            return self.ask_vision(question, image, max_tokens, temperature)
        else:
            # Text-only task - use Gemma text model
            return self.ask_text(question, max_tokens, temperature)
    
    def unload_vision(self):
        """Unload vision model to free memory."""
        if self.vision_llm is not None:
            print("[DualModel] Unloading vision model...")
            self.vision_llm = None
            self.vision_chat_handler = None
            import gc
            gc.collect()
            print("[DualModel] ✅ Vision model unloaded")

