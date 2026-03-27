"""
Phi-3-Vision Assistant - Clean implementation following official Microsoft documentation
Uses transformers library for proper Phi-3-Vision support
"""
import os
import time
import numpy as np
import cv2


class Phi3VisionAssistant:
    """
    Clean Phi-3-Vision implementation using transformers library.
    Based on official Microsoft Phi-3-Vision documentation.
    """
    
    def __init__(self, model_name="microsoft/Phi-3.5-vision-instruct", lazy_load=True):
        """
        Initialize Phi-3-Vision assistant.
        
        Args:
            model_name: Hugging Face model name
            lazy_load: If True, only load model when needed
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        print(f"[Phi3-Vision] Model: {model_name}")
        
        if not lazy_load:
            self._load_model()
    
    def _load_model(self):
        """Load Phi-3-Vision model using transformers (official approach)."""
        if self.model is not None:
            return
        
        print("[Phi3-Vision] Loading model with transformers...")
        start = time.time()
        
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch
            from PIL import Image
            
            self.PIL_Image = Image
            
            # Load processor (handles tokenization and image processing)
            print("[Phi3-Vision] Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                _attn_implementation='eager'  # Use eager for Jetson compatibility
            )
            
            # Load model with FP16 for memory efficiency
            print("[Phi3-Vision] Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cuda",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                _attn_implementation='eager'
            )
            
            elapsed = time.time() - start
            print(f"[Phi3-Vision] ✅ Model loaded in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"[Phi3-Vision] ❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def ask(self, question, image, max_tokens=150, temperature=0.7):
        """
        Ask a question about an image using Phi-3-Vision.
        
        Args:
            question: Question about the image
            image: OpenCV image (numpy array) or PIL Image
            max_tokens: Maximum response length
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            
        Returns:
            str: Model's response
        """
        # Load model if needed
        if self.model is None:
            self._load_model()
        
        print(f"[Phi3-Vision] Query: {question}")
        start = time.time()
        
        try:
            import torch
            
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = self.PIL_Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # Prepare messages in Phi-3-Vision format (official documentation)
            # Use <|image_1|> token to indicate where image should be processed
            messages = [
                {
                    "role": "user",
                    "content": f"<|image_1|>\n{question}"
                }
            ]
            
            # Apply chat template to format the prompt correctly
            prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs (both text and image)
            inputs = self.processor(
                prompt,
                [pil_image],
                return_tensors="pt"
            ).to("cuda")
            
            # Generation parameters
            generation_args = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True if temperature > 0 else False,
                "top_p": 0.9,
            }
            
            # Generate response
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **generation_args
                )
            
            # Remove input tokens from output (only get generated text)
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            
            # Decode the generated tokens
            answer = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            elapsed = time.time() - start
            
            # Clean up response
            answer = answer.strip()
            
            print(f"[Phi3-Vision] ⚡ Response in {elapsed:.2f}s")
            print(f"[Phi3-Vision] Answer: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            print(f"[Phi3-Vision] ❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def unload(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            print("[Phi3-Vision] Unloading model...")
            self.model = None
            self.processor = None
            
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("[Phi3-Vision] ✅ Model unloaded")

