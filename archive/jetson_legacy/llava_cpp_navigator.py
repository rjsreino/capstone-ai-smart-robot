"""
LLaVA navigator using llama-cpp-python with vision support
Efficient GPU-accelerated inference on Jetson Orin
"""
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import json
import re
from PIL import Image
import numpy as np

class LLaVACppNavigator:
    """
    LLaVA navigator using llama-cpp-python for fast GPU inference.
    """
    
    def __init__(self,
                 model_path="/home/jetson/.cache/llava-v1.5-7b-q4.gguf",
                 mmproj_path="/home/jetson/.cache/llava-mmproj-fixed.gguf",
                 n_gpu_layers=99):
        """
        Initialize LLaVA with llama-cpp-python.
        
        Args:
            model_path: Path to GGUF model
            mmproj_path: Path to vision projector GGUF
            n_gpu_layers: Number of layers to offload to GPU (99 = all)
        """
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        
        print(f"[LLaVA-cpp] Loading model with {n_gpu_layers} GPU layers...")
        
        # Initialize chat handler with vision support
        self.chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
        
        # Load model with GPU acceleration (reduced context for memory)
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=1024,  # Reduced context to fit in memory
            logits_all=True,
            verbose=False,
            n_threads=4
        )
        
        print("[LLaVA-cpp] Model loaded successfully on GPU!")
    
    def get_navigation_command(self, image, custom_prompt=None):
        """
        Get navigation command from image.
        
        Args:
            image: PIL Image or numpy array
            custom_prompt: Optional custom prompt for goal-based navigation
            
        Returns:
            dict: Navigation command
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to base64 data URI
        import io
        import base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_uri = f"data:image/jpeg;base64,{img_str}"
        
        # Create prompt - use custom if provided, otherwise default
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = "Describe this scene briefly. What do you see?"
        
        # Query model
        try:
            response = self.llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            # Extract response
            answer = response['choices'][0]['message']['content']
            
            # Filter out hash marks
            answer = answer.replace('#', '').strip()
            answer_lower = answer.lower()
            
            # Extract action from natural language response
            if 'stop' in answer_lower or 'do not' in answer_lower or 'blocked' in answer_lower or 'obstacle' in answer_lower:
                action = 'stop'
                speed = 'slow'
                distance = 0.0
            elif 'left' in answer_lower or 'turn left' in answer_lower:
                action = 'left'
                if 'clear' in answer_lower or 'open' in answer_lower:
                    speed = 'medium'
                    distance = 0.4
                else:
                    speed = 'slow'
                    distance = 0.2
            elif 'right' in answer_lower or 'turn right' in answer_lower:
                action = 'right'
                if 'clear' in answer_lower or 'open' in answer_lower:
                    speed = 'medium'
                    distance = 0.4
                else:
                    speed = 'slow'
                    distance = 0.2
            elif 'backward' in answer_lower or 'back' in answer_lower or 'reverse' in answer_lower:
                action = 'backward'
                speed = 'slow'
                distance = 0.3
            else:
                # Default forward
                action = 'forward'
                if 'clear' in answer_lower or 'safe' in answer_lower or 'open' in answer_lower:
                    speed = 'medium'
                    distance = 0.5
                elif 'caution' in answer_lower or 'careful' in answer_lower:
                    speed = 'slow'
                    distance = 0.3
                else:
                    speed = 'slow'
                    distance = 0.3
            
            # Clean reasoning text
            reasoning = answer[:100] if answer and len(answer) > 3 else 'AI analysis'
            
            return {
                'action': action,
                'distance': distance,
                'speed': speed,
                'reasoning': reasoning
            }
                
        except Exception as e:
            print(f"[LLaVA-cpp] Error: {e}")
            return {
                'action': 'stop',
                'distance': 0.0,
                'speed': 'slow',
                'reasoning': f'Error: {str(e)[:50]}'
            }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'llm'):
            del self.llm
        if hasattr(self, 'chat_handler'):
            del self.chat_handler
        print("[LLaVA-cpp] Cleaned up")

