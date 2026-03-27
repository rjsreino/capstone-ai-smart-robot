"""
Vision Processing - Qwen VLM Integration
Captures images from OAK-D camera and analyzes them using Qwen2-VL.
Similar to ChatGPT's camera mode - show an image and get AI analysis.
"""
import os
import io
import base64
import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger('Vision')

# Try to import Qwen VLM
QWEN_OK = False
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch
    QWEN_OK = True
except ImportError:
    logger.warning("Qwen VLM not available. Install: pip install transformers torch qwen-vl-utils")

# Try to import camera support
try:
    from app.camera import CameraService, DepthAICameraSource, OpenCVCameraSource, CameraError
    CAMERA_OK = True
except ImportError:
    CAMERA_OK = False
    logger.warning("Camera module not available")


class VisionProcessor:
    """Process images using Qwen2-VL Vision Language Model.
    
    This enables functionality similar to ChatGPT's camera mode:
    - Show a problem written on paper (e.g., "2+2=?")
    - Get AI analysis and answer
    - Works with OAK-D or USB camera
    """
    
    def __init__(
        self, 
        vision_model: str = "Qwen/Qwen2-VL-7B-Instruct",
        use_oakd: bool = True
    ):
        """Initialize Vision Processor.
        
        Args:
            vision_model: Qwen VLM model to use
            use_oakd: Try to use OAK-D camera first (True), or USB camera (False)
        """
        self.model_instance = None
        self.processor = None
        self.vision_model = vision_model
        self.camera_service = None
        self.use_oakd = use_oakd
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup Qwen VLM
        if QWEN_OK:
            try:
                logger.info(f"Loading Qwen VLM: {vision_model} on {self.device}...")
                self.model_instance = Qwen2VLForConditionalGeneration.from_pretrained(
                    vision_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto"
                )
                self.processor = AutoProcessor.from_pretrained(vision_model)
                logger.info(f"✅ Qwen VLM ready (model: {vision_model})")
            except Exception as e:
                logger.error(f"Failed to load Qwen VLM: {e}")
                self.model_instance = None
                self.processor = None
        else:
            logger.error("Qwen VLM dependencies not installed")
    
    def _init_camera(self) -> bool:
        """Initialize camera service (lazy initialization).
        
        Returns:
            True if camera initialized successfully
        """
        if self.camera_service is not None:
            return True
        
        if not CAMERA_OK:
            logger.error("Camera support not available")
            return False
        
        try:
            primary_source = None
            fallback_source = None
            
            # Try OAK-D first if requested
            if self.use_oakd and DepthAICameraSource.is_available():
                try:
                    logger.info("Initializing OAK-D camera...")
                    primary_source = DepthAICameraSource(
                        preview_width=640,
                        preview_height=480,
                        fps=15.0
                    )
                    logger.info("✅ OAK-D camera initialized")
                except Exception as e:
                    logger.warning(f"OAK-D initialization failed: {e}")
            
            # Try USB camera as fallback or primary
            if OpenCVCameraSource.is_available():
                try:
                    logger.info("Initializing USB camera...")
                    usb_camera = OpenCVCameraSource(
                        device=0,
                        jpeg_quality=90,
                        frame_width=640,
                        frame_height=480
                    )
                    if primary_source is None:
                        primary_source = usb_camera
                        logger.info("✅ USB camera initialized (primary)")
                    else:
                        fallback_source = usb_camera
                        logger.info("✅ USB camera initialized (fallback)")
                except Exception as e:
                    logger.warning(f"USB camera initialization failed: {e}")
            
            if primary_source is None:
                logger.error("No camera sources available")
                return False
            
            self.camera_service = CameraService(
                primary=primary_source,
                fallback=fallback_source
            )
            logger.info("✅ Camera service ready")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    async def capture_image(self) -> Optional[bytes]:
        """Capture a JPEG image from the camera.
        
        Returns:
            JPEG image bytes, or None if capture failed
        """
        if not self._init_camera():
            logger.error("Camera not available")
            return None
        
        try:
            logger.info("📸 Capturing image...")
            jpeg_bytes = await self.camera_service.get_frame()
            logger.info(f"✅ Image captured ({len(jpeg_bytes)} bytes)")
            return jpeg_bytes
        except CameraError as e:
            logger.error(f"Failed to capture image: {e}")
            return None
    
    async def analyze_image(
        self, 
        image_bytes: bytes = None, 
        question: str = "What do you see in this image? If there's a problem or question written, solve it.",
        max_tokens: int = 500,
        detail: str = "auto"
    ) -> Optional[str]:
        """Analyze an image using Qwen VLM.
        
        Args:
            image_bytes: JPEG image bytes (if None, captures from camera)
            question: Question to ask about the image
            max_tokens: Maximum response length
            detail: Ignored (kept for compatibility)
        
        Returns:
            AI response text, or None if analysis failed
            
        Example:
            # Capture from camera and analyze
            response = await processor.analyze_image(
                question="What's written on this paper? If it's a math problem, solve it."
            )
            
            # Analyze existing image
            with open("problem.jpg", "rb") as f:
                response = await processor.analyze_image(
                    image_bytes=f.read(),
                    question="Solve this problem"
                )
        """
        if not self.model_instance or not self.processor:
            logger.error("Qwen VLM not initialized")
            return None
        
        # Capture image if not provided
        if image_bytes is None:
            logger.info("No image provided, capturing from camera...")
            image_bytes = await self.capture_image()
            if image_bytes is None:
                return None
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Resize if too large
            max_size = 1024
            w, h = image.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            logger.info(f"🔍 Analyzing image with Qwen VLM...")
            logger.info(f"   Question: {question}")
            logger.info(f"   Model: {self.vision_model}")
            
            # Prepare messages for Qwen VLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Apply chat template and process
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model_instance.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
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
            )[0].strip()
            
            if answer:
                logger.info(f"✅ Vision Analysis Response: {answer[:100]}...")
                return answer
            else:
                logger.warning("Qwen VLM returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return None
    
    async def solve_problem(self, image_bytes: bytes = None) -> Optional[str]:
        """Solve a problem shown in an image (e.g., math problem on paper).
        
        This is a convenience method optimized for problem-solving.
        
        Args:
            image_bytes: JPEG image bytes (if None, captures from camera)
        
        Returns:
            Solution and explanation
            
        Example:
            # Show paper with "2+2=?" to camera
            solution = await processor.solve_problem()
            # Returns: "The problem shows 2+2. The answer is 4."
        """
        question = (
            "Look at this image carefully. "
            "If there's a problem, question, or puzzle written on it, "
            "please solve it step by step and explain your answer. "
            "If it's a math problem, show your work. "
            "If it's text or other content, describe what you see and provide any relevant information."
        )
        
        return await self.analyze_image(
            image_bytes=image_bytes,
            question=question,
            max_tokens=800
        )
    
    async def describe_scene(self, image_bytes: bytes = None) -> Optional[str]:
        """Describe what's in the camera view.
        
        Args:
            image_bytes: JPEG image bytes (if None, captures from camera)
        
        Returns:
            Scene description
        """
        question = (
            "Describe what you see in this image in detail. "
            "Include objects, people, text, and any other relevant information."
        )
        
        return await self.analyze_image(
            image_bytes=image_bytes,
            question=question,
            max_tokens=500
        )
    
    async def read_text(self, image_bytes: bytes = None) -> Optional[str]:
        """Read and extract text from an image (OCR).
        
        Args:
            image_bytes: JPEG image bytes (if None, captures from camera)
        
        Returns:
            Extracted text
        """
        question = (
            "Please read and transcribe all text visible in this image. "
            "Maintain the original formatting as much as possible."
        )
        
        return await self.analyze_image(
            image_bytes=image_bytes,
            question=question,
            max_tokens=1000
        )
    
    def save_image(self, image_bytes: bytes, filepath: str) -> bool:
        """Save captured image to file.
        
        Args:
            image_bytes: JPEG image bytes
            filepath: Path to save image
        
        Returns:
            True if saved successfully
        """
        try:
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            logger.info(f"✅ Image saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    async def close(self):
        """Clean up resources."""
        if self.camera_service:
            try:
                await self.camera_service.close()
                logger.info("Camera service closed")
            except Exception as e:
                logger.error(f"Error closing camera service: {e}")


# Convenience function for quick use
async def analyze_from_camera(
    question: str = "What do you see?",
    use_oakd: bool = True,
    save_to: str = None
) -> Optional[str]:
    """Quick function to capture and analyze an image from camera.
    
    Args:
        question: Question to ask about the image
        use_oakd: Use OAK-D camera if available
        save_to: Optional path to save the captured image
    
    Returns:
        AI analysis response
        
    Example:
        result = await analyze_from_camera(
            question="Solve the math problem in this image",
            save_to="captured.jpg"
        )
        print(result)
    """
    processor = VisionProcessor(use_oakd=use_oakd)
    
    try:
        # Capture image
        image_bytes = await processor.capture_image()
        if image_bytes is None:
            return None
        
        # Save if requested
        if save_to:
            processor.save_image(image_bytes, save_to)
        
        # Analyze
        result = await processor.analyze_image(image_bytes, question)
        return result
        
    finally:
        await processor.close()


if __name__ == "__main__":
    # Test/Demo
    import asyncio
    
    async def test_vision():
        """Test vision processing."""
        print("=" * 60)
        print("  Qwen VLM Vision Test")
        print("=" * 60)
        
        processor = VisionProcessor()
        
        try:
            print("\n📸 Capturing image from camera...")
            print("💡 TIP: Hold up a paper with a problem like '2+2=?'")
            print("    or any text you want the AI to read.\n")
            
            # Wait a moment for user to position their paper
            await asyncio.sleep(3)
            
            # Solve problem shown to camera
            result = await processor.solve_problem()
            
            if result:
                print("\n✅ AI Response:")
                print("=" * 60)
                print(result)
                print("=" * 60)
            else:
                print("\n❌ Failed to analyze image")
                
        finally:
            await processor.close()
    
    # Run test
    asyncio.run(test_vision())

