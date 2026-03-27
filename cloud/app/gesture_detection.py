"""
Gesture Detection Service using MediaPipe Hands
Detects hand gestures like 'like' (thumbs up) and 'heart' shape
"""
import logging
import numpy as np
from typing import Optional, Literal

LOGGER = logging.getLogger(__name__)

# Try to import MediaPipe Hands
_mediapipe_available = False
_hands_solution = None

try:
    import mediapipe as mp
    _mediapipe_available = True
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    _hands_solution = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    LOGGER.info("MediaPipe Hands loaded successfully")
except ImportError:
    LOGGER.warning("MediaPipe not available. Install with: pip install mediapipe")
except Exception as e:
    LOGGER.warning(f"Failed to initialize MediaPipe: {e}")


GestureType = Literal['like', 'heart', 'none']


class GestureDetector:
    """Detects hand gestures from images using MediaPipe Hands."""
    
    def __init__(self):
        if not _mediapipe_available or not _hands_solution:
            raise RuntimeError("MediaPipe Hands not available")
        self.hands = _hands_solution
    
    def detect_gesture(self, image: np.ndarray) -> tuple[GestureType, float]:
        """
        Detect gesture in image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Tuple of (gesture_type, confidence)
        """
        if not _mediapipe_available:
            return 'none', 0.0
        
        try:
            results = self.hands.process(image)
            
            if not results.multi_hand_landmarks:
                return 'none', 0.0
            
            # Process each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                gesture, confidence = self._classify_gesture(hand_landmarks)
                if gesture != 'none':
                    return gesture, confidence
            
            return 'none', 0.0
            
        except Exception as e:
            LOGGER.error(f"Gesture detection error: {e}")
            return 'none', 0.0
    
    def _classify_gesture(self, landmarks) -> tuple[GestureType, float]:
        """
        Classify gesture based on hand landmarks.
        
        MediaPipe Hand Landmark indices:
        0: WRIST
        4: THUMB_TIP
        8: INDEX_FINGER_TIP
        12: MIDDLE_FINGER_TIP
        16: RING_FINGER_TIP
        20: PINKY_TIP
        """
        # Get landmark coordinates
        landmarks_list = []
        for landmark in landmarks.landmark:
            landmarks_list.append([landmark.x, landmark.y, landmark.z])
        
        landmarks_array = np.array(landmarks_list)
        
        # Check for "like" gesture (thumbs up)
        like_score = self._detect_like_gesture(landmarks_array)
        if like_score > 0.7:
            return 'like', like_score
        
        # Check for "heart" gesture (two hands forming a heart)
        heart_score = self._detect_heart_gesture(landmarks_array)
        if heart_score > 0.6:
            return 'heart', heart_score
        
        return 'none', 0.0
    
    def _detect_like_gesture(self, landmarks: np.ndarray) -> float:
        """
        Detect thumbs up gesture.
        Criteria:
        - Thumb is extended upward
        - Other fingers are closed
        """
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        # Thumb is extended if tip is above IP joint
        thumb_extended = thumb_tip[1] < thumb_ip[1]
        
        # Other fingers are closed if tips are below PIP joints
        index_closed = index_tip[1] > index_pip[1]
        middle_closed = middle_tip[1] > middle_pip[1]
        
        if thumb_extended and index_closed and middle_closed:
            # Calculate confidence based on how clear the gesture is
            thumb_extension = thumb_ip[1] - thumb_tip[1]
            return min(0.9, 0.6 + thumb_extension * 3)
        
        return 0.0
    
    def _detect_heart_gesture(self, landmarks: np.ndarray) -> float:
        """
        Detect heart gesture (two hands forming heart shape).
        For single hand: curved fingers forming heart-like shape.
        """
        # Check if index and middle fingers are curved together
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]
        ring_tip = landmarks[16]
        
        # Fingers should be curved inward
        index_curved = index_tip[1] > index_mcp[1]  # Tip below MCP
        middle_curved = middle_tip[1] > middle_mcp[1]
        
        # Tips should be close together (forming point)
        finger_distance = np.linalg.norm(index_tip[:2] - middle_tip[:2])
        close_together = finger_distance < 0.15
        
        # Ring finger should also be curved
        ring_curved = ring_tip[1] > landmarks[13][1]
        
        if index_curved and middle_curved and close_together:
            # Calculate confidence
            confidence = 0.5
            if ring_curved:
                confidence += 0.2
            if finger_distance < 0.1:
                confidence += 0.2
            return min(0.9, confidence)
        
        return 0.0


# Global detector instance
_gesture_detector: Optional[GestureDetector] = None


def get_gesture_detector() -> Optional[GestureDetector]:
    """Get or create gesture detector instance."""
    global _gesture_detector
    
    if _gesture_detector is None and _mediapipe_available:
        try:
            _gesture_detector = GestureDetector()
            LOGGER.info("Gesture detector initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize gesture detector: {e}")
    
    return _gesture_detector


def detect_gesture_from_image(image_bytes: bytes) -> tuple[GestureType, float]:
    """
    Detect gesture from image bytes (JPEG/PNG).
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        Tuple of (gesture_type, confidence)
    """
    if not _mediapipe_available:
        return 'none', 0.0
    
    try:
        import cv2
        from PIL import Image
        import io
        
        # Decode image
        image = Image.open(io.BytesIO(image_bytes))
        image_rgb = np.array(image.convert('RGB'))
        
        detector = get_gesture_detector()
        if not detector:
            return 'none', 0.0
        
        return detector.detect_gesture(image_rgb)
        
    except Exception as e:
        LOGGER.error(f"Error detecting gesture from image: {e}")
        return 'none', 0.0

