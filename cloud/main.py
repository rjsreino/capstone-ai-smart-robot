#!/usr/bin/env python3
"""
Rovy Cloud Server - Unified Entry Point
Runs both:
- FastAPI REST API (port 8000) - for mobile app
- WebSocket server (port 8765) - for robot communication

Usage:
    python main.py
"""
import asyncio
import json
import time
import base64
import signal
import sys
import logging
import threading
from datetime import datetime
from typing import Set, Optional

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('RovyCloud')

# Check dependencies
try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    WEBSOCKETS_OK = True
except ImportError:
    WEBSOCKETS_OK = False
    logger.error("websockets not installed. Run: pip install websockets")

try:
    import uvicorn
    UVICORN_OK = True
except ImportError:
    UVICORN_OK = False
    logger.warning("uvicorn not installed. REST API disabled. Run: pip install uvicorn")

# Import AI modules
try:
    from ai import CloudAssistant
    AI_OK = True
except ImportError as e:
    AI_OK = False
    logger.warning(f"AI module not available: {e}")
    CloudAssistant = None

try:
    from speech import SpeechProcessor
    SPEECH_OK = True
except ImportError as e:
    SPEECH_OK = False
    logger.warning(f"Speech module not available: {e}")
    SpeechProcessor = None


class RobotConnection:
    """Manages WebSocket connection to the robot (Raspberry Pi)."""
    
    def __init__(self, server: 'RovyCloudServer'):
        self.server = server
        self.clients: Set[WebSocketServerProtocol] = set()
        self.last_image: Optional[bytes] = None
        self.last_image_time: float = 0.0
        self.last_sensors = {}
        self.exploring: bool = False
        self.exploration_task: Optional[asyncio.Task] = None
        self.exploration_websocket: Optional[WebSocketServerProtocol] = None
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str = "/"):
        """Handle robot WebSocket connection."""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🤖 Robot connected: {client_addr}")
        
        self.clients.add(websocket)
        
        try:
            await self.send_speak(websocket, f"Connected to {config.ASSISTANT_NAME} cloud")
            
            async for message in websocket:
                try:
                    await self.handle_message(websocket, message)
                except Exception as e:
                    logger.error(f"Message error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Robot disconnected: {client_addr}")
        finally:
            # Stop exploration if this was the exploring robot
            if self.exploring and self.exploration_websocket == websocket:
                await self.stop_exploration()
            self.clients.discard(websocket)
    
    async def handle_message(self, websocket: WebSocketServerProtocol, raw_message: str):
        """Process message from robot."""
        msg = json.loads(raw_message)
        msg_type = msg.get('type', '')
        
        if msg_type == 'audio_data':
            await self.handle_audio(websocket, msg)
        elif msg_type == 'image_data':
            await self.handle_image(websocket, msg)
        elif msg_type == 'text_query':
            await self.handle_text_query(websocket, msg)
        elif msg_type == 'sensor_data':
            self.handle_sensor_data(msg)
        elif msg_type == 'wake_word':
            await self.send_speak(websocket, "Yes? I'm listening.")
        elif msg_type == 'ping':
            await websocket.send(json.dumps({"type": "pong"}))
    
    async def handle_audio(self, websocket: WebSocketServerProtocol, msg: dict):
        """Process audio from robot microphone."""
        if not self.server.speech:
            return
        
        logger.info("🎤 Processing audio...")
        
        try:
            audio_bytes = base64.b64decode(msg.get('audio_base64', ''))
            sample_rate = msg.get('sample_rate', 16000)
            
            # Run Whisper STT
            text = await asyncio.get_event_loop().run_in_executor(
                None, self.server.speech.transcribe, audio_bytes, sample_rate
            )
            
            if text:
                logger.info(f"📝 Heard: '{text}'")
                
                # Check for wake word
                text_lower = text.lower()
                wake_detected = any(w in text_lower for w in config.WAKE_WORDS)
                
                if wake_detected:
                    query = text_lower
                    for wake in config.WAKE_WORDS:
                        query = query.replace(wake, "").strip()
                    
                    if query:
                        await self.process_query(websocket, query)
                    else:
                        await self.send_speak(websocket, "Yes? How can I help?")
                else:
                    await self.process_query(websocket, text)
                    
        except Exception as e:
            logger.error(f"Audio error: {e}")
    
    async def handle_image(self, websocket: WebSocketServerProtocol, msg: dict):
        """Store latest camera frame from robot."""
        try:
            img_data = msg.get('image_base64', '')
            if img_data:
                self.last_image = base64.b64decode(img_data)
                self.last_image_time = time.time()
                logger.debug(f"📷 Image received ({len(self.last_image)} bytes)")
            else:
                logger.warning("Empty image_base64 in message")
        except Exception as e:
            logger.error(f"Image error: {e}")
    
    async def handle_text_query(self, websocket: WebSocketServerProtocol, msg: dict):
        """Handle text query from robot."""
        text = msg.get('text', '')
        use_vision = msg.get('include_vision', False)
        logger.info(f"💬 Query: '{text}'")
        await self.process_query(websocket, text, use_vision=use_vision)
    
    def handle_sensor_data(self, msg: dict):
        """Store sensor readings from robot."""
        self.last_sensors = {
            'battery_voltage': msg.get('battery_voltage'),
            'battery_percent': msg.get('battery_percent'),
            'temperature': msg.get('temperature'),
            'imu': {
                'roll': msg.get('imu_roll'),
                'pitch': msg.get('imu_pitch'),
                'yaw': msg.get('imu_yaw')
            }
        }
    
    async def process_query(self, websocket: WebSocketServerProtocol, query: str, use_vision: bool = None):
        """Process query using AI models."""
        if not self.server.assistant:
            await self.send_speak(websocket, "AI not available")
            return
        
        query_lower = query.lower().strip()
        
        # Check for navigation commands
        if ('start' in query_lower or 'begin' in query_lower) and ('auto' in query_lower or 'autonomous' in query_lower) and 'navigation' in query_lower:
            logger.info(f"🤖 Auto navigation command detected: '{query}'")
            await self.send_navigation(websocket, action='start_explore', duration=None)
            await self.send_speak(websocket, "Starting autonomous navigation. I will explore and avoid obstacles.")
            return
        
        if ('start' in query_lower or 'begin' in query_lower) and 'explore' in query_lower:
            logger.info(f"🤖 Explore command detected: '{query}'")
            await self.send_navigation(websocket, action='start_explore', duration=None)
            await self.send_speak(websocket, "Starting exploration mode.")
            return
        
        # Check for stop exploring command (explore start is handled by tool system)
        if ('stop' in query_lower and 'explor' in query_lower) or ('stop explor' in query_lower):
            logger.info(f"🛑 Stop explore command detected: '{query}'")
            await self.stop_exploration()
            await self.send_speak(websocket, "Stopping exploration.")
            return
        
        if ('stop' in query_lower or 'end' in query_lower) and 'navigation' in query_lower:
            logger.info(f"🛑 Stop navigation command detected: '{query}'")
            await self.send_navigation(websocket, action='stop')
            await self.send_speak(websocket, "Stopping navigation.")
            return
        
        # Check for dance command
        if 'dance' in query_lower or 'bust a move' in query_lower or 'show me your moves' in query_lower:
            logger.info(f"💃 Dance command detected: '{query}'")
            
            # Extract dance style if mentioned
            style = 'party'  # default
            if 'wiggle' in query_lower:
                style = 'wiggle'
            elif 'spin' in query_lower:
                style = 'spin'
            elif 'party' in query_lower:
                style = 'party'
            
            # Check if music should be included (default: yes for dance commands)
            with_music = True
            music_genre = 'dance'
            
            # Extract music genre if specified
            if 'classical' in query_lower:
                music_genre = 'classical'
            elif 'jazz' in query_lower:
                music_genre = 'jazz'
            elif 'rock' in query_lower:
                music_genre = 'rock'
            elif 'electronic' in query_lower or 'edm' in query_lower:
                music_genre = 'electronic'
            
            await self.send_dance(websocket, style=style, duration=10, with_music=with_music, music_genre=music_genre)
            await self.send_speak(websocket, f"Let me show you my {style} dance moves with {music_genre} music!")
            return
        
        # Check for music command (without dancing)
        if 'play music' in query_lower or 'play some music' in query_lower:
            logger.info(f"🎵 Music command detected: '{query}'")
            
            # Extract genre
            genre = 'fun'  # default for general music requests
            if 'classical' in query_lower:
                genre = 'classical'
            elif 'jazz' in query_lower:
                genre = 'jazz'
            elif 'rock' in query_lower:
                genre = 'rock'
            elif 'pop' in query_lower:
                genre = 'pop'
            elif 'dance' in query_lower or 'party' in query_lower:
                genre = 'dance'
            elif 'chill' in query_lower or 'relax' in query_lower:
                genre = 'chill'
            elif 'electronic' in query_lower or 'edm' in query_lower:
                genre = 'electronic'
            
            await self.send_music(websocket, action='play', genre=genre)
            await self.send_speak(websocket, f"Playing {genre} music for you!")
            return
        
        # Check for stop music command
        if ('stop' in query_lower or 'pause' in query_lower) and 'music' in query_lower:
            logger.info(f"⏹️ Stop music command detected: '{query}'")
            await self.send_music(websocket, action='stop')
            await self.send_speak(websocket, "Stopping music.")
            return
        
        logger.info(f"🎯 Processing: '{query}'")
        
        # Auto-detect vision need
        if use_vision is None:
            vision_keywords = ['see', 'look', 'what is', 'who is', 'describe', 'camera', 'front']
            use_vision = any(kw in query_lower for kw in vision_keywords)
        
        # Log vision status
        if use_vision:
            if self.last_image:
                logger.info(f"👁️  Using vision (image: {len(self.last_image)} bytes)")
            else:
                logger.warning("👁️  Vision requested but no image available!")
        
        try:
            if use_vision and self.last_image:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.server.assistant.ask_with_vision, query, self.last_image
                )
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.server.assistant.ask, query
                )
            
            # Check if response is from translation tool (needs target language TTS)
            response_language = self.server.assistant.get_response_language()
            await self.send_speak(websocket, response, language=response_language)
            
            # Check for movement commands
            movement = self.server.assistant.extract_movement(response, query)
            if movement:
                await self.send_move(websocket, **movement)
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            await self.send_speak(websocket, "Sorry, I had trouble with that.")
    
    # === Commands to robot ===
    
    async def send_speak(self, websocket: WebSocketServerProtocol, text: str, language: str = "en"):
        """
        Send TTS to robot.
        
        Args:
            text: Text to speak
            language: Language code for TTS (default: 'en')
        """
        audio_b64 = None
        if self.server.speech:
            try:
                audio_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self.server.speech.synthesize, text, language
                )
                if audio_bytes:
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            except Exception as e:
                logger.warning(f"TTS failed: {e}")
        
        msg = {"type": "speak", "text": text, "audio_base64": audio_b64}
        await websocket.send(json.dumps(msg))
        if language != "en":
            logger.info(f"🔊 Sent ({language}): '{text[:50]}...'")
        else:
            logger.info(f"🔊 Sent: '{text[:50]}...'")
    
    async def send_move(self, websocket: WebSocketServerProtocol,
                        direction: str, distance: float = 0.5, speed: str = "medium"):
        """Send movement command to robot."""
        msg = {"type": "move", "direction": direction, "distance": distance, "speed": speed}
        await websocket.send(json.dumps(msg))
        logger.info(f"🚗 Move: {direction} {distance}m")
    
    async def send_gimbal(self, websocket: WebSocketServerProtocol, pan: float, tilt: float):
        """Send gimbal command to robot."""
        msg = {"type": "gimbal", "pan": pan, "tilt": tilt, "action": "move"}
        await websocket.send(json.dumps(msg))
    
    async def send_navigation(self, websocket: WebSocketServerProtocol, action: str, duration: float = None, x: float = None, y: float = None):
        """Send navigation command to robot."""
        msg = {"type": "navigation", "action": action}
        if duration is not None:
            msg["duration"] = duration
        if x is not None:
            msg["x"] = x
        if y is not None:
            msg["y"] = y
        await websocket.send(json.dumps(msg))
        logger.info(f"🧭 Navigation: {action}")
    
    async def send_dance(self, websocket: WebSocketServerProtocol, style: str = 'party', duration: float = 10, 
                        with_music: bool = False, music_genre: str = 'dance'):
        """Send dance command to robot."""
        msg = {
            "type": "dance", 
            "style": style, 
            "duration": duration,
            "with_music": with_music,
            "music_genre": music_genre
        }
        await websocket.send(json.dumps(msg))
        logger.info(f"💃 Dance: {style} for {duration}s" + (f" with {music_genre} music" if with_music else ""))
    
    async def send_music(self, websocket: WebSocketServerProtocol, action: str = 'play', genre: str = 'dance'):
        """Send music command to robot."""
        msg = {"type": "music", "action": action, "genre": genre}
        await websocket.send(json.dumps(msg))
        logger.info(f"🎵 Music: {action}" + (f" ({genre})" if action == 'play' else ""))
    
    async def start_exploration(self, websocket: WebSocketServerProtocol):
        """Start exploration mode - uses camera to navigate autonomously."""
        if self.exploring:
            try:
                await self.send_speak(websocket, "Already exploring!")
            except:
                pass  # Ignore if WebSocket is closed
            return
        
        logger.info("🚀 Starting exploration mode")
        
        # Check if WebSocket is still open
        if websocket.closed:
            raise ConnectionError("WebSocket connection is closed")
        
        self.exploring = True
        self.exploration_websocket = websocket
        
        # Start exploration task - MUST be created on the main server event loop
        # The tool executor runs in a thread pool with a temporary loop, so we need
        # to schedule the task on the main server loop
        main_loop = self.server.main_loop
        if main_loop and main_loop.is_running():
            logger.info(f"🔧 Scheduling exploration task on main server loop: {main_loop}")
            # Use run_coroutine_threadsafe to schedule on main loop from any thread
            future = asyncio.run_coroutine_threadsafe(
                self.exploration_loop(websocket),
                main_loop
            )
            # Store the future as our task reference
            self.exploration_task = future
            logger.info(f"✅ Exploration task scheduled on main loop: {future}")
        else:
            # Fallback: try to create task on current loop (might not work if temporary)
            try:
                loop = asyncio.get_running_loop()
                logger.warning(f"⚠️ Main loop not available, using current loop: {loop}")
                self.exploration_task = loop.create_task(self.exploration_loop(websocket))
            except RuntimeError:
                logger.error("❌ Cannot create exploration task - no event loop available")
                self.exploring = False
                raise RuntimeError("Cannot start exploration - no event loop available")
        
        # Try to send message, but don't fail if WebSocket is closed
        try:
            await self.send_speak(websocket, "Starting exploration mode. I'll navigate using my camera.")
        except Exception as e:
            logger.warning(f"Could not send speak message via WebSocket: {e}, but exploration will continue")
    
    async def stop_exploration(self):
        """Stop exploration mode."""
        if not self.exploring:
            return
        
        logger.info("🛑 Stopping exploration mode")
        self.exploring = False
        
        # Cancel exploration task
        if self.exploration_task:
            if isinstance(self.exploration_task, asyncio.Task):
                if not self.exploration_task.done():
                    self.exploration_task.cancel()
                    try:
                        await self.exploration_task
                    except asyncio.CancelledError:
                        pass
            elif hasattr(self.exploration_task, 'cancel'):  # Future from run_coroutine_threadsafe
                if not self.exploration_task.done():
                    self.exploration_task.cancel()
        
        # Stop robot movement
        if self.exploration_websocket:
            await self.send_move(self.exploration_websocket, direction='stop')
        
        self.exploration_websocket = None
        self.exploration_task = None
    
    async def exploration_loop(self, websocket: WebSocketServerProtocol):
        """Main exploration loop - analyzes images while stationary, then moves smoothly."""
        logger.info("🔍 Exploration loop started - smooth continuous movement mode")
        
        iteration = 0
        current_direction = 'forward'  # Track current movement direction
        
        while self.exploring:
            try:
                iteration += 1
                logger.info(f"🔄 Exploration iteration {iteration}")
                
                # Step 1: Stop the robot briefly to capture a clear, stable image
                logger.debug("🛑 Stopping robot for clear image capture...")
                await self.send_move(websocket, direction='stop', distance=0, speed='medium')
                
                # Step 2: Wait for robot to settle (brief pause)
                await asyncio.sleep(0.3)
                
                # Step 3: Wait for a fresh image to arrive (robot streams continuously)
                image_wait_start = time.time()
                initial_image_time = self.last_image_time
                max_wait = 2.0  # Maximum time to wait for new image
                
                logger.debug("📷 Waiting for fresh image...")
                while self.exploring:
                    if self.last_image and self.last_image_time > initial_image_time:
                        logger.debug(f"✅ Fresh image received (waited {time.time() - image_wait_start:.2f}s)")
                        break
                    
                    if time.time() - image_wait_start > max_wait:
                        logger.warning("⏱️ Timeout waiting for fresh image, using existing")
                        break
                    
                    await asyncio.sleep(0.1)
                
                if not self.exploring:
                    break
                
                if not self.last_image:
                    logger.warning("⚠️ No image available, skipping this iteration")
                    await asyncio.sleep(1.0)
                    continue
                
                # Step 4: Analyze the clear, stationary image
                logger.info(f"📸 Analyzing image for navigation decision (size: {len(self.last_image)} bytes)...")
                
                navigation_prompt = """You are navigating a robot. Look at this camera image and decide the best direction to move to explore the area while avoiding obstacles.

Analyze the image and respond with ONLY one of these words:
- "forward" - if the path ahead is clear
- "left" - if you should turn left to avoid obstacles or explore
- "right" - if you should turn right to avoid obstacles or explore
- "stop" - if there are obstacles blocking all directions

Consider:
- Avoid walls, furniture, and other obstacles
- Prefer open spaces
- If forward is blocked, choose left or right based on which side looks more open
- Be cautious and safe

Respond with only the word: forward, left, right, or stop"""
                
                # Get navigation decision from AI
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.server.assistant.ask_with_vision, navigation_prompt, self.last_image
                )
                
                # Extract direction from response
                response_lower = response.lower().strip()
                direction = None
                
                if 'forward' in response_lower or 'ahead' in response_lower:
                    direction = 'forward'
                elif 'left' in response_lower:
                    direction = 'left'
                elif 'right' in response_lower:
                    direction = 'right'
                elif 'stop' in response_lower or 'blocked' in response_lower:
                    direction = 'stop'
                else:
                    # Default to forward if unclear
                    logger.warning(f"Unclear navigation response: '{response}', defaulting to forward")
                    direction = 'forward'
                
                logger.info(f"🧭 Navigation decision: {direction} (AI: '{response[:60]}')")
                
                # Step 5: Execute movement smoothly
                if direction == 'stop':
                    # Obstacle detected - stay stopped briefly, then try turning
                    logger.info("🛑 Obstacle detected, will try alternate direction")
                    await asyncio.sleep(1.0)
                    # Try turning left on next iteration
                    if current_direction == 'forward':
                        direction = 'left'
                    elif current_direction == 'left':
                        direction = 'right'
                    elif current_direction == 'right':
                        direction = 'forward'
                
                # Move in the chosen direction for a longer distance/time for smooth continuous movement
                if direction != 'stop':
                    current_direction = direction
                    # Use longer distance for smoother continuous movement
                    distance = 0.5 if direction == 'forward' else 0.3
                    await self.send_move(websocket, direction=direction, distance=distance, speed='medium')
                    logger.info(f"🚗 Moving {direction} ({distance}m)")
                    
                    # Continue moving for a bit before next check
                    # This allows smooth continuous movement between image checks
                    await asyncio.sleep(1.5)
                
            except asyncio.CancelledError:
                logger.info("Exploration loop cancelled")
                break
            except Exception as e:
                logger.error(f"Exploration loop error: {e}", exc_info=True)
                await asyncio.sleep(1.0)
        
        # Ensure robot stops when exploration ends
        try:
            await self.send_move(websocket, direction='stop', distance=0, speed='medium')
        except:
            pass
        
        logger.info("🔍 Exploration loop ended")
    
    async def broadcast(self, msg: dict):
        """Send to all connected robots."""
        for client in self.clients:
            try:
                await client.send(json.dumps(msg))
            except:
                pass


class RovyCloudServer:
    """
    Unified cloud server for Rovy robot.
    - REST API for mobile app (FastAPI on port 8000)
    - WebSocket for robot communication (port 8765)
    - AI processing (LLM, Vision, Speech)
    """
    
    def __init__(self):
        self.running = False
        self.assistant = None
        self.speech = None
        self.robot = RobotConnection(self)
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None
        
        logger.info("=" * 60)
        logger.info("  ROVY CLOUD SERVER")
        logger.info(f"  REST API: http://0.0.0.0:{config.API_PORT}")
        logger.info(f"  WebSocket: ws://0.0.0.0:{config.WS_PORT}")
        logger.info(f"  Tailscale: {config.PC_SERVER_IP}")
        logger.info("=" * 60)
        
        # Initialize AI
        self._init_ai()
    
    def _init_ai(self):
        """Initialize AI models."""
        logger.info("Loading AI models...")
        
        if AI_OK and CloudAssistant:
            try:
                # Qwen VLM - pass configuration from config
                self.assistant = CloudAssistant(
                    model=config.QWEN_MODEL,
                    vision_model=config.QWEN_VISION_MODEL,
                    temperature=config.QWEN_TEMPERATURE,
                    enable_tools=True
                )
                # Pass server reference to tool executor
                if self.assistant.tool_executor:
                    self.assistant.tool_executor.server = self
                    logger.info(f"✅ Server reference set on tool executor: {self.assistant.tool_executor.server is not None}")
                logger.info(f"✅ AI assistant ready (Qwen {config.QWEN_MODEL})")
            except Exception as e:
                logger.error(f"AI init failed: {e}")
        
        if SPEECH_OK and SpeechProcessor:
            try:
                self.speech = SpeechProcessor(
                    whisper_model=config.WHISPER_MODEL,
                    tts_engine=config.TTS_ENGINE,
                    piper_voices=config.PIPER_VOICES
                )
                logger.info("✅ Speech processor ready")
            except Exception as e:
                logger.error(f"Speech init failed: {e}")
    
    async def run_websocket_server(self):
        """Run WebSocket server for robot connection."""
        if not WEBSOCKETS_OK:
            logger.error("WebSocket server disabled - websockets not installed")
            return
        
        async with serve(
            self.robot.handle_connection,
            config.HOST,
            config.WS_PORT,
            ping_interval=30,
            ping_timeout=10
        ):
            logger.info(f"✅ WebSocket server running on ws://{config.HOST}:{config.WS_PORT}")
            while self.running:
                await asyncio.sleep(1)
    
    def run_api_server(self):
        """Run FastAPI REST server for mobile app."""
        if not UVICORN_OK:
            logger.error("REST API disabled - uvicorn not installed")
            return
        
        # Import FastAPI app
        try:
            from app.main import app
            logger.info(f"✅ REST API running on http://{config.HOST}:{config.API_PORT}")
            uvicorn.run(app, host=config.HOST, port=config.API_PORT, log_level="warning")
        except Exception as e:
            logger.error(f"REST API failed: {e}")
    
    async def start(self):
        """Start all servers."""
        self.running = True
        
        # Run REST API in separate thread
        if UVICORN_OK:
            api_thread = threading.Thread(target=self.run_api_server, daemon=True)
            api_thread.start()
        
        # Run WebSocket server in main async loop
        await self.run_websocket_server()
    
    def stop(self):
        """Stop all servers."""
        self.running = False
        logger.info("🛑 Server stopping...")


server: RovyCloudServer = None


def get_server() -> RovyCloudServer:
    """Get the global server instance (for use by FastAPI endpoints)."""
    return server


async def broadcast_to_robot(text: str):
    """Send a speak command to all connected robots."""
    if server and server.robot.clients:
        msg = {"type": "speak", "text": text}
        await server.robot.broadcast(msg)
        logger.info(f"🔊 Broadcast to robot: '{text[:50]}...'")
        return True
    return False


def signal_handler(sig, frame):
    logger.info("\n👋 Shutting down...")
    if server:
        server.stop()
    sys.exit(0)


async def main():
    global server
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    ROVY CLOUD SERVER                          ║
    ║              Unified AI + API + Robot Hub                     ║
    ║                                                               ║
    ║  Services:                                                    ║
    ║  • REST API (port 8000) - Mobile app connection              ║
    ║  • WebSocket (port 8765) - Robot connection                  ║
    ║  • AI: LLM + Vision + Speech (Qwen 2 VLM)                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server = RovyCloudServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())

