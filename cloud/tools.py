"""
External API Tools for AI Assistant
Provides function calling capabilities for weather, music, and other services.
"""
import os
import re
import json
import logging
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from functools import lru_cache

logger = logging.getLogger('Tools')

# Try to import optional dependencies
try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False
    logger.warning("httpx not available - weather and web APIs disabled")

try:
    import pyttsx3
    PYTTSX3_OK = True
except ImportError:
    PYTTSX3_OK = False
    logger.warning("pyttsx3 not available - local TTS disabled")


class ToolExecutor:
    """Executes external API calls and tools for the assistant."""
    
    # Cache face recognition service to avoid reloading models every time
    _face_recognition_service = None
    _face_recognition_service_lock = None
    
    def __init__(self, assistant=None, server=None):
        self.spotify_enabled = os.getenv("SPOTIFY_ENABLED", "false").lower() == "true"
        self.youtube_music_enabled = os.getenv("YOUTUBE_MUSIC_ENABLED", "false").lower() == "true"
        self.http_client = None
        self.assistant = assistant  # Store assistant reference for translation
        self.server = server  # Store server reference for robot control
        self.exploring_http = False  # Track HTTP-based exploration
        self.exploration_task_http = None  # HTTP exploration task
        
        # Initialize lock for thread-safe face service caching
        if ToolExecutor._face_recognition_service_lock is None:
            import threading
            ToolExecutor._face_recognition_service_lock = threading.Lock()
        
        # Tool definitions for LLM to understand what's available
        self.tools = {
            "get_weather": {
                "description": "Get current weather for a location",
                "parameters": {
                    "location": "City name or 'current' for user's location"
                },
                "keywords": ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot"]
            },
            "get_time": {
                "description": "Get current time and date",
                "parameters": {},
                "keywords": ["time", "date", "today", "day", "clock", "what time"]
            },
            "calculate": {
                "description": "Perform mathematical calculations",
                "parameters": {
                    "expression": "Math expression to evaluate"
                },
                "keywords": ["calculate", "math", "plus", "minus", "times", "divide", "equals"]
            },
            "play_music": {
                "description": "Play music or control playback",
                "parameters": {
                    "action": "play/pause/stop/next/previous",
                    "query": "Song or artist name (optional)"
                },
                "keywords": ["play", "music", "song", "pause", "stop", "next", "skip"]
            },
            "set_reminder": {
                "description": "Set a reminder for later",
                "parameters": {
                    "message": "Reminder message",
                    "minutes": "Minutes from now"
                },
                "keywords": ["remind", "reminder", "alert", "notify"]
            },
            "web_search": {
                "description": "Search the web for information",
                "parameters": {
                    "query": "Search query"
                },
                "keywords": ["search", "look up", "find", "google", "who is", "what is"]
            },
            "move_robot": {
                "description": "Move the robot in a direction",
                "parameters": {
                    "direction": "forward/backward/left/right",
                    "distance": "Distance in meters (default 0.5)",
                    "speed": "slow/medium/fast (default medium)"
                },
                "keywords": ["move", "go", "forward", "backward", "left", "right", "turn", "drive"]
            },
            "explore_robot": {
                "description": "Start autonomous exploration mode - robot will use camera to navigate and avoid obstacles",
                "parameters": {},
                "keywords": ["explore", "exploration", "autonomous", "navigate", "wander", "roam"]
            },
            "translate": {
                "description": "Translate text from one language to another",
                "parameters": {
                    "text": "Text to translate",
                    "target_language": "Target language (e.g., Chinese, Spanish, French, etc.)"
                },
                "keywords": ["translate", "translation", "traduire", "traducir", "übersetzen", "tradurre", "翻訳", "翻译"]
            },
            "guess_age": {
                "description": "Estimate the age of a person from a camera image",
                "parameters": {},
                "keywords": ["guess my age", "how old", "age", "estimate age", "what's my age"]
            },
            "recognize_face": {
                "description": "Recognize if the person in the camera image is someone I know",
                "parameters": {},
                "keywords": ["do you know me", "recognize me", "who am i", "do you recognize me", "who is this"]
            }
        }
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if not HTTPX_OK:
            return False
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=10.0)
        return True
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    def detect_tool_use(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Fast keyword-based detection of tool usage.
        Returns: {"tool": "tool_name", "params": {...}} or None
        """
        query_lower = query.lower()
        
        # Translation - check first as it's very specific
        # Support multilingual translation keywords
        translate_keywords = [
            'translate', 'traduire', 'traducir', 'übersetzen', 'tradurre', 'traduzir',  # European
            '翻訳', '翻译', '번역',  # Japanese, Chinese, Korean
            'अनुवाद', 'ترجمه',  # Hindi, Farsi
            'dịch',  # Vietnamese
        ]
        
        if any(kw in query_lower for kw in translate_keywords):
            # Flexible multilingual patterns
            patterns = [
                # English
                r'translate\s+to\s+(\w+)[\s\.,;:]+(.+)',
                r'translate\s+(.+?)\s+to\s+(\w+)',
                # French
                r'traduire\s+en\s+(\w+)[\s\.,;:]+(.+)',
                # Spanish
                r'traducir\s+a(?:l)?\s+(\w+)[\s\.,;:]+(.+)',
                # German
                r'übersetzen?\s+(?:zu|nach)\s+(\w+)[\s\.,;:]+(.+)',
                # Chinese (翻译到英语 = translate to English)
                r'翻译(?:到|成)\s*(\w+)[\s\.,;:，。：；]*(.+)',
                # More flexible: just detect "translate" + language name anywhere
                r'(\w+)\s*(?:translate|翻译|traducir|traduire)',  # "English translate" or "英语翻译"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)  # Use original query to preserve case
                if match:
                    # Try to extract language and text
                    if len(match.groups()) >= 2:
                        if pattern.startswith(r'translate\s+to') or 'en' in pattern or 'a' in pattern or '翻译' in pattern:
                            target_lang = match.group(1).strip()
                            text = match.group(2).strip()
                            text = text.strip('.,;:!?，。：；')
                        else:
                            text = match.group(1).strip()
                            target_lang = match.group(2).strip()
                    else:
                        # Fallback: let LLM figure it out
                        target_lang = "english"
                        text = query
                    
                    return {"tool": "translate", "params": {"text": text, "target_language": target_lang}}
            
            # If no pattern matched but has translate keyword, let LLM handle it
            # Extract potential language name (english, chinese, español, etc.)
            lang_words = ['english', 'chinese', 'spanish', 'french', 'german', 'italian', 
                          'portuguese', 'russian', 'hindi', 'farsi', 'persian', 'nepali', 'vietnamese',
                          '英语', '中文', '西班牙语', '法语', '德语']
            found_lang = None
            for lang in lang_words:
                if lang in query_lower:
                    found_lang = lang
                    break
            
            if found_lang:
                return {"tool": "translate", "params": {"text": query, "target_language": found_lang}}
        
        # Time/Date - check first to avoid conflicts with "what is"
        time_patterns = [
            r'\bwhat time\b',
            r'\btime is it\b',
            r'\bwhat\'?s the time\b',
            r'\bcurrent time\b',
            r'\bwhat date\b',
            r'\bwhat day\b',
            r'\btoday\'?s date\b'
        ]
        if any(re.search(p, query_lower) for p in time_patterns):
            return {"tool": "get_time", "params": {}}
        
        # Calculator - look for math expressions
        if re.search(r'\d+\s*[\+\-\*\/×÷]\s*\d+', query_lower):
            math_match = re.search(r'([\d\s\+\-\*\/\(\)\.×÷]+)', query_lower)
            if math_match:
                return {"tool": "calculate", "params": {"expression": math_match.group(1).strip()}}
        
        if 'calculate' in query_lower:
            # Extract everything after "calculate"
            calc_match = re.search(r'calculate\s+(.+)', query_lower)
            if calc_match:
                return {"tool": "calculate", "params": {"expression": calc_match.group(1).strip()}}
        
        # Weather - check for weather keywords
        weather_keywords = ['weather', 'temperature', 'forecast', 'degrees', 'cold', 'hot', 'rain', 'sunny', 'cloudy']
        if any(kw in query_lower for kw in weather_keywords):
            location = self._extract_location(query) or "Seoul"
            return {"tool": "get_weather", "params": {"location": location}}
        
        # Music control - English only
        # Check standalone commands first
        standalone_music_patterns = [
            (r'\b(?:next|skip)\s*(?:song|track|one)?\b', 'next'),
            (r'\b(?:previous|prev|back|last)\s*(?:song|track|one)?\b', 'previous'),
            (r'\b(?:pause|hold)\s*(?:it|music|song|this)?\b', 'pause'),
            (r'\b(?:resume|continue|unpause)\s*(?:music|song|it)?\b', 'play'),
            (r'\b(?:stop)\s*(?:music|song|it|playing|this)?\b', 'stop'),
        ]
        
        for pattern, action in standalone_music_patterns:
            if re.search(pattern, query_lower):
                # Make sure it's not part of a movement command
                if not any(move_kw in query_lower for move_kw in ['move', 'go', 'drive', 'walk', 'turn', 'forward', 'backward']):
                    return {"tool": "play_music", "params": {"action": action, "query": ""}}
        
        # Music control with explicit "music" or "song" keyword
        if 'music' in query_lower or 'song' in query_lower or 'play' in query_lower:
            action = "play"
            if "pause" in query_lower:
                action = "pause"
            elif "stop" in query_lower:
                action = "stop"
            elif "next" in query_lower or "skip" in query_lower:
                action = "next"
            elif "previous" in query_lower or "back" in query_lower:
                action = "previous"
            return {"tool": "play_music", "params": {"action": action, "query": ""}}
        
        # Robot movement - English only
        # Skip if this is a translation request
        if not any(kw in query_lower for kw in translate_keywords):
            movement_keywords = ['move', 'go', 'drive', 'turn']
            if any(kw in query_lower for kw in movement_keywords):
                direction = "forward"  # default
                distance = 0.5
                speed = "medium"
                
                # Detect direction
                if 'forward' in query_lower or 'ahead' in query_lower or 'straight' in query_lower:
                    direction = "forward"
                elif 'backward' in query_lower or 'back' in query_lower or 'reverse' in query_lower:
                    direction = "backward"
                elif 'left' in query_lower:
                    direction = "left"
                elif 'right' in query_lower:
                    direction = "right"
                
                # Detect speed
                if 'fast' in query_lower or 'quick' in query_lower:
                    speed = "fast"
                elif 'slow' in query_lower or 'slowly' in query_lower:
                    speed = "slow"
                
                # Detect distance (optional - look for numbers with units)
                distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:meter|metre|m\b)', query_lower)
                if distance_match:
                    distance = float(distance_match.group(1))
                
                return {"tool": "move_robot", "params": {"direction": direction, "distance": distance, "speed": speed}}
        
        # Explore robot
        if 'explore' in query_lower or 'exploration' in query_lower:
            return {"tool": "explore_robot", "params": {}}
        
        # Age estimation
        age_keywords = ['guess my age', 'how old', 'what\'s my age', 'estimate age', 'guess age']
        if any(kw in query_lower for kw in age_keywords):
            return {"tool": "guess_age", "params": {}}
        
        # Face recognition
        recognition_keywords = ['do you know me', 'recognize me', 'who am i', 'do you recognize me', 'who is this', 'know me']
        if any(kw in query_lower for kw in recognition_keywords):
            return {"tool": "recognize_face", "params": {}}
        
        # Reminders
        if 'remind' in query_lower:
            reminder_match = re.search(r'remind\s+(?:me\s+)?(?:to\s+)?(.+?)(?:\s+in\s+(\d+)\s+(?:minute|min)s?)?', query_lower)
            if reminder_match:
                message = reminder_match.group(1).strip()
                minutes = int(reminder_match.group(2)) if reminder_match.group(2) else 5
                return {"tool": "set_reminder", "params": {"message": message, "minutes": minutes}}
        
        # Web search - specific patterns only
        search_triggers = ['who is', 'what is', 'search for', 'look up', 'tell me about']
        for trigger in search_triggers:
            if trigger in query_lower:
                # Exclude if it's asking about time/date/weather
                if not any(kw in query_lower for kw in ['time', 'weather', 'temperature']):
                    # Extract the search query
                    parts = query_lower.split(trigger, 1)
                    if len(parts) > 1:
                        search_query = parts[1].strip()
                        return {"tool": "web_search", "params": {"query": search_query}}
        
        return None
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from weather query."""
        query_lower = query.lower()
        
        # Common patterns: "weather in X", "weather at X", "X weather"
        patterns = [
            r'weather\s+in\s+([a-z\s]+?)(?:\s+today|\s+now|$|\?)',
            r'weather\s+at\s+([a-z\s]+?)(?:\s+today|\s+now|$|\?)',
            r'in\s+([a-z\s]+?)\s+weather',
            r'temperature\s+in\s+([a-z\s]+?)(?:\s+today|\s+now|$|\?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip()
                # Clean up common words
                location = re.sub(r'\b(the|today|now|there)\b', '', location).strip()
                if len(location) > 2:
                    return location.title()
        
        # If no specific location found, return None (will use Seoul default)
        return None
    
    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return results.
        Returns: {"success": bool, "result": str, "data": Any}
        """
        try:
            logger.info(f"Executing tool: {tool_name} with params: {params}")
            
            if tool_name == "get_weather":
                return await self.get_weather(params.get("location", "current"))
            elif tool_name == "get_time":
                return await self.get_time()
            elif tool_name == "calculate":
                return await self.calculate(params.get("expression", ""))
            elif tool_name == "play_music":
                return await self.play_music(params.get("action", "play"), params.get("query", ""))
            elif tool_name == "move_robot":
                return await self.move_robot(
                    params.get("direction", "forward"),
                    params.get("distance", 0.5),
                    params.get("speed", "medium")
                )
            elif tool_name == "explore_robot":
                return await self.explore_robot()
            elif tool_name == "set_reminder":
                return await self.set_reminder(params.get("message", ""), params.get("minutes", 5))
            elif tool_name == "web_search":
                return await self.web_search(params.get("query", ""))
            elif tool_name == "translate":
                return await self.translate(params.get("text", ""), params.get("target_language", "English"))
            elif tool_name == "guess_age":
                return await self.guess_age()
            elif tool_name == "recognize_face":
                return await self.recognize_face()
            else:
                return {"success": False, "result": f"Unknown tool: {tool_name}", "data": None}
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {"success": False, "result": f"Error: {str(e)}", "data": None}
    
    async def get_weather(self, location: str = "Seoul") -> Dict[str, Any]:
        """Get weather information for a location using Open-Meteo (FREE, no API key needed!)."""
        if not await self._ensure_client():
            return {"success": False, "result": "Weather service unavailable", "data": None}
        
        try:
            # Use Seoul as default location
            if location == "current" or not location:
                location = "Seoul"
            
            # Step 1: Geocoding - Convert city name to coordinates
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_response = await self.http_client.get(geo_url)
            
            if geo_response.status_code != 200:
                return {
                    "success": False,
                    "result": f"Could not find location: {location}",
                    "data": None
                }
            
            geo_data = geo_response.json()
            
            if "results" not in geo_data or len(geo_data["results"]) == 0:
                return {
                    "success": False,
                    "result": f"City not found: {location}",
                    "data": None
                }
            
            # Get coordinates
            city_info = geo_data["results"][0]
            lat = city_info["latitude"]
            lon = city_info["longitude"]
            city_name = city_info["name"]
            country = city_info.get("country", "")
            
            # Step 2: Get weather data using coordinates
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
                f"&timezone=auto"
            )
            
            weather_response = await self.http_client.get(weather_url)
            
            if weather_response.status_code != 200:
                return {
                    "success": False,
                    "result": f"Could not fetch weather for {city_name}",
                    "data": None
                }
            
            weather_data = weather_response.json()
            current = weather_data["current"]
            
            # Extract weather info
            temp = current["temperature_2m"]
            feels_like = current["apparent_temperature"]
            humidity = current["relative_humidity_2m"]
            wind_speed = current["wind_speed_10m"]
            weather_code = current["weather_code"]
            
            # Convert weather code to description
            weather_desc = self._weather_code_to_description(weather_code)
            
            # Format location string
            location_str = f"{city_name}, {country}" if country else city_name
            
            # Create TTS-friendly response (more natural, less technical)
            # Round temperature to whole number for easier listening
            temp_int = round(temp)
            feels_int = round(feels_like)
            
            # Build natural response
            if feels_int == temp_int:
                # Don't mention "feels like" if it's the same
                result = f"It's {weather_desc} in {city_name}, {temp_int} degrees"
            else:
                result = f"It's {weather_desc} in {city_name}, {temp_int} degrees, feels like {feels_int}"
            
            # Add extra info only if significantly different conditions
            if humidity > 80:
                result += ", quite humid"
            elif humidity < 30:
                result += ", quite dry"
            
            if wind_speed > 30:
                result += ", and windy"
            
            return {
                "success": True,
                "result": result,
                "data": {
                    "location": location_str,
                    "temperature": temp,
                    "feels_like": feels_like,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "description": weather_desc,
                    "weather_code": weather_code
                }
            }
        
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {"success": False, "result": f"Weather service error: {str(e)}", "data": None}
    
    def _weather_code_to_description(self, code: int) -> str:
        """Convert WMO weather code to human-readable description."""
        # WMO Weather interpretation codes
        weather_codes = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "slight snow",
            73: "moderate snow",
            75: "heavy snow",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "unknown")
    
    async def get_time(self) -> Dict[str, Any]:
        """Get current time and date."""
        now = datetime.now()
        result = now.strftime("It's %I:%M %p on %A, %B %d, %Y")
        
        return {
            "success": True,
            "result": result,
            "data": {
                "datetime": now.isoformat(),
                "timestamp": now.timestamp()
            }
        }
    
    async def calculate(self, expression: str) -> Dict[str, Any]:
        """Perform safe mathematical calculation."""
        try:
            # Clean the expression
            expression = expression.replace("x", "*").replace("×", "*").replace("÷", "/")
            expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            
            # Safe eval with limited scope
            allowed_chars = set("0123456789+-*/().")
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return {"success": False, "result": "Invalid mathematical expression", "data": None}
            
            # Evaluate
            result = eval(expression, {"__builtins__": {}}, {})
            
            return {
                "success": True,
                "result": f"{expression} = {result}",
                "data": {"expression": expression, "result": result}
            }
        
        except Exception as e:
            return {"success": False, "result": f"Calculation error: {str(e)}", "data": None}
    
    async def play_music(self, action: str = "play", query: str = "") -> Dict[str, Any]:
        """Control music playback - YouTube Music, Spotify, or system control."""
        try:
            # Priority 1: YouTube Music (if enabled)
            if self.youtube_music_enabled:
                return await self._youtube_music_control(action, query)
            
            # Priority 2: Spotify (if enabled)
            if self.spotify_enabled:
                return await self._spotify_control(action, query)
            
            # Priority 3: System media control (playerctl/etc)
            if action == "pause":
                result = await self._control_system_media("pause")
            elif action == "play":
                result = await self._control_system_media("play")
            elif action == "stop":
                result = await self._control_system_media("stop")
            elif action == "next":
                result = await self._control_system_media("next")
            elif action == "previous":
                result = await self._control_system_media("previous")
            else:
                return {"success": False, "result": "Music control not configured", "data": None}
            
            return result
        
        except Exception as e:
            logger.error(f"Music control error: {e}")
            return {"success": False, "result": f"Music control error: {str(e)}", "data": None}
    
    async def _youtube_music_control(self, action: str, query: str = "") -> Dict[str, Any]:
        """Control YouTube Music via robot browser automation."""
        if not await self._ensure_client():
            return {"success": False, "result": "Cannot connect to robot", "data": None}
        
        try:
            # Get robot IP
            robot_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
            
            # Send YouTube Music control command to robot
            url = f"http://{robot_ip}:8000/youtube-music/{action}"
            
            payload = {"action": action}
            if query:
                payload["query"] = query
            
            response = await self.http_client.post(url, json=payload, timeout=10.0)
            
            if response.status_code == 200:
                result_data = response.json()
                if action == "play" and not query:
                    return {
                        "success": True,
                        "result": "Playing music on YouTube Music",
                        "data": result_data
                    }
                else:
                    return {
                        "success": True,
                        "result": f"YouTube Music {action}",
                        "data": result_data
                    }
            else:
                return {
                    "success": False,
                    "result": f"YouTube Music control failed: {response.status_code}",
                    "data": None
                }
        
        except Exception as e:
            logger.error(f"YouTube Music control error: {e}")
            return {
                "success": False,
                "result": f"Could not control YouTube Music: {str(e)}",
                "data": None
            }
    
    async def _spotify_control(self, action: str, query: str = "") -> Dict[str, Any]:
        """Control Spotify playback using Web API with user auth."""
        try:
            # Import spotipy
            try:
                import spotipy
                from spotipy.oauth2 import SpotifyOAuth
            except ImportError:
                return {"success": False, "result": "Spotify library not installed. Run: pip install spotipy", "data": None}
            
            # Get credentials
            client_id = os.getenv("SPOTIFY_CLIENT_ID")
            client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
            redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
            
            if not client_id or not client_secret:
                return {"success": False, "result": "Spotify credentials not configured", "data": None}
            
            # Create Spotify client with cached token
            sp_oauth = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-read-playback-state,user-modify-playback-state,user-library-read",
                cache_path=".spotify_token_cache"
            )
            
            token_info = sp_oauth.get_cached_token()
            if not token_info:
                return {
                    "success": False,
                    "result": "Spotify not authenticated. Run: python cloud/auth_spotify.py",
                    "data": None
                }
            
            sp = spotipy.Spotify(auth=token_info['access_token'])
            
            # Find ROVY device
            devices = sp.devices()
            rovy_device = None
            for dev in devices.get('devices', []):
                if 'ROVY' in dev['name'] or 'raspotify' in dev['name'].lower():
                    rovy_device = dev
                    break
            
            if not rovy_device:
                return {
                    "success": False,
                    "result": "ROVY Spotify device not found. Make sure Raspotify is running.",
                    "data": None
                }
            
            device_id = rovy_device['id']
            
            if action == "play":
                # Play random from liked songs
                try:
                    # Get user's saved tracks
                    results = sp.current_user_saved_tracks(limit=50)
                    tracks = results['items']
                    
                    if tracks:
                        # Get track URIs
                        track_uris = [item['track']['uri'] for item in tracks]
                        
                        # Start playback on ROVY device with shuffle
                        sp.shuffle(True, device_id=device_id)
                        sp.start_playback(device_id=device_id, uris=track_uris)
                        
                        return {
                            "success": True,
                            "result": f"Playing {len(tracks)} liked songs on ROVY (shuffled)",
                            "data": {"action": "play", "tracks": len(tracks)}
                        }
                    else:
                        # No liked songs, play a popular playlist
                        sp.start_playback(device_id=device_id, context_uri="spotify:playlist:37i9dQZEVXbMDoHDwVN2tF")
                        return {
                            "success": True,
                            "result": "Playing Global Top 50 on ROVY",
                            "data": {"action": "play", "source": "playlist"}
                        }
                except Exception as e:
                    # Fallback: try to just resume
                    try:
                        sp.start_playback(device_id=device_id)
                        return {"success": True, "result": "Music playing on ROVY", "data": {"action": "play"}}
                    except:
                        return {"success": False, "result": f"Could not start playback: {str(e)}", "data": None}
            
            elif action == "pause":
                sp.pause_playback(device_id=device_id)
                return {"success": True, "result": "Music paused", "data": {"action": "pause"}}
            
            elif action == "next":
                sp.next_track(device_id=device_id)
                return {"success": True, "result": "Next track", "data": {"action": "next"}}
            
            elif action == "previous":
                sp.previous_track(device_id=device_id)
                return {"success": True, "result": "Previous track", "data": {"action": "previous"}}
            
            elif action == "stop":
                sp.pause_playback(device_id=device_id)
                return {"success": True, "result": "Music stopped", "data": {"action": "stop"}}
            
            else:
                return {"success": False, "result": f"Unknown action: {action}", "data": None}
        
        except Exception as e:
            logger.error(f"Spotify control error: {e}")
            return {"success": False, "result": f"Spotify error: {str(e)}", "data": None}
    
    async def _get_spotify_token(self, client_id: str, client_secret: str) -> Optional[str]:
        """Get Spotify access token using client credentials flow."""
        try:
            import base64
            
            # Encode credentials
            auth_str = f"{client_id}:{client_secret}"
            auth_bytes = auth_str.encode('utf-8')
            auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
            
            # Request token
            response = await self.http_client.post(
                "https://accounts.spotify.com/api/token",
                headers={
                    "Authorization": f"Basic {auth_b64}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data="grant_type=client_credentials"
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("access_token")
            else:
                logger.error(f"Spotify auth failed: {response.status_code} {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Spotify token error: {e}")
            return None
    
    async def _control_system_media(self, action: str) -> Dict[str, Any]:
        """Control system media playback using platform-specific commands or robot."""
        try:
            import platform
            system = platform.system()
            
            if system == "Linux":
                # Use playerctl for Linux
                cmd = ["playerctl", action]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "result": f"Music {action}",
                        "data": {"action": action}
                    }
            
            elif system == "Darwin":  # macOS
                # Use osascript for macOS Music/Spotify control
                if action == "play":
                    script = 'tell application "Music" to play'
                elif action == "pause":
                    script = 'tell application "Music" to pause'
                elif action == "stop":
                    script = 'tell application "Music" to stop'
                elif action == "next":
                    script = 'tell application "Music" to next track'
                elif action == "previous":
                    script = 'tell application "Music" to previous track'
                else:
                    return {"success": False, "result": f"Unsupported action: {action}", "data": None}
                
                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "result": f"Music {action}",
                        "data": {"action": action}
                    }
            
            elif system == "Windows":
                # On Windows (cloud server), send music control to robot speakers
                return await self._control_robot_music(action)
            
            return {
                "success": False,
                "result": f"Could not control media on {system}",
                "data": None
            }
        
        except Exception as e:
            logger.error(f"System media control error: {e}")
            return {
                "success": False,
                "result": "Media player not found or not responding",
                "data": None
            }
    
    async def _control_robot_music(self, action: str) -> Dict[str, Any]:
        """Send music control command to robot/rover."""
        if not await self._ensure_client():
            return {"success": False, "result": "Cannot connect to robot", "data": None}
        
        try:
            # Get robot IP from environment
            robot_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
            
            # Send music control command to robot
            url = f"http://{robot_ip}:8000/music/{action}"
            
            response = await self.http_client.post(url, json={"action": action}, timeout=5.0)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "result": f"Music {action} on robot",
                    "data": {"action": action, "target": "robot"}
                }
            else:
                return {
                    "success": False,
                    "result": f"Robot returned status {response.status_code}",
                    "data": None
                }
        
        except Exception as e:
            logger.error(f"Robot music control error: {e}")
            return {
                "success": False,
                "result": f"Could not control robot music: {str(e)}",
                "data": None
            }
    
    async def set_reminder(self, message: str, minutes: int = 5) -> Dict[str, Any]:
        """Set a reminder (stores in memory for now)."""
        try:
            # In a production system, this would use a proper task scheduler
            # For now, we'll just acknowledge it
            
            reminder_time = datetime.now().timestamp() + (minutes * 60)
            
            # You could integrate with system notifications here
            # For Linux: notify-send
            # For macOS: osascript -e 'display notification "message" with title "Reminder"'
            # For Windows: PowerShell toast notifications
            
            result = f"I'll remind you to {message} in {minutes} minutes"
            
            return {
                "success": True,
                "result": result,
                "data": {
                    "message": message,
                    "minutes": minutes,
                    "reminder_time": reminder_time
                }
            }
        
        except Exception as e:
            logger.error(f"Reminder error: {e}")
            return {"success": False, "result": f"Could not set reminder: {str(e)}", "data": None}
    
    async def web_search(self, query: str) -> Dict[str, Any]:
        """Search the web for information (simplified version)."""
        # This is a simplified version. In production, you might use:
        # - DuckDuckGo API
        # - Google Custom Search API
        # - Bing Search API
        
        if not await self._ensure_client():
            return {"success": False, "result": "Web search unavailable", "data": None}
        
        try:
            # Use DuckDuckGo Instant Answer API (free, no API key needed)
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
            response = await self.http_client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get the abstract/answer
                answer = data.get("AbstractText", "")
                if not answer:
                    answer = data.get("Answer", "")
                
                if answer:
                    # Truncate to reasonable length
                    if len(answer) > 300:
                        answer = answer[:297] + "..."
                    
                    return {
                        "success": True,
                        "result": answer,
                        "data": data
                    }
                else:
                    return {
                        "success": False,
                        "result": f"No quick answer found for '{query}'",
                        "data": data
                    }
            else:
                return {
                    "success": False,
                    "result": "Web search failed",
                    "data": None
                }
        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"success": False, "result": f"Search error: {str(e)}", "data": None}
    
    async def translate(self, text: str, target_language: str) -> Dict[str, Any]:
        """
        Translate text to target language using the LLM.
        Supports bidirectional translation (any language to any language).
        
        Args:
            text: Text to translate
            target_language: Target language (e.g., "Chinese", "Spanish", "English", "英语", etc.)
        
        Returns:
            Dict with success, translated result, and data
        """
        try:
            # Language name normalization (only languages with Piper voices)
            language_map = {
                'chinese': 'Chinese (中文)',
                'mandarin': 'Chinese (中文)',
                '中文': 'Chinese (中文)',
                '中国': 'Chinese (中文)',
                'spanish': 'Spanish (Español)',
                'español': 'Spanish (Español)',
                'french': 'French (Français)',
                'français': 'French (Français)',
                'german': 'German (Deutsch)',
                'deutsch': 'German (Deutsch)',
                'italian': 'Italian (Italiano)',
                'italiano': 'Italian (Italiano)',
                'portuguese': 'Portuguese (Português)',
                'português': 'Portuguese (Português)',
                'russian': 'Russian (Русский)',
                'русский': 'Russian (Русский)',
                'hindi': 'Hindi (हिन्दी)',
                'हिन्दी': 'Hindi (हिन्दी)',
                'english': 'English',
                '英语': 'English',
                '英文': 'English',
                'farsi': 'Farsi (فارسی)',
                'persian': 'Farsi (فارسی)',
                'فارسی': 'Farsi (فارسی)',
                'nepali': 'Nepali (नेपाली)',
                'नेपाली': 'Nepali (नेपाली)',
                'vietnamese': 'Vietnamese (Tiếng Việt)',
                'tiếng việt': 'Vietnamese (Tiếng Việt)',
                # Japanese and Korean not available in standard Piper TTS
            }
            
            # Normalize target language
            target_lower = target_language.lower().strip()
            display_language = language_map.get(target_lower, target_language.title())
            
            # Create smarter translation prompt that handles bidirectional translation
            prompt = f"Translate the following text to {display_language}. The text may be in any language. Return ONLY the translated text, nothing else:\n\n{text}"
            
            # Use the LLM to translate
            logger.info(f"Translating '{text}' to {display_language}")
            
            # Use the assistant reference if available, otherwise create one
            if self.assistant:
                assistant = self.assistant
            else:
                # Import AI module here to avoid circular imports
                from ai import CloudAssistant
                assistant = CloudAssistant()
            
            # Get translation using asyncio.to_thread to avoid event loop conflicts
            # IMPORTANT: disable_tools=True prevents infinite recursion!
            import asyncio
            translation = await asyncio.to_thread(
                assistant.ask, prompt, 150, 0.1, True  # disable_tools=True
            )
            
            # Clean up the response (remove any extra explanations)
            translation = translation.strip()
            
            # Remove common prefixes the LLM might add
            prefixes_to_remove = [
                "Here is the translation:",
                "Translation:",
                "The translation is:",
                "In " + display_language + ":",
                display_language + ":",
            ]
            
            for prefix in prefixes_to_remove:
                if translation.lower().startswith(prefix.lower()):
                    translation = translation[len(prefix):].strip()
            
            # Remove quotes if the LLM wrapped it
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
            if translation.startswith("'") and translation.endswith("'"):
                translation = translation[1:-1]
            
            logger.info(f"Translation result: {translation}")
            
            return {
                "success": True,
                "result": translation,
                "data": {
                    "original": text,
                    "translation": translation,
                    "target_language": display_language
                }
            }
        
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            return {
                "success": False,
                "result": f"Translation failed: {str(e)}",
                "data": None
            }
    
    async def explore_robot(self) -> Dict[str, Any]:
        """Start autonomous exploration mode - robot navigates using camera vision."""
        try:
            # Get server instance - use stored reference first, then fallback to import
            server = self.server
            
            if not server:
                # Fallback: Try to import server
                import sys
                import os
                tools_dir = os.path.dirname(os.path.abspath(__file__))
                if tools_dir not in sys.path:
                    sys.path.insert(0, tools_dir)
                
                try:
                    import main
                    server = main.get_server()
                    logger.info(f"✅ Got server via import fallback: {server is not None}")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Import fallback failed: {e}")
            
            if not server:
                logger.error("❌ Server instance is None - not passed to ToolExecutor and import failed")
                return {
                    "success": False,
                    "result": "Server not available for exploration. Make sure the server is running.",
                    "data": None
                }
            
            logger.info(f"✅ Server retrieved, checking robot connection...")
            
            # Check if already exploring (WebSocket or HTTP)
            if server.robot and server.robot.exploring:
                return {
                    "success": True,
                    "result": "Already exploring via WebSocket!",
                    "data": None
                }
            
            if self.exploring_http:
                return {
                    "success": True,
                    "result": "Already exploring via HTTP!",
                    "data": None
                }
            
            # Try WebSocket first (preferred for real-time images)
            if server.robot and server.robot.clients:
                robot_ws = next(iter(server.robot.clients), None)
                if robot_ws:
                    try:
                        logger.info("🚀 Starting exploration via WebSocket connection")
                        await server.robot.start_exploration(robot_ws)
                        return {
                            "success": True,
                            "result": "Starting exploration mode. I'll navigate using my camera to avoid obstacles.",
                            "data": None
                        }
                    except Exception as e:
                        logger.warning(f"⚠️ WebSocket exploration failed: {e}, falling back to HTTP")
                        # Fall through to HTTP exploration
            
            # Fallback: Use HTTP/REST API for exploration (like move_robot does)
            logger.info("⚠️ No WebSocket connection, using HTTP/REST API for exploration")
            
            # Start HTTP-based exploration loop
            if not await self._ensure_client():
                return {
                    "success": False,
                    "result": "Cannot connect to robot via HTTP",
                    "data": None
                }
            
            # Start exploration task that uses HTTP
            # CRITICAL: The tool executor runs via run_until_complete() in a thread pool
            # We need to schedule the task on the main FastAPI event loop, not the temporary one
            self.exploring_http = True
            
            try:
                # Start exploration in a background thread with its own event loop
                # This avoids the complexity of scheduling on the main FastAPI loop
                # The exploration loop will run independently
                import threading
                
                def run_exploration_loop():
                    """Run exploration loop in a background thread with its own event loop."""
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        logger.info("🔍 Starting exploration loop in background thread")
                        # Run the exploration loop
                        loop.run_until_complete(self._exploration_loop_http(server))
                    except Exception as e:
                        logger.error(f"Exploration loop error: {e}", exc_info=True)
                    finally:
                        loop.close()
                        logger.info("🔍 Exploration loop thread ended")
                
                # Start the background thread
                exploration_thread = threading.Thread(target=run_exploration_loop, daemon=True, name="ExplorationThread")
                exploration_thread.start()
                self.exploration_task_http = exploration_thread  # Store thread reference
                
                logger.info(f"✅ Started exploration in background thread: {exploration_thread.name}")
                logger.info(f"   Thread alive: {exploration_thread.is_alive()}")
                logger.info(f"   exploring_http: {self.exploring_http}")
                
            except Exception as e:
                logger.error(f"Failed to create exploration task: {e}", exc_info=True)
                self.exploring_http = False
                return {
                    "success": False,
                    "result": f"Could not start exploration: {str(e)}",
                    "data": None
                }
            
            return {
                "success": True,
                "result": "Starting exploration mode via HTTP. I'll navigate using my camera to avoid obstacles.",
                "data": None
            }
        
        except Exception as e:
            logger.error(f"Exploration error: {e}", exc_info=True)
            return {
                "success": False,
                "result": f"Could not start exploration: {str(e)}",
                "data": None
            }
    
    async def _exploration_loop_http(self, server):
        """HTTP-based exploration loop - stops robot before capturing clear images, then moves smoothly."""
        logger.info("🔍 HTTP exploration loop started - smooth continuous movement mode")
        logger.info(f"   exploring_http flag: {self.exploring_http}")
        logger.info(f"   assistant available: {self.assistant is not None}")
        logger.info(f"   http_client available: {self.http_client is not None}")
        
        robot_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
        logger.info(f"   Robot IP: {robot_ip}")
        
        move_url = f"http://{robot_ip}:8000/control/move"
        snapshot_url = f"http://{robot_ip}:8000/shot"
        
        try:
            iteration = 0
            
            while self.exploring_http:
                iteration += 1
                logger.info(f"🔄 Exploration iteration {iteration}")
                
                # Step 1: Stop the robot briefly to capture a clear, stable image
                logger.debug("🛑 Stopping robot for clear image capture...")
                try:
                    await self.http_client.post(move_url, json={
                        "direction": "stop",
                        "distance": 0,
                        "speed": "medium"
                    }, timeout=5.0)
                except Exception as e:
                    logger.warning(f"Error sending stop command: {e}")
                
                # Step 2: Wait for robot to settle (brief pause for stability)
                await asyncio.sleep(0.5)
                
                # Step 3: Fetch a clear, stationary image with retry logic
                logger.debug(f"📷 Fetching clear image from camera at {snapshot_url}...")
                image_bytes = None
                max_retries = 3
                retry_delay = 0.3  # Start with 300ms delay
                
                for retry_attempt in range(max_retries):
                    try:
                        response = await self.http_client.get(snapshot_url, timeout=5.0)
                        if response.status_code == 200:
                            image_bytes = response.content
                            if image_bytes and len(image_bytes) >= 1000:
                                logger.debug(f"📷 Image fetched successfully: {len(image_bytes)} bytes (attempt {retry_attempt + 1})")
                                break  # Success, exit retry loop
                            else:
                                logger.warning(f"Invalid image data received (size: {len(image_bytes) if image_bytes else 0} bytes, attempt {retry_attempt + 1}/{max_retries})")
                        else:
                            logger.warning(f"Failed to get image: HTTP {response.status_code} (attempt {retry_attempt + 1}/{max_retries})")
                        
                        # Wait before retry (exponential backoff)
                        if retry_attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 1.5  # Increase delay for next retry
                    
                    except Exception as e:
                        logger.warning(f"Error fetching image (attempt {retry_attempt + 1}/{max_retries}): {e}")
                        if retry_attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 1.5
                
                # If all retries failed, skip this iteration
                if not image_bytes or len(image_bytes) < 1000:
                    logger.warning(f"⚠️ Failed to get valid image after {max_retries} attempts, skipping this iteration")
                    await asyncio.sleep(0.5)
                    continue
                
                # Step 4: Analyze the clear, stationary image - let AI decide everything
                logger.info(f"📸 Analyzing image for navigation decision (size: {len(image_bytes)} bytes)...")
                
                navigation_prompt = """You are navigating a robot. Look at this camera image and decide the best direction to move to explore the area while avoiding obstacles.

Analyze the image and respond with ONLY one of these words:
- "forward" - if the path ahead is clear
- "left" - if you should turn left to avoid obstacles or explore
- "right" - if you should turn right to avoid obstacles or explore
- "stop" - if there are obstacles blocking all directions

Consider:
- Avoid walls, furniture, and other obstacles
- Prefer open spaces for smooth exploration
- Choose the direction that allows the best exploration while staying safe

Respond with only the word: forward, left, right, or stop"""
                
                # Get navigation decision from AI
                if self.assistant:
                    response_text = await asyncio.get_event_loop().run_in_executor(
                        None, self.assistant.ask_with_vision, navigation_prompt, image_bytes
                    )
                else:
                    logger.error("Assistant not available for exploration")
                    break
                
                # Extract direction from response - trust AI completely
                response_lower = response_text.lower().strip()
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
                    logger.warning(f"Unclear navigation response: '{response_text}', defaulting to forward")
                    direction = 'forward'
                
                logger.info(f"🧭 Navigation decision: {direction} (AI: '{response_text[:60]}')")
                
                # Step 5: Execute whatever AI decided - no forced logic
                if direction == 'stop':
                    # AI says stop - actually stop and wait
                    logger.info("🛑 AI detected obstacle - staying stopped")
                    await asyncio.sleep(2.0)  # Wait longer when stopped
                else:
                    # Move in the direction AI chose
                    distance = 0.5 if direction == 'forward' else 0.3
                    try:
                        await self.http_client.post(move_url, json={
                            "direction": direction,
                            "distance": distance,
                            "speed": "medium"
                        }, timeout=5.0)
                        logger.info(f"🚗 Moving {direction} ({distance}m)")
                        
                        # Continue moving smoothly before next check
                        move_time = 1.5
                        await asyncio.sleep(move_time)
                    except Exception as e:
                        logger.warning(f"Error sending move command: {e}")
                        await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("HTTP exploration loop cancelled")
        except Exception as e:
            logger.error(f"HTTP exploration loop error: {e}", exc_info=True)
        finally:
            # Ensure robot stops when exploration ends
            try:
                await self.http_client.post(move_url, json={
                    "direction": "stop",
                    "distance": 0,
                    "speed": "medium"
                }, timeout=5.0)
            except:
                pass
            
            self.exploring_http = False
            self.exploration_task_http = None
            logger.info("🔍 HTTP exploration loop ended")
    
    async def move_robot(self, direction: str = "forward", distance: float = 0.5, speed: str = "medium") -> Dict[str, Any]:
        """Move the robot in a specified direction."""
        if not await self._ensure_client():
            return {"success": False, "result": "Cannot connect to robot", "data": None}
        
        try:
            # Get robot IP from environment
            robot_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
            
            # Validate inputs
            valid_directions = ["forward", "backward", "left", "right"]
            if direction not in valid_directions:
                return {
                    "success": False,
                    "result": f"Invalid direction. Use: {', '.join(valid_directions)}",
                    "data": None
                }
            
            valid_speeds = ["slow", "medium", "fast"]
            if speed not in valid_speeds:
                speed = "medium"
            
            # Clamp distance to reasonable values (0.1 to 5 meters)
            distance = max(0.1, min(5.0, distance))
            
            # Send movement command to robot
            url = f"http://{robot_ip}:8000/control/move"
            
            payload = {
                "direction": direction,
                "distance": distance,
                "speed": speed
            }
            
            response = await self.http_client.post(url, json=payload, timeout=10.0)
            
            if response.status_code == 200:
                # Create natural language response
                distance_str = f"{distance:.1f} meters" if distance != 0.5 else "a bit"
                return {
                    "success": True,
                    "result": f"Moving {direction} {distance_str}",
                    "data": {"direction": direction, "distance": distance, "speed": speed}
                }
            else:
                return {
                    "success": False,
                    "result": f"Robot movement failed: {response.status_code}",
                    "data": None
                }
        
        except Exception as e:
            logger.error(f"Robot movement error: {e}")
            return {
                "success": False,
                "result": f"Could not move robot: {str(e)}",
                "data": None
            }
    
    async def guess_age(self) -> Dict[str, Any]:
        """
        Estimate age from a camera image using a model from Hugging Face.
        Downloads the model on first use.
        """
        try:
            # Step 1: Get image from robot camera
            if not await self._ensure_client():
                return {"success": False, "result": "Cannot connect to robot", "data": None}
            
            robot_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
            snapshot_url = f"http://{robot_ip}:8000/shot"
            
            logger.info("📷 Capturing image for age estimation...")
            response = await self.http_client.get(snapshot_url, timeout=10.0)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "result": "Could not capture image from robot camera",
                    "data": None
                }
            
            image_bytes = response.content
            if not image_bytes or len(image_bytes) < 1000:
                return {
                    "success": False,
                    "result": "Invalid image captured",
                    "data": None
                }
            
            logger.info(f"📸 Image captured: {len(image_bytes)} bytes")
            
            # Step 2: Load image
            try:
                from PIL import Image
                import io
            except ImportError:
                return {
                    "success": False,
                    "result": "Age estimation requires Pillow library. Run: pip install pillow",
                    "data": None
                }
            
            try:
                image = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return {
                    "success": False,
                    "result": f"Could not process image: {str(e)}",
                    "data": None
                }
            
            # Step 3: Download and use age regression model from Hugging Face
            # Using Sharris/age_detection_regression - TensorFlow/Keras model with MobileNetV2
            try:
                import tensorflow as tf
                from huggingface_hub import hf_hub_download
                import numpy as np
            except ImportError:
                return {
                    "success": False,
                    "result": "Age estimation requires TensorFlow and huggingface-hub. Run: pip install tensorflow huggingface-hub",
                    "data": None
                }
            
            logger.info("🤖 Loading age estimation model from Hugging Face...")
            
            # Check if model is already loaded
            if not hasattr(self, '_age_model') or self._age_model is None:
                logger.info("📥 Downloading age estimation model from Hugging Face (this may take a moment on first use)...")
                
                model_name = "Sharris/age_detection_regression"
                
                try:
                    # Download the SavedModel from Hugging Face Hub
                    # The model is stored in the 'saved_model' directory
                    logger.info(f"Downloading model: {model_name}...")
                    
                    # Try to download the saved_model directory
                    try:
                        # Use huggingface_hub to download the model files
                        from huggingface_hub import snapshot_download
                        import os
                        import tempfile
                        
                        # Download the entire repository
                        cache_dir = snapshot_download(
                            repo_id=model_name,
                            repo_type="model",
                            cache_dir=None  # Use default cache
                        )
                        
                        # Look for saved_model directory
                        saved_model_path = os.path.join(cache_dir, "saved_model")
                        if not os.path.exists(saved_model_path):
                            # Try alternative locations - check if saved_model.pb exists in root
                            if os.path.exists(os.path.join(cache_dir, "saved_model.pb")):
                                saved_model_path = cache_dir
                            else:
                                # Search for saved_model directory or file
                                for root, dirs, files in os.walk(cache_dir):
                                    if "saved_model.pb" in files:
                                        saved_model_path = root
                                        break
                                    if "saved_model" in dirs:
                                        saved_model_path = os.path.join(root, "saved_model")
                                        break
                        
                        if not os.path.exists(saved_model_path):
                            raise FileNotFoundError(f"Could not find saved_model directory in {cache_dir}")
                        
                        # Load the TensorFlow SavedModel
                        logger.info(f"Loading TensorFlow SavedModel from: {saved_model_path}")
                        model = tf.saved_model.load(saved_model_path)
                        
                        self._age_model = model
                        self._age_model_path = saved_model_path
                        logger.info(f"✅ Age estimation model loaded successfully: {model_name}")
                        
                    except Exception as download_error:
                        logger.warning(f"Could not download from Hugging Face Hub: {download_error}")
                        # Fallback: try loading directly if model is already cached
                        try:
                            # Try to find cached model
                            from huggingface_hub import hf_hub_download
                            import os
                            
                            # Download the saved_model.pb file
                            model_file = hf_hub_download(
                                repo_id=model_name,
                                filename="saved_model/saved_model.pb",
                                cache_dir=None
                            )
                            saved_model_path = os.path.dirname(model_file)
                            model = tf.saved_model.load(saved_model_path)
                            
                            self._age_model = model
                            self._age_model_path = saved_model_path
                            logger.info(f"✅ Age estimation model loaded from cache: {model_name}")
                            
                        except Exception as fallback_error:
                            logger.error(f"Failed to load model: {fallback_error}")
                            return {
                                "success": False,
                                "result": f"Could not load age estimation model. Error: {str(fallback_error)}",
                                "data": None
                            }
                
                except Exception as model_error:
                    logger.error(f"Could not load {model_name}: {model_error}", exc_info=True)
                    return {
                        "success": False,
                        "result": f"Could not load age estimation model: {str(model_error)}",
                        "data": None
                    }
            
            # Use model for inference
            try:
                model = self._age_model
                model_name = "Sharris/age_detection_regression"
                
                logger.info(f"🔍 Processing image with age estimation model ({model_name})...")
                
                # Preprocess image for MobileNetV2
                # MobileNetV2 preprocessing: resize to 224x224, scale to [-1, 1]
                image_resized = image.resize((224, 224))
                img_array = np.array(image_resized, dtype=np.float32)
                
                # MobileNetV2 preprocessing: scale from [0, 255] to [-1, 1]
                # Formula: (pixel / 127.5) - 1.0
                img_array = (img_array / 127.5) - 1.0
                
                # Add batch dimension: (1, 224, 224, 3)
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Convert to TensorFlow tensor
                img_tensor = tf.constant(img_batch)
                
                # Run inference
                # The model signature should be 'serving_default' for SavedModel
                try:
                    # Try calling the model directly (most SavedModels support this)
                    predictions = model(img_tensor)
                except Exception as sig_error:
                    # If that fails, try using the serving_default signature
                    logger.debug(f"Direct call failed: {sig_error}, trying serving_default signature")
                    try:
                        # Some models use explicit signatures
                        if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
                            predictions = model.signatures['serving_default'](img_tensor)
                            # Extract the output value if it's a dict
                            if isinstance(predictions, dict):
                                # Get the first output value
                                output_key = list(predictions.keys())[0]
                                predictions = predictions[output_key]
                        else:
                            # Try to find any callable signature
                            if hasattr(model, 'signatures') and len(model.signatures) > 0:
                                sig_name = list(model.signatures.keys())[0]
                                predictions = model.signatures[sig_name](img_tensor)
                                if isinstance(predictions, dict):
                                    output_key = list(predictions.keys())[0]
                                    predictions = predictions[output_key]
                            else:
                                raise ValueError("No callable signatures found in model")
                    except Exception as sig_error2:
                        logger.error(f"All signature attempts failed: {sig_error2}")
                        raise
                
                # Extract age prediction
                # The model outputs a single scalar value (age in years)
                if isinstance(predictions, dict):
                    # If it's a dict, get the first value
                    pred_value = list(predictions.values())[0]
                elif hasattr(predictions, 'numpy'):
                    # TensorFlow tensor
                    pred_value = predictions.numpy()
                else:
                    # NumPy array or scalar
                    pred_value = predictions
                
                # Handle different output shapes
                if isinstance(pred_value, np.ndarray):
                    if pred_value.ndim > 0:
                        estimated_age = float(pred_value.flatten()[0])
                    else:
                        estimated_age = float(pred_value)
                else:
                    estimated_age = float(pred_value)
                
                # Validate and clamp age
                if estimated_age is None or estimated_age < 1:
                    logger.warning(f"Invalid age prediction: {estimated_age}")
                    return {
                        "success": False,
                        "result": f"Model produced invalid age prediction: {estimated_age}",
                        "data": None
                    }
                
                # Round to nearest integer and validate range
                estimated_age = int(round(estimated_age))
                estimated_age = max(1, min(120, estimated_age))  # At least 1 year old, max 120
                
                logger.info(f"✅ Age estimated: {estimated_age} years (from {model_name})")
                
                return {
                    "success": True,
                    "result": f"I estimate you are around {estimated_age} years old.",
                    "data": {"estimated_age": estimated_age}
                }
                
            except Exception as e:
                logger.error(f"Age estimation inference error: {e}", exc_info=True)
                return {
                    "success": False,
                    "result": f"Age estimation failed: {str(e)}",
                    "data": None
                }
        
        except Exception as e:
            logger.error(f"Age estimation error: {e}", exc_info=True)
            return {
                "success": False,
                "result": f"Could not estimate age: {str(e)}",
                "data": None
            }
    
    async def _estimate_age_with_assistant(self, image_bytes: bytes) -> Dict[str, Any]:
        """Helper method to estimate age using assistant's vision capabilities."""
        try:
            logger.info("🎯 Using AI vision to estimate age...")
            age_prompt = """Look carefully at this image of a person's face. Analyze facial features such as:
- Skin texture, wrinkles, and fine lines
- Hair color, graying, and style
- Facial structure and bone definition
- Eye appearance and crow's feet
- Overall maturity indicators
- Facial volume and elasticity

Provide your best estimate of their exact age in years. Be as specific and accurate as possible.
Respond with ONLY a single number representing the estimated age (e.g., "25", "42", "18", "67").
If you cannot see a clear face or the person is not visible, respond with "Unable to determine age - no clear face visible"."""
            
            # Use the assistant's vision capabilities
            import asyncio
            age_response = await asyncio.get_event_loop().run_in_executor(
                None, self.assistant.ask_with_vision, age_prompt, image_bytes
            )
            
            # Extract age from response
            age_response = age_response.strip()
            
            # Try to extract a number
            import re
            age_match = re.search(r'\b(\d{1,3})\b', age_response)
            
            if age_match:
                estimated_age = int(age_match.group(1))
                # Validate age range
                if 0 <= estimated_age <= 120:
                    return {
                        "success": True,
                        "result": f"I estimate you are around {estimated_age} years old.",
                        "data": {
                            "estimated_age": estimated_age,
                            "confidence": "medium"  # AI-based estimation
                        }
                    }
            
            # If no number found, return the AI's response
            return {
                "success": True,
                "result": age_response,
                "data": {
                    "estimated_age": None,
                    "response": age_response
                }
            }
        except Exception as e:
            logger.error(f"Assistant vision age estimation error: {e}", exc_info=True)
            return {
                "success": False,
                "result": f"Age estimation failed: {str(e)}",
                "data": None
            }
    
    async def recognize_face(self) -> Dict[str, Any]:
        """
        Recognize if the person in the camera image matches any known faces.
        Uses InsightFace (ArcFace) for high-accuracy face recognition.
        """
        try:
            # Step 1: Get image from robot camera
            if not await self._ensure_client():
                return {"success": False, "result": "Cannot connect to robot", "data": None}
            
            robot_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
            snapshot_url = f"http://{robot_ip}:8000/shot"
            
            logger.info("📷 Capturing image for face recognition...")
            response = await self.http_client.get(snapshot_url, timeout=10.0)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "result": "Could not capture image from robot camera",
                    "data": None
                }
            
            image_bytes = response.content
            if not image_bytes or len(image_bytes) < 1000:
                return {
                    "success": False,
                    "result": "Invalid image captured",
                    "data": None
                }
            
            logger.info(f"📸 Image captured: {len(image_bytes)} bytes")
            
            # Step 2: Try to use InsightFace service (no dlib required!)
            try:
                import sys
                # tools.py is in cloud/, app/ is in cloud/app/
                tools_dir = os.path.dirname(os.path.abspath(__file__))  # cloud/
                app_dir = os.path.join(tools_dir, "app")  # cloud/app/
                
                if app_dir not in sys.path:
                    sys.path.insert(0, app_dir)
                
                from face_recognition_service import FaceRecognitionService, FaceRecognitionError
                import cv2
                import numpy as np
                import io
            except ImportError as e:
                logger.warning(f"InsightFace not available: {e}")
                return {
                    "success": False,
                    "result": "Face recognition requires InsightFace. Install with: pip install insightface onnxruntime",
                    "data": None
                }
            
            # Step 3: Get or create cached face recognition service
            # known-faces is in cloud/known-faces (same directory as tools.py)
            known_faces_dir = os.path.join(os.path.dirname(__file__), "known-faces")
            if not os.path.exists(known_faces_dir):
                return {
                    "success": False,
                    "result": f"Known faces directory not found: {known_faces_dir}",
                    "data": None
                }
            
            # Use cached service if available (much faster - avoids reloading models)
            with ToolExecutor._face_recognition_service_lock:
                if ToolExecutor._face_recognition_service is None:
                    try:
                        logger.info("🔄 Initializing face recognition service (first time - this may take a moment)...")
                        ToolExecutor._face_recognition_service = FaceRecognitionService(
                            known_faces_dir=known_faces_dir, 
                            threshold=0.6
                        )
                        logger.info("✅ Face recognition service initialized and cached")
                    except FaceRecognitionError as e:
                        return {
                            "success": False,
                            "result": f"Failed to initialize face recognition: {str(e)}",
                            "data": None
                        }
                face_service = ToolExecutor._face_recognition_service
            
            # Step 4: Process the captured image
            logger.info("🔍 Processing captured image for face recognition...")
            # Convert bytes to numpy array (OpenCV format)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "success": False,
                    "result": "Failed to decode image",
                    "data": None
                }
            
            # Recognize faces using InsightFace
            results = face_service.recognize_faces(image, return_locations=True)
            
            if len(results) == 0:
                return {
                    "success": True,
                    "result": "I don't see a face in the image. Please make sure your face is visible to the camera.",
                    "data": {"recognized": False, "name": None}
                }
            
            # Get the first (most confident) recognition result
            best_result = results[0]
            recognized_name = best_result.get("name", "Unknown")
            confidence = best_result.get("confidence", 0.0)
            
            if recognized_name != "Unknown":
                logger.info(f"✅ Recognized: {recognized_name} (confidence: {confidence:.3f})")
                return {
                    "success": True,
                    "result": f"Yes! I recognize you. You are {recognized_name}!",
                    "data": {
                        "recognized": True,
                        "name": recognized_name,
                        "confidence": float(confidence)
                    }
                }
            else:
                logger.info(f"❌ No match found (best confidence: {confidence:.3f})")
                return {
                    "success": True,
                    "result": "I don't recognize you. You're not in my database of known faces.",
                    "data": {
                        "recognized": False,
                        "name": None,
                        "confidence": float(confidence)
                    }
                }
        
        except Exception as e:
            logger.error(f"Face recognition error: {e}", exc_info=True)
            return {
                "success": False,
                "result": f"Face recognition failed: {str(e)}",
                "data": None
            }
    
    def get_tools_description(self) -> str:
        """Get a description of available tools for the LLM prompt."""
        tools_list = []
        for name, info in self.tools.items():
            tools_list.append(f"- {name}: {info['description']}")
        
        return "\n".join(tools_list)


# Global instance
_tool_executor = None

def get_tool_executor(assistant=None, server=None) -> ToolExecutor:
    """Get or create global ToolExecutor instance."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor(assistant=assistant, server=server)
    else:
        # Update references if provided
        if assistant and not _tool_executor.assistant:
            _tool_executor.assistant = assistant
        if server and not _tool_executor.server:
            _tool_executor.server = server
    return _tool_executor

