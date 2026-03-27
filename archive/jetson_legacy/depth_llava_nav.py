"""
Oak-D 3D Depth + LLaVA AI Navigation
Professional autonomous system combining:
- 3D stereo depth for real-time obstacle avoidance (30 FPS)
- LLaVA AI for scene understanding and strategic planning
"""
import time
import threading
from queue import Queue
from pathlib import Path
import signal
import sys

from rover_controller import Rover
from oakd_depth_navigator import OakDDepthCamera, DepthNavigator
from llava_cpp_navigator import LLaVACppNavigator


class DepthLLaVARover:
    """
    Professional autonomous navigation system.
    """
    
    def __init__(self, port='/dev/ttyACM0', llava_interval=15.0, safe_distance_mm=800):
        self.rover = None
        self.camera = None
        self.depth_nav = None
        self.llava_nav = None
        self.running = False
        self.port = port
        self.llava_interval = llava_interval
        self.safe_distance_mm = safe_distance_mm
        
        # Frame queue - "общий стол" для кадров
        self.frame_queue = Queue(maxsize=2)
        
        self.llava_guidance = None
        self.guidance_lock = threading.Lock()  # Замок для защиты llava_guidance
        
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        print("\n[System] Stopping...")
        self.stop()
        sys.exit(0)
    
    def initialize(self):
        print("=" * 70)
        print("PROFESSIONAL AUTONOMOUS NAVIGATION SYSTEM")
        print("3D Depth Perception + AI Vision")
        print("=" * 70)
        
        print("\n[1/4] Connecting to rover...")
        self.rover = Rover(port=self.port)
        
        print("\n[2/4] Starting Oak-D stereo camera...")
        self.camera = OakDDepthCamera(resolution=(640, 480))
        self.camera.start()
        
        print("\n[3/4] Initializing 3D depth navigator...")
        # Use safe_distance from args if provided  
        safe_dist = getattr(self, 'safe_distance_mm', 500)  # 500mm for indoor spaces
        self.depth_nav = DepthNavigator(safe_distance_mm=safe_dist)
        
        print("\n[4/4] LLaVA AI will load in background...")
        self.llava_nav = None  # Will be loaded by LLaVA thread
        
        print("\n" + "=" * 70)
        print("SYSTEM READY")
        print("=" * 70 + "\n")
    
    def _capture_thread(self):
        """'Поставщик' - единственный поток, который захватывает кадры."""
        while self.running:
            try:
                rgb, depth = self.camera.capture_frames()
                
                # Положить свежие кадры в очередь
                if self.frame_queue.full():
                    self.frame_queue.get()  # Удалить старый кадр
                self.frame_queue.put((rgb, depth))
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"[Capture] Error: {e}")
                time.sleep(0.5)
    
    def _llava_thread(self):
        """LLaVA analyzes scenes for high-level understanding."""
        # Load LLaVA in this thread so it doesn't block startup
        print("[AI] Loading LLaVA in background...")
        try:
            self.llava_nav = LLaVACppNavigator(n_gpu_layers=99)
            print("[AI] LLaVA loaded and ready!")
        except Exception as e:
            print(f"[AI] Failed to load LLaVA: {e}")
            return
        
        while self.running:
            try:
                # Взять кадр из очереди
                if not self.frame_queue.empty():
                    rgb, _ = self.frame_queue.get()
                    
                    print(f"[AI] Analyzing scene with LLaVA...")
                    guidance = self.llava_nav.get_navigation_command(rgb)
                    
                    # Безопасная запись с использованием lock
                    with self.guidance_lock:
                        self.llava_guidance = guidance
                    
                    print(f"[AI] Recommendation: {guidance['action']} - {guidance['reasoning'][:50]}")
                
                time.sleep(self.llava_interval)
                
            except Exception as e:
                print(f"[AI] Error: {e}")
                time.sleep(5)
    
    def _depth_navigation_thread(self):
        """Real-time 3D depth-based navigation with intelligent evasion."""
        last_action = None
        last_clearance = 1.0  # Track clearance for emergency detection
        
        while self.running:
            try:
                # ПРОСТО ПОЛУЧАЕМ КАДР - автоматически ждет если пусто
                rgb, depth = self.frame_queue.get()
                
                # Get depth-based obstacle avoidance
                depth_cmd = self.depth_nav.get_navigation_command(rgb, depth)
                
                # Безопасное чтение LLaVA guidance
                local_llava_guidance = None
                with self.guidance_lock:
                    if self.llava_guidance:
                        local_llava_guidance = self.llava_guidance.copy()
                
                # Combine with LLaVA strategic guidance (используем локальную копию!)
                metrics = depth_cmd.get('metrics', {})
                if local_llava_guidance and depth_cmd['action'] != 'stop' and metrics:
                    llava_action = local_llava_guidance.get('action')
                    
                    if llava_action == 'forward':
                        suggested_clearance = metrics.get('front', 0)
                    elif llava_action == 'left':
                        suggested_clearance = metrics.get('left', 0)
                    elif llava_action == 'right':
                        suggested_clearance = metrics.get('right', 0)
                    else:
                        suggested_clearance = 0
                    
                    reference_clearance = metrics.get('front', 0)
                    if suggested_clearance >= max(reference_clearance * 0.9, self.depth_nav.blocked_distance_mm):
                        cmd = {
                            'action': llava_action,
                            'speed': depth_cmd['speed'],
                            'distance': depth_cmd['distance'],
                            'steering_bias': depth_cmd.get('steering_bias', 0.0),
                            'reasoning': local_llava_guidance.get('reasoning', 'LLaVA guidance'),
                            'state': depth_cmd.get('state', 'EXPLORING')
                        }
                    else:
                        cmd = depth_cmd
                else:
                    cmd = depth_cmd
                
                # Execute smooth movement
                action = cmd['action']
                
                # VERY REDUCED SPEEDS for better reaction time
                # Add PREVENTIVE slowdown based on path clearance
                base_speed_lookup = {'slow': 0.12, 'medium': 0.18, 'fast': 0.25}
                base_speed = base_speed_lookup.get(cmd['speed'], 0.12)
                
                # Get clearance for current path
                if action == 'forward':
                    clearance = cmd.get('scores', {}).get('center', 0.5)
                else:
                    clearance = max([v for v in cmd.get('scores', {}).values() if v > 0] or [0.5])
                
                # EMERGENCY STOP if clearance drops suddenly (collision imminent!)
                clearance_drop = last_clearance - clearance
                if clearance_drop > 0.30 and clearance < 0.40:
                    # Sudden drop + low clearance = EMERGENCY!
                    print(f"[Nav] 🚨 EMERGENCY STOP - Clearance dropped {int(clearance_drop*100)}% (now {int(clearance*100)}%)")
                    self.rover.stop()
                    last_action = 'stop'
                    last_clearance = clearance
                    time.sleep(0.3)
                    continue
                
                last_clearance = clearance
                
                # PREVENTIVE SLOWDOWN: reduce speed if clearance is marginal
                if clearance < 0.65:
                    speed_modifier = clearance / 0.65  # Scale down speed
                    speed_val = base_speed * max(speed_modifier, 0.4)  # At least 40% speed
                else:
                    speed_val = base_speed
                
                if action == 'stop':
                    if last_action != 'stop':
                        reason_text = cmd.get('reasoning', depth_cmd.get('reasoning', 'Idle'))
                        print(f"[Nav] STOP - {reason_text}")
                        self.rover.stop()
                        last_action = 'stop'
                    time.sleep(0.2)
                    
                else:
                    # Execute movement command
                    bias = float(cmd.get('steering_bias', 0.0)) if isinstance(cmd, dict) else 0.0
                    bias = max(min(bias, 0.8), -0.8)
                    
                    if action == 'forward':
                        adjust = bias * 0.5
                        left_scale = max(min(1.0 + adjust, 1.6), 0.4)
                        right_scale = max(min(1.0 - adjust, 1.6), 0.4)
                        L = speed_val * left_scale
                        R = speed_val * right_scale
                    elif action == 'backward':
                        L = -speed_val
                        R = -speed_val
                    elif action == 'left':
                        turn_power = 0.15  # VERY SLOW turn for better depth analysis
                        L = -turn_power
                        R = turn_power
                    elif action == 'right':
                        turn_power = 0.15  # VERY SLOW turn for better depth analysis
                        L = turn_power
                        R = -turn_power
                    else:
                        L = 0.0
                        R = 0.0
                    
                    self.rover._send(L, R)
                    
                    if action != last_action:
                        reason_text = cmd.get('reasoning', depth_cmd.get('reasoning', ''))
                        print(f"[Nav] {action.upper()} - {reason_text}")
                        last_action = action
                    
                    time.sleep(0.05)  # 20Hz for very smooth control
                
            except Exception as e:
                print(f"[Nav] Error: {e}")
                try:
                    self.rover.stop()
                except:
                    pass
                time.sleep(1)
    
    def run(self, duration=60):
        """Run the navigation system."""
        self.running = True
        
        print(f"\n🚀 Starting navigation for {duration} seconds")
        print(f"  • Camera: Capturing at 30 FPS")
        print(f"  • 3D Depth: Real-time obstacle avoidance (20 FPS)")
        print(f"  • LLaVA AI: Scene understanding (every {self.llava_interval}s)")
        print("  • Press Ctrl+C to stop\n")
        
        # Start all threads - capture FIRST!
        capture_thread = threading.Thread(target=self._capture_thread, daemon=True)
        llava_thread = threading.Thread(target=self._llava_thread, daemon=True)
        depth_thread = threading.Thread(target=self._depth_navigation_thread, daemon=True)
        
        capture_thread.start()  # Поставщик начинает первым
        time.sleep(0.5)  # Дать время заполнить очередь
        llava_thread.start()
        depth_thread.start()
        
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\n[System] Interrupted")
        finally:
            self.stop()
    
    def stop(self):
        self.running = False
        time.sleep(0.3)
        
        if self.rover:
            try:
                self.rover.stop()
                self.rover.cleanup()
            except:
                pass
        
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
        
        if self.depth_nav:
            self.depth_nav.cleanup()
        
        if self.llava_nav:
            self.llava_nav.cleanup()
        
        print("\n[System] Shutdown complete")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Oak-D 3D Depth + LLaVA AI Autonomous Navigation'
    )
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--llava-interval', type=float, default=15.0,
                       help='LLaVA analysis interval (seconds)')
    parser.add_argument('--safe-distance', type=int, default=500,
                       help='Safe distance to obstacles (mm)')
    parser.add_argument('--port', default='/dev/ttyACM0')
    
    args = parser.parse_args()
    
    rover = DepthLLaVARover(
        port=args.port,
        llava_interval=args.llava_interval,
        safe_distance_mm=args.safe_distance
    )
    
    rover.initialize()
    rover.run(duration=args.duration)

