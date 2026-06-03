#!/usr/bin/env python3
"""
ZED Room Guidance System
Uses depth data to calculate safe paths and provide room guidance instructions.
Provides a premium visual dashboard and a clean callback hook for future wake-word
and speech synthesis integration.
"""

import cv2
import numpy as np
import time
import sys
import os
from typing import Dict, Tuple

# Add current folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zed_depth_processor import ZedDepthProcessor, ZedDepthConfig


# ==========================================
# FUTURE INTEGRATION HOOK
# ==========================================
def on_guidance_update(command: str, zones_data: Dict[str, float], safe_dist: float):
    """
    Hook for future integrations.
    This function is called every time a new guidance decision is computed.
    You can easily plug this into your:
    - Wake-word model / conversation manager
    - Local Text-To-Speech (TTS) library or server API
    - Robot velocity command publisher (ROS/Serial)
    """
    # Example printed guidance logs:
    # Print only when command changes or at a throttled rate
    now = time.time()
    if not hasattr(on_guidance_update, "last_print_time"):
        on_guidance_update.last_print_time = 0
        on_guidance_update.last_cmd = ""
        
    if command != on_guidance_update.last_cmd or (now - on_guidance_update.last_print_time) > 2.0:
        print(f"[GUIDANCE] Command: {command:<12} | L: {zones_data['left']:.0f}mm | C: {zones_data['center']:.0f}mm | R: {zones_data['right']:.0f}mm (Safe Thresh: {safe_dist:.0f}mm)")
        on_guidance_update.last_print_time = now
        on_guidance_update.last_cmd = command


def main():
    print("=" * 60)
    print("        ZED 1 SPATIAL ROOM GUIDANCE SYSTEM")
    print("=" * 60)
    print("Keyboard Controls:")
    print("  'q' : Quit application")
    print("  '+' : Increase Safe Distance Threshold (+100mm)")
    print("  '-' : Decrease Safe Distance Threshold (-100mm)")
    print("  'r' : Restart ZED camera connection")
    print("=" * 60)

    # Configure the processor for ZED 1 over USB 2.0
    config = ZedDepthConfig(
        resolution="vga",       # For USB 2.0 connection (720p causes Error 1893)
        fps=15,                 # Bandwidth friendly
        depth_mode="ULTRA",     # CUDA-based, highly robust, no compiling hangs
        min_depth=400,          # 40cm (ignore close reflections)
        max_depth=5000          # 5m (indoor range)
    )
    
    # Safe distance threshold parameters
    safe_distance_threshold = 1200.0  # Default safe distance in millimeters (1.2 meters)
    
    try:
        processor = ZedDepthProcessor(config)
        processor.start()
    except Exception as e:
        print(f"\n[FATAL] Connection error: {e}")
        print("Please check USB port and connection, then try again.")
        sys.exit(1)

    print("\nSystem running! Opening visual dashboard. Press 'q' to quit.")
    
    cv2.namedWindow("ZED Spatial Room Guidance Dashboard")
    
    try:
        while True:
            # Grab frame
            if not processor.grab_frame():
                # If grab fails, sleep briefly and continue
                time.sleep(0.01)
                continue
                
            # Get data
            rgb_frame = processor.get_rgb_frame()
            depth_frame = processor.get_depth_frame()
            
            if rgb_frame is None or depth_frame is None:
                continue
                
            # Process depth data for navigation
            nav_data = processor.process_depth_for_navigation(depth_frame)
            if nav_data is None:
                continue
                
            # Extract statistics
            zones = nav_data['zones']
            left_dist = zones['left']['median']
            center_dist = zones['center']['median']
            right_dist = zones['right']['median']
            
            # Determine guidance instruction
            # Logic:
            # 1. If center is clear (> safe_distance_threshold), check if we can go forward.
            # 2. If center is blocked:
            #    - Check if left is clear.
            #    - Check if right is clear.
            #    - Turn towards the direction with more clearance.
            # 3. If all sides are blocked under a critical threshold, command STOP.
            
            CRITICAL_STOP_DIST = 500.0  # 50cm danger close
            
            # Check for Doorway / Frontier Peak state
            gdm = zones.get('global_depth_max')
            is_doorway = False
            if gdm:
                val = gdm.get('value', 0.0)
                left_anomaly = gdm.get('left_wall_anomaly_mm', 0.0)
                right_anomaly = gdm.get('right_wall_anomaly_mm', 0.0)
                # Check if values match the doorway/frontier peak state
                if (1800.0 <= val <= 2200.0 and 
                    400.0 <= left_anomaly <= 550.0 and 
                    700.0 <= right_anomaly <= 900.0 and
                    gdm.get('zone') == 'center'):
                    is_doorway = True

            if is_doorway:
                cmd_text = "ENTER DOORWAY"
                cmd_color = (0, 255, 0)  # Green
            elif center_dist < CRITICAL_STOP_DIST or (left_dist < CRITICAL_STOP_DIST and right_dist < CRITICAL_STOP_DIST):
                cmd_text = "STOP! DANGER"
                cmd_color = (0, 0, 255)  # Red
            elif center_dist >= safe_distance_threshold:
                # Path ahead is clear
                cmd_text = "GO FORWARD"
                cmd_color = (0, 255, 0)  # Green
            else:
                # Center is blocked, determine turn direction
                if left_dist > right_dist:
                    cmd_text = "TURN LEFT"
                    cmd_color = (0, 255, 255)  # Yellow
                else:
                    cmd_text = "TURN RIGHT"
                    cmd_color = (255, 128, 0)  # Orange

            # Call the future integration hook
            zones_data = {
                'left': left_dist,
                'center': center_dist,
                'right': right_dist
            }
            on_guidance_update(cmd_text, zones_data, safe_distance_threshold)
            
            # ----------------------------------------------------
            # RENDER PREMIUM VISUAL DASHBOARD
            # ----------------------------------------------------
            h, w, _ = rgb_frame.shape
            
            # Resize source frames for compact side-by-side display (640x360 each)
            display_w = 640
            display_h = 360
            
            rgb_small = cv2.resize(rgb_frame, (display_w, display_h))
            depth_colored = processor.visualize_depth(depth_frame)
            depth_small = cv2.resize(depth_colored, (display_w, display_h))
            
            # Draw boundary zones overlay on RGB frame
            # 3 zones: Left (0 to w//3), Center (w//3 to 2*w//3), Right (2*w//3 to w)
            col_w = display_w // 3
            
            # Zone alpha overlays
            overlay = rgb_small.copy()
            
            # Left zone color (red if blocked, green if clear)
            left_color = (0, 0, 255) if left_dist < safe_distance_threshold else (0, 255, 0)
            cv2.rectangle(overlay, (0, 0), (col_w, display_h), left_color, -1)
            
            # Center zone color
            center_color = (0, 0, 255) if center_dist < safe_distance_threshold else (0, 255, 0)
            cv2.rectangle(overlay, (col_w, 0), (col_w * 2, display_h), center_color, -1)
            
            # Right zone color
            right_color = (0, 0, 255) if right_dist < safe_distance_threshold else (0, 255, 0)
            cv2.rectangle(overlay, (col_w * 2, 0), (display_w, display_h), right_color, -1)
            
            # Blend overlay with RGB
            cv2.addWeighted(overlay, 0.15, rgb_small, 0.85, 0, rgb_small)
            
            # Draw zone boundaries on both frames
            for col in [col_w, col_w * 2]:
                cv2.line(rgb_small, (col, 0), (col, display_h), (255, 255, 255), 1)
                cv2.line(depth_small, (col, 0), (col, display_h), (255, 255, 255), 1)
                
            # Add text markers for distance in each zone
            cv2.putText(rgb_small, f"L: {left_dist:.0f}mm", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(rgb_small, f"C: {center_dist:.0f}mm", (col_w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(rgb_small, f"R: {right_dist:.0f}mm", (col_w * 2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Combine side-by-side (1280x360)
            main_panels = np.hstack((rgb_small, depth_small))
            
            # Create a dedicated dashboard HUD footer panel (1280x120)
            hud_h = 120
            hud_w = display_w * 2
            hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
            # Fill with subtle slate dark color
            hud[:] = [24, 20, 18]
            
            # Draw borders
            cv2.rectangle(hud, (0, 0), (hud_w - 1, hud_h - 1), (50, 50, 50), 1)
            
            # Draw guidance direction block in the center
            cmd_box_x1 = hud_w // 2 - 180
            cmd_box_x2 = hud_w // 2 + 180
            cv2.rectangle(hud, (cmd_box_x1, 15), (cmd_box_x2, 105), cmd_color, -1)
            # Add contrast shadow to text for premium feel
            text_size = cv2.getTextSize(cmd_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)[0]
            text_x = hud_w // 2 - text_size[0] // 2
            text_y = hud_h // 2 + text_size[1] // 2
            # Text shadow
            cv2.putText(hud, cmd_text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 3)
            cv2.putText(hud, cmd_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3)
            
            # Left panel metadata (Telemetry)
            cv2.putText(hud, "SYSTEM TELEMETRY", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(hud, f"Model: ZED 1 Camera", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"FPS:   {processor.fps:.1f} Hz", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if processor.fps > 10 else (0, 150, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"Mode:  USB 2.0 (VGA)", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Right panel settings
            cv2.putText(hud, "THRESHOLD SETTINGS", (hud_w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(hud, f"Safe Distance: {safe_distance_threshold:.0f} mm", (hud_w - 280, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, "Adjust: [+] / [-] keys", (hud_w - 280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
            
            # Combine panel and footer vertically to create final dashboard (1280x480)
            dashboard = np.vstack((main_panels, hud))
            
            # Show output
            cv2.imshow("ZED Spatial Room Guidance Dashboard", dashboard)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                safe_distance_threshold = min(4000.0, safe_distance_threshold + 100)
                print(f"[SETTING] Increased Safe Distance Threshold to {safe_distance_threshold:.0f}mm")
            elif key == ord('-') or key == ord('_'):
                safe_distance_threshold = max(500.0, safe_distance_threshold - 100)
                print(f"[SETTING] Decreased Safe Distance Threshold to {safe_distance_threshold:.0f}mm")
            elif key == ord('r'):
                print("[SYSTEM] Restarting ZED SDK context...")
                processor.restart()
                
    finally:
        processor.stop()
        cv2.destroyAllWindows()
        print("\nGuidance System closed cleanly.")


if __name__ == "__main__":
    main()
