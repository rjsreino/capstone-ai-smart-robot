#!/usr/bin/env python3
"""
User-friendly status display for Waveshare UGV Rover.
Shows clean, easy-to-read information on the OLED screen.
"""

from rover_controller import Rover
import time
import datetime
import argparse

class DisplayMode:
    """Different display modes for the rover."""
    
    @staticmethod
    def simple(rover):
        """Simple mode - just the essentials."""
        status = rover.get_status()
        voltage = status['voltage']
        temp = status['temperature']
        
        # Battery percentage and status
        if voltage > 12.4:
            battery = "[100%] Full"
        elif voltage > 12.0:
            battery = "[75%] Good"
        elif voltage > 11.6:
            battery = "[50%] OK"
        elif voltage > 11.2:
            battery = "[25%] Low"
        else:
            battery = "[CRITICAL!]"
        
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        
        return [
            f"== ROVER READY ==",
            f"Power: {battery}",
            f"Temp: {temp:.0f}C",
            f"Time: {current_time}"
        ]
    
    @staticmethod
    def detailed(rover):
        """Detailed mode - more technical info."""
        status = rover.get_status()
        voltage = status['voltage']
        temp = status['temperature']
        left = status['left_speed']
        right = status['right_speed']
        
        is_moving = abs(left) > 0.01 or abs(right) > 0.01
        state = "MOVING" if is_moving else "IDLE"
        
        # Get battery percentage
        if voltage > 12.4:
            pct = "100%"
        elif voltage > 12.0:
            pct = "75%"
        elif voltage > 11.6:
            pct = "50%"
        elif voltage > 11.2:
            pct = "25%"
        else:
            pct = "LOW"
        
        return [
            f"Batt: {voltage:.1f}V ({pct})",
            f"Temp: {temp:.0f}C",
            f"State: {state}",
            f"Speed L{left:.1f} R{right:.1f}"
        ]
    
    @staticmethod
    def navigation(rover):
        """Navigation mode - orientation info."""
        status = rover.get_status()
        voltage = status['voltage']
        yaw = status['yaw']
        pitch = status['pitch']
        roll = status['roll']
        
        # Compass direction
        if -22.5 <= yaw < 22.5:
            direction = "N"
            arrow = "^"
        elif 22.5 <= yaw < 67.5:
            direction = "NE"
            arrow = "/"
        elif 67.5 <= yaw < 112.5:
            direction = "E"
            arrow = ">"
        elif 112.5 <= yaw < 157.5:
            direction = "SE"
            arrow = "\\"
        elif 157.5 <= yaw < 180 or -180 <= yaw < -157.5:
            direction = "S"
            arrow = "v"
        elif -157.5 <= yaw < -112.5:
            direction = "SW"
            arrow = "\\"
        elif -112.5 <= yaw < -67.5:
            direction = "W"
            arrow = "<"
        else:
            direction = "NW"
            arrow = "/"
        
        return [
            f"Heading: {direction} {arrow}",
            f"Angle: {yaw:.0f} deg",
            f"Tilt: {pitch:.1f}",
            f"Battery: {voltage:.1f}V"
        ]
    
    @staticmethod
    def custom_message(rover, messages):
        """Display custom messages."""
        return messages[:4]  # Max 4 lines

def main():
    parser = argparse.ArgumentParser(description='Display rover status on OLED screen')
    parser.add_argument('--mode', choices=['simple', 'detailed', 'nav'], 
                       default='simple',
                       help='Display mode: simple (default), detailed, or nav (navigation)')
    parser.add_argument('--port', default='/dev/ttyACM0',
                       help='Serial port (default: /dev/ttyACM0)')
    args = parser.parse_args()
    
    print("=== Rover Display ===")
    print(f"Mode: {args.mode}")
    print("Press Ctrl+C to stop\n")
    
    # Connect to rover
    rover = Rover(port=args.port, baudrate=115200)
    time.sleep(1)
    
    # Select display mode
    if args.mode == 'simple':
        display_func = DisplayMode.simple
    elif args.mode == 'detailed':
        display_func = DisplayMode.detailed
    else:
        display_func = DisplayMode.navigation
    
    try:
        while True:
            # Get display lines from selected mode
            lines = display_func(rover)
            
            # Display on OLED
            rover.display_multiline(lines)
            
            # Print to terminal
            print(f"{lines[0]:21} | {lines[1]:21} | {lines[2]:21} | {lines[3]:21}", end='\r')
            
            # Update every second
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n\nStopping display...")
        rover.display_reset()
        rover.cleanup()
        print("Display reset to default.")

if __name__ == '__main__':
    main()

