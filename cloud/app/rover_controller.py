"""
Rover Controller - Adapter for ugv_rpi BaseController
Provides the Rover interface expected by main.py
"""
import sys
import os
import time

# Add ugv_rpi to path
UGV_RPI_PATH = "/home/rovy/ugv_rpi"
if UGV_RPI_PATH not in sys.path:
    sys.path.insert(0, UGV_RPI_PATH)

import serial

try:
    from base_ctrl import BaseController
    BASE_CTRL_OK = True
except ImportError as e:
    print(f"[rover_controller] Failed to import BaseController: {e}")
    BASE_CTRL_OK = False
    BaseController = None


class Rover:
    """
    Rover adapter class for the ugv_rpi BaseController.
    Provides the interface expected by the main API.
    """
    
    def __init__(self, port='/dev/ttyAMA0', baudrate=115200):
        """Initialize rover controller.
        
        Args:
            port: Serial port (default /dev/ttyAMA0 for Pi5)
            baudrate: Baud rate (default 115200)
        """
        self.port = port
        self.baudrate = baudrate
        self.base = None
        
        if BASE_CTRL_OK and BaseController:
            try:
                self.base = BaseController(port, baudrate)
                print(f"[Rover] Connected on {port}")
            except Exception as e:
                print(f"[Rover] Failed to connect on {port}: {e}")
        else:
            print("[Rover] BaseController not available")
    
    def set_motor(self, left: float, right: float):
        """Set motor speeds.
        
        Args:
            left: Left motor speed (-1 to 1)
            right: Right motor speed (-1 to 1)
        """
        if self.base:
            self.base.base_speed_ctrl(left, right)
    
    def stop(self):
        """Stop all motors."""
        if self.base:
            self.base.base_speed_ctrl(0, 0)
    
    def lights_ctrl(self, pwmA: int, pwmB: int):
        """Control LED lights.
        
        Args:
            pwmA: Front light PWM (0-255)
            pwmB: Back light PWM (0-255)
        """
        if self.base:
            self.base.lights_ctrl(pwmA, pwmB)
    
    def gimbal_ctrl(self, x: float, y: float, speed: int = 10, acceleration: int = 0):
        """Control gimbal position.
        
        Args:
            x: Pan angle (-180 to 180)
            y: Tilt angle (-30 to 90)
            speed: Movement speed
            acceleration: Movement acceleration
        """
        if self.base:
            self.base.gimbal_ctrl(x, y, speed, acceleration)
    
    def display_text(self, line: int, text: str):
        """Display text on OLED.
        
        Args:
            line: Line number (0-3)
            text: Text to display (max 21 chars)
        """
        if self.base:
            self.base.base_oled(line, text[:21])
    
    def display_reset(self):
        """Reset OLED to default display."""
        if self.base:
            self.base.base_default_oled()
    
    def nod(self, times: int = 2, center_tilt: int = 0, delta: int = 20, pan: int = 0, delay: float = 0.3):
        """Nod the gimbal (yes gesture).
        
        Args:
            times: Number of nods
            center_tilt: Center tilt position
            delta: Nod amplitude
            pan: Pan position
            delay: Delay between movements
        """
        if self.base:
            self.base.gimbal_ctrl(pan, center_tilt, 10, 0)
            time.sleep(0.5)
            for _ in range(times):
                self.base.gimbal_ctrl(pan, center_tilt - delta, 10, 0)
                time.sleep(delay)
                self.base.gimbal_ctrl(pan, center_tilt + delta, 10, 0)
                time.sleep(delay)
            self.base.gimbal_ctrl(pan, center_tilt, 10, 0)
    
    def get_status(self):
        """Get rover status.
        
        Returns:
            dict with voltage, temperature, roll, pitch, yaw
        """
        if self.base:
            try:
                # Read multiple messages to try to find a T:1001 telemetry message
                max_attempts = 10
                for attempt in range(max_attempts):
                    data = self.base.feedback_data()
                    
                    if data and isinstance(data, dict):
                        msg_type = data.get('T')
                        
                        # T:1001 contains full telemetry with voltage
                        if msg_type == 1001 and 'v' in data:
                            voltage_centivolts = data.get('v', 0.0)
                            # Voltage is sent in centivolts (hundredths of a volt)
                            # Convert to volts: 1215 centivolts = 12.15V
                            voltage = voltage_centivolts / 100.0
                            print(f"[Rover] Got T:1001 telemetry with voltage: {voltage}V ({voltage_centivolts} centivolts)")
                            
                            return {
                                'voltage': voltage,
                                'temperature': data.get('temp', 0.0),
                                'roll': data.get('r', 0.0),
                                'pitch': data.get('p', 0.0),
                                'yaw': data.get('y', 0.0),
                                'cpu': 0,
                                'ai_state': 'idle',
                            }
                        
                        print(f"[Rover] Attempt {attempt+1}: Got T:{msg_type}, data keys: {list(data.keys())}, data: {data}")
                    
                    time.sleep(0.05)  # Wait a bit before trying again
                
                # If we didn't get T:1001, use the last data we got
                if self.base.base_data and isinstance(self.base.base_data, dict):
                    data = self.base.base_data
                    voltage_centivolts = data.get('v', 0.0)
                    voltage = voltage_centivolts / 100.0 if voltage_centivolts else 0.0
                    print(f"[Rover] Using cached data T:{data.get('T')} after {max_attempts} attempts, voltage: {voltage}V")
                    return {
                        'voltage': voltage,
                        'temperature': data.get('temp', 0.0),
                        'roll': data.get('r', 0.0),
                        'pitch': data.get('p', 0.0),
                        'yaw': data.get('y', 0.0),
                        'cpu': 0,
                        'ai_state': 'idle',
                    }
                    
            except Exception as e:
                print(f"[Rover] get_status error: {e}")
        
        return {
            'voltage': 0.0,
            'temperature': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'cpu': 0,
            'ai_state': 'idle',
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self.base:
            self.base.gimbal_dev_close()


# Test
if __name__ == "__main__":
    rover = Rover()
    print("Status:", rover.get_status())
    rover.display_text(0, "ROVY")
    rover.display_text(1, "Cloud Mode")
    time.sleep(2)
    rover.cleanup()

