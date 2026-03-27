#!/usr/bin/env python3
import time
from rover_controller import Rover  # update this import if filename is different


def main():
    # Adjust port/baudrate if needed
    rover = Rover(port="/dev/ttyACM0", baudrate=115200)

    try:
        # This already:
            # - reads voltage
            # - converts to %
            # - gets WiFi SSID
            # - gets IP
            # - draws:
            #   line 0: ROVY
            #   line 1: Bat: XX%
            #   line 2: WiFi: <ssid>
            #   line 3: IP: <ip>
            rover.display_reset()

            # Update every 2 seconds
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[Autorun] Stopped by user (Ctrl+C).")
    finally:
        rover.cleanup()


if __name__ == "__main__":
    main()
