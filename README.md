# capstone-ai-smart-robot

A ROS2-based robotic platform currently undergoing hardware restoration and software integration. This project involves a dual-controller architecture (ESP32 for low-level control and Raspberry Pi/Jetson for high-level logic).

## 🛠 Project Status: Hardware Restoration
The project is currently in the **diagnostic and repair phase** following a power system failure.

### Completed Milestones
* **Hardware Diagnostic:** Identified a blown PCB trace on the UPS Module 3S; confirmed ESP32 main board and peripherals are functional.
* **Serial Communications:** Established 115200 baud link via Arduino IDE to monitor boot sequences and system logs.
* **Peripheral Mapping:** Verified Pan-Tilt servo daisy-chain (ID 11 detected) and mounted IMX335 5MP camera.
* **Firmware Verification:** Confirmed successful LittleFS mounting and Access Point initialization via `ugv_base` logs.

## 🚀 12-Week Roadmap

### Phase 1: Power & Local Control (Weeks 1-2)
* Replace UPS Module 3S and verify 18650 battery health.
* Restore 12V rail and initialize QMI8658 IMU.
* Develop a `pyserial` wrapper to send JSON movement commands to the ESP32.

### Phase 2: Vision & Host Integration (Weeks 3-4)
* Integrate IMX335 camera via SH1.0 to USB-A adapter.
* Flash Host Computer (Raspberry Pi/Jetson) with Waveshare-optimized Ubuntu/ROS2 image.
* Establish internal communication between Host and ESP32.

### Phase 3: Logic & ROS2 Environment (Weeks 5-8)
* Configure headless SSH access and ROS2 workspace.
* Bridge ROS2 `Twist` messages to ESP32 JSON API.
* Implement remote teleoperation via keyboard/gamepad.

### Phase 4: Autonomy & Computer Vision (Weeks 9-12)
* Implement OpenCV-based object/color tracking.
* Tune PID controllers for smooth motion and acceleration.
* Stress-test battery life and finalize chassis cable management.

## 🔌 Hardware Specs
* **Controller:** ESP32-WROOM-32
* **Camera:** IMX335 5MP (160° FOV)
* **Servos:** ST3215 Serial Bus Servos
* **Power:** 3S 18650 Lithium Batteries (UPS Module 3S)