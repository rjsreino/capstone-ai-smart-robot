class DummyRover:
    def _send_direct(self, left, right):
        print(f"Left wheel: {left:.2f}, Right wheel: {right:.2f}")

class SimpleNavigator:
    def __init__(self):
        self.rover = DummyRover()
        self.wheel_base = 0.17
        self.max_wheel_speed = 0.7

    def velocity_callback(self, linear_vel, angular_vel):
        left_speed = linear_vel - (angular_vel * self.wheel_base / 2.0)
        right_speed = linear_vel + (angular_vel * self.wheel_base / 2.0)

        left_speed = max(-self.max_wheel_speed, min(self.max_wheel_speed, left_speed))
        right_speed = max(-self.max_wheel_speed, min(self.max_wheel_speed, right_speed))

        self.rover._send_direct(left_speed, right_speed)

nav = SimpleNavigator()

while True:
    cmd = input("Command (forward/left/right/stop/exit): ").strip().lower()

    if cmd == "forward":
        nav.velocity_callback(0.3, 0.0)
    elif cmd == "left":
        nav.velocity_callback(0.2, 0.5)
    elif cmd == "right":
        nav.velocity_callback(0.2, -0.5)
    elif cmd == "stop":
        nav.velocity_callback(0.0, 0.0)
    elif cmd == "exit":
        break
    else:
        print("Unknown command")