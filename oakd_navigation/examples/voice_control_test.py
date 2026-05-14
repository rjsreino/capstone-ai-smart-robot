import speech_recognition as sr
    
class DummyRover:
    def _send_direct(self, left, right):
        print(f"[ROVER] Left: {left:.2f}, Right: {right:.2f}")

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

def handle_voice_command(text: str, nav: SimpleNavigator):
    text = text.lower().strip()
    print("Recognized:", text)

    if "forward" in text:
        nav.velocity_callback(0.3, 0.0)
    elif "backward" in text or "back" in text:
        nav.velocity_callback(-0.2, 0.0)
    elif "left" in text:
        nav.velocity_callback(0.15, 0.6)
    elif "right" in text:
        nav.velocity_callback(0.15, -0.6)
    elif "stop" in text:
        nav.velocity_callback(0.0, 0.0)
    else:
        print("Command not recognized")

recognizer = sr.Recognizer()
nav = SimpleNavigator()

with sr.Microphone(device_index=1) as source:
    recognizer.adjust_for_ambient_noise(source)
    print("Listening... say forward, left, right, backward, or stop")

    while True:
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            handle_voice_command(text, nav)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Speech recognition error:", e)
            break
        except KeyboardInterrupt:
            break