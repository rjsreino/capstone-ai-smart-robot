import pyttsx3

engine = pyttsx3.init(driverName="sapi5")
engine.setProperty("rate", 170)

voices = engine.getProperty("voices")

for i, voice in enumerate(voices):
    print(f"{i}: {voice.id}")

if voices:
    engine.setProperty("voice", voices[1].id)

engine.say("Hello. This is a test.")
engine.runAndWait()

print("Done")