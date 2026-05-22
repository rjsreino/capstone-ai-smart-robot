from fastapi import FastAPI
from live_vision_assistant import (
    latest_detections,
    guidance_cmd,
    zones_data
)
import uvicorn
import threading
from live_vision_assistant import vision_loop

app = FastAPI()

latest_data = {
    "guidance": "WAIT",
    "left_distance": 0,
    "center_distance": 0,
    "right_distance": 0,
    "detections": []
}

@app.get("/")
def root():
    return {
        "status": "AI server running"
    }

@app.get("/status")
def status():

    detections = []

    for d in latest_detections:

        detections.append({
            "object": d["class_name"],
            "position": d["position"],
            "distance": d["distance"]
        })

    return {
    "guidance": guidance_cmd,
    "left_distance": zones_data.get("left", 0),
    "center_distance": zones_data.get("center", 0),
    "right_distance": zones_data.get("right", 0),
    "detections": detections
}

@app.post("/command")
def command(data: dict):

    user_command = data.get("command", "")

    print(f"[COMMAND] {user_command}")

    return {
        "response": f"Received command: {user_command}"
    }

if __name__ == "__main__":
    vision_thread = threading.Thread(
    target=vision_loop,
    daemon=True
)
    vision_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)