from fastapi import FastAPI
import uvicorn
import threading
import zed_vision_assistant as zva
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File
import tempfile
import whisper
from llm_reasoner import ask_llm

whisper_model = whisper.load_model("tiny")
app = FastAPI()

@app.get("/")
def root():
    return {"status": "AI server running"}

@app.get("/status")
def status():
    with zva.frame_lock:
        detections = [
            {
                "object": d["class_name"],
                "position": d["position"],
                "distance": d["distance"],
                "depth_meters": d.get("depth_meters"),
                "confidence": round(d["confidence"], 2)
            }
            for d in zva.latest_detections
        ]

        return {
            "guidance": zva.guidance_cmd,
            "left_distance": zva.zones_data.get("left", 0),
            "center_distance": zva.zones_data.get("center", 0),
            "right_distance": zva.zones_data.get("right", 0),
            "detections": detections
        }
        
@app.get("/map", response_class=HTMLResponse)
def map_view():
    return """
    <html>
    <body style="background:#0f172a;color:white;font-family:Arial;text-align:center;">
        <h2>Supervisor HUD</h2>
        <p>10m x 10m Bird's Eye View Grid</p>

        <div style="
            width:300px;
            height:300px;
            margin:auto;
            display:grid;
            grid-template-columns:repeat(3,1fr);
            grid-template-rows:repeat(3,1fr);
            gap:4px;
        ">
            <div style="background:#1e293b;padding:20px;">NW</div>
            <div style="background:#1e293b;padding:20px;">Front</div>
            <div style="background:#1e293b;padding:20px;">NE</div>
            <div style="background:#1e293b;padding:20px;">Left</div>
            <div style="background:#dc2626;padding:20px;">Robot</div>
            <div style="background:#1e293b;padding:20px;">Right</div>
            <div style="background:#1e293b;padding:20px;">SW</div>
            <div style="background:#16a34a;padding:20px;">Back</div>
            <div style="background:#1e293b;padding:20px;">SE</div>
        </div>
    </body>
    </html>
    """

@app.post("/voice-command")
async def voice_command(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".caf") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    result = whisper_model.transcribe(temp_audio_path, language="en")
    command_text = result["text"].strip().lower()

    with zva.frame_lock:
        detections = list(zva.latest_detections)
        guidance = zva.guidance_cmd
        zones_data = dict(zva.zones_data)

    llm_detections = [
        {
            "class": d["class_name"],
            "position": d["position"],
            "distance": d["distance"],
            "confidence": round(d["confidence"], 2),
            "depth_meters": d.get("depth_meters"),
        }
        for d in detections
    ]

    response = ask_llm(
        question=command_text,
        detections=llm_detections,
        direction_summary={
            "left_distance_mm": float(zones_data.get("left", 0)),
            "center_distance_mm": float(zones_data.get("center", 0)),
            "right_distance_mm": float(zones_data.get("right", 0)),
            "best_direction": guidance,
        },
        ocr_text=""
    )

    return {
        "transcript": command_text,
        "response": response
    }

if __name__ == "__main__":
    vision_thread = threading.Thread(
        target=zva.vision_loop,
        daemon=True
    )

    vision_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)