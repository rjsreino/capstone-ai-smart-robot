import whisper
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_model = whisper.load_model(
    "tiny"
).to(device)