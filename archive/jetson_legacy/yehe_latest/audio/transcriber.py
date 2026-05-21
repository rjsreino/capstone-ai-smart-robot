import io
import numpy as np
import soundfile as sf
import whisper

from models.whisper_model import whisper_model

from config.settings import SAMPLE_RATE

def transcribe_audio(wav_buffer: io.BytesIO) -> str:

    audio_array, sample_rate = sf.read(
        wav_buffer,
        dtype="float32"
    )

    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)

    if sample_rate != SAMPLE_RATE:
        import librosa

        audio_array = librosa.resample(
            audio_array,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    if audio_array.size == 0:
        return ""

    audio_array = whisper.pad_or_trim(audio_array)

    mel = whisper.log_mel_spectrogram(
        audio_array
    ).to(whisper_model.device)

    options = whisper.DecodingOptions(
        language="en",
        fp16=False
    )

    result = whisper.decode(
        whisper_model,
        mel,
        options
    )

    return result.text.strip()