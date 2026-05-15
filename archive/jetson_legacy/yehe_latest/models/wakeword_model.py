from openwakeword.model import Model

from config.settings import WAKEWORD_NAME


wakeword_model = Model(
    wakeword_models=[WAKEWORD_NAME],
    inference_framework="onnx"
)