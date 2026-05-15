import shared.state as state

from utils.text_utils import (
    normalize_text,
    is_repeat_command,
    remember_response,
)

try:
    from assistant.llm_reasoner import ask_llm

    print("[LLM] Connected to llm_reasoner.py")

except Exception as e:

    print(
        "[LLM ERROR] Could not import llm_reasoner.py:",
        e
    )

    def ask_llm(command: str, detections: list[dict]) -> str:
        return "LLM is not connected right now."


def handle_vision_query(
    command: str,
    detections: list[dict],
    frame,
):

    command = normalize_text(command)

    if command in {
        "stop",
        "exit",
        "quit",
    }:
        return "__EXIT__"

    if (
        "stop vision mode" in command
        or "exit vision mode" in command
    ):
        return "__EXIT__"

    if is_repeat_command(command):

        if state.last_response_text:
            return state.last_response_text

        return "There is nothing recent to repeat."

    llm_input = []

    for detection in detections:

        llm_input.append({
            "class": detection["class_name"],
            "position": detection["position"],
            "distance": detection["distance"],
            "confidence": round(
                detection["confidence"],
                2
            ),
        })

    try:

        answer = ask_llm(
            command,
            llm_input
        )

        remember_response(answer)

        return answer

    except Exception as e:

        print("[LLM ERROR]", e)

        answer = (
            "I could not process "
            "that question right now."
        )

        remember_response(answer)

        return answer