import json

from ollama import chat


SYSTEM_PROMPT = """
You are an assistive navigation AI.

Rules:
1. Answer in one short sentence only.
2. Use only the provided context.
3. Prefer the direction with fewer close obstacles.
4. Do not invent objects or distances.
5. Do not mention OCR unless OCR text exists.
"""


def ask_llm(
    question: str,
    detections: list[dict],
    direction_summary: dict | None = None,
    scene_mode: str = "navigation assistance",
    goal: str = "help the user move safely",
    ocr_text: str = "",
) -> str:

    payload = {
        "question": question,
        "scene_mode": scene_mode,
        "goal": goal,
        "detections": detections,
    }

    if direction_summary:
        payload["direction_summary"] = direction_summary

    if ocr_text.strip():
        payload["ocr_text"] = ocr_text

    response = chat(
        model="phi3",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(payload),
            },
        ],
    )

    return response["message"]["content"].strip()