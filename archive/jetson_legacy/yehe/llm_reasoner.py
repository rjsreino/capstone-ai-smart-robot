import json
from ollama import chat

SYSTEM_PROMPT = """
You are an assistive vision robot.

Rules:
1. Answer in one short sentence only.
2. Use only the provided detections.
3. Do not mention OCR unless OCR text is explicitly provided in the prompt.
4. Do not guess actions, emotions, intentions, or future behavior.
5. If no relevant object is present, say so directly.
6. No extra explanation.
7. No safety advice.
8. No assumptions.
"""

def ask_llm(question: str, detections: list[dict], ocr_text: str = "") -> str:
    payload = {
        "question": question,
        "detections": detections,
        "ocr_text": ocr_text
    }

    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    response = chat(
        model="phi3",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": 0.1,
            "num_predict": 30
        }
    )

    return response.message.content.strip()