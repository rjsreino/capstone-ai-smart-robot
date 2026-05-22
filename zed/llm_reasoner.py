import json
import logging

try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

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
    """
    Queries local Ollama using phi3 with scene context and user question.
    Falls back gracefully to rule-based responses if Ollama is unreachable.
    """
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

    # 1. Attempt to use Ollama if available
    if OLLAMA_AVAILABLE:
        try:
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
        except Exception as e:
            print(f"[LLM WARNING] Ollama query failed (Ensure Ollama is running and 'phi3' is pulled): {e}")

    # 2. Rule-based fallback if Ollama fails or is not installed
    print("[LLM FALLBACK] Generating a rule-based fallback scene description.")
    
    # Analyze detections
    if not detections:
        scene_desc = "The path in front of you looks completely clear."
    else:
        obj_strings = []
        for d in detections[:3]:  # Top 3 closest/largest objects
            pos = d.get("position", "ahead")
            dist = d.get("distance", "some distance away")
            cls = d.get("class_name", "object")
            obj_strings.append(f"a {cls} on your {pos} which is {dist}")
        scene_desc = f"I detect " + ", ".join(obj_strings) + "."

    # Include safety direction guidance
    if direction_summary and "best_direction" in direction_summary:
        best_dir = direction_summary["best_direction"].replace("_", " ")
        scene_desc += f" The safest path appears to be to the {best_dir}."

    if ocr_text.strip():
        scene_desc += f" I also read the text: '{ocr_text}'."

    return scene_desc
