import json

try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


SYSTEM_PROMPT = """
You are Vicky, a wearable navigation assistant for a blind user.

Answer naturally, like a helpful human guide.
Keep the answer short: maximum 2 sentences.
Use only the provided detections, depth, and direction data.
Do not invent objects.
If the user asks about a specific object and it is not detected, say it is not detected.
If there is danger, prioritize safety first.
Avoid robotic phrases like "Current guidance is" or "I detect".
"""


def _get_class_name(d: dict) -> str:
    return d.get("class", d.get("class_name", "object")).lower()


def _fallback_response(
    question: str,
    detections: list[dict],
    direction_summary: dict | None,
    ocr_text: str = "",
) -> str:
    q = question.lower()

    object_words = [
        "bottle", "person", "chair", "couch", "laptop", "cup",
        "phone", "cell phone", "book", "table", "dining table",
        "backpack", "bag", "handbag", "door", "keyboard", "mouse",
        "remote", "tv", "bed", "plant"
    ]

    requested = None
    for word in object_words:
        if word in q:
            requested = word
            break

    if requested:
        matches = [
            d for d in detections
            if requested in _get_class_name(d)
        ]

        if not matches:
            return f"No, I do not see a {requested} right now."

        target = matches[0]
        obj = _get_class_name(target)
        pos = target.get("position", "ahead")
        dist = target.get("distance", "nearby")
        depth = target.get("depth_meters")

        if depth is not None:
            return f"Yes, I see a {obj} on your {pos}, about {depth:.1f} meters away."
        return f"Yes, I see a {obj} on your {pos}."

    if not detections:
        return "I do not see any major obstacle right now."

    nearest = detections[0]
    obj = _get_class_name(nearest)
    pos = nearest.get("position", "ahead")
    dist = nearest.get("distance", "nearby")
    depth = nearest.get("depth_meters")

    guidance = ""
    if direction_summary:
        guidance = direction_summary.get("best_direction", "")

    if "safe" in q or "path" in q or "forward" in q or "walk" in q:
        if guidance == "GO FORWARD":
            return "The path ahead looks clear. Move forward carefully."
        if "STOP" in guidance:
            return f"Stop for now. There is a {obj} too close near the {pos}."
        if guidance:
            return f"The path is not fully clear. {guidance.title().replace('_', ' ')}."

    if "left" in q and direction_summary:
        left = direction_summary.get("left_distance_mm", 0)
        return f"Your left side has about {left:.0f} millimeters of clearance."

    if "right" in q and direction_summary:
        right = direction_summary.get("right_distance_mm", 0)
        return f"Your right side has about {right:.0f} millimeters of clearance."

    if depth is not None:
        return f"There is a {obj} {dist} on your {pos}, about {depth:.1f} meters away."

    return f"There is a {obj} {dist} on your {pos}."


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
        "direction_summary": direction_summary or {},
        "ocr_text": ocr_text,
    }

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
                        "content": (
                            "Answer the user's question using only this JSON scene data:\n"
                            + json.dumps(payload, indent=2)
                        ),
                    },
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": 70,
                },
            )

            answer = response["message"]["content"].strip()

            if answer:
                print("[OLLAMA RESPONSE]", answer)
                return answer

        except Exception as e:
            print(f"[LLM WARNING] Ollama failed: {e}")

    print("[LLM FALLBACK] Using local fallback response.")
    return _fallback_response(question, detections, direction_summary, ocr_text)