import json
from ollama import chat


SYSTEM_PROMPT = """
You are an assistive vision robot.

Rules:
1. Answer in maximum 1 short sentence.
2. Use only provided detections.
3. Never guess intentions, emotions, actions, or future behavior.
4. If information is unavailable, say you cannot determine that.
5. Be direct and practical.
6. No extra explanation.
7. No safety lecture.
8. No assumptions.

Examples:
Q: What is in front of me?
A: A person is in front of you.

Q: What is the guy doing?
A: I can only detect that a person is standing in front of you.

Q: Where is the bottle?
A: The bottle is on the right.

Q: Is path clear?
A: The path ahead is blocked by a chair.
"""

def ask_llm(question: str, detections: list[dict]) -> str:
    detections_json = json.dumps(detections, ensure_ascii=False)

    user_prompt = f"""
User question:
{question}

Current camera detections:
{detections_json}

Answer as an assistive robot.
"""

    response = chat(
    model='phi3',
    messages=[
        {'role':'system','content':SYSTEM_PROMPT},
        {'role':'user','content':user_prompt}
    ],
    options={
        "temperature": 0.2,
        "num_predict": 40
    }
)

    return response.message.content.strip()