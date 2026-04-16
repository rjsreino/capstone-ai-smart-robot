from llm_reasoner import ask_llm

detections = [
    {"class": "chair", "position": "center", "distance": "close", "confidence": 0.91},
    {"class": "bottle", "position": "right", "distance": "far", "confidence": 0.82}
]

question = "Can I walk forward?"

answer = ask_llm(question, detections)
print(answer)