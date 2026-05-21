import re

import shared.state as state

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_tts_text(text: str) -> str:
    replacements = {
        "it's": "it is",
        "It's": "It is",
        "there's": "there is",
        "There's": "There is",
        "you're": "you are",
        "You're": "You are",
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def is_repeat_command(command: str) -> bool:
    repeat_keywords = [
        "repeat",
        "say again",
        "one more time",
        "repeat it",
        "repeat that",
        "can you repeat",
        "say it again",
    ]
    return any(keyword in command for keyword in repeat_keywords)


def remember_response(text: str):
    global last_response_text
    last_response_text = text