import cv2

from models.ocr_model import ocr_reader

def extract_text_from_frame(frame) -> str:
    if frame is None:
        return ""

    h, w = frame.shape[:2]

    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    y1 = int(h * 0.15)
    y2 = int(h * 0.75)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = ocr_reader.readtext(thresh)

    filtered_lines = []
    seen = set()

    for _, text, conf in results:
        text = text.strip()
        if len(text) < 3:
            continue
        if conf < 0.30:
            continue

        lowered = text.lower()
        if lowered in seen:
            continue

        seen.add(lowered)
        filtered_lines.append(text)

    return " ".join(filtered_lines)