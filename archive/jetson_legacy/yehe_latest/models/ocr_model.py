import easyocr

try:
    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=True
    )

except Exception:

    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=False
    )