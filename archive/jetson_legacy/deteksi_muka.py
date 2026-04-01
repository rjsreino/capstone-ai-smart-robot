import cv2
from face_recognizer import FaceRecognizer

recognizer = FaceRecognizer("known-faces")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # sesuaikan index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Webcam tidak terdeteksi")
    exit()

frame_count = 0
last_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # recognition hanya tiap 5 frame
    if frame_count % 8 == 0:
        last_results = recognizer.recognize_face(frame, is_rgb=False)

    for name, conf, (left, top, right, bottom) in last_results:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()