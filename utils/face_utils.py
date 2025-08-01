import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import uuid
import os

face_session = ort.InferenceSession("models/face_detector.onnx", providers=["CPUExecutionProvider"])

def detect_and_crop_face(file, save_path="static"):
    # Görseli byte olarak oku ve cv2 ile çöz
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    input_tensor = img_rgb.astype(np.uint8)  # CHW değil! Direkt HWC → uint8

    input_name = face_session.get_inputs()[0].name
    outputs = face_session.run(None, {input_name: input_tensor})
    # İlk iki çıktıyı varsayıyoruz: scores, bboxes
    scores = outputs[0]
    bboxes = outputs[1]

    if len(scores) == 0:
        return None, None

    best_idx = np.argmax(scores)
    x1, y1, x2, y2 = bboxes[best_idx]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    face = img_rgb[y1:y2, x1:x2]
    pil_face = Image.fromarray(face)

    # Görsel üzerine bbox çiz
    img_draw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

    filename = f"face_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(save_path, filename)
    cv2.imwrite(filepath, img_draw)
    print("Model outputs:", [o.name for o in face_session.get_outputs()])

    return pil_face, filename
