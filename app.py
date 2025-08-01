from flask import Flask, request, render_template
from utils.face_utils import detect_and_crop_face
from utils.inference import predict, ArcFaceBackbone
import torch
import os

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArcFaceBackbone().to(DEVICE)
model.load_state_dict(torch.load("Models/arcface_ranknet_final1.pth", map_location=DEVICE))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        cropped_face, image_filename = detect_and_crop_face(file)

        if cropped_face is None:
            return render_template("index.html", error="Yüz bulunamadı.")

        # Yüz kırpıldıktan sonra diske kaydedilen görselin yolunu al
        face_path = os.path.join("static", image_filename)

        # Tahmini yap (Aynen terminaldeki predict fonksiyonu gibi)
        score = predict(face_path, model)

        return render_template("index.html", score=score, image_file=image_filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
