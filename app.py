import os
import io
import base64

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# ── Config desde variables de entorno ──
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
MODEL_ID         = os.environ.get("ROBOFLOW_MODEL", "goat-looker/6")

# ── Inicializar cliente Roboflow Hosted API ──
CLIENT = None

def init_client():
    global CLIENT
    if not ROBOFLOW_API_KEY:
        print("⚠️  Sin ROBOFLOW_API_KEY — detección deshabilitada")
        return
    try:
        from inference_sdk import InferenceHTTPClient
        CLIENT = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        print(f"✅ Cliente Roboflow listo — modelo: {MODEL_ID}")
    except Exception as e:
        print(f"❌ Error iniciando cliente Roboflow: {e}")

init_client()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if not CLIENT:
        return jsonify({"error": "Sin API key configurada", "predictions": []}), 503

    try:
        # Obtener imagen
        if request.content_type and "application/json" in request.content_type:
            data      = request.get_json()
            b64       = data.get("image", "").split(",")[-1]
            img_bytes = base64.b64decode(b64)
        else:
            f = request.files.get("image")
            if not f:
                return jsonify({"error": "No image provided"}), 400
            img_bytes = f.read()

        conf = float(request.args.get("confidence", 0.35))
        conf = max(0.1, min(0.95, conf))

        # Guardar imagen temporal
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tmp_path = "/tmp/frame.jpg"
        img.save(tmp_path, "JPEG", quality=85)

        # Llamar a Roboflow Hosted API
        result = CLIENT.infer(tmp_path, model_id=MODEL_ID)

        # Filtrar por confianza
        predictions = []
        for p in result.get("predictions", []):
            if p["confidence"] < conf:
                continue
            predictions.append({
                "x":          p["x"],
                "y":          p["y"],
                "width":      p["width"],
                "height":     p["height"],
                "confidence": p["confidence"],
                "class":      p.get("class", "goat"),
            })

        return jsonify({
            "predictions": predictions,
            "image":       {"width": img.width, "height": img.height},
            "model":       MODEL_ID
        })

    except Exception as e:
        print(f"Error en /detect: {e}")
        return jsonify({"error": str(e), "predictions": []}), 500

@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model":        MODEL_ID,
        "client_ready": CLIENT is not None,
        "model_type":   "roboflow-hosted" if CLIENT else "no-model"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
