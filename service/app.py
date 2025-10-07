from flask import Flask, request, jsonify
from pathlib import Path
from model_utils import train_and_save, predict_for_month, load_model

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train():
    payload = request.get_json(silent=True) or {}
    data_dir = payload.get("data_dir", "cs-train")
    meta = train_and_save(data_dir)
    return jsonify({"message": "model trained", "meta": meta})

@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    target = data.get("date") or data.get("month")  # acceptă {"date":"YYYY-MM"}
    if not target:
        return jsonify({"error": "Provide 'date' as 'YYYY-MM'"}), 400
    try:
        result = predict_for_month(target)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Model not trained. Call /train first."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # pentru dezvoltare locală
    app.run(host="0.0.0.0", port=8000, debug=True)
