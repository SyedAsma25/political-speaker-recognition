from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Political Speaker Recognition API is LIVE"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    
    # dummy prediction (replace later)
    return jsonify({
        "input_text": text,
        "predicted_speaker": "Unknown (model not loaded yet)"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
