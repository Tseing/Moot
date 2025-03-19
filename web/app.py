from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from utils import ModelHost, get_result, validate_prediction_form

app = Flask(__name__)
CORS(app)
MOOT_APP_PATH = "moot"
MODEL_HOST = ModelHost("all", "cuda")


@app.route("/", methods=["GET", "POST"])
def index():
    return jsonify(
        {
            "type": "info",
            "message": "Server is working. Moot Application URL: http://127.0.0.1:5000/app/index.html",
        }
    )


@app.route("/app/<path:filename>", methods=["GET"])
def moot_app(filename):
    try:
        return send_from_directory(MOOT_APP_PATH, filename)
    except FileNotFoundError:
        return "File not found", 404


@app.route("/predict", methods=["POST"])
def predict():
    prediction_form = request.get_json()
    is_success, message = validate_prediction_form(prediction_form)
    if not is_success:
        return jsonify(
            {
                "type": "error",
                "message": message,
            }
        )

    # smiles_list, beam_size, mode = (
    #     prediction_form["smiles"],
    #     prediction_form["beam_size"],
    #     prediction_form["mode"],
    # )
    is_success, result = get_result(prediction_form["_cleaned_data"], MODEL_HOST)
    if not is_success:
        return jsonify({"type": "error", "message": "Internal error."})

    return jsonify({"type": "info", "result": result})


if __name__ == "__main__":
    print("Moot Application URL: http://127.0.0.1:5000/app/index.html")
    app.run(debug=True)
