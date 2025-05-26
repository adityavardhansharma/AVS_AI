import os
import json
from dotenv import load_dotenv
from flask import (
    Flask, request, jsonify,
    Response, stream_with_context, send_from_directory
)
from flask_cors import CORS
import requests

# Load environment variables
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise RuntimeError("SARVAM_API_KEY not set. Check your .env file.")

SARVAM_CHAT_ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/chat": {"origins": "*"}})

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/chat", methods=["POST"])
def chat_completion_proxy():
    try:
        payload_in = request.get_json()
        if not payload_in or "message" not in payload_in:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        user_msg = payload_in["message"]
        messages = [{"role": "user", "content": user_msg}]
        sarvam_payload = {
            "model": "sarvam-m",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {SARVAM_API_KEY}",
            "Content-Type": "application/json",
        }

        sarvam_response = requests.post(SARVAM_CHAT_ENDPOINT, headers=headers, json=sarvam_payload, stream=True)
        sarvam_response.raise_for_status()

        def generate():
            for chunk in sarvam_response.iter_content(chunk_size=None):
                if chunk:
                    decoded = chunk.decode('utf-8')
                    for line in decoded.splitlines():
                        if line.startswith('data: '):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                yield ""
                                return
                            try:
                                data_obj = json.loads(data_str)
                                delta = data_obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                            except (json.JSONDecodeError, IndexError):
                                continue

        return Response(stream_with_context(generate()), mimetype='text/plain')

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)