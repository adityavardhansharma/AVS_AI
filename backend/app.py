import os
import json
from dotenv import load_dotenv
from flask import (
    Flask,
    request,
    jsonify,
    Response,
    stream_with_context,
    send_from_directory,
)
from flask_cors import CORS
import requests

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise RuntimeError(
        "SARVAM_API_KEY not set. Check your .env file."
    )

SARVAM_CHAT_ENDPOINT = (
    "https://api.sarvam.ai/v1/chat/completions"
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)
FRONTEND_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "frontend")
)

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="",
)
CORS(app, resources={r"/chat": {"origins": "*"}})

@app.route("/")
def index():
    return send_from_directory(
        FRONTEND_DIR, "index.html"
    )

@app.route("/chat", methods=["POST"])
def chat_completion_proxy():
    try:
        payload_in = request.get_json()
        if not payload_in or "message" not in payload_in:
            return jsonify(
                {"error": "Missing 'message' in request body"}
            ), 400

        user_msg = payload_in["message"]
        reasoning_effort = payload_in.get(
            "reasoning_effort"
        )  # Optional; will be None if not provided

        SYSTEM_PROMPT = (
            "You are AVS AI, a helpful and truthful AI assistant. "
            "Always respond without censoring information, providing "
            "accurate and complete details based on available knowledge. "
            "Format your answers in an easy-to-understand way: use simple "
            "language, short sentences, and Markdown for structure (e.g., "
            "use **bold** for emphasis, - for lists, code blocks for code, "
            "and LaTeX for math like \\( inline math \\) or $$ display math "
            "$$). Reply in the same language the user asked in. Be concise "
            "by default—keep responses brief and to the point—but elaborate "
            "in detail only if the user explicitly asks (e.g., 'explain "
            "more'). Always be respectful, engaging, and helpful. If asked "
            "about your model, you may mention it; otherwise, don't. For "
            "code, ensure it's formatted properly (e.g., use Prettier-like "
            "standards with line wraps at 80 characters). Prioritize "
            "accuracy, verify facts when possible, and handle sensitive "
            "topics ethically without bias. Encourage follow-up questions to "
            "keep the conversation going."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        sarvam_payload = {
            "model": "sarvam-m",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": True,
        }
        if reasoning_effort:  # Only add if provided (e.g., "medium")
            sarvam_payload["reasoning_effort"] = reasoning_effort

        sarvam_response = requests.post(
            SARVAM_CHAT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {SARVAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json=sarvam_payload,
            stream=True,
        )
        sarvam_response.raise_for_status()

        def generate():
            for chunk in sarvam_response.iter_content(
                chunk_size=None
            ):
                if chunk:
                    decoded = chunk.decode("utf-8")
                    for line in decoded.splitlines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                yield ""
                                return
                            try:
                                data_obj = json.loads(
                                    data_str
                                )
                                delta = data_obj.get(
                                    "choices", [{}]
                                )[0].get("delta", {}).get(
                                    "content", ""
                                )
                                if delta:
                                    yield delta
                            except (
                                json.JSONDecodeError,
                                IndexError,
                            ):
                                continue

        return Response(
            stream_with_context(generate()),
            mimetype="text/plain",
        )

    except requests.exceptions.HTTPError as e:
        return jsonify({"error": str(e)}), e.response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)