from flask import Flask, request, jsonify
from fastmrz import FastMRZ
from flask_cors import CORS
import sys
app = Flask(__name__)
fast_mrz = FastMRZ()

## allow CORS
CORS(app)

@app.route('/extract_mrz_from_base64', methods=['POST'])
def extract_mrz_from_base64():
    if 'base64' not in request.json:
        return jsonify({'error': 'No base64 image provided'}), 400

    base64_image = request.json['base64']
    text_raw, parsed = fast_mrz.get_mrz(base64_image, raw=True)
    api_output = {
        "mrz_text": text_raw,
        "status": parsed["status"]
    }
    if "message" in parsed.keys():
        api_output["message"] = parsed["message"]
    
    return jsonify(api_output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)