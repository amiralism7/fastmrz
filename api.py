from flask import Flask, request, jsonify
from fastmrz import FastMRZ
from flask_cors import CORS
import sys
import base64
import cv2
app = Flask(__name__)

CORS(app)

@app.route('/extract_mrz_from_base64', methods=['POST'])
def extract_mrz_from_base64():
    fast_mrz = FastMRZ()
    if 'base64' not in request.json:
        return jsonify({'error': 'No base64 image provided'}), 400

    base64_image = request.json['base64']
    text_raw, parsed = fast_mrz.get_mrz(base64_image, raw=True)
    api_output = parsed
    api_output["mrz_text"] = text_raw
    # get the proper_threshold in order to recreate the processed image
    proper_threshold = fast_mrz._proper_threshold
    processed_image = fast_mrz._apply_threshold(fast_mrz.image, proper_threshold)
    # encode the processed image to base64 format using 
    base64_image = base64.b64encode(cv2.imencode('.jpg', processed_image)[1]).decode()
    api_output["processed_image"] = base64_image
    
    return jsonify(api_output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)