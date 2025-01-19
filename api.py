from flask import Flask, request, jsonify
from fastmrz import FastMRZ
from flask_cors import CORS
import sys
import base64
import cv2
import numpy as np
app = Flask(__name__)

CORS(app)

def process_confidence_data(confidence_data):
    ## find the 2 text with the highest length, and return their confidence
    indices = np.argsort([len(text) for text in confidence_data["text"]])[-2:]
    output = list(np.array(confidence_data["conf"])[indices])
    output = [int(x) for x in output]
    return output

@app.route('/extract_mrz_from_base64', methods=['POST'])
def extract_mrz_from_base64():
    try:
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
        api_output["confidence"] = process_confidence_data(fast_mrz.confidence_data)
        return jsonify(api_output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)