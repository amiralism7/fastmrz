### test api

import requests
import json
import base64
import os

def test_extract_mrz_from_base64():
    url = 'http://127.0.0.1:5001/extract_mrz_from_base64'
    with open("/app/data/passport_uk.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    req = requests.post(url, json={'base64': encoded_string.decode('utf-8')})
    print("THIS IS THE RESPONSE of BASE64")
    print(req.text)
    print("--------------------------")
    assert req.status_code == 200
if __name__ == '__main__':
    test_extract_mrz_from_base64()
    
    