### test api

import requests
import json
import base64
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def test_extract_mrz_from_base64():
    url = 'http://127.0.0.1:5001/extract_mrz_from_base64'
    with open("./data/passport_uk.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    req = requests.post(url, json={'base64': encoded_string.decode('utf-8')})
    print("THIS IS THE RESPONSE of BASE64")
    print(req.text)
    ## get the image from the response and plot 
    # image_data = base64.b64decode(json.loads(req.text)["processed_image"])
    # # decode the base64 image to a numpy array
    # image_array = np.frombuffer(image_data, np.uint8)
    # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # # plot the image
    # plt.imshow(image)
    # plt.show()
    print("--------------------------")
    assert req.status_code == 200
if __name__ == '__main__':
    test_extract_mrz_from_base64()
    
    