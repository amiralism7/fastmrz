import numpy as np
import cv2
import pytesseract
from datetime import datetime
import os
import base64
import binascii
import sys
import io

class FastMRZ:
    def __init__(self, tesseract_path=""):
        self.tesseract_path = tesseract_path
        self.net = cv2.dnn.readNetFromONNX(
            os.path.join(os.path.dirname(__file__), "model/mrz_seg.onnx")
        )
        self.image = None

    def _cleanse_roi(self, raw_text):
        input_list = raw_text.replace(" ", "").split("\n")

        selection_length = next(
            (
                len(item)
                for item in input_list
                if "<" in item and len(item) in {30, 36, 44}
            ),
            None,
        )
        if selection_length is None:
            return ""
        new_list = [item for item in input_list if len(item) >= selection_length]
        return "\n".join(new_list)

    def _get_final_check_digit(self, input_string, input_type):
        if input_type == "TD3":
            return self._get_check_digit(
                input_string[:10] + input_string[13:20] + input_string[21:43]
            )
        elif input_type == "TD2":
            return self._get_check_digit(
                input_string[:10] + input_string[13:20] + input_string[21:35]
            )
        else:
            return self._get_check_digit(
                input_string[0][5:]
                + input_string[1][:7]
                + input_string[1][8:15]
                + input_string[1][18:29]
            )

    def _get_check_digit(self, input_string):
        weights_pattern = [7, 3, 1]

        total = 0
        for i, char in enumerate(input_string):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - ord("A") + 10
            else:
                value = 0
            total += value * weights_pattern[i % len(weights_pattern)]

        check_digit = total % 10

        return str(check_digit)

    def _format_date(self, input_date):
        formatted_date = str(datetime.strptime(input_date, "%y%m%d").date())
        return formatted_date
    
    def _is_valid(self, image_input):
        if isinstance(image_input, str):
            if os.path.isfile(image_input):
                return True
            else:
                try:
                    base64.b64decode(image_input)
                    return True
                except binascii.Error:
                    return False
        elif isinstance(image_input, np.ndarray):
            return image_input.shape[-1] == 3
        else:
            return False
        
    def _process_image(self, threshold=255):
        image = self.image.copy()
        # filter out high intensity pixels
        image = self._apply_threshold(image, threshold)
        # Proceed with existing processing
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        image = np.asarray(np.float32(image / 255))
        if image.shape[-1] > 3:
            image = image[:, :, :3]
        image = np.reshape(image, (1, 256, 256, 3))
        return image
    
    def _get_roi(self, output_data, threshold=255):
        if self.tesseract_path != "":
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        image = self.image.copy()
        output_data = (output_data[0, :, :, 0] > 0.35).astype(np.uint8) * 255
        altered_image = cv2.resize(output_data, (image.shape[1], image.shape[0]))

        kernel = np.ones((5, 5), dtype=np.uint8)
        altered_image = cv2.erode(altered_image, kernel, iterations=3)
        contours, _ = cv2.findContours(
            altered_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            return ""

        c_area = np.array([cv2.contourArea(c) for c in contours])
        x, y, w, h = cv2.boundingRect(contours[np.argmax(c_area)])
        roi_arr = image[y : y + h, x : x + w].copy()

        #### Filter out high intensity pixels
        roi_arr = self._apply_threshold(roi_arr, threshold)
        return pytesseract.image_to_string(roi_arr, lang="mrz")
    
    def _get_raw_mrz(self, threshold=255):
        image_array = self._process_image(threshold)   
        self.net.setInput(image_array)
        output_data = self.net.forward()
        
        raw_roi = self._get_roi(output_data, threshold)
        return self._cleanse_roi(raw_roi)
    
    def get_mrz_with_threshold(self, image, raw=False, threshold=255):
        if not self._is_valid(image):
            return {"status": "FAILURE", "message": "Invalid input image"}
        
        if not isinstance(threshold, int) or threshold < 0 or threshold > 255:
            threshold = 255
            print("Invalid threshold value. Setting threshold to 255")
            
        self._load_image(image)
        mrz_text = self._get_raw_mrz(threshold=threshold)
        ### rotate image if no MRZ detected 
        if not mrz_text:
            for i in range(1, 4):
                self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
                mrz_text = self._get_raw_mrz(threshold=threshold)
                if mrz_text:
                    break
        ###
        return mrz_text if raw else self._parse_mrz(mrz_text)
    
    def get_mrz(self, image, raw=False):
        if not self._is_valid(image):
            return {"status": "FAILURE", "message": "Invalid input image"}
        
        self._load_image(image)
        ## we want to find the best rotation (we check for multiple thresholds), which will return a MRZ of length at least 20
        thresholds = [20, 90, 210]
        
        for threshold in thresholds:
            for rotation in range(4):
                mrz_text = self._get_raw_mrz(threshold=threshold)
                if len(mrz_text) > 20:
                    break
                self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE * rotation)
            if len(mrz_text) > 20:
                break
            
        ## check if the loops where broken (a sign that we found a MRZ)
        if len(mrz_text) <= 20:
            mrz_text = ""
            parsed_mrz = {"status": "FAILURE", "message": "No MRZ detected"}
        else:
            ## a correct mrz will also have a valid parsed mrz
            ## we check for different thresholds to find the one that has a valid parsed mrz
            ## if we don't find a valid parsed mrz, we return the one that has the correct length (89 = 44 + 44 + 1)
            ## if we don't find a correct length, we return the last one
            correct_len = None
            correct_len_parsed = None
            for threshold in thresholds:
                mrz_text = self._get_raw_mrz(threshold=threshold)
                parsed_mrz = self._parse_mrz(mrz_text)
                if parsed_mrz["status"] == "SUCCESS":
                    break
                if len(mrz_text) == 89:
                    correct_len = mrz_text
                    correct_len_parsed = parsed_mrz
                    
            if parsed_mrz["status"] != "SUCCESS":
                if correct_len is not None:
                    mrz_text = correct_len
                    parsed_mrz = correct_len_parsed

        return (mrz_text, parsed_mrz) if raw else parsed_mrz

    def _apply_threshold(self, image, threshold=255):
        mask = cv2.inRange(image, (threshold, threshold, threshold), (255, 255, 255))
        mask = np.stack([mask]*3, axis=-1)
        filtered_image = np.where(mask, 255, image).astype(np.uint8)
        return filtered_image
    
    def _load_image(self, image_input):
        if isinstance(image_input, str):
            if os.path.isfile(image_input):
                # It's a file path
                image = cv2.imread(image_input, cv2.IMREAD_COLOR)
            else:
                # Assume it's a base64-encoded string
                try:
                    image_data = base64.b64decode(image_input)
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except binascii.Error:
                    raise ValueError("Invalid base64 image data")
        elif isinstance(image_input, np.ndarray):
            # It's already an image array
            image = image_input
        else:
            raise ValueError("Invalid image input type")
        self.image = image   

    def _get_date_of_birth(self, date_of_birth_str, date_of_expiry_str):
        birth_year = int(date_of_birth_str[:4])
        expiry_year = int(date_of_expiry_str[:4])

        if expiry_year > birth_year:
            return date_of_birth_str
        adjusted_year = birth_year - 100
        return f"{adjusted_year}-{date_of_birth_str[5:]}"

    def _parse_mrz(self, mrz_text):
        if not mrz_text:
            return {"status": "FAILURE", "message": "No MRZ detected"}
        mrz_lines = mrz_text.strip().split("\n")
        if len(mrz_lines) not in [2, 3]:
            return {"status": "FAILURE", "message": "Invalid MRZ format"}

        mrz_code_dict = {}
        if len(mrz_lines) == 2:
            mrz_code_dict["mrz_type"] = "TD2" if len(mrz_lines[0]) == 36 else "TD3"

            # Line 1
            mrz_code_dict["document_type"] = mrz_lines[0][:2].strip("<")
            mrz_code_dict["country_code"] = mrz_lines[0][2:5]
            if not mrz_code_dict["country_code"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ format"}
            names = mrz_lines[0][5:].split("<<")
            mrz_code_dict["surname"] = names[0].replace("<", " ")
            mrz_code_dict["given_name"] = names[1].replace("<", " ")

            # Line 2
            mrz_code_dict["document_number"] = mrz_lines[1][:9].replace("<", "")
            if (
                self._get_check_digit(mrz_code_dict["document_number"])
                != mrz_lines[1][9]
            ):
                return {
                    "status": "FAILURE",
                    "message": "document number checksum is not matching",
                }
            mrz_code_dict["nationality"] = mrz_lines[1][10:13]
            if not mrz_code_dict["nationality"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ format"}
            mrz_code_dict["date_of_birth"] = mrz_lines[1][13:19]
            if (
                self._get_check_digit(mrz_code_dict["date_of_birth"])
                != mrz_lines[1][19]
            ):
                return {
                    "status": "FAILURE",
                    "message": "date of birth checksum is not matching",
                }
            mrz_code_dict["date_of_birth"] = self._format_date(
                mrz_code_dict["date_of_birth"]
            )
            mrz_code_dict["sex"] = mrz_lines[1][20]
            mrz_code_dict["date_of_expiry"] = mrz_lines[1][21:27]
            if (
                self._get_check_digit(mrz_code_dict["date_of_expiry"])
                != mrz_lines[1][27]
            ):
                return {
                    "status": "FAILURE",
                    "message": "date of expiry checksum is not matching",
                }
            mrz_code_dict["date_of_expiry"] = self._format_date(
                mrz_code_dict["date_of_expiry"]
            )
            mrz_code_dict["date_of_birth"] = self._get_date_of_birth(
                mrz_code_dict["date_of_birth"], mrz_code_dict["date_of_expiry"]
            )
            if mrz_code_dict["mrz_type"] == "TD3":
                mrz_code_dict["optional_data"] = mrz_lines[1][28:35].strip("<")

            mrz_code_dict["optional_data"] = (
                mrz_lines[1][28:35].strip("<")
                if mrz_code_dict["mrz_type"] == "TD2"
                else mrz_lines[1][28:42].strip("<")
            )
            if mrz_lines[1][-1] != self._get_final_check_digit(
                mrz_lines[1], mrz_code_dict["mrz_type"]
            ):
                return {
                    "status": "FAILURE",
                    "message": "final checksum is not matching",
                }

        else:
            mrz_code_dict["mrz_type"] = "TD1"

            # Line 1
            mrz_code_dict["document_type"] = mrz_lines[0][:2].strip("<")
            mrz_code_dict["country_code"] = mrz_lines[0][2:5]
            if not mrz_code_dict["country_code"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ format"}
            mrz_code_dict["document_number"] = mrz_lines[0][5:14]
            if (
                self._get_check_digit(mrz_code_dict["document_number"])
                != mrz_lines[0][14]
            ):
                return {
                    "status": "FAILURE",
                    "message": "document number checksum is not matching",
                }
            mrz_code_dict["optional_data_1"] = mrz_lines[0][15:].strip("<")

            # Line 2
            mrz_code_dict["date_of_birth"] = mrz_lines[1][:6]
            if self._get_check_digit(mrz_code_dict["date_of_birth"]) != mrz_lines[1][6]:
                return {
                    "status": "FAILURE",
                    "message": "date of birth checksum is not matching",
                }
            mrz_code_dict["date_of_birth"] = self._format_date(
                mrz_code_dict["date_of_birth"]
            )
            mrz_code_dict["sex"] = mrz_lines[1][7]
            mrz_code_dict["date_of_expiry"] = mrz_lines[1][8:14]
            if (
                self._get_check_digit(mrz_code_dict["date_of_expiry"])
                != mrz_lines[1][14]
            ):
                return {
                    "status": "FAILURE",
                    "message": "date of expiry checksum is not matching",
                }
            mrz_code_dict["date_of_expiry"] = self._format_date(
                mrz_code_dict["date_of_expiry"]
            )
            mrz_code_dict["date_of_birth"] = self._get_date_of_birth(
                mrz_code_dict["date_of_birth"], mrz_code_dict["date_of_expiry"]
            )
            mrz_code_dict["nationality"] = mrz_lines[1][15:18]
            if not mrz_code_dict["nationality"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ format"}
            mrz_code_dict["optional_data_2"] = mrz_lines[0][18:29].strip("<")
            if mrz_lines[1][-1] != self._get_final_check_digit(
                mrz_lines, mrz_code_dict["mrz_type"]
            ):
                return {
                    "status": "FAILURE",
                    "message": "final checksum is not matching",
                }

            # Line 3
            names = mrz_lines[2].split("<<")
            mrz_code_dict["surname"] = names[0].replace("<", " ")
            mrz_code_dict["given_name"] = names[1].replace("<", " ")

        # Final status
        mrz_code_dict["status"] = "SUCCESS"
        return mrz_code_dict

