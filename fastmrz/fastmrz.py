
import numpy as np
import cv2
import pytesseract
from datetime import datetime
import os
import base64
import binascii
import pickle

class FastMRZ:
    def __init__(self, tesseract_path=""):
        self.tesseract_path = tesseract_path
        self.net = cv2.dnn.readNetFromONNX(
            os.path.join(os.path.dirname(__file__), "model/mrz_seg.onnx")
        )
        self.image = None
        self._thresholds = [20, 90, 210]
        self._proper_threshold = 255
        self.confidence_data = None

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
        # Filter out high intensity pixels
        image = self._apply_threshold(image, threshold)
        # Resize to 256x256 (required by the model) and scale [0,1]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        image = np.asarray(np.float32(image / 255))
        # If alpha channel is present, remove it
        if image.shape[-1] > 3:
            image = image[:, :, :3]
        # Reshape as the model expects BCHW
        image = np.reshape(image, (1, 256, 256, 3))
        return image
    
    def _apply_threshold(self, image, threshold=255):
        mask = cv2.inRange(image, (threshold, threshold, threshold), (255, 255, 255))
        mask = np.stack([mask]*3, axis=-1)
        filtered_image = np.where(mask, 255, image).astype(np.uint8)
        return filtered_image

    def _detect_and_correct_skew(self, roi):
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Increase edge contrast slightly if needed
        # Tune alpha/beta later # todo
        adjusted = cv2.convertScaleAbs(gray_roi, alpha=1.5, beta=15)

        # Canny edge detection
        # tweak thresholds as needed # todo
        edges = cv2.Canny(adjusted, 50, 200)

        # Use HoughLinesP to detect line segments
        # - Adjust minLineLength to capture diagonal text lines
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi/180, 
            threshold=30,       # Lower if your lines are short or faint, Needs tuning # todo
            minLineLength=roi.shape[1] // 4,  # shorter min lines can catch diagonal text lines
            maxLineGap=20
        )

        if lines is None or len(lines) == 0:
            # No lines to analyze; return original
            return roi

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle_deg = np.degrees(np.arctan2(dy, dx))
            angles.append(angle_deg)
        
        if not angles:
            # No valid angles found
            return roi

        # Median angle is often more robust than mean if outliers exist
        median_angle = np.median(angles)

        # Adjust if angle is near -180 or +180
        if median_angle < -90:
            median_angle += 180
        elif median_angle > 90:
            median_angle -= 180

        # If the detected angle is very small, skip rotation
        # (like 1-2 degrees might be negligible, but adjust if you want finer corrections)
        if abs(median_angle) < 1.0:
            return roi

        # Rotate around ROI center
        h, w = roi.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def _get_roi(self, output_data, threshold=255):
        """
        Modified to auto-detect MRZ region skew and correct it before Tesseract OCR.
        """
        if self.tesseract_path != "":
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        image = self.image.copy()
            
        # From the segmentation mask, keep pixels > 0.35
        output_data = (output_data[0, :, :, 0] > 0.35).astype(np.uint8) * 255
            
        # Resize the mask to original image size
        altered_image = cv2.resize(output_data, (image.shape[1], image.shape[0]))

        # Morphological erode to remove small noise
        kernel = np.ones((5, 5), dtype=np.uint8)
        altered_image = cv2.erode(altered_image, kernel, iterations=3)

        contours, _ = cv2.findContours(
            altered_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            return ""

        # Pick the largest contour
        c_area = np.array([cv2.contourArea(c) for c in contours])
        x, y, w, h = cv2.boundingRect(contours[np.argmax(c_area)])
        roi_arr = image[y : y + h, x : x + w].copy()
            
        # Filter out high intensity from ROI
        roi_arr = self._apply_threshold(roi_arr, threshold)

        roi_arr = self._detect_and_correct_skew(roi_arr)

        # Finally, run Tesseract with the MRZ language
        raw_text = pytesseract.image_to_string(roi_arr, lang="mrz")
        custom_config = r'--oem 3 --psm 6'
        confidence_data = pytesseract.image_to_data(roi_arr, lang="mrz", output_type=pytesseract.Output.DICT, config=custom_config)
        return raw_text, confidence_data
    
    def _get_raw_mrz(self, threshold=255):
        """
        Runs the net on the preprocessed image and returns the cleansed MRZ text.
        """
        image_array = self._process_image(threshold)
        
        with open("image_array.pkl", "wb") as f:
            pickle.dump(image_array, f)
            
        self.net.setInput(image_array)
        output_data = self.net.forward()
        
        raw_roi, confidence_data = self._get_roi(output_data, threshold)
        return self._cleanse_roi(raw_roi), confidence_data
    
    # def get_mrz_with_threshold(self, image, raw=False, threshold=255):
    #     if not self._is_valid(image):
    #         return {"status": "FAILURE", "message": "Invalid input image"}
        
    #     if not isinstance(threshold, int) or threshold < 0 or threshold > 255:
    #         threshold = 255
    #         print("Invalid threshold value. Setting threshold to 255")
            
    #     self._load_image(image)
    #     mrz_text, confidence_data = self._get_raw_mrz(threshold=threshold)

    #     # If no result found, fallback to 90° increments
    #     if not mrz_text:
    #         for i in range(1, 4):
    #             self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
    #             mrz_text, confidence_data = self._get_raw_mrz(threshold=threshold)
    #             if mrz_text:
    #                 break

    #     return mrz_text if raw else self._parse_mrz(mrz_text)
    
    def get_mrz(self, image, raw=False):
        if not self._is_valid(image):
            return {"status": "FAILURE", "message": "Invalid input image"}
        
        self._load_image(image)
        thresholds = self._thresholds  # [20, 90, 210]
        
        mrz_text = ""
        parsed_mrz = {"status": "FAILURE", "message": "No MRZ detected"}
        
        # Attempt multiple thresholds
        for rotation in range(4):
            for threshold in thresholds:
                mrz_text, confidence_data = self._get_raw_mrz(threshold=threshold)
                if len(mrz_text) > 20:
                    break
            if len(mrz_text) > 20:
                break
            # Try next rotation if no success
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE * rotation) 
            

        if len(mrz_text) <= 20:
            mrz_text = ""
            self.confidence_data = confidence_data
            parsed_mrz = {"status": "FAILURE", "message": "No MRZ detected"}
        else:
            correct_len = None
            correct_len_parsed = None
            for threshold in thresholds:
                mrz_text, confidence_data = self._get_raw_mrz(threshold=threshold)
                parsed_mrz = self._parse_mrz(mrz_text)
                if parsed_mrz["status"] == "SUCCESS":
                    self._proper_threshold = threshold
                    self.confidence_data = confidence_data
                    break
                if len(mrz_text) == 89:
                    correct_len = mrz_text
                    correct_len_parsed = parsed_mrz
                    self._proper_threshold = threshold
                    self.confidence_data = confidence_data

            if parsed_mrz["status"] != "SUCCESS":
                if correct_len is not None:
                    mrz_text = correct_len
                    parsed_mrz = correct_len_parsed
                    self.confidence_data = confidence_data
        
        return (mrz_text, parsed_mrz) if raw else parsed_mrz

    def _load_image(self, image_input):
        if isinstance(image_input, str):
            if os.path.isfile(image_input):
                image = cv2.imread(image_input, cv2.IMREAD_COLOR)
            else:
                try:
                    image_data = base64.b64decode(image_input)
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except binascii.Error:
                    raise ValueError("Invalid base64 image data")
        elif isinstance(image_input, np.ndarray):
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
            return {"status": "FAILURE", "message": "Invalid MRZ format (invalid number of lines)"}

        mrz_code_dict = {}
        if len(mrz_lines) == 2:
            mrz_code_dict["mrz_type"] = "TD2" if len(mrz_lines[0]) == 36 else "TD3"
            mrz_code_dict["document_type"] = mrz_lines[0][:2].strip("<")
            mrz_code_dict["country_code"] = mrz_lines[0][2:5].replace("<", "")
            if mrz_code_dict["country_code"] == "D":
                mrz_code_dict["country_code"] = "DEU"
            if not mrz_code_dict["country_code"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ country_code format"}

            names = mrz_lines[0][5:].split("<<")
            mrz_code_dict["surname"] = names[0].replace("<", " ")
            mrz_code_dict["given_name"] = names[1].replace("<", " ")

            mrz_code_dict["document_number"] = mrz_lines[1][:9].replace("<", "")
            if self._get_check_digit(mrz_code_dict["document_number"]) != mrz_lines[1][9]:
                return {"status": "FAILURE", "message": "document number checksum is not matching"}

            mrz_code_dict["nationality"] = mrz_lines[1][10:13].replace("<", "")
            if mrz_code_dict["nationality"] == "D":
                mrz_code_dict["nationality"] = "DEU"
            if not mrz_code_dict["nationality"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ nationality format"}

            mrz_code_dict["date_of_birth"] = mrz_lines[1][13:19]
            if self._get_check_digit(mrz_code_dict["date_of_birth"]) != mrz_lines[1][19]:
                return {"status": "FAILURE", "message": "date of birth checksum is not matching"}
            mrz_code_dict["date_of_birth"] = self._format_date(mrz_code_dict["date_of_birth"])
            mrz_code_dict["sex"] = mrz_lines[1][20]
            mrz_code_dict["date_of_expiry"] = mrz_lines[1][21:27]
            if self._get_check_digit(mrz_code_dict["date_of_expiry"]) != mrz_lines[1][27]:
                return {"status": "FAILURE", "message": "date of expiry checksum is not matching"}
            mrz_code_dict["date_of_expiry"] = self._format_date(mrz_code_dict["date_of_expiry"])
            mrz_code_dict["date_of_birth"] = self._get_date_of_birth(
                mrz_code_dict["date_of_birth"], mrz_code_dict["date_of_expiry"]
            )

            mrz_code_dict["optional_data"] = (
                mrz_lines[1][28:35].strip("<")
                if mrz_code_dict["mrz_type"] == "TD2"
                else mrz_lines[1][28:42].strip("<")
            )
            if mrz_lines[1][-1] != self._get_final_check_digit(
                mrz_lines[1], mrz_code_dict["mrz_type"]
            ):
                return {"status": "FAILURE", "message": "final checksum is not matching"}

        else:
            # TD1 format
            mrz_code_dict["mrz_type"] = "TD1"
            mrz_code_dict["document_type"] = mrz_lines[0][:2].strip("<")
            mrz_code_dict["country_code"] = mrz_lines[0][2:5].replace("<", "")
            if mrz_code_dict["country_code"] == "D":
                mrz_code_dict["country_code"] = "DEU"
            if not mrz_code_dict["country_code"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ country_code format"}

            mrz_code_dict["document_number"] = mrz_lines[0][5:14]
            if self._get_check_digit(mrz_code_dict["document_number"]) != mrz_lines[0][14]:
                return {"status": "FAILURE", "message": "document number checksum is not matching"}
            mrz_code_dict["optional_data_1"] = mrz_lines[0][15:].strip("<")

            mrz_code_dict["date_of_birth"] = mrz_lines[1][:6]
            if self._get_check_digit(mrz_code_dict["date_of_birth"]) != mrz_lines[1][6]:
                return {"status": "FAILURE", "message": "date of birth checksum is not matching"}
            mrz_code_dict["date_of_birth"] = self._format_date(mrz_code_dict["date_of_birth"])
            mrz_code_dict["sex"] = mrz_lines[1][7]
            mrz_code_dict["date_of_expiry"] = mrz_lines[1][8:14]
            if self._get_check_digit(mrz_code_dict["date_of_expiry"]) != mrz_lines[1][14]:
                return {"status": "FAILURE", "message": "date of expiry checksum is not matching"}
            mrz_code_dict["date_of_expiry"] = self._format_date(mrz_code_dict["date_of_expiry"])
            mrz_code_dict["date_of_birth"] = self._get_date_of_birth(
                mrz_code_dict["date_of_birth"], mrz_code_dict["date_of_expiry"]
            )

            mrz_code_dict["nationality"] = mrz_lines[1][15:18].replace("<", "")
            if mrz_code_dict["nationality"] == "D":
                mrz_code_dict["nationality"] = "DEU"
            if not mrz_code_dict["nationality"].isalpha():
                return {"status": "FAILURE", "message": "Invalid MRZ nationality format"}

            mrz_code_dict["optional_data_2"] = mrz_lines[0][18:29].strip("<")
            if mrz_lines[1][-1] != self._get_final_check_digit(mrz_lines, mrz_code_dict["mrz_type"]):
                return {"status": "FAILURE", "message": "final checksum is not matching"}

            names = mrz_lines[2].split("<<")
            mrz_code_dict["surname"] = names[0].replace("<", " ")
            mrz_code_dict["given_name"] = names[1].replace("<", " ")

        mrz_code_dict["status"] = "SUCCESS"
        return mrz_code_dict


