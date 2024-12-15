FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean

RUN apt-get update && apt-get install -y \
    python3-opencv \
    && apt-get clean

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MRZ trained data to the tessdata directory
COPY tessdata/mrz.traineddata /usr/share/tesseract-ocr/4.00/tessdata/

COPY . /app 
EXPOSE 5001

CMD ["python", "api.py"]

