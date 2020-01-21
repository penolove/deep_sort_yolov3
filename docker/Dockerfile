FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt update
RUN apt install -y libsm6 libxext6 libxrender1 ffmpeg
COPY requirements.txt /requrements.txt
RUN pip install -r /requrements.txt