FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    aria2 \
    python3-pip \
    python3-numpy

RUN aria2c -x8 http://crcv.ucf.edu/data/Selfie/Selfie-dataset.tar.gz
RUN mkdir /work/
RUN tar zxvf Selfie-dataset.tar.gz -C /work/

WORKDIR /work/
RUN aria2c -x8 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
RUN bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

COPY requirements.txt ./
RUN apt-get install -y cmake
RUN pip3 install --requirement requirements.txt

RUN apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

COPY labels_selfie.csv ./
COPY labels_test.csv ./
COPY expression_classifier-0000.params ./
COPY expression_classifier-symbol.json ./
COPY ./*.py ./
