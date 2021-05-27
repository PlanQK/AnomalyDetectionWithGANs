#FROM tensorflow/tensorflow:2.1.0-gpu
#FROM ubuntu:latest
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

RUN mkdir /quantum-anomaly
WORKDIR /quantum-anomaly

ADD requirements.txt /quantum-anomaly/
RUN mkdir /quantum-anomaly/input-data
RUN mkdir /quantum-anomaly/model

RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --upgrade setuptools pip; python3 -m pip install -r /quantum-anomaly/requirements.txt

ADD GanClassifiers/ /quantum-anomaly/GanClassifiers
ADD run_simple.py /quantum-anomaly/
RUN chmod +x /quantum-anomaly/run_simple.py

ADD saveaccount.py /quantum-anomaly/
RUN python3 saveaccount.py

ADD qiskit_device.py /usr/local/lib/python3.6/dist-packages/pennylane_qiskit/qiskit_device.py

ENTRYPOINT [ "python3", "run_simple.py"]