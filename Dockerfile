#FROM tensorflow/tensorflow:2.1.0-gpu
FROM ubuntu:latest

RUN mkdir /quantum-anomaly
WORKDIR /quantum-anomaly

ADD requirements.txt /quantum-anomaly/
RUN mkdir /quantum-anomaly/input-data
RUN mkdir /quantum-anomaly/model

RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --upgrade setuptools pip; python3 -m pip install -r /quantum-anomaly/requirements.txt
ADD GanClassifiers/ /quantum-anomaly/GanClassifiers
ADD run_simple.py /quantum-anomaly/


ENTRYPOINT [ "python3", "run_simple.py"]