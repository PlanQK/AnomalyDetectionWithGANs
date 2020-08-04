#FROM tensorflow/tensorflow:2.1.0-gpu
FROM ubuntu:18.04

RUN mkdir /quantum-anomaly
WORKDIR /quantum-anomaly

ADD code/ /quantum-anomaly/code
ADD requirements.txt /quantum-anomaly/
ADD run_simple.py /quantum-anomaly/
RUN mkdir /quantum-anomaly/input-data
RUN mkdir /quantum-anomaly/model

RUN apt-get update && apt-get install -y python3-pip git
RUN python3 -m pip install --upgrade setuptools pip; python3 -m pip install -r /quantum-anomaly/requirements.txt --use-feature=2020-resolver


ENTRYPOINT [ "python3", "run_simple.py"]