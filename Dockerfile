#FROM tensorflow/tensorflow:2.1.0-gpu
#FROM ubuntu:latest
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

RUN mkdir /quantum-anomaly
WORKDIR /quantum-anomaly

ADD requirements.txt /quantum-anomaly/
RUN mkdir /quantum-anomaly/input-data
RUN mkdir /quantum-anomaly/model

RUN apt-get update

# temp for testing
ADD labfile_temp_3.py /quantum-anomaly/
RUN mkdir /quantum-anomaly/model/input-data
ADD model/input-data/predictionSet.csv /quantum-anomaly/model/input-data
ADD model/input-data/trainSet.csv /quantum-anomaly/model/input-data

ADD forest-sdk-2.23.0-linux-deb.run /quantum-anomaly/
RUN yes Y | ./forest-sdk-2.23.0-linux-deb.run

# python3.8-way
RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN python3.8 -m pip install pip
RUN python3.8 -m pip install --upgrade setuptools pip
RUN apt-get install -y git
#RUN python3.8 -m pip install cirq==0.11.0
RUN python3.8 -m pip install -r /quantum-anomaly/requirements.txt
#RUN python3.8 -m pip install -e "git+https://github.com/maximilianraaff/Cirq.git@v0.13.1-dev#egg=cirq_rigetti&subdirectory=cirq-rigetti"
#RUN python3.8 -m pip install -e "git+https://github.com/maximilianraaff/pyquil.git@pyquil_compatibility_mod_3.0.1#egg=pyquil"
# Alternatively
#RUN git clone -b pyquil_compatibility_mod_3.0.1 https://github.com/maximilianraaff/pyquil.git
#RUN pip install -e /quantum-anomaly/pyquil
#RUN git clone -b v0.13.1-dev https://github.com/maximilianraaff/Cirq.git
#RUN pip install -e /quantum-anomaly/Cirq/cirq-rigetti
#
#RUN python3.8 -m pip install PennyLane_qiskit==0.14.0

# python3.6-way
#RUN apt-get install -y python3
#RUN apt-get install -y python3-pip
#RUN apt-get install -y git
#RUN python3 -m pip install --upgrade setuptools pip
#RUN python3 -m pip install cirq==0.11.0
#RUN python3 -m pip install -e "git+https://github.com/maximilianraaff/Cirq.git@v0.13.1-dev#egg=cirq_rigetti&subdirectory=cirq-rigetti"
#RUN python3 -m pip install -r /quantum-anomaly/requirements.txt
#RUN python3 -m pip install -e "git+https://github.com/maximilianraaff/pyquil.git@pyquil_compatibility_mod#egg=pyquil"
#RUN python3 -m pip install PennyLane_qiskit==0.14.0
#
#
ADD GanClassifiers/ /quantum-anomaly/GanClassifiers
ADD run_simple.py /quantum-anomaly/
RUN chmod +x /quantum-anomaly/run_simple.py

ADD saveaccount.py /quantum-anomaly/

#RUN qvm -S > /dev/null 2>&1 &
#RUN quilc -S > /dev/null 2>&1 &


#RUN python3.8 saveaccount.py
#
ADD qiskit_device.py /usr/local/lib/python3.8/dist-packages/pennylane_qiskit/qiskit_device.py
#
#ENTRYPOINT [ "python3", "run_simple.py"]