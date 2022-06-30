# Define parent image
FROM ubuntu:latest

# Create directories, set working directory
RUN mkdir /quantum-anomaly
RUN mkdir /quantum-anomaly/input_data
RUN mkdir /quantum-anomaly/model
WORKDIR /quantum-anomaly
ADD requirements.txt /quantum-anomaly/

# Install & update required applications
RUN apt-get update
RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN apt-get install -y virtualenv
RUN python3.8 -m pip install pip
RUN python3.8 -m pip install --upgrade setuptools pip

# Download forked python packages and set up virtual python3.8-environment
RUN virtualenv venv -p python3.8
ENV PATH="/quantum-anomaly/venv/bin:$PATH"
RUN python -m pip install --upgrade setuptools pip
RUN python -m pip install -r /quantum-anomaly/requirements.txt

# Execute final tasks and copy relevant files
ADD entrypoint_script.sh /quantum-anomaly/
ADD gan_classifiers/ /quantum-anomaly/gan_classifiers
ADD run_gan_classifier.py /quantum-anomaly/
RUN chmod +x /quantum-anomaly/run_gan_classifier.py

# Define entry-point
ENTRYPOINT ["bash", "entrypoint_script.sh"]