
# @author FK
# this script is intended to get run on an AWS EC2 instance
# it setups all packages needed for the fake news anomaly detection
# !! the directory QuantumClassifierDocker/ is needed for the script to execute completely
sudo apt update

sudo apt install python3-dev python3-pip python3-venv

python3 -m venv planqk

source planqk/bin/activate

cd QuantumClassifierDocker/
python3 -m pip install -r requirements.txt
python3 -m pip install gensim nltk