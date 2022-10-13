#!/bin/bash
echo "starting ui"
cd /QuantumClassifierDocker/
nohup streamlit run ui.py --server.port 80 &