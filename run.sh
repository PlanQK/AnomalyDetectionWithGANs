docker run \
--env-file run_config.txt \
--mount type=bind,source=/mnt/c/Users/d92510/Desktop/planqk/quantum_classifier_docker/QuantumClassifierDocker/model,target=/quantum-anomaly/model \
--mount type=bind,source=/mnt/c/Users/d92510/Desktop/planqk/quantum_classifier_docker/QuantumClassifierDocker/input_data,target=/quantum-anomaly/input_data \
qanomaly:1.2
