docker run \
--env-file run_config.txt \
--mount type=bind,source=/mnt/c/Maximilian_Hoeschler/Projekte/04_PlanQK/Repo/QuantumClassifierDocker/model,target=/quantum-anomaly/model \
--mount type=bind,source=/mnt/c/Maximilian_Hoeschler/Projekte/04_PlanQK/Repo/QuantumClassifierDocker/input_data,target=/quantum-anomaly/input_data \
qanomaly:1.1
