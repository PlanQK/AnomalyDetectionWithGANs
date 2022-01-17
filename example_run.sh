#sh ./build.sh
docker run --gpus=0 \
    --env ibmqx_token="notused"\
    --env trainingSteps=10 --env batchSize=4 --env latentDim=5 --env adamTrainingRate=0.08 \
    --env discriminatorIterations=2 --env latentVariableOptimizationIterations=300 \
    --env backend="ibmqx2" \
    --mount type=bind,source=`pwd`/model/,target=/quantum-anomaly/model \
    -it qanomalyexample_run.sh:1.0 \
    pennylaneIBMQ train
#    --env latentVariableOptimizer=TF \
    