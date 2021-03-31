# Quantum GAN: Outlier Detection in a Docker Cotainer
This repository contains a semi supervised method to perform outlier detecion with a (simulated) quantum computer.
The core algorithm returns an outlier score, but a threshold of the outlier score is optimized with the labelled input.

## Prerequisites
A system running Docker.


## How to build
Run the command
```
docker build -t quantumAnomaly:1.0 .
```
This builds the docker image `quantumAnomaly:1.0`.

## How to run
After the image was created. A container can be started by:
```
docker run
    --mount type=bind,source=/path/to/input/data/,target=/quantum-anomaly/input-data \
    --mount type=bind,source=/path/to/save/model/,target=/quantum-anomaly/model \
    quantumAnomaly:1.0 classical|tfqSimulator|pennylaneSimulator|pennylaneIBMQ train|predict
```
The container can be run in training mode by specifying the `train` flag. This will train the model and save the trained parameters in the mounted volume.
In `predict` mode the container loads the trained parameters, that were previously saved to the mounted `model` folder.

For both runs the following files need to be present in the `input-data` directory:
- `trainSet.csv`
- `predictionSet.csv`

The input data needs to be normalized and it is assumed that the columns containing the labels is called `Class`. A class value of 1 indicates an outlier. Furthermore, if the number of features significantly differ from around 30 some fine-tuning might be required in the meta parameters and neural network structure.

There are many settings that can be adjusted through environment variables. Add them to the docker run command using e.g. `--env trainingSteps=100`. The following lists the settings with their default value and a short explanation.
```
    trainingSteps: 1000  Number of iteration for the training of the GAN
    totalDepth: 4  Depth of the circuit or number of layers in the generator
    thresholdSearchIterations: 10  Number of optimization steps to find the threshold
    thresholdSearchBatchSize: batch size for the threshold optimization
    batchSize: 64  Number of samples per training step
    discriminatorIterations: 5  How often does the discriminator update its weights vs Generator
    gpWeight: 10  Weight factor for the gradient Penalty (Wasserstein Loss specific parameter)
    latentVariableOptimizationIterations: 30  Number of optimization iterations to obtain the latent variables
```