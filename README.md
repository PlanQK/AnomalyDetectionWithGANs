# Quantum GAN: Outlier Detection in a Docker Container
This repository contains a semi supervised method to perform outlier detection with a (simulated) quantum computer.
The core algorithm returns an outlier score, but a threshold of the outlier score is optimized with the labelled input.

## Prerequisites
A system running Docker.


## How to build
Run the command
```
docker build -t qanomaly:1.1 .
```
This builds the docker image `qanomaly:1.1`.

## How to run
Before the container can be started and the training/prediction is executed, the source-path in run.sh needs to be
adapted to the /path/to/save/your/model. All parameters for the run can be adapted in the file run_config.txt.
They are described below. A container and hence a run can be started with the command:
```
bash run.sh
```

The `model` folder will store all data relevant to a model. This includes the training and prediction data and the trained weights. The weights will be created during the `train` run mode. During the training and predictions steps the following files need to be present in the `model/input-data` directory:
- `trainSet.csv`
- `predictionSet.csv`

The input data needs to be normalized and if a column called `Class` is present it will be removed, because this is an unsupervised method. For a different neural network structure the code in the file `GanClassifiers.py` needs to be adjusted.

### Parameter description
In this section, the run parameters in run_config.txt are described. The `method` variable describes the used python
package to train/predict the model. The supported methods are:
```
    classical: Purely classical generator using Tensorflow
    tfqSimulator: Parametrized circuit with tensorflow quantum
    pennylaneSimulator: Parametrized circuit with Pennylane
    pennylaneIBMQ: Same as pennylaneSimulator, but with the IBM Q backend
```
If `method=tfqSimulator` can also be used together with a rigetti-backend, simulating the quantum-circuit on a QVM.
To this end, the parameter `backend` needs to be set to `backend=rigetti`. The container can be run in training mode by
specifying the `trainOrpredict=train` flag. This will train the model and save the trained parameters in the mounted volume.
In `trainOrpredict=predict` mode the container loads the trained parameters, that were previously saved to the mounted `model` folder.
The anomaly scores on the prediction set are obtained this way. In prediction mode, a new file is created in the
input-data folder with the name `anoScoreResults.csv`. Some of the parameters change the network size and therefore
must have identical values in both the training step and the prediction step. It is thus highly recommended,
to keep these environment variables set identical between training and prediction steps.

There are many further settings that can be adjusted in run_config.txt. The following list contains the settings with
their default value (if the variable is not supplied in run_config.txt) and a short explanation.
```
    trainingSteps: 1000  Number of iteration for the training of the GAN
    latentDim: 10  size of the latent space = num qubits
    totalDepth: 4  Depth of the circuit or number of layers in the generator
    batchSize: 64  Number of samples per training step
    adamTrainingRate: 0.01  Training rate for the Adam optimizer
    discriminatorIterations: 5  How often does the discriminator update its weights vs Generator
    gpWeight: 1.0  Weight factor for the gradient Penalty (Wasserstein Loss specific parameter)
    latentVariableOptimizer: "forest_minimize"  Optimizer to use
    latentVariableOptimizationIterations: 30  Number of optimization iterations to obtain the latent variables
    ibmqx_token: ""  Token to access IBM Quantum experience
    backend: "ibmq_16_melbourne"  If backend="rigetti", a rigetti QVM is going to be used to evaluate the cirq-circuits
```



## Example
A sample command for a training run is therefore:
```
docker run \
    --mount type=bind,source=/path/to/input/model/,target=/quantum-anomaly/model \
    --env latentDim=5 --env trainingSteps=500 \
    qanomaly:1.0 classical train
```

The anomaly scores on the prediction set are then obtained by the following command:
```
docker run \
    --mount type=bind,source=/path/to/input/model/,target=/quantum-anomaly/model \
    --env latentDim=5 --env trainingSteps=500 \
    qanomaly:1.0 classical predict
```
During this step a new file is created in the input-data folder with the name `anoScoreResults.csv`.
Some of the environment variables change the network size and therefore must have identical values in both the training step and the prediction step. I highly encourage to keep these environment variables set between training and prediction steps, even if they might not be needed.
