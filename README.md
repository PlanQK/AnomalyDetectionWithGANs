# Quantum GAN: Outlier Detection in a Docker Container
This repository contains a semi supervised method to perform outlier detection with a (simulated) quantum computer.
The core algorithm returns an outlier score, but a threshold of the outlier score is optimized with the labelled input.

## Prerequisites
A system running Docker.


## How to build
Run the command
```
docker build -t qanomaly:1.2 .
```
This builds the docker image `qanomaly:1.2` in the current directory. Alternatively, simply run `build.sh`.

## Provide input-data
The input-data to train and test on has either to be supplied in the `input_data` folder by placing one or several csv-files
in the folder and leaving the `data_filepath` empty, or by specifying the path of a specific csv-file by filling said parameter.

The supplied files have to obey a specific csv format with "," as a delimiter, where the first row holds the feature
dimension labels and the last column with the name `Class` represents the true label, signaling a normal sample or an
anomaly. All supplied values have to be in the range [0.0, 1.0]. Example:

|V1|V2|V3|Class|
|---|---|---|---|
|0.8|0.9|0.1|0|
|0.8|0.2|0.9|1|

or shown as plain text:

V1,V2,V3,Class\
0.8,0.9,0.1,0\
0.8,0.2,0.9,1

The partition of all aggregated datafiles into training-/validation-/test-dataset amounts to 80 %, 10 % and 10 %.
During the training phase, only normal samples are used, during validation and testing the ratio of normal to anormal
samples is 50:50.

## How to run
Before the container can be started and the training/prediction is executed, the source-path in `run.sh` needs to be
adapted to the /path/to/repo/model and the /path/to/repo/input_data respectively (where model and input_data are folders
at the top level of the repo). All parameters for the run can be adapted in the file `run_config.txt`.
They are described below. A container and hence a run can be started with the command:
```
bash run.sh
```

The `model` folder will during and at the end of the training process store the weights of the trained model as well as
plots of the training progress in the `model\train_history` folder. After the `prediction` mode of execution, the trained model is tested
against the test-data and the performance of the model is stored in the `model` folder.

### Parameter description
In this section, the run parameters in run_config.txt are described. The `method` variable describes the type of the GAN.
The supported GANs are:
```
    classical: Purely classical AutoEncoder, additional Encoder and Discriminator using Tensorflow
    quantum: quantum-classical hybrid Decoder in the AutoEncoder; all other networks stay classical
```

If using the quantum `method`, the `quantum_circuit_type` parameter determines the design of the quantum circuit used for the Decoder.
The following values are possible:

- standard
- CompleteRotationCircuitIdentity
- CompleteRotationCircuitRandom
- StrongEntanglementIdentity
- StrongEntanglementRandom
- LittleEntanglementIdentity
- LittleEntanglementRandom
- SemiClassicalIdentity
- SemiClassicalRandom

The depth of the quantum-circuit is set by the variable `quantum_depth`. The structure of the respective quantum circuits can be viewed in the file `QuantumCircuits.py`. By subclassing, new
decoder circuits can additionally be implemented.

The container can be run in training mode by specifying the `train_or_predict=train` flag. This will train the model and save the trained parameters in the mounted volume.
In `train_or_predict=predict` mode the container loads the trained parameters that were previously saved to the mounted `model` folder.
The anomaly scores on the prediction set and the resulting performance of the GAN are obtained this way.

Some of the parameters change the network size and therefore
must have identical values in both the training step and the prediction step. It is thus highly recommended,
to keep the environment variables in `run_config.txt` set identical between training and prediction.

There are many further settings that can be adjusted in run_config.txt. The following list contains the settings with
their default value (if the variable is not supplied in run_config.txt) and a short explanation.
```
    data_filepath: optional filepath if a single csv-file shall be used as input-data
    method: architecture of the GAN; see above
    quantum_circuit_type: architecture of the quantum decoder, only used if method=quantum; see above
    quantum_depth: depth of quantum decoder circuit, only used if method=quantum
    shots: number of repeated measurements taken for one parameter sweep of the quantum circuit, only used if method=quantum 
    train_or_predict: either train or predict, modus to run script in
    training_steps: 1000  Number of iterations for the training of the GAN
    latent_dimensions: 10  size of the latent space -> output dim of the encoder/input dim of the decoder (= num qubits in quantum case)
    batch_size: 64  number of samples picked randomly out of training set per training step
    discriminator_iterations: 5  how often does the discriminator update its weights per trainingStep (always one generator weights update per training step)
    gradient_penalty_weight: 10.0  weight factor for the gradient Penalty (Wasserstein Loss specific parameter)
    discriminator_training_rate: 0.02 initial learning rate of discriminator Adam Optimizer
    generator_training_rate: 0.02  initial learning rate of generator Adam Optimizer (generator = autoencoder + additional encoder)
    adv_loss_weight: 1 weight of the adversarial generator loss
    con_loss_weight: 50 weight of the contextual generator loss
    enc_loss_weight: 1 weight of the encoding generator loss
    validation_interval: how often the performance of the GAN is checked against the validation set and the optimal anomaly threshold is determined
    validation_samples: 100 number of validation samples used (e.g. if 100, then 100 normal and 100 unnormal samples will be used)
```
