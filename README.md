# Quantum GAN: Outlier Detection in a Docker Container
This repository contains a semi supervised method to perform outlier detection with a (simulated) quantum computer.
The core algorithm returns an outlier score, but a threshold of the outlier score is optimized with the labelled input.

## Prerequisites
A system running Docker.
## How to build
Run the command
```
docker build -t planqk-service .
```
This builds the docker image `planqk-service` in the current directory. Alternatively, simply run `build.sh`.

## Provide input-data
The input-data to train or test the classifier to be supplied in the `input` folder in json format. 
The data values are given as a vector of data points. Each data point consist of a vector of values 
in the range [0.0, 1.0] plus the true label, signaling a normal sample or an anomaly. Example:
```
    "data": {
        "values": [
            [
                0.9370186098551015,
                0.921981567635444,
                0.8985929672365479,
                0.10102346455136567,
                0.09772290503185904,
                0.1100610989159374,
                0.06477057388020797,
                0.0964527986715833,
                0.12739252806545098,
                0.0
            ], ...
            [
                0.919818707941956,
                0.8504884183157868,
                0.8969647505052396,
                0.9529457948863298,
                0.9868972941346208,
                0.9417877009290838,
                0.004120323665909765,
                0.1260498866684419,
                0.04495871230873992,
                0.0
            ],
        ] 
    }
```
In addition to the input-data the json contains information on the parameters used for the run. These are provided within the "params" object of the input-data file.


### Parameter description

A sample of the params section is given in the following.

```
    "params": {
        "method": "classical",
        "train_or_predict": "predict",
        "is_supervised": true,
        "training_steps": 3,
        "quantum_circuit_type": "StandardCircuit",
        "quantum_depth": 3,
        "batch_size": 16,
        "discriminator_iterations": 5,
        "validation_interval": 1,
        "validation_samples": 100,
        "discriminator_training_rate": 0.02,
        "generator_training_rate": 0.02,
        "gradient_penalty_weight": 10.0,
        "shots": 100,
        "latent_dimensions": 6,
        "adv_loss_weight": 1,
        "con_loss_weight": 50,
        "enc_loss_weight": 1,
        "trained_model": {...}
```
We now describe each of the parameters in more detail.

By specifying the `train_or_predict` flag the corresponding process (`train` or `predict`) will be triggered.

If `train_or_predict` is set to `predict` we need to supply the weights and parameters of the trained model. This is done under the keyword `trained_model`, where we can copy and paste the output of the training run. It is important to keep other parameters consistent to the training run, as these might change the dimensionality of the weights. A sample structure of the trained model is given in the following:
```
"params":{
    ...
    "trained_model": {
        "auto_encoder_weights": [[...]],
        "auto_decoder_weights": [[...]],
        "encoder_weights": [[...]],
        "discriminator_weights": [[...]],
        "threshold": 0.0025526642574161293
    }
}
```


The `method` variable describes the type of the GAN.The supported GANs are:
```
    classical: Purely classical AutoEncoder, additional Encoder and Discriminator using Tensorflow
    quantum: quantum-classical hybrid Decoder in the AutoEncoder; all other networks stay classical
```

The `is_supervised` variable can be either `true` or `false`. It describes if the algorithm is run in a supervised setting. This is useful to obtain more detailed performance metrics but requires labeled data as input. The last feature is assumed to be the class/anomaly information. During the prediction step the `threshold` parameter needs to be supplied as well. To keep the correct dimensionality we recommend not to mix supervised training with unsupervised predictions.


If using the quantum `method`, the `quantum_circuit_type` parameter determines the design of the quantum circuit used for the Decoder.
The following values are possible:

```
    StandardCircuit
    CompleteRotationCircuitIdentity
    CompleteRotationCircuitRandom
    StrongEntanglementIdentity
    StrongEntanglementRandom
    LittleEntanglementIdentity
    LittleEntanglementRandom
    SemiClassicalIdentity
   SemiClassicalRandom
```


The depth of the quantum-circuit is set by the variable `quantum_depth`. The structure of the respective quantum circuits can be viewed in the file `QuantumCircuits.py`. By subclassing, new decoder circuits can additionally be implemented.

There are further settings to be determined. Below is a list of the required parameters and a short description for each of them.
```
    train_or_predict: either train or predict
    method: architecture of the GAN; see above
    quantum_circuit_type: architecture of the quantum decoder, only used if method=quantum; see above
    quantum_depth: depth of quantum decoder circuit, only used if method=quantum
    shots: number of repeated measurements taken for one parameter sweep of the quantum circuit, only used if method=quantum 
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
    trained_model: The weights and threshold from the output of the training run. For the training run this keyword is not necessary.
```
Some of the parameters change the network size and therefore must have identical values in both the training step and the prediction step. It is thus highly recommended, to keep the parameter set identical between training and prediction.

## How to run
Before the container can be started and the training/prediction is executed, the path to the input file `path_to_json` in `run.sh` needs to be adapted. There are three input files available: `train.json`, `test_classical.json` and `test_quantum.json` to train and test a classical and quantum version respectively of the classifier. These can be used as model to construct new examples.

A container and hence a run can be started with the command:
```
bash run.sh
```
Once the run is concluded the result is printed in json format. The result of the training contains the weights of the trained classifier (auto_encoder_weights, auto_decoder_weights, encoder_weights and discriminator_weights) and the optimized threshold. On the other hand, the result of the testing contains the confusion matrix computed for the predictions and the corresponding MCC.