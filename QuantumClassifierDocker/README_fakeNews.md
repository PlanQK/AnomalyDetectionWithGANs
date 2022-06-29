
# Fake news GANanomaly detection
This project is an extension to the GANomaly fraud detection project.
It extends the existing project to work on news as input data.

Both the classical as well as the quantum version have been changed to work with fake news. 

## Input data
Since the project is anomaly detection, an appropriate data set has to be prepared.

The **Liar dataset** has been created as a benchmark dataset: https://arxiv.org/abs/1705.00648
The dataset can be downloaded as a zip directory via the following link: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
There are 4529 true news and 2511 false news in the data set. The news contain an average o f20 (true) and 19 (false) words per news
The false news are manually checked and labeled by journalists.

## Input preprocessing
Naturally, the news can not be processed as input in text form. Therfore, they will be encoded with *paragraph embeddings*.
These are a specialized type of *word embeddings* and encode paragraphs, meaning sentences or phrases, in numeric vectors.
This is implemented with the framework `gensim`, which based their algorithm on this article http://arxiv.org/abs/1405.4053v2. For this, the following command can be run:
```
python3 doc2vec_FK.py
```
The following variables can be set in `doc2vec_FK.py`:
```
    dimensions: Specify the amount of dimensions of the paragraph embeddings.
    dm_or_dbow: Specify the method with which the paragraph embeddings will be created. "dm" or "dbow". Check http://arxiv.org/abs/1405.4053v2 for a detailed overview.
```
The result of this script are paragraph embeddings of the **Liar dataset** in a json file.


## Classical approach
The classical approach is nearly the same as it is for the fraud detection project. The GAN model consists of a generator, which tries to create a fake sample out of a training sample to trick the discriminator, and a discriminator, which tries to distinguish between the real training sample and the fakely generated one.
The discriminator is a simple deep neural network containing 8 layers with one output node for a binary decision. However, the layer depth can be adjusted.
The generator consists of one auto-encoder, one decoder and one encoder. The auto-encoder compresses the input into a latent space, which the decoder tries to decompress. The encoder, having the same architecture as the auto-encoder, compresses the decompressed sample once again. All three instances are deep neural networks with a layer depth of 8. As in the discriminator, the depth can be adjusted.


## Quantum approach
Additionally to the classical approach, a semi-quantum version of the GAN model was implemented. The decoder (as part of the generator) had a quantum component in the fraud detection project. In there, the latent space was processed through quantum circuits and then upscaled to the input dimensionality through a classical neural network.
For the fake news GANomaly detection, this results in some problems. The input dimensionality is much higher for fake news with 150 paragraph embedding dimensions. Therefore, the latent space should be not lower than around 30 to 50 to have the same ratio as in the fraud detection project. However, the simulation of quantum cirucits is not possible on most computers for more than 20 qubits simultaneously.

With this, a new approach had been implemented to allow the usage of a higher-dimensional latent space.
1. Simple and basic version
One simple solution is to sequentially compute the circuits. Take the example of 150 input dimensions and 50 latent dimensions. For this, 15 circuits with 10 qubits each are created and computed sequentially. Each qubit is measured once which results in 150 decompressed dimensions. Each qubit gets one input, with every input dimension being used thrice.
2. More complex multi-input qubits
The dimensions of the paragraph embeddings correlate with each other. Therefore, it can be assumed, that the latent dimensions do too. To enhance this correlation, multiple dimensions can serve as the input for one qubit. There still have to be 15 circuits with 10 qubits each to generate 150 measurement. As an example, on the first qubit in the first circuit, the first and second dimension of the latent space are used as input. Additionally, there are 4 trainable parameters on each qubit.

The two solutions for creating a solely quantum-based decoder are put on separate branches.


## Branches
There are two distinct branches regarding the fake news GANomaly detection. The classical approach does not differ between the branches.
1. fake_news
This branch is the 'main' fake news branch. It holds the most recent version of the algorithms. On there, the **multi-input qubit** approach has been implemented, which means that 2 input dimensions of the paragraph embeddings are used on each qubit. Therefore, 2^2=4 trainable parameters are also set on each qubit.
2. fake_news_basicV
This branch is an offspring of the fake_news branch. It contains a more basic approach for the quantum decoder. This approach uses the structure-wise same quantum circuits 15 times. Also, only one input dimensions of the paragraph embeddings is used on each qubit, together with one trainable parameter.


## Testing
How to use and apply the theoretical models explained above?
The script `run_gan_classifier.py`, which is the main executed script in the fraud detection project, can also be used in this project. Upon execution, one training and one prediction are done sequentially.

To not manually, but more statistically test different fake news algorithms or input data, a test script `test_gan_classifier.py` has been written. The process of training and prediction, as done by the `run_gan_classifier.py` script, can be executed multiple times in there. The results of the runs are saved in a json file and can then be graphically displayed with the script `plot_results.py`.
In there, both the MCC as well as the threshold of every run are displayed in scatter plots and boxplots. In addition, it is possible to also plot all different losses of the GAN model.


## Executing on AWS
Since the quantum approach takes a lot of computing time, it has been outsourced to AWS instances.
The dependencies of the project can be python version dependent. The default instance configuration for the OS image is: *Ubuntu Server 20.04 LTS (HVM)*
Regarding the instance type, there are several possiblities. For computing with quantum components, the **c5.4xlarge** instance was used.
The script `aws_setup.sh` can be used to setup the basis needed for executing the scripts on AWS.