
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


## Classical training

## Quantum training

## Testing