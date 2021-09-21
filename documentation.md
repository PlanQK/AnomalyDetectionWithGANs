### Folder QuantumClassifierDocker contains all ressources for Fraud Detection on the Xente-dataset 'training.csv'

# First step: TRANSFORMING RAW DATA

Raw data is stored in 'training.csv'. To transform it into processable dataset execute:

'python3 transform_data.py training.csv'

Now, processable dataset 'training.csv' is in 'transformed'-Folder.

# Second step: PREPROCESSING OF PROCESSABLE DATA

*NOTE: Before every new preprocessing execute from terminal: 'make cleaning', so that the data from previous usage is deleted*

For the second step, use the 'Makefile'. In the Makefile, change the 'INPUT_FILE' parameter to

'YourPath/QuantumClassifierDocker/transformed/training.csv'

Now set the 'REPETITIONS' parameter to the number of times you want to run the program.

From Terminal execute command

'make -jN preprocess'

where N is the number of jobs you want to run in parallel. 
Now, every model-folder contains a training set 'trainSet.csv' and a predicition set 'predicitionSet.csv'. 

By default, the trainset contains 90 per cent, the prediction set 10 per cent of the data. You can alter this
distribution by setting a different parameter for the 'test_size' parameter in the 'sklearn.model_selection.train_test_split()' function used in the preprocessData.py script.

Also, you might want to oversample the prediction set to contain more fraudulent transactions. For that, change 'testing.to_csv' to 'testing_os.to_csv' at the end of the preprocessData.py script.

# Third step: TRAINING OF THE MODEL

You can run the programm in different modi, classical or quantum. To choose the classical option, set the
'MODUS' parameter in the Makefile to 'classical', to run the quantum version implemented with Tensorflow set it
to 'TfqSimulator'.

To start the training of the GAN execute the terminal command

'make -jN training' 

where N is the number of jobs you want to run in parallel.
Now, in every model-folder the parameters optimized during training are stored in the 'checkpoint' subfolder. 

# Fourth step: PREDICTION

From Terminal execute command

'make -jN results'

where N is the number of jobs you want to run in parallel. 
Now, for every input in the 'predictionSet.csv'-file an anomaly score is calculated. The predictions are stored
as 'j.csv' in the 'results' folder, where j is the j-th model-folder (corresponding to the j-th repetition).


# Fifth step: EVALUATION 

First, you'll need to set the 'l'-parameter in the 'metrics_new.py' script to the number of repetitions you ran.
After that, for evaluating the results of the prediction, execute from terminal:

'make metric'

This command will lead to the execution of the 'metrics_new.py' program. For every repetition (that is for 
every model$-Folder), a Folder metric$ is created that contains a .csv-file 'metrics.csv'. 
For every threshold in the threshold-grid, this file contains the False Positive Rate, False Negative Rate, Precision, Recall and F1 Scores. 

Additionally, in the QuantumClassifierDocker folder a .csv-file 'optimal_metrics.csv' is created containing the maximal F1-Score and corresponding threshold for every repetition respectively.


# Sixth step: COMPARISON WITH OTHER UNSUPERVISED METHODS

To compare the results of the evaluation of AnoWGan with that of other unsupervised ML methods, the 
'IsolationForest' and 'OneClassSVM' method can be evaluated. 

*CAVE: Preprocessing must be done by excecuting 'make preprocess' if not already done for previous execution of the main program*

To choose from one of the two methods, set the 'OTHER_METHOD' parameter in the Makefile. For the number of times the method should be execeuted, set the 'REPETITIONS' parameter.

To get the evaluation metrics for this method, execute from terminal:

'make other_metric'

Now, for every repetition there is a 'other_metric$(no. repitition)' folder with the results of the 
evaluation of the respective method in the '$(method)/metrics.csv' file.

Additionally, a 'f1_mean_$(method).csv' file is created in QuantumClassifierDocker which stores the 
mean F1-Value over all repetitions for the respective method.


## ADDITONAL FUNCTIONALITY

# Plotting

Using the 'plot_helper.py' script, you can plot the results of the evaluation quickly. You have to specifiy
input and output paths by setting the variables "INPUT_FILE" and "OUTPUT_FILE" in the script.

# Run on virtual machine

(Shell-script runOnVM.sh)

--> Docker-Commands 
--> License-Text