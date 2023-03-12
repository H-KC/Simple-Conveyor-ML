# Simple-Conveyor-ML


## data_creation.py
Creates datasets and saves them to "train" and "test" folders. Each folder must contain at least 2 data files.

## model_preprocessing.py
Performs data preprocessing, for example, using sklearn.preprocessing.StandardScaler.

## model_preparation.py
Creates and trains a machine learning model on the built data from the “train” folder. Saves the trained model to a file.

## model_testing.py
Checks the machine learning model on the built data from the “test” folder. Displays model quality metrics.

## pipeline.sh
Runs all python scripts in sequence.
