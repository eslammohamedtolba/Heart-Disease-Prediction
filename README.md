# Heart-Disease-Prediction
This repository contains a Python-based machine learning model to predict the presence of heart disease based on a dataset. 
The model achieves an accuracy of 85% on both the training and testing data.

## Prerequisites
Before using this model, make sure you have the following prerequisites:
1- Python Environment: You should have Python installed on your system. You can download it from python.org.
2- Required Libraries: Install the necessary libraries by running the following command:pip install pandas matplotlib seaborn scikit-learn numpy
3- Dataset: The dataset is included in the repository as heart_disease_data.csv. You don't need to download it separately.

## Overview
The provided Python script, `heart_disease_prediction.py`, trains a logistic regression model to predict the presence of heart disease. The code performs the following steps:
1. **Data Loading**: The heart disease dataset is loaded from the included CSV file.
2. **Data Exploration**: Various aspects of the dataset are explored, including shape, statistical information, and correlations between features. This exploratory analysis helps gain insights into the data.
3. **Data Preprocessing**: The data is split into input (X) and label (Y) variables. It is further divided into training and testing sets using the `train_test_split` function. The model is trained on the training set.
4. **Model Training**: A logistic regression model is created and trained on the training data using `LogisticRegression()` from scikit-learn.
5. **Model Evaluation**: The model's accuracy is evaluated on both the training and testing sets using the `accuracy_score` function.
6. **Making Predictions**: The code includes a section to make predictions on new data. It allows you to input a set of features, and the model will predict whether the individual has heart disease or not.

## Contributions
Contributions and suggestions from the community are highly encouraged. 
If you have any ideas for improvements, bug fixes, or new features, please feel free to contribute.
