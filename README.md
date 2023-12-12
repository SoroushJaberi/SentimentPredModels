# Model Performance

Here are the performance metrics for each model implemented in the Sentiment Analysis Suite:

## Naive Bayes Model
- **Accuracy**: 0.8571
- **Precision**: 0.9192
- **Recall**: 0.7832
- **F1 Score**: 0.8458

## Linear SVC Model
- **Accuracy**: 0.8610
- **Precision**: 0.8910
- **Recall**: 0.8228
- **F1 Score**: 0.8556

## SGD Model
- **Accuracy**: 0.8617
- **Precision**: 0.9245
- **Recall**: 0.7879
- **F1 Score**: 0.8508

## Random Forest Model
- **Accuracy**: 0.8587
- **Precision**: 0.9020
- **Recall**: 0.8050
- **F1 Score**: 0.8507



# Sentiment Analysis Suite

This repository houses a machine-learning-based sentiment analysis toolkit, designed to classify sentiment of text reviews into positive and negative categories. It leverages several machine learning algorithms including Naive Bayes, Support Vector Machines (SVC), Stochastic Gradient Descent (SGD), and Random Forest classifiers.


## Overview
The code provides a framework to pre-process review datasets, build a sentiment analysis model, and evaluate the performance of the model using various metrics. The project makes use of NLP techniques such as TF-IDF for feature extraction and various classifiers from the popular `scikit-learn` library to perform sentiment analysis on text data.


## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.x installed
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn` libraries installed
