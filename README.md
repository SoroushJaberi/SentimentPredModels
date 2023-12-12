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


# Model Explanations and Analysis

Here we describe the machine learning models used in our Sentiment Analysis Suite, their mechanics, benefits, and limitations.

## LinearSVC (Linear Support Vector Classification)
**Explained**: LinearSVC implements support vector classification for linear kernels. It relies on liblinear, unlike SVC which uses libsvm, and focuses on maximizing the margin between the nearest points of each class—the support vectors.

**Benefits**:
- Performs well in high-dimensional spaces, especially when dimensions outnumber samples.
- Flexible in penalities and loss function choices, suitable for sensitive applications.
- Ideal for hard-margin classification in linearly separable data.

**Cons**:
- Recommended for smaller datasets because of long training times.
- Sensitive to anomalies and noisy data.

## Stochastic Gradient Descent (SGD)
**Explained**: SGD is effective at fitting linear classifiers and regressors under convex loss functions. It estimates the loss gradient for each sample and updates the model accordingly, achieving swift convergence.

**Benefits**:
- Ideal for large-scale and online learning tasks due to its efficiency and ease of implementation.
- Highly flexible, can be extended to deep learning and other parametric models.

**Cons**:
- Requires tuning of hyperparameters such as iterations and regularization.
- Sensitive to feature scaling.
- Less stable and can be affected by noisy data compared to batch optimization techniques.

## Naive Bayes
**Explained**: Based on the naive conditional independence assumption, naive Bayes methods apply Bayes’ theorem to build probabilistic classifiers.

**Benefits**:
- Requires a small amount of training data if the independence assumption holds.
- Works better with categorical inputs.
- Extremely fast in making predictions.

**Cons**:
- The independence assumption is often unrealistic, limiting accuracy.
- Tends to be outperformed by more complex models.
- Struggles with zero-frequency values, cannot handle unseen categories in the test data.

## RandomForestClassifier
**Explained**: RandomForestClassifier employs an ensemble of decision trees, averaging their predictions. Introducing randomness during tree growth helps to diversify the models and reduce overfitting.

**Benefits**:
- Handles various feature types without needing scaling.
- Predictive power is generally high, with good generalization.
- Gives a good indicator of feature importance.

**Cons**:
- Can overfit with noisy datasets.
- Slower predictions due to aggregating multiple trees.
- Complex, requiring more time to train compared to simpler models.

## General Comparative Analysis
Each model has its own set of strengths and weaknesses that impact its performance depending on the task and data specifics.
- Linear SVC excels with high-dimensional data, like text classification, where overfitting isn't a major concern.
- SGD is scalable for linear learning where speed and model stability need balancing.
- Naive Bayes is best for quick baseline models and small datasets with roughly met independence assumptions.
- RandomForestClassifier often shows strong performance and is particularly effective when interpretability is important.

Ultimately, empirical testing, cross-validation, and considering trade-offs between performance, scalability, and interpretability should guide model selection.
