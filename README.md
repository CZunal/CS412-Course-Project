# CS412 Course Project

## Overview

This project focuses on enhancing the performance of a natural language processing (NLP) model for analyzing and predicting scores based on prompts. The primary goal is to explore various techniques related to text preprocessing, feature engineering, model tuning, and experimenting with different models.

## Contributors
- Yağız Toprak Işık 29174

## Table of Contents
- [GradientBoosting](#adaboost)
  - [Model Training](#model-training)
  - [Solution Motivation](#solution-motivation)
  - [Results](#results)
  - [Codes](#codes)

## Model Training:

- The provided code utilizes the XGBoost model for predicting student grades based on prompts. The XGBoost model is configured with n_estimators = 100 and learning_rate = 0.1.
- The model is trained using a subset of the data, specifically X_train and y_train, and subsequently validated on the test set (X_test and y_test).

## Solution Motivation:

- XGBoost is a robust ensemble learning method that combines the predictive power of multiple weak learners, often decision trees, to create a highly accurate and robust predictive model. 
- The key concept behind XGBoost is to sequentially train weak models, focusing on correcting the errors made by the preceding ones. By assigning more weight to misclassified observations in each iteration, XGBoost iteratively enhances its predictive capabilities, contributing to improved overall model performance.

- The model's configuration, with 100 estimators and a learning rate of 0.1, reflects a balance between model complexity and generalization. The use of cross-validation provides a robust assessment of the model's performance across various subsets of the combined training and test data, offering insights into its ability to generalize to unseen data.

- The iterative and boosting nature of XGBoost makes it well-suited for capturing complex relationships within the data while mitigating overfitting. The provided code not only trains the model but also evaluates its performance through cross-validation, contributing to a more comprehensive understanding of its predictive capabilities.

```plaintext
Cross-Validation R2 Scores: [0.99046837 0.66777219 0.55252456 0.97269405 0.79382785]
Mean R2 Score: 0.7954574044560845

XGBoost - MSE Train: 0.0021904191782652427
XGBoost - MSE TEST: 30.330423434854602
XGBoost - R2 Train: 0.9999865964105298
XGBoost - R2 TEST: 0.7298333288660587
```

- These metrics provide insights into the model's performance on both the training and test sets. The Mean Squared Error (MSE) values indicate the average squared difference between predicted and actual values, while the R-squared (R2) values measure the proportion of variance explained by the model.

- The achieved R2 scores, particularly 0.9999 for the training set and 0.729 for the test set, demonstrate a high level of predictive accuracy. These results suggest that the AdaBoost model effectively captures the relationships within the data, providing reliable predictions.
