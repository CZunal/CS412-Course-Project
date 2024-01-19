# CS412 Course Project

## Overview

This project focuses on enhancing the performance of a natural language processing (NLP) model for analyzing and predicting scores based on prompts. The primary goal is to explore various techniques related to text preprocessing, feature engineering, model tuning, and experimenting with different models.

## Contributors
- Fikret Kayra Yilmaz
- Gorkem Topcu

## Table of Contents
- [Linear Regression](#linear-regression)
  - [Model Training](#model-training)
  - [Solution Motivation](#solution-motivation)
  - [Results](#results)

## Linear Regression

### Model Training
- The program utilizes Linear Regression, a simple yet effective regression model, to predict student grades based on the provided features.
- The model is trained on a subset of the data (X_train and y_train) and evaluated on both the training and test sets.
- Cross-validation is applied to obtain more reliable performance metrics, considering different subsets of the training data.

### Solution Motivation
- Linear Regression is a suitable choice for predicting numeric values (like grades) based on input features.
- The approach is useful as it provides a baseline model to understand the relationship between input features and grades.
- Cross-validation ensures that the model's performance is not overly optimistic or pessimistic, providing a more accurate assessment.
- The simplicity of Linear Regression makes it interpretable, helping to identify the most influential features on the predicted grades.

### Results
- Mean Square Error: 7.209050111588801e-28
- R2 Score: 1.0
- Performance Graph:
![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/717978b1-62a7-48e5-964e-79d98e08e7a0)

