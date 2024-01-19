# CS412 Course Project

## Overview

This project focuses on enhancing the performance of a natural language processing (NLP) model for analyzing and predicting scores based on prompts. The primary goal is to explore various techniques related to text preprocessing, feature engineering, model tuning, and experimenting with different models.

## Contributors
- Fikret Kayra Yilmaz 29371
- Gorkem Topcu 28862

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
- The Linear Regression model was trained and evaluated, producing the following performance metrics:
```plaintext
Mean Squared Error(Train): 7.032085026046438e-28
Mean Squared Error(Test): 1.211690350419474e-28
R2 Score(Train): 1.0
R2 Score(Test): 1.0

Cross-Validated Performance:
Mean Squared Error (CV): 7.209050111588801e-28
R2 Score (CV): 1.0
```
- The Mean Squared Error (MSE) values, being extremely close to zero, indicate very low prediction errors. In the context of R2 scores, a perfect score of 1.0 suggests that the model precisely predicts the target variable.
- These results imply that the Linear Regression model fits the data exceptionally well, both in the training and test sets. The cross-validated performance reinforces the model's robustness, indicating its ability to generalize effectively to different subsets of the training data.
- It's crucial to note that such perfect scores might raise concerns about overfitting, especially if the dataset is small or the model is overly complex.
  
![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/d1ae4666-25b1-45f0-afb1-a981305b9358)

![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/3fe42047-2963-418c-b1a2-d46f3602cfc3)

### Code
```python
# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the train set
y_train_pred = model.predict(X_train)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
