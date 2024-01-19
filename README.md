# CS412 Course Project

## Overview

This project focuses on enhancing the performance of a natural language processing (NLP) model for analyzing and predicting scores based on prompts. The primary goal is to explore various techniques related to text preprocessing, feature engineering, model tuning, and experimenting with different models.

## Contributors
- Gorkem Topcu

## Table of Contents
- [Linear Regression](#linear-regression)
  - [Model Training](#model-training)
  - [Solution Motivation](#solution-motivation)
  - [Results](#results)
  - [Code](#code)

## Support Vector Regressor

### Model Training
- The program uses Support Vector Regression (SVR) with a linear kernel (kernel='linear') and a regularization parameter (C=1.0).
- The SVR model is trained on the provided training data (X_train, y_train) using the fit method.
- The trained SVR model is used to make predictions on both the training set (y_train_pred) and the test set (y_test_pred).
- Mean Squared Error (MSE) and R-squared (R2) are calculated to evaluate the performance of the model on both the training and test sets.
- k-fold cross-validation (cv=5) is applied to the training set using cross_val_predict. It provides cross-validated predictions (y_cv_pred_svr), which are used to evaluate the performance of the model during training.
- MSE and R2 scores are computed for the cross-validated predictions to assess the model's generalization performance.

### Solution Motivation
- The program utilizes Support Vector Regressor (SVR), specifically with a linear kernel and C=1.0, for predicting student grades based on input features.
- The SVR model is trained on a subset of the data (X_train and y_train) and evaluated on both the training and test sets.
- Cross-validation is applied to obtain more reliable performance metrics, considering different subsets of the training data.

### Results
- The Support Vector Regressor model was trained and evaluated, producing the following results:
```plaintext
Mean Squared Error(Train): 0.005130020227711553
Mean Squared Error(Test): 0.0047693131526653386
R2 Score(Train): 0.99995751759085
R2 Score(Test): 0.99995751759085

Cross-Validated Performance (SVR):
Mean Squared Error (CV): 0.15945861545308992
R2 Score (CV): 0.999024242555844
```

- The Mean Squared Error (MSE) values measure the average squared difference between predicted and actual grades. Lower MSE values indicate better model performance. The R2 scores represent the proportion of variance explained by the model, with values closer to 1 indicating a high level of predictive accuracy.

- The SVR model exhibits exceptional performance on both the training and test sets, with very low MSE and high R2 scores. This suggests that the SVR model effectively captures the underlying patterns in the data, providing accurate predictions. The cross-validated results further validate the model's robustness, indicating its ability to generalize well to different subsets of the training data.

![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/b2d043f9-13a6-4639-a52d-b82c28469553)

![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/fba93847-bc32-407b-81d9-05f6df8b3cb0)

### Code
```python
# Create an SVR model
svr_model = SVR(kernel='linear', C=1.0)

# Fit the SVR model on the training data
svr_model.fit(X_train, y_train)

# Make predictions on the test set
y_train_pred = svr_model.predict(X_train)
y_test_pred = svr_model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Apply k-fold cross-validation with SVR
y_cv_pred_svr = cross_val_predict(svr_model, X_train, y_train, cv=5)

# Evaluate performance using cross-validated predictions
mse_cv_svr = mean_squared_error(y_train, y_cv_pred_svr)
r2_cv_svr = r2_score(y_train, y_cv_pred_svr)