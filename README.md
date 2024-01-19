# CS412 Course Project

## Overview

This project focuses on enhancing the performance of a natural language processing (NLP) model for analyzing and predicting scores based on prompts. The primary goal is to explore various techniques related to text preprocessing, feature engineering, model tuning, and experimenting with different models.

## Contributors
- Can Zunal 29453

## Table of Contents
- [Neural Network](#neural-network)
  - [Model Training](#model-training)
  - [Solution Motivation](#solution-motivation)
  - [Results](#results)
  - [Codes](#codes)

## Neural Network

### Model Training
- The code employs a Neural Network architecture for predicting student grades based on prompts.
  - The architecture includes a Sequential model with dense layers and dropout for regularization.
  - Input features are normalized using StandardScaler to enhance convergence during training.
  - The model is trained on a subset of the data (X_train and y_train) and validated on X_test and y_test.
  - Training details include the optimizer, loss function, and evaluation metric.

### Solution Motivation
- The choice of a Neural Network is motivated by the need to capture complex, non-linear relationships between input features and grades.
- Neural networks excel in handling intricate feature interactions, providing flexibility in learning patterns.
- Compared to Linear Regression, neural networks can discover hierarchical representations, potentially improving predictive performance.

### Results
- The Neural Network model was trained and evaluated, producing the following results:

```plaintext
MSE Train: 8.6942377150357
MSE TEST: 8.94185264789965
R2 Train: 0.9467983141105091
R2 TEST: 0.9203509120523148
```

- These metrics provide insights into the model's performance on both the training and test sets. The Mean Squared Error (MSE) values indicate the average squared difference between predicted and actual values, while the R-squared (R2) values measure the proportion of variance explained by the model.

- The achieved R2 scores, particularly 0.946 for the training set and 0.920 for the test set, demonstrate a high level of predictive accuracy. These results suggest that the neural network effectively captures the relationships within the data, providing reliable predictions.

![MSE Plot](nn_mse_plot.png)
### Codes

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Normalize/Scale input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a more complex model architecture with dropout and regularization
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))  # Adding dropout for regularization
model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))  # Adding another dropout layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Fit the model without early stopping and model checkpointing
history = model.fit(X_train_scaled, y_train, epochs=10000, batch_size=32, validation_data=(X_test_scaled, y_test))

# Plotting the training and validation MSE over epochs
plt.figure(figsize=(12, 6))

# Plot training & validation MSE values
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])

plt.title('Model MSE over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Prediction
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculation of Mean Squared Error (MSE)
print("MSE Train:", mean_squared_error(y_train, y_train_pred))
print("MSE TEST:", mean_squared_error(y_test, y_test_pred))

print("R2 Train:", r2_score(y_train, y_train_pred))
print("R2 TEST:", r2_score(y_test, y_test_pred))
```
