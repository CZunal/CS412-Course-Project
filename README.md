# CS412 Course Project

## Overview

This project focuses on enhancing the performance of a natural language processing (NLP) model for analyzing and predicting scores based on prompts. The primary goal is to explore various techniques related to text preprocessing, feature engineering, model tuning, and experimenting with different models.

## Contributors
- Can Zunal 29453
- Gorkem Topcu 28862
- Berk Ay 29026
- Fikret Kayra Yılmaz 29371
- Yagiz Toprak Isik 29174

## Table of Contents
- [Overview of The Repository](#overview-of-the-repository)
  - [Methodology](#methodology)
    - [Neural Network](#neural-network)
    - [Decision Tree Regressor](#decision-tree-regressor)
    - [Linear Regression](#linear-regression)
    - [Ada Boost Regressor](#ada-boost-regressor)
    - [Lasso Regression](#lasso-regression)
    - [Ridge Regression](#ridge-regression)
    - [Support Vector Regressor](#support-vector-regressor)
  - [Results](#results)
    - [Decision Tree Regressor](#decision-tree-regressor-1)
    - [Neural Network](#neural-network-1)
    - [Linear Regression](#linear-regression-1)
    - [Ada Boost Regressor](#ada-boost-regressor-1)
    - [Lasso Regression](#lasso-regression-1)
    - [Ridge Regression](#ridge-regression-1)
    - [Support Vector Regressor](#support-vector-regressor-1)
  - [Team contributions](#team-contributions)

  
### Overview of The Repository

Each team member created model specific branch for evaluating different models performances. For further information you can check other branches readme files and codes.

### Methodology

#### Neural Network
- The choice of a Neural Network is motivated by the need to capture complex, non-linear relationships between input features and grades.
- Neural networks excel in handling intricate feature interactions, providing flexibility in learning patterns.
- Compared to Linear Regression, neural networks can discover hierarchical representations, potentially improving predictive performance.

#### Decision Tree Regressor
- It will be used as a baseline to improve upon as it is the improved version of what the instructor has given.
- Decision Tree Regressor is chosen for its ability to capture non-linear relationships in the data.
- The squared error criterion and limited maximum depth help prevent overfitting while allowing the model to learn meaningful patterns.

#### Linear Regression
- Linear Regression is a suitable choice for predicting numeric values (like grades) based on input features.
- The approach is useful as it provides a baseline model to understand the relationship between input features and grades.
- Cross-validation ensures that the model's performance is not overly optimistic or pessimistic, providing a more accurate assessment.
- The simplicity of Linear Regression makes it interpretable, helping to identify the most influential features on the predicted grades.

#### Ada Boost Regressor
- AdaBoost focuses on combining multiple weak learners (simple models) to form a strong learner.
- The key idea behind AdaBoost is to focus on the weaknesses of individual models and give more weight to the observations that are misclassified by previous models. 
- This iterative process helps improve overall model performance.

#### Lasso Regression
- Lasso Regression is a linear regression technique that adds a penalty term to the standard linear regression cost function.
- The penalty is proportional to the absolute values of the coefficients, encouraging the model to shrink some coefficients to exactly zero.
- This can be particularly useful for feature selection, as Lasso tends to produce sparse models by effectively setting some feature coefficients to zero.

#### Ridge Regression
- Ridge Regression, also known as Tikhonov regularization or L2 regularization, is a linear regression technique that adds a penalty term to the standard linear regression cost function.
- The penalty is proportional to the squared values of the coefficients, encouraging the model to shrink the coefficients toward zero. 
- This regularization term helps prevent overfitting and can be particularly useful when dealing with multicollinearity (high correlation among predictor variables).

#### Support Vector Regressor
- The program utilizes Support Vector Regressor (SVR), specifically with a linear kernel and C=1.0, for predicting student grades based on input features.
- The SVR model is trained on a subset of the data (X_train and y_train) and evaluated on both the training and test sets.
- Cross-validation is applied to obtain more reliable performance metrics, considering different subsets of the training data.
  
High-level explanation of things considered and solutions offered.

### Results

#### Decision Tree Regressor
```plaintext
MSE Train: 0.11475208640157092
MSE TEST: 11.900277777777779
R2 Train: 0.9992978102674439
R2 TEST: 0.8939988938928952
```

![image](https://github.com/CZunal/CS412-Course-Project/blob/3ee3c2372f996587645f278e379e176d1da23822/dr_mse_plot.png)
#### Neural Network
```plaintext
MSE Train: 8.6942377150357
MSE TEST: 8.94185264789965
R2 Train: 0.9467983141105091
R2 TEST: 0.9203509120523148
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/af26ca9fce6b2e078d94326f56ac75612949a1eb/nn_mse_plot.png)

#### Linear Regression
```plaintext
Mean Squared Error(Train): 7.032085026046438e-28
Mean Squared Error(Test): 1.211690350419474e-28
R2 Score(Train): 1.0
R2 Score(Test): 1.0

Cross-Validated Performance:
Mean Squared Error (CV): 7.209050111588801e-28
R2 Score (CV): 1.0
```
![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/d1ae4666-25b1-45f0-afb1-a981305b9358)

![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/3fe42047-2963-418c-b1a2-d46f3602cfc3)

#### Ada Boost Regressor
```plaintext
MSE Train: 0.7160604006452509
MSE TEST: 12.665975190894672
R2 Train: 0.9956182909000579
R2 TEST: 0.8871784839621872
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/a64c40a4f41143956fe6021969c5053794ed2f14/output.png)

#### Lasso Regression
```plaintext
MSE Train: 0.0018853801435518567
MSE TEST: 0.001653435732602358
R2 Train: 0.9999884630021093
R2 TEST: 0.9999852721071049
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/8b0544d5d384d382702e46e777792d4960c0c24c/output.png)

#### Ridge Regression
```plaintext
MSE Train: 2.414411922772e-07
MSE TEST: 2.920148371051739e-07
R2 Train: 0.9999999985225757
R2 TEST: 0.999999997398893
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/623c1c80f172ee9cd61c830be71572eea5ddb623/output.png)

#### Support Vector Regressor
```plaintext
Mean Squared Error(Train): 0.005130020227711553
Mean Squared Error(Test): 0.0047693131526653386
R2 Score(Train): 0.99995751759085
R2 Score(Test): 0.99995751759085

Cross-Validated Performance (SVR):
Mean Squared Error (CV): 0.15945861545308992
R2 Score (CV): 0.999024242555844
```
![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/b2d043f9-13a6-4639-a52d-b82c28469553)

![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/fba93847-bc32-407b-81d9-05f6df8b3cb0)


Experimental findings supported by figures,
tables etc.

### Team contributions
Each team member created branches

- Decision Tree Regressor - Baseline
- Neural Network - Can Zunal
- Linear Regression - Fikret Kayra Yılmaz, Gorkem Topcu 
- Ada Boost Regressor - Berk Ay
- Lasso Regression - Berk Ay
- Ridge Regression - Berk Ay
- Support Vector Regressor - Gorkem Topcu
- Gradient Boosting Regression - Yagiz Toprak Isik
