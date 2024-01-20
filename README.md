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
    - [Gradient Boosting Regression](#gradient-boosting-regressor)
  - [Results](#results)
    - [Decision Tree Regressor](#decision-tree-regressor-1)
    - [Ada Boost Regressor](#ada-boost-regressor-1)
    - [Neural Network](#neural-network-1)
    - [Lasso Regression](#lasso-regression-1)
    - [Ridge Regression](#ridge-regression-1)
    - [Support Vector Regressor](#support-vector-regressor-1)
    - [Linear Regression](#linear-regression-1)
    - [Gradient Boosting Regression](#gradient-boosting-regressor-1)

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

#### Gradient Boosting Regressor
- Gradient Boosting Regressor is an ensemble learning technique that sequentially combines weak learners to form a robust predictive model.
High-level explanation of things considered and solutions offered.
- This method excels in capturing complex relationships and non-linear patterns in the data, making it a powerful tool for predicting numeric values.
- The iterative nature and adaptability of Gradient Boosting Regressor make it versatile for various predictive modeling scenarios.

### Results

Ranked by Test R2 Square Values


#### Decision Tree Regressor (Baseline)
```plaintext
MSE Train: 23.792569028909234
MSE TEST: 123.90309194337131
R2 Train: 0.8544087675690079
R2 TEST: -0.1036603549383901
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/39063fc474ace31de8bed9df1016470ed848f8e8/dr_mse_plot.png)

#### Ada Boost Regressor
```plaintext
MSE Train: 23.512951460074028
MSE TEST: 99.94532471726836
R2 Train: 0.856119800387978
R2 TEST: 0.10974221206435131
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/e52e999f4f3fba64306f507734d3b9319c6ab48d/output.png)

#### Neural Network
```plaintext
MSE Train: 54.67724691269477
MSE TEST: 111.86605245506158
R2 Train: 0.6654195789332196
R2 TEST: 0.0035589490007483793
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/1f1ea7d8a20b2a323b148354454f0796be208ea3/nn_mse_plot.png)


#### Lasso Regression
```plaintext
MSE Train: 121.84475807965048
MSE TEST: 109.24607685765808
R2 Train: 0.2544088672174103
R2 TEST: 0.026896245531506602
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/fc21d4efb5bcd165b7d25283df46c763aeee6e9a/output.png)

#### Ridge Regression
```plaintext
MSE Train: 102.43671492188103
MSE TEST: 123.8936916254293
R2 Train: 0.37317035610834226
R2 TEST: -0.10357662209465146
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/c197a960845540069640de409db6107d393cfb15/output.png)

#### Support Vector Regressor
```plaintext
Mean Squared Error(Train): 129.45437423099355
Mean Squared Error(Test): 122.81119238753843
R2 Score(Train): -0.09393431636706562
R2 Score(Test): -0.09393431636706562

Cross-Validated Performance (SVR):
Mean Squared Error (CV): 495.71812899949924
R2 Score (CV): -2.0333930418245107
```
![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/b2d043f9-13a6-4639-a52d-b82c28469553)

![image](https://github.com/CZunal/CS412-Course-Project/assets/73399460/fba93847-bc32-407b-81d9-05f6df8b3cb0)

#### Linear Regression
```plaintext
Mean Squared Error(Train): 100.87553202163336
Mean Squared Error(Test): 135.36989735207453
R2 Score(Train): 0.3827235297156596
R2 Score(Test): -0.20580032843608853

Cross-Validated Performance:
Mean Squared Error (CV): 744.1548410832953
R2 Score (CV): -3.553624297618481
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/c45edb6916dcb1d122405f794d591b70183f2174/image-1.png)

![image](https://github.com/CZunal/CS412-Course-Project/blob/c45edb6916dcb1d122405f794d591b70183f2174/image.png)


#### Gradient Boosting Regression
```plaintext
XGBoost - MSE Train: 0.022012364528882715
XGBoost - MSE TEST: 141.72754318997497
XGBoost - R2 Train: 0.9998653021757934
XGBoost - R2 TEST: -0.26243072846869375

Cross-Validation R2 Scores: [-1.10587873 -0.13118338 -8.12836632 -0.51004095 -1.72988704]
Mean R2 Score: -2.3210712836062344
```
![image](https://github.com/CZunal/CS412-Course-Project/blob/d0a5368e39d8de3fbf3582796b1494bb7829f7fc/output.png)


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
