# CS412 Course Project

## Overview

This project focuses on enhancing the performance of a natural language processing (NLP) model for analyzing and predicting scores based on prompts. The primary goal is to explore various techniques related to text preprocessing, feature engineering, model tuning, and experimenting with different models.

## Table of Contents
- [Text Preprocessing](#text-preprocessing)
  - [Lowercasing](#lowercasing)
  - [Removing Punctuation and Special Characters](#removing-punctuation-and-special-characters)
  - [Tokenization](#tokenization)
  - [Stemming and Lemmatization](#stemming-and-lemmatization)
  - [Normalizing Grades](#normalizing-grades)
  - [Keyword Selection](#keyword-selection)
- [Feature Engineering](#feature-engineering)
  - [Vectorization Techniques](#vectorization-techniques)
  - [Text Length](#text-length)
- [Model Tuning](#model-tuning)
  - [Architecture Adjustments](#architecture-adjustments)
  - [Regularization](#regularization)
  - [Learning Rate](#learning-rate)
- [Different Models to Try](#different-models-to-try)
  - [Tfidf Vectorization with Linear Regression](#tfidf-vectorization-with-linear-regression)
  - [Neural Network with Word Embeddings](#neural-network-with-word-embeddings)
  - [Gradient Boosting Regression with Feature Importance](#gradient-boosting-regression-with-feature-importance)
  - [Support Vector Regression (SVR) with Kernel Tricks](#support-vector-regression-svr-with-kernel-tricks)
  - [Neural Network Adjustments](#neural-network-adjustments)
- [Stacking](#stacking)
## Text Preprocessing

### Lowercasing
- **Advantages:**
  - Standardizes text for consistent analysis.
- **Disadvantages:**
  - Can lead to the loss of information, especially sentiment or emphasis indicated by punctuation marks.

### Removing Punctuation and Special Characters
- **Advantages:**
  - Cleans text for better analysis.
- **Disadvantages:**
  - May result in the loss of information, altering the meaning based on word context.

### Tokenization
- **Advantages:**
  - Breaks text into meaningful units.
- **Disadvantages:**
  - May lead to information loss as the context of words affects meaning.

### Stemming and Lemmatization
- **Advantages:**
  - Reduces words to their base form.
- **Disadvantages:**
  - Potential loss of information, as the meaning can change based on word context.

### Normalizing Grades
- **Techniques:**
  - Min-max Scaling
  - Z-Score Standardization
  - Log Transformation (useful for skewed grades)
  - Box-Cox Transformation

### Keyword Selection
- Explore most used keywords for better search results.

## Feature Engineering

### Vectorization Techniques
- Check Word2Vec and GloVe for semantic relationship capture.

### Text Length
- Experiment with features related to the length of prompts or responses.

## Model Tuning

### Architecture Adjustments
- Experiment with changing neural network architecture (e.g., increasing layers or units).

### Regularization
- Introduce techniques like dropout layers to prevent overfitting.

### Learning Rate
- Experiment with different learning rates during training.

## Different Models to Try

### Tfidf Vectorization with Linear Regression
- Use Tfidf vectorization to represent prompts and apply linear regression for modeling.

### Neural Network with Word Embeddings
- Utilize Word2Vec or GloVe embeddings to represent prompts in a neural network.

### Gradient Boosting Regression with Feature Importance
- Use gradient boosting regression for improved model performance, analyze feature importance.

### Support Vector Regression (SVR) with Kernel Tricks
- Implement SVR with kernel tricks for predicting scores.

### Neural Network Adjustments
- Experiment with changing epoch and batch size in neural network training.
## Stacking
- Try different merging different methods
