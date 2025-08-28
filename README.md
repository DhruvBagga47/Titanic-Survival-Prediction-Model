# Titanic-Survival-Prediction-Model
This project predicts passenger survival on the Titanic using machine learning. The workflow includes data preprocessing, feature engineering, model selection, hyperparameter tuning, and submission generation.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Dependencies](#dependencies)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Feature Engineering](#feature-engineering)  
6. [Model Training & Evaluation](#model-training--evaluation)  
7. [Hyperparameter Tuning](#hyperparameter-tuning)  
8. [Final Model & Predictions](#final-model--predictions)  
9. [Results](#results)

## Project Overview
The Titanic dataset is a classic beginner-level competition on Kaggle. The goal is to build a model that predicts whether a passenger survived or not based on attributes such as age, gender, class, and embarkation port. This project uses Decision Trees and Random Forest Classifiers, evaluating them with stratified train/validation splits and selecting the best model based on validation accuracy.

## Dataset
The dataset comes from [Kaggle’s Titanic competition.](https://www.kaggle.com/competitions/titanic/overview "Go to Kaggle website")

``train.csv``: Training data with labels (Survived).  
``test.csv``: Test data used for predictions.

## Dependencies
Imports used:
```python
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score
```

## Data Preprocessing

- Dropped unnecessary columns: ``Name, Ticket, Cabin.``
- Filled missing values:
  - Age → replaced with mean.
  - Embarked → replaced with mode.
  - Fare (test set) → replaced with mean.
- Converted categorical variables (``Sex, Embarked``) into dummy/indicator variables.
- Ensured consistent columns between train and test datasets.

## Feature Engineering
- Excluded non-informative columns (``PassengerId, Survived``).
- Selected remaining features as input (``X``).
- Target variable (``y``) was Survived.

## Model Training & Evaluation
- Performed train/validation split (80/20, stratified to balance classes).
- Defined evaluation functions for:
  - ``Decision Tree``
  - ``Random Forest``
- Used accuracy_score to compare validation performance.

## Hyperparameter Tuning
- Iterated over different values of ``max_leaf_nodes`` (range 4–80).
- Selected best-performing hyperparameters for each model.
- Compared validation accuracies of Decision Tree vs Random Forest.

## Final Model & Predictions
- Trained the best Random Forest model on the full training set.
- Generated predictions on the test dataset.
- Exported results to submission.csv in the format required by Kaggle.

## Results
- Decision Tree: best accuracy at chosen leaf nodes.
- Random Forest: higher validation accuracy overall.
- Final predictions saved successfully as submission.csv.
