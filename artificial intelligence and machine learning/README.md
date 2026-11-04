# Carcinoma Classification — Machine Learning Exam Project

## Project Overview

This project was developed as part of a university examination in Machine Learning and Data Science.

The objective is to classify histopathological images stained using the **Hematoxylin–Eosin–Saffron (HES)** technique, identifying the presence of carcinoma and assigning each image to one of three classes defined by the competition.

The task is formulated as a **three-class image classification problem**:

| Label | Class | Description |
|---|---|---|
| -1 | Carcinoma neg | Negative |
| 0 | Carcinoma pos, benign | Positive, benign category |
| 1 | Carcinoma pos, malignant | Positive, malignant category |

## Dataset

The dataset provided by the competition consists of **186 histopathological images**, divided into:

- **62 labeled images** for supervised learning.
- **124 unlabeled images** for prediction.

The limited availability of labeled data introduces several methodological challenges, including:

- Preventing overfitting.
- Designing an appropriate validation strategy.
- Evaluating data augmentation techniques.
- Exploring semi-supervised learning approaches.

## Project Objectives

The main objectives of the project are:

1. **Exploratory Data Analysis (EDA):** investigate dataset characteristics and class distribution.
2. **Data Preprocessing:** prepare histopathological images for model training.
3. **Model Development:** implement and compare different classification approaches.
4. **Model Evaluation:** assess predictive performance using appropriate classification metrics.
5. **Competition Submission:** generate predictions for unlabeled images and prepare the submission file according to the competition requirements.

## Competition

**Platform:** Kaggle

**Challenge:** [OX-ML Carcinoma Classification](https://www.kaggle.com/competitions/oxml-carinoma-classification)

## Academic Disclaimer

This project was developed exclusively for academic and educational purposes.

The models and predictions are not clinically validated and must not be used for medical diagnosis or clinical decision-making.
