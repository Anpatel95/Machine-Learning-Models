---

# Wine Quality Analysis using various ML Models

This project utilizes various classification and regression models to analyze the chemical properties of wines, specifically comparing the least-rated wines (quality = 3) and the best-rated wines (quality = 8). The aim is to extract insights and knowledge from the dataset by identifying the key chemical characteristics that influence wine quality.

## Dataset

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/code/georgyzubkov/wine-quality-exploratory-data-analysis-ml). It consists of the chemical features of wines along with their corresponding quality ratings.

## Objective

The objective of this project is to analyze the chemical properties of wines with the lowest quality rating (3) versus those with the highest quality rating (8) using various machine learning models and optimization techniques. The dataset is split into 67% for training and 33% for testing.

## Authors' Model Accuracy - Default Algorithms

The initial phase evaluates model performance using default hyperparameters and SMOTE (Synthetic Minority Over-sampling Technique) for oversampling.

| Method Used                                 | Testing Accuracy Score |
|----------------------------------------------|-------------------------|
| Random Forest with SMOTE oversampling        | 86.0%                   |
| XGBoost with SMOTE oversampling              | 84.0%                   |
| K-nearest neighbors with SMOTE oversampling  | 75.0%                   |
| Support Vector Classifier with SMOTE oversampling | 70.0%             |
| Logistic Regression with SMOTE oversampling  | 58.0%                   |

## Phase 1 - Default Algorithms

This phase establishes baseline performance using different machine learning models with their default settings:

| Method Used                      | Hyperparameters       | Standardized | Training Accuracy | Testing Accuracy |
|----------------------------------|-----------------------|--------------|-------------------|------------------|
| Naive Bayes                      | Default Settings      | No           | 45.38%            | 48.40%           |
| Decision Tree                    | max_depth = 9         | No           | 89.38%            | 52.77%           |
| Logistic Regression              | Default Settings      | Yes          | 60.62%            | 55.98%           |
| Randomized Forest Classifier     | random_state = 0      | No           | 100%              | 64.43%           |

## Phase 2 - Exploration with Various Algorithms and Optimization Methods

The second phase explores different algorithms and optimization techniques to improve performance:

| Algorithm Used                             | Hyperparameters                                             | Standardized | Training Accuracy | Testing Accuracy |
|---------------------------------------------|-------------------------------------------------------------|--------------|-------------------|------------------|
| Decision Tree with AdaBoost                 | n_estimators = 400                                           | No           | 100%              | 62.68%           |
| Decision Tree with AdaBoost                 | n_estimators = 300                                           | No           | 100%              | 62.68%           |
| Decision Tree with AdaBoost                 | n_estimators = 200                                           | No           | 100%              | 63.27%           |
| Decision Tree with AdaBoost                 | n_estimators = 220                                           | No           | 100%              | 63.85%           |
| Decision Tree with PCA                      | max_depth = 9, random_state = 0                              | No           | 86.5%             | 56.27%           |
| Naive Bayes with AdaBoost                   | n_estimators = 500                                           | No           | 55.00%            | 54.81%           |
| Logistic Regression (Polynomial)            | degree = 2, max_iter = 2000                                  | Yes          | 67.62%            | 56.85%           |
| Logistic Regression (Polynomial, OVR)       | degree = 3, max_iter = 2000, multi_class = 'ovr'             | Yes          | 84.12%            | 58.60%           |
| RandomForest with PCA                       | random_state = 10, n_estimators = 200, criterion = 'entropy' | Yes          | 100.0%            | 63.85%           |
| RandomForest with Feature Selection and PCA | max_features = 7, random_state = 10, n_estimators = 200      | Yes          | 100.0%            | 58.89%           |

## Conclusion

The project demonstrates that different algorithms and optimization methods provide varying levels of accuracy in classifying wine quality. While some models like the Random Forest achieved a perfect training accuracy of 100%, the testing accuracy varied, indicating potential overfitting. Future work can focus on hyperparameter tuning, feature engineering, and additional comparative analysis to improve model performance.

---
