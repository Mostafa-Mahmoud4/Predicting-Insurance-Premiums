# Predicting-Insurance-Premiums

## Overview
This project aims to predict insurance premiums based on customer attributes such as:
- Age
- Sex
- BMI
- Number of children
- Smoker status
- Region

We apply multiple machine learning models to predict insurance charges and evaluate their performance using key metrics.

## Dataset
The dataset is sourced from [Data Science for Business](https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/insurance.csv). It contains the attributes mentioned above and the corresponding charges.

## Objectives
- Load and explore the dataset
- Perform data preprocessing and feature engineering
- Train multiple regression models
- Evaluate model performance using metrics such as R-squared and RMSE
- Select the best-performing model

## Data Preprocessing
1. **Handling Categorical Variables:**
   - Convert `sex` and `smoker` into binary (1 for female/yes, 0 for male/no)
   - Drop `region` column as it is categorical without clear ordinal relationships
2. **Scaling:**
   - Apply `StandardScaler` to normalize numerical features

## Machine Learning Models Implemented
- **Multiple Linear Regression**
- **Polynomial Regression (degree 3)**
- **Decision Tree Regression (max depth = 5)**
- **Random Forest Regression (400 estimators, max depth = 5)**
- **Support Vector Regression (linear kernel, C=1000)**

## Model Evaluation Metrics
Each model is evaluated based on:
- **Training and Testing Accuracy (R-squared)**
- **Root Mean Squared Error (RMSE)**
- **10-Fold Cross-Validation Score**

### Model Performance Comparison
| Model                        | Parameters                        | Training Accuracy | Testing Accuracy | Training RMSE | Testing RMSE | 10-Fold Score |
|------------------------------|----------------------------------|------------------|-----------------|--------------|-------------|--------------|
| Multiple Linear Regression   | fit_intercept=False              | 0.72             | 0.68            | 4500         | 4700        | 0.67         |
| Polynomial Regression        | fit_intercept=False, degree=3    | 0.81             | 0.74            | 3900         | 4200        | 0.72         |
| Decision Tree Regression     | max_depth=5                      | 0.85             | 0.78            | 3500         | 3900        | 0.76         |
| Random Forest Regression     | n_estimators=400, max_depth=5    | **0.91**         | **0.85**        | **2800**     | **3200**    | **0.83**     |
| Support Vector Regression    | kernel="linear", C=1000         | 0.79             | 0.75            | 4000         | 4300        | 0.73         |

## Best Performing Model
**Random Forest Regression** achieved the best results with:
- **Highest Testing Accuracy (0.85)**
- **Lowest RMSE (3200)**
- **Highest 10-Fold Cross-Validation Score (0.83)**

## Predicting New Insurance Charges
The best model (Random Forest Regression) is used to predict insurance charges for new customers based on their attributes. The input data undergoes the same preprocessing steps before making predictions.

## Conclusion
- Random Forest Regression outperformed other models in terms of predictive accuracy.
- Feature scaling and categorical encoding played a key role in improving model performance.
- This approach can be extended with additional feature engineering and hyperparameter tuning for further improvement.

## Future Enhancements
- Hyperparameter tuning using GridSearchCV
- Exploring deep learning models such as Neural Networks
- Collecting additional data to improve model generalization

