# DVA263_INL1_4: FIFA 18 Player Overall Rating Prediction

## Introduction

This project aims to predict the overall rating of players in the FIFA 18 dataset using supervised machine learning algorithms. The overall rating is a critical metric that encapsulates a player's skills and performance in the game. We explore various regression models to achieve this prediction accurately.

## Dataset

The dataset comprises over 17,000 players, each with more than 70 attributes, including player statistics, personal data, and categorical information.

## Methodology

### Data Preprocessing

- **Data Cleaning:** Removed irrelevant columns such as 'Photo', 'Flag', and 'Club Logo'. Handled missing values and cleaned statistical attributes.
- **Feature Engineering:** Created new features like 'Attacking Score', 'Defensive Score', and others to enhance model performance.

### Model Selection

- **Algorithms Used:**
  - RandomForestRegressor
  - LinearRegression
  - SVR
  - LGBMRegressor
- **Hyperparameter Tuning:** Used GridSearchCV with 5-fold cross-validation to optimize hyperparameters.

### Evaluation Metrics

- **Metrics:** Mean Squared Error (MSE) and R² score for regression tasks.

## Models

### 1. RandomForestRegressor

- **Hyperparameters:**
  - `n_estimators`: 100, 200
  - `max_depth`: None, 10, 20
- **Best Parameters:**
  - `n_estimators`: 200
  - `max_depth`: 20

### 2. LinearRegression

- **Hyperparameters:**
  - `fit_intercept`: True, False
- **Best Parameters:**
  - `fit_intercept`: True

### 3. SVR

- **Hyperparameters:**
  - `C`: 0.1, 1.0
  - `kernel`: 'linear', 'rbf'
- **Best Parameters:**
  - `C`: 1.0
  - `kernel`: 'rbf'

### 4. LGBMRegressor

- **Hyperparameters:**
  - `learning_rate`: 0.01, 0.1
  - `n_estimators`: 100, 200
- **Best Parameters:**
  - `learning_rate`: 0.1
  - `n_estimators`: 200

## Evaluation

### Results

- **RandomForestRegressor:**
  - **MSE:** 0.123064
  - **R²:** 0.998152
  - **Best Parameters**
    - `model__max_depth`: None
    - `model__n_estimators`: 200
- **LinearRegression:**
  - **MSE:** 3.522638	
  - **R²:** 0.924367
  - **Best Parameters**
    - `model__fit_intercept`: True
- **SVR:** 
  - **MSE:** 1.033004	
  - **R²:** 0.982995
  - **Best Parameters**
    - `model__C`: 1.0
    - `model__kernel`: rbf
- **LGBMRegressor:**
  - **MSE:** 0.149932	
  - **R²:** 0.997527
  - **Best Parameters**
    - `model__learning_rate`: 0.1
    - `model__n_estimators`: 200

### Conclusion

TO BE DETERMINED 

## Future Work

- Explore classification tasks by binning the 'Overall' ratings into categories.
- Investigate the impact of different sampling techniques on model performance.
- Experiment with ensemble methods to further improve prediction accuracy.

## References

- Scikit-learn documentation for model selection and evaluation.
- SMOTE for imbalanced data handling.
