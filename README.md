# FIRST ASSIGNEMENT : INL1 SUPERVISED LEARNING

## DVA263_INL1_Group_4: FIFA 18 Player Overall Rating Prediction

## Introduction

This project aims to predict the overall rating of players in the FIFA 18 dataset using supervised machine learning algorithms. The overall rating is a critical metric that encapsulates a player's skills and performance in the game. We explore various regression models to achieve this prediction accurately.

## Dataset

The dataset comprises over 17981 players, each with more than 74 attributes, including player statistics, personal data, and categorical information.

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

## Results

The performance of various regression models was evaluated using cross-validation (CV) and tested on a hold-out test set. Below is a summary of the results:

- **Random Forest Regressor**: This model achieved the best performance with a cross-validation score (Negative MSE) of **0.1231**, a Mean Squared Error (MSE) of **0.0882**, and an R² score of **0.9982** on the test set. The optimal parameters included `max_depth=None` and `n_estimators=200`.

- **Linear Regression**: While computationally simple, this model had a higher test MSE of **3.6114** and an R² score of **0.9244**. The model performed best with `fit_intercept=True`.

- **Support Vector Regressor (SVR)**: The SVR model demonstrated strong performance with a test MSE of **0.8120** and an R² score of **0.9830**, using a Radial Basis Function (RBF) kernel and `C=1.0`.

- **LightGBM Regressor**: This gradient boosting model also performed well, achieving a test MSE of **0.1181** and an R² score of **0.9975**, with optimal parameters including `learning_rate=0.1` and `n_estimators=200`.

Overall, the **Random Forest Regressor** provided the best balance of accuracy and predictive power, closely followed by the **LightGBM Regressor**, while the **SVR** offered similar performance in a simpler model.

## Future Work

- Explore classification tasks by binning the 'Overall' ratings into categories.
- Investigate the impact of different sampling techniques on model performance.
- Experiment with ensemble methods to further improve prediction accuracy.

## References

- Scikit-learn documentation for model selection and evaluation.
- SMOTE for imbalanced data handling.



# SECOND ASSIGNEMENT : INL2 UNSUPERVISED LEARNING

## DVA263_INL2_Group_4: Market Segmentation Data



# Setup

To reproduce the results and run the code in this repository, follow these steps:

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/VNAZZARENO/DVA263-group-4-2024.git
cd DVA263-group-4-2024
```

### 2. Create a Virtual Environment (Optional but Recommended)
Create a virtual environment to manage dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm imbalanced-learn joblib tqdm
```

### 4. Run the Code
Open the `assignement_1_no_gpu.ipynb` to reproduce the results:

### 5. Deactivate the Virtual Environment (Optional)
Once you're done, you can deactivate the virtual environment:
```bash
deactivate
```

### Requirements
The project requires the following Python packages:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `imbalanced-learn`
- `joblib`
- `tqdm`
