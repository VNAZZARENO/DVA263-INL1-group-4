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

## Introduction

This project focuses on **unsupervised learning techniques** to analyze customer behaviors from the Market Segmentation dataset. Using clustering algorithms, we segmented customers into meaningful groups to derive actionable insights for targeted marketing strategies.

## Dataset

The **Market Segmentation dataset** contains behavioral statistics for 8,950 unique customers across 17 original features (excluding an ID column). To enhance the analysis, we engineered 8 new dependent features.

### Key Data Characteristics:
- Missing values imputed using clustering-based methods (K-Means).
- Features include statistics like `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`, and others.

## Objectives

1. Segment customers into distinct behavioral groups.
2. Provide actionable marketing recommendations for each group.
3. Visualize and analyze feature importance for segmentation.

## Methodology

### Data Cleaning and Preprocessing

1. **Missing Value Imputation**: Used a clustering-based imputation strategy for `CREDIT_LIMIT` and `MINIMUM_PAYMENTS`.
2. **Scaling**: Z-score normalization was applied for consistent clustering results.
3. **Correlation Analysis**: Hierarchical clustering of feature correlations was performed to understand relationships among features.

### Clustering

1. **K-Means**:
   - Optimal clusters determined using silhouette analysis.
   - Chose **4 clusters** based on silhouette scores and interpretability.
2. **Dimensionality Reduction**:
   - PCA for visualization of clusters.
   - t-SNE for a more detailed 2D embedding of clusters.

### Visualization

- **PCA Biplot** and **Cluster Overlay**: Demonstrated cluster separability.
- **t-SNE Plot**: Showed inter-cluster relationships.
- **Cluster-Wise Heatmap**: Highlighted normalized feature importance.
- **Radar Chart**: Provided an intuitive view of feature significance for each cluster.

## Results

### Customer Segments and Recommendations

| Cluster | Consumer Type           | Key Characteristics                                                                 | Marketing Recommendations                                                                 |
|---------|--------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 0       | Balanced Consumers      | Average `BALANCE` and `CREDIT_LIMIT`.                                              | Promote basic financial products and incentives for improved engagement.                 |
| 1       | Highly Engaged Consumers| High `PURCHASES`, `ENGAGEMENT_INDEX`, `PURCHASE_TO_LIMIT`.                         | Offer cashback rewards and installment plans, monitor repayment behavior.               |
| 2       | Outstanding Consumers   | High `PAYMENTS`, low `CREDIT_UTILIZATION`.                                         | Cross-sell premium financial products and savings plans.                                 |
| 3       | Risky Consumers         | High `CREDIT_UTILIZATION`, `DEBT_TO_PAYMENT_RATIO`, `CASH_ADVANCE_TO_PURCHASES`.   | Provide credit counseling, debt consolidation plans, and support to manage finances.     |

## Ethics of Clustering in Marketing

- Customers should be informed about the use of their data.
- Data privacy must comply with regulations like GDPR.
- Bias in data and recommendations should be avoided to ensure fairness.

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
Open `DVA263_INL1_Group_4.ipynb` or `DVA263_INL2_Group_4.ipynb` to reproduce the results:

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
