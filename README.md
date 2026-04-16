

## Problem Statement

The objective of this project is to develop and evaluate multiple supervised machine learning classification models to predict whether a customer will subscribe to a term deposit offered by a financial institution.

Using the Bank Marketing Dataset (Moro et al., 2014), this study analyzes customer demographic information, financial attributes, and previous marketing campaign interactions to identify patterns that influence subscription decisions.

This is a **binary classification problem**, where the target variable `deposit` indicates whether a client subscribed to a term deposit (Yes/No).

## Dataset Description

**Dataset Name:** Bank Marketing Dataset  
**Original Study:** Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22–31.  
**Repository:** UCI Machine Learning Repository  
**Access Source:** Kaggle (Bank Marketing Dataset) https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

The dataset contains information related to direct marketing campaigns (phone calls) conducted by a banking institution. The objective of the campaign was to determine whether a client would subscribe to a term deposit.

It includes demographic attributes, financial information, and campaign-related variables for each client. The dataset is commonly used for binary classification tasks in marketing analytics and predictive modeling.

### Dataset Characteristics

- **Number of Features:** 16 input variables  
- **Target Variable:** `deposit` (Yes/No)  
- **Problem Type:** Binary Classification  
- **Domain:** Banking / Marketing Analytics  

The features can be broadly categorized into:

- **Client Information:** age, job, marital status, education, balance, etc.  
- **Campaign Information:** contact type, duration, campaign frequency, previous outcomes  
- **Financial Indicators:** housing loan, personal loan, default status  

The dataset is moderately imbalanced, with a higher proportion of clients not subscribing to a term deposit.

## Model Training & Data Splitting

The raw dataset was split into training and testing sets using an 80:20 ratio 
with stratified sampling to preserve the class distribution of the target variable (`deposit`).

The split was performed before feature engineering to prevent data leakage. 
All preprocessing steps (imputation, encoding, clipping, scaling, and feature derivation) 
were fitted only on the training data and then applied to the test data using the 
saved preprocessing parameters (`preprocessing_params.pkl`).

A fixed `random_state=42` was used to ensure reproducibility of results.

The test split generated during training is saved as `data/bank_test.csv`. 
This file is available for download and can either be directly uploaded or imported using the built-in import option in the Streamlit interface for consistent model evaluation and demonstration.

## Models Used

### Model Performance Comparison

| Model Name                    | Accuracy |  AUC   | Precision | Recall | F1 Score |  MCC   |
| :---------------------------- | :------: | :----: | :-------: | :----: | :------: | :----: |
| **Logistic Regression**       |  0.6740  | 0.7319 |  0.6640   | 0.6314 |  0.6473  | 0.3449 |
| **Decision Tree**             |  0.7031  | 0.7595 |  0.8025   | 0.4953 |  0.6125  | 0.4231 |
| **K-Nearest Neighbors (KNN)** |  0.6803  | 0.7351 |  0.6870   | 0.5974 |  0.6390  | 0.3573 |
| **Naive Bayes**               |  0.6359  | 0.7023 |  0.6758   | 0.4452 |  0.5368  | 0.2725 |
| **Random Forest (Ensemble)**  |  0.7241  | 0.7784 |  0.7500   | 0.6267 |  0.6828  | 0.4478 |
| **XGBoost (Ensemble)**        |  0.7349  | 0.7862 |  0.7589   | 0.6456 |  0.6977  | 0.4691 |

### Model Observations

| ML Model Name                 | Observation                                                                                                                                                                                                                                                                               |
| :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**       | Serves as a linear baseline model. It achieves moderate performance (Accuracy: 0.6740, F1: 0.6473), indicating reasonable predictive ability but limited capacity to model complex nonlinear relationships among features.                                                                |
| **Decision Tree**             | Achieves higher precision (0.8025) but lower recall (0.4953), suggesting that while it makes confident positive predictions, it misses a significant number of actual subscribers. This imbalance reduces its overall F1-score.                                                           |
| **K-Nearest Neighbors (KNN)** | Shows moderate performance (F1: 0.6390). As a distance-based algorithm, it may struggle with mixed feature types and encoded categorical variables, making it less effective than tree-based ensemble methods on this dataset.                                                            |
| **Naive Bayes**               | Shows the lowest performance across all metrics (Accuracy: 63.59%, F1: 53.68%).  This is likely due to its strong conditional independence assumption, which may not hold in a dataset where financial and campaign-related features are interrelated.                                    |
| **Random Forest (Ensemble)**  | Closely follows XGBoost with an Accuracy of 72.41% and strong F1-score (68.28%). As an ensemble method, it offers robust predictions and reduces overfitting compared to single decision trees.                                                                                           |
| **XGBoost (Ensemble)**        | Demonstrates the best overall performance with the highest Accuracy (73.49%) and ROC-AUC (0.7862). Its gradient boosting framework effectively captures complex feature interactions, making it the strongest model among those evaluated and the most suitable candidate for deployment. |

This study implemented and evaluated six classification models on the Bank Marketing dataset to predict customer subscription to term deposits. Based on comparative analysis across Accuracy, AUC, Precision, Recall, F1-score, and MCC, ensemble methods (Random Forest and XGBoost) outperformed single-model classifiers. 

XGBoost achieved the highest overall performance, indicating its effectiveness in modeling complex nonlinear relationships within structured marketing data. The results demonstrate that ensemble learning improves generalization and predictive stability, making XGBoost the most suitable model for deployment in the Streamlit application.


##  Project Structure

```
ML_Classification_Models/
├── app.py                      # Streamlit web application
├── train.py                    # Main training script
├── data_pipeline.py            # Data processing and feature engineering
├── model/                      # Model implementations
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── xgboost.py
│   └── ...
├── trained_models/             # Trained models and metrics
│   ├── preprocessing_params.pkl
│   └── ...
├── data/                       # Dataset directory
│   ├── bank.csv               # Training dataset (Bank Marketing)
│   └── bank_test.csv          # Test dataset (Raw)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Dataset

The project uses the **Bank Marketing Dataset** with the following features:

- **Bank Client Data**:
    - `age`: Age in years
    - `job`: Type of job (admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)
    - `marital`: Marital status (divorced, married, single, unknown)
    - `education`: Education level (primary, secondary, tertiary, unknown)
    - `default`: Has credit in default? (yes/no)
    - `balance`: Average yearly balance, in euros
    - `housing`: Has housing loan? (yes/no)
    - `loan`: Has personal loan? (yes/no)

- **Communication Data**:
    - `contact`: Contact communication type (cellular, telephone)
    - `day`: Last contact day of the month
    - `month`: Last contact month of year
    - `duration`: Last contact duration, in seconds

- **Campaign Data**:
    - `campaign`: Number of contacts performed during this campaign and for this client
    - `pdays`: Number of days that passed by after the client was last contacted from a previous campaign
    - `previous`: Number of contacts performed before this campaign and for this client
    - `poutcome`: Outcome of the previous marketing campaign (failure, other, success, unknown)

- **Target**:
    - `deposit`: Has the client subscribed to a term deposit? (yes/no)
  
## Data Pipeline & Feature Engineering

The `data_pipeline.py` module implements a **12-step feature engineering pipeline** that transforms the raw Bank Marketing dataset into model-ready features. All fitted parameters (clip bounds, scaler, mode values, etc.) are persisted to `trained_models/preprocessing_params.pkl` so the same transforms are applied consistently during inference.

## Model Training & Data Splitting

The raw dataset was split into training and testing sets using an 80:20 ratio 
with stratified sampling to preserve the class distribution of the target variable (`deposit`).

The split was performed before feature engineering to prevent data leakage. 
All preprocessing steps (imputation, encoding, clipping, scaling, and feature derivation) 
were fitted only on the training data and then applied to the test data using the 
saved preprocessing parameters (`preprocessing_params.pkl`).

A fixed `random_state=42` was used to ensure reproducibility of results.

The test split generated during training is saved as `data/bank_test.csv`. 
This file is available for download and can either be directly uploaded or imported using the built-in import option in the Streamlit interface for consistent model evaluation and demonstration.

### Pipeline Steps

|   #   | Step                           | Details                                                                                                                                                                                                               |
| :---: | :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1   | **Drop Leaky / Noisy Columns** | Removes `duration` (leaks outcome) and `day` (noisy).                                                                                                                                                                 |
|   2   | **Impute Unknowns**            | Replaces `unknown` in `job` and `education` with the training-set mode. `contact` and `poutcome` unknowns are kept as valid categories.                                                                               |
|   3   | **Encode Target**              | Maps `deposit` → 1 (yes) / 0 (no).                                                                                                                                                                                    |
|   4   | **Binary Label Encoding**      | Maps `default`, `housing`, and `loan` → 1 / 0.                                                                                                                                                                        |
|   5   | **Derived Features**           | `was_contacted` — 1 if `pdays ≥ 0` (client was previously contacted). `balance_per_age` — wealth-to-age ratio. `poutcome_success_contacted` — interaction of prior contact × successful outcome.                      |
|   6   | **Clip Outliers**              | Clips `balance`, `campaign`, and `previous` at the 1st and 99th percentile bounds (fitted on training data).                                                                                                          |
|   7   | **Log Transforms**             | Applies `log1p` to reduce skewness: shifted log for `balance` and `balance_per_age` (handles negatives), standard `log1p` for `campaign` and `previous`, `pdays` (−1 replaced with 0 first).                          |
|   8   | **Ordinal Encoding**           | Encodes `education` as primary → 0, secondary → 1, tertiary → 2.                                                                                                                                                      |
|   9   | **Cyclical Encoding**          | Converts `month` to `month_sin` and `month_cos` using sine/cosine with period 12. The original `month` column is dropped.                                                                                             |
|  10   | **One-Hot Encoding**           | One-hot encodes `job`, `marital`, `contact`, and `poutcome` (with `drop_first=True`). Missing columns during inference are zero-filled for consistency.                                                               |
|  11   | **Standard Scaling**           | Applies `StandardScaler` (zero mean, unit variance) to numerical features: `age`, `balance`, `campaign`, `pdays`, `previous`, `education`, `month_sin`, `month_cos`, `balance_per_age`, `poutcome_success_contacted`. |
|  12   | **Save Parameters**            | Persists all fitted parameters (`preprocessing_params.pkl`) for reproducible inference.                                                                                                                               |

### Engineered Features Summary

| Feature                      | Source                                  | Type                         |
| :--------------------------- | :-------------------------------------- | :--------------------------- |
| `was_contacted`              | `pdays`                                 | Binary (0/1)                 |
| `balance_per_age`            | `balance / age`                         | Continuous (log-transformed) |
| `poutcome_success_contacted` | `was_contacted × (poutcome == success)` | Binary interaction           |
| `month_sin`                  | `month`                                 | Cyclical (sin)               |
| `month_cos`                  | `month`                                 | Cyclical (cos)               |

> **Note:** After one-hot encoding the final feature set contains ~30 columns (varies slightly by dataset cardinality). Two original columns (`duration`, `day`) are dropped, and 5 new derived/cyclical features are added alongside the one-hot expansions.
>

## Web Application Features
- **Quick Test Data Import:** One-click option to load the provided test dataset directly
- **Download Test Dataset:** Download the sample test dataset for external evaluation
- **Dataset Upload**: Upload CSV files for evaluation (supports raw or preprocessed)
- **Model Selection**: Choose from trained models via sidebar
- **Evaluation Metrics**: View accuracy, precision, recall, F1-score, ROC-AUC
- **Confusion Matrix**: Interactive heatmap visualization
- **Classification Report**: Detailed per-class metrics
- **Download Predictions**: Export results as CSV
  
## Project Features

- **Modular Training Framework**: Easy-to-extend architecture for multiple ML models
- **Data Pipeline**: Automated 12-step feature engineering pipeline with derived features, encoding, outlier clipping, and scaling
- **Interactive Web App**: Streamlit-based UI for model evaluation
- **Model Persistence**: Save and load trained models with metadata
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, MCC

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/username/ML_Classification_Models.git
cd ML_Classification_Models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

Train a model using the command-line interface:

```bash
# Train a specific model (e.g., XGBoost)
python train.py --model xgboost

# Train ALL models sequentially
python train.py --model all

# List available models
python train.py --list-models

# Custom configuration (test size, random state)
python train.py --model random_forest --test-size 0.3 --random-state 123
```

### Running the Web Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

##  Streamlit Cloud Deployment

### Deploy to Streamlit Cloud:

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy!

**Note:** Make sure to include `trained_models/` in your repository or run the training script as part of the deployment process.



## Adding New Models

1. Create a new model script in `model/` directory (e.g., `model/svm.py`):

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, ...

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    
    # ... evaluation logic ...
    
    return model_data, metrics
```

2. Update `train.py` to import and register the new model:

```python
from model import svm

# In train_model function:
match model_name:
    case 'svm':
        return svm.train_and_evaluate(X_train, y_train, X_test, y_test)
```

3. Train the new model:

```bash
python train.py --model svm
```

## Technologies Used

- **Python 3.x**
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization
- **XGBoost**: Gradient boosting framework

## License

This project is open source and available under the MIT License.



##  Contributing

Contributions, issues, and feature requests are welcome!

## Show your support

Give a star if this project helped you!
