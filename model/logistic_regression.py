"""
Logistic Regression Model Training and Evaluation.

This module provides training and evaluation for Logistic Regression classifier
with GridSearchCV hyperparameter tuning.
"""

import time
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef
)


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Logistic Regression model with GridSearchCV tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (model_data, metrics)
            - model_data: Dictionary containing model, scaler, and metadata
            - metrics: Dictionary with 6 mandatory metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
    """
    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION MODEL (GridSearchCV)")
    print("=" * 70)
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    start_time = time.time()
    
    # Scale features (LR benefits from scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l1', 'l2'],
        'class_weight': [None, 'balanced'],
        'max_iter': [1000],
    }
    
    # Note: lbfgs does not support l1 penalty — GridSearchCV will skip
    # invalid combos via error_score='raise' being default; we handle
    # this by filtering valid combos only or catching errors.
    # Use error_score=0 to silently skip invalid combos.
    
    base_model = LogisticRegression(random_state=42)
    
    print("\nRunning GridSearchCV (5-fold stratified CV, scoring=f1)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0,
        error_score=0,  # Skip invalid param combos (e.g. lbfgs + l1)
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    training_time = time.time() - start_time
    
    print(f"GridSearchCV completed in {training_time:.2f} seconds")
    print(f"Best CV F1 Score: {best_cv_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Make predictions with best model
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    
    # Calculate 6 mandatory metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Display metrics
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {metrics['accuracy']:.4f}")
    print(f"{'AUC':<20} {metrics['auc']:.4f}")
    print(f"{'Precision':<20} {metrics['precision']:.4f}")
    print(f"{'Recall':<20} {metrics['recall']:.4f}")
    print(f"{'F1 Score':<20} {metrics['f1_score']:.4f}")
    print(f"{'MCC':<20} {metrics['mcc']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm[0][0]:<4} FP={cm[0][1]:<4}]")
    print(f"   [FN={cm[1][0]:<4} TP={cm[1][1]:<4}]]")
    print("=" * 70)
    
    # Package model data
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'model_name': 'LogisticRegressionModel',
        'trained': True,
        'training_time': training_time,
        'metrics': metrics,
        'best_params': best_params,
        'best_cv_f1': best_cv_score,
        'timestamp': datetime.now().isoformat()
    }
    
    return model_data, metrics
