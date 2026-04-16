"""
Main Training Script for bank-marketing-subscription-prediction.

This script orchestrates the entire training pipeline:
1. Load and prepare data using data_pipeline.py
2. Train the selected model using match statements
3. Evaluate model performance
4. Save the trained model

Dataset: Bank Marketing (data/bank.csv)
Target: deposit (yes/no -> 1/0)
"""

import argparse
import sys
import os
from datetime import datetime
import json
import pickle

# Import data pipeline functions
from data_pipeline import load_data, feature_engineering, split_data, save_test_data, split_raw_data

# Import model training functions
from model import logistic_regression
from model import decision_tree
from model import knn
from model import naive_bayes
from model import random_forest
from model import xgboost


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def prepare_data(data_path='data/bank.csv', test_size=0.2, random_state=42):
    """
    Prepare data using the data pipeline.
    
    Flow: Split RAW data first, save raw test data, then apply feature engineering
    with train-only fitting for all transforms.
    
    Args:
        data_path (str): Path to the data file
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - feature-engineered data
    """
    print("=" * 70)
    print("DATA PREPARATION PIPELINE")
    print("=" * 70)
    
    # Load RAW data
    print("\n[Step 1/4] Loading raw data...")
    df = load_data(data_path)
    
    # Split RAW data FIRST (before feature engineering)
    print("\n[Step 2/4] Splitting raw data...")
    df_train_raw, df_test_raw = split_raw_data(
        df, test_size=test_size, random_state=random_state
    )
    
    # Save RAW test data
    print("\n[Step 3/4] Saving raw test data...")
    save_test_data(df_test_raw, 'data/bank_test.csv')
    
    # Apply feature engineering: fit on TRAIN, transform both
    print("\n[Step 4/4] Applying feature engineering...")
    print("  - Engineering training data (fit=True)...")
    df_train_engineered = feature_engineering(df_train_raw, fit=True)
    print("  - Engineering test data (fit=False, using train params)...")
    df_test_engineered = feature_engineering(df_test_raw, fit=False)
    
    # Prepare features and targets
    X_train = df_train_engineered.drop('deposit', axis=1)
    y_train = df_train_engineered['deposit']
    X_test = df_test_engineered.drop('deposit', axis=1)
    y_test = df_test_engineered['deposit']
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETED")
    print("=" * 70)
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(model_name, X_train, y_train, X_test, y_test):
    """
    Train the selected model using match statement.
    
    Args:
        model_name (str): Name of the model to train
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (model_data, metrics)
            - model_data: Dictionary containing model and metadata
            - metrics: Dictionary with 6 mandatory metrics
    """
    match model_name:
        case 'logistic_regression':
            return logistic_regression.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        case 'decision_tree':
            return decision_tree.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        case 'knn':
            return knn.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        case 'naive_bayes':
            return naive_bayes.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        case 'random_forest':
            return random_forest.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        case 'xgboost':
            return xgboost.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        case _:
            available_models = [
                'logistic_regression', 'decision_tree', 'knn', 
                'naive_bayes', 'random_forest', 'xgboost'
            ]
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {', '.join(available_models)}"
            )


def save_model_and_metrics(model_data, metrics, output_dir='trained_models'):
    """
    Save the trained model and its metrics.
    
    Uses fixed filenames (without timestamps) so new training overwrites old models.
    
    Args:
        model_data (dict): Model data dictionary
        metrics (dict): Evaluation metrics
        output_dir (str): Directory to save the model
        
    Returns:
        tuple: (model_path, metrics_path)
    """
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use fixed filename (no timestamp) - will overwrite existing
    model_filename = f"{model_data['model_name']}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {model_path}")
    
    # Save metrics with same fixed naming
    metrics_filename = f"{model_data['model_name']}_metrics.json"
    metrics_path = os.path.join(output_dir, metrics_filename)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("SAVING COMPLETED")
    print("=" * 70)
    
    return model_path, metrics_path


def main(args):
    """
    Main training pipeline.
    
    Args:
        args: Command-line arguments
    """
    print("\n" + "=" * 70)
    print("BANK MARKETING CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Test Size: {args.test_size}")
    print(f"Random State: {args.random_state}")
    
    try:
        # Step 1: Prepare data (once for all models)
        X_train, X_test, y_train, y_test = prepare_data(
            data_path=args.data,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        models_to_train = []
        if args.model == 'all':
            models_to_train = [
                'logistic_regression', 'decision_tree', 'knn', 
                'naive_bayes', 'random_forest', 'xgboost'
            ]
            print(f"\nTraining ALL {len(models_to_train)} models sequentially...")
        else:
            models_to_train = [args.model]
            
        # Store results for summary
        all_results = []
        
        for model_name in models_to_train:
            print(f"\n\n{'='*30} TRAINING: {model_name} {'='*30}")
            
            # Step 2: Train model
            model_data, metrics = train_model(
                model_name, 
                X_train, y_train, 
                X_test, y_test
            )
            
            # Step 3: Save model and metrics always for 'all' or if args.save is True
            if args.save:
                model_path, metrics_path = save_model_and_metrics(
                    model_data, metrics, output_dir=args.output_dir
                )
            
            # Collect results
            all_results.append({
                'model': model_data['model_name'],
                'accuracy': metrics['accuracy'],
                'auc': metrics['auc'],
                'f1': metrics['f1_score'],
                'mcc': metrics['mcc'],
                'confusion_matrix': metrics['confusion_matrix'],
                'path': model_path if args.save else "Not saved"
            })
            
        # Print Final Summary if multiple models were trained
        if len(all_results) > 1:
            print("\n" + "=" * 80)
            print("FINAL MODEL COMPARISON SUMMARY")
            print("=" * 80)
            print(f"{'Model Name':<25} {'Accuracy':<10} {'AUC':<10} {'F1 Score':<10} {'MCC':<10}")
            print("-" * 80)
            
            for res in all_results:
                print(f"{res['model']:<25} {res['accuracy']:.4f}     {res['auc']:.4f}     {res['f1']:.4f}     {res['mcc']:.4f}")
            
            print("=" * 80)
            
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        return all_results
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def list_available_models():
    """Display all available models."""
    print("\nAvailable Models:")
    print("-" * 60)
    models = [
        ('logistic_regression', 'Logistic Regression'),
        ('decision_tree', 'Decision Tree Classifier'),
        ('knn', 'K-Nearest Neighbors Classifier'),
        ('naive_bayes', 'Naive Bayes Classifier'),
        ('random_forest', 'Random Forest Classifier (Ensemble)'),
        ('xgboost', 'XGBoost Classifier (Ensemble)')
    ]
    for i, (name, description) in enumerate(models, 1):
        print(f"{i}. {name:<25} - {description}")
    print("-" * 60)
    print("\nAll models return 6 metrics:")
    print("  * Accuracy  * AUC  * Precision  * Recall  * F1 Score  * MCC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models for bank marketing deposit prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model logistic_regression
  python train.py --model decision_tree --test-size 0.3
  python train.py --model random_forest
  python train.py --model knn
  python train.py --list-models
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='logistic_regression',
        help='Model to train (default: logistic_regression)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/bank.csv',
        help='Path to the data file (default: data/bank.csv)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Save the trained model (default: True)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_false',
        dest='save',
        help='Do not save the trained model'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trained_models',
        help='Directory to save models (default: trained_models)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list models command
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    # Run main training pipeline
    main(args)
