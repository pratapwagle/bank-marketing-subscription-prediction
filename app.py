"""
Bank Marketing Deposit Prediction - Streamlit Application

A web application for evaluating models in the
bank-marketing-subscription-prediction project. Features a sidebar-driven workflow with idle state.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, matthews_corrcoef
)
from data_pipeline import prepare_features_for_inference, get_feature_engineering_info

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Bank Marketing Deposit Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
/* ---- Theme-aware CSS custom properties ---- */
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fb;
    --bg-card: #ffffff;
    --bg-badge: #eef2ff;
    --text-primary: #1a1a2e;
    --text-secondary: #6b7280;
    --text-muted: #9ca3af;
    --text-button: #374151;
    --border-color: #e8ecf1;
    --border-button: #d1d5db;
    --bar-bg: #e5e7eb;
    --accent: #4361ee;
    --accent-light: #a5b4fc;
    --shadow: rgba(0,0,0,0.04);
    --idle-circle-fill: #f0f4ff;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #1e1e2e;
        --bg-secondary: #181825;
        --bg-card: #262637;
        --bg-badge: #2a2a4a;
        --text-primary: #e0e0ef;
        --text-secondary: #a0a0b8;
        --text-muted: #7a7a95;
        --text-button: #c0c0d5;
        --border-color: #3a3a50;
        --border-button: #4a4a60;
        --bar-bg: #3a3a50;
        --accent: #6880f0;
        --accent-light: #818cf8;
        --shadow: rgba(0,0,0,0.2);
        --idle-circle-fill: #2a2a4a;
    }
}

/* ---- Top Navbar ---- */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: var(--bg-primary);
    border-bottom: 2px solid var(--border-color);
    border-radius: 0.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px var(--shadow);
}
.navbar-brand {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.navbar-brand .accent { color: var(--accent); }
.navbar-info {
    font-size: 0.82rem;
    color: var(--text-secondary);
    text-align: right;
    line-height: 1.5;
}
.navbar-info strong { color: var(--text-primary); }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    min-width: 500px ;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 0.6rem;
    font-weight: 600;
    padding: 0.6rem 1rem;
    margin-bottom: 0.25rem;
    transition: all 0.15s ease;
}
section[data-testid="stSidebar"] .stDownloadButton > button {
    width: 100%;
    border-radius: 0.6rem;
    font-weight: 500;
    border: 1.5px solid var(--border-button);
    background: var(--bg-primary);
    color: var(--text-button);
    padding: 0.6rem 1rem;
    margin-bottom: 0.25rem;
}
section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* ---- Step labels ---- */
.step-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.step-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
}

/* ---- Idle state ---- */
.idle-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 55vh;
    text-align: center;
    color: var(--text-muted);
}
.idle-icon {
    width: 100px;
    height: 100px;
    margin-bottom: 1.5rem;
}
.idle-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.idle-desc {
    font-size: 0.95rem;
    color: var(--text-secondary);
    max-width: 420px;
    line-height: 1.6;
}
.idle-bars {
    display: flex;
    gap: 0.5rem;
    margin-top: 1.5rem;
}
.idle-bar {
    height: 6px;
    border-radius: 3px;
    background: var(--bar-bg);
}

/* ---- Results metrics cards ---- */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 0.75rem;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 1px 3px var(--shadow);
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 0.25rem;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
}

/* ---- Model info badge ---- */
.model-badge {
    display: inline-block;
    background: var(--bg-badge);
    color: var(--accent);
    font-weight: 600;
    padding: 0.35rem 0.85rem;
    border-radius: 2rem;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

REQUIRED_COLUMNS = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
    'loan', 'contact', 'month', 'campaign', 'pdays', 'previous', 'poutcome'
]

def validate_dataset(df):
    """
    Validate that the dataframe contains all required columns.
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        tuple: (is_valid, missing_columns)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


@st.cache_data
def load_csv_file(uploaded_file):
    """Load and cache uploaded dataset."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate schema
        is_valid, missing = validate_dataset(df)
        if not is_valid:
            st.error(
                f"**Invalid Dataset**: Missing required columns: {', '.join(missing)}.\n"
                "Please upload a valid Bank Marketing dataset."
            )
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


@st.cache_resource
def load_model(model_path):
    """Load and cache trained model."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def get_available_models(models_dir='trained_models'):
    """Get available trained models."""
    model_mapping = {
        'Logistic Regression': 'LogisticRegressionModel.pkl',
        'Decision Tree': 'DecisionTreeModel.pkl',
        'K-Nearest Neighbors': 'KNNModel.pkl',
        'Naive Bayes': 'NaiveBayesModel.pkl',
        'Random Forest': 'RandomForestModel.pkl',
        'XGBoost': 'XGBoostModel.pkl'
    }
    available = {}
    for display_name, filename in model_mapping.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            available[display_name] = filename
    return available


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba[:, 1]) if y_pred_proba is not None else None,
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Create confusion matrix heatmap with white background for visibility."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.set_facecolor('white')
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        cbar_kws={'label': 'Count'}, ax=ax,
        square=True, linewidths=1, linecolor='gray'
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(['No Deposit (0)', 'Deposit (1)'])
    ax.set_yticklabels(['No Deposit (0)', 'Deposit (1)'])
    plt.tight_layout()
    return fig


def display_classification_report(y_true, y_pred):
    """Display classification report as a formatted table."""
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(4)
    df_report.index = df_report.index.map({
        '0': 'No Deposit (0)',
        '1': 'Deposit (1)',
        'accuracy': 'Accuracy',
        'macro avg': 'Macro Average',
        'weighted avg': 'Weighted Average'
    })
    return df_report


def render_idle_state():
    """Render the idle/awaiting computation state in the main area."""
    # Add vertical spacing to center visually
    st.write("")
    st.write("")
    st.write("")

    # Center content using columns
    spacer_l, center, spacer_r = st.columns([1, 2, 1])

    with center:
        # SVG rendered via st.components (self-contained dark mode CSS for iframe)
        import streamlit.components.v1 as components
        idle_svg = """
        <style>
            :root {
                --circle-fill: #f0f4ff;
                --accent: #4361ee;
                --accent-light: #a5b4fc;
            }
            @media (prefers-color-scheme: dark) {
                :root {
                    --circle-fill: #2a2a4a;
                    --accent: #6880f0;
                    --accent-light: #818cf8;
                }
                body { background: transparent; }
            }
            body { margin: 0; background: transparent; }
        </style>
        <div style="text-align: center; padding: 2rem 0;">
            <svg width="100" height="100" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="60" cy="60" r="56" fill="var(--circle-fill)" stroke="var(--accent)" stroke-width="2"/>
              <rect x="30" y="55" width="6" height="20" rx="3" fill="var(--accent-light)"/>
              <rect x="42" y="40" width="6" height="40" rx="3" fill="#818cf8"/>
              <rect x="54" y="48" width="6" height="30" rx="3" fill="#6366f1"/>
              <rect x="66" y="35" width="6" height="50" rx="3" fill="#818cf8"/>
              <rect x="78" y="50" width="6" height="25" rx="3" fill="var(--accent-light)"/>
              <circle cx="95" cy="55" r="12" fill="var(--accent)" fill-opacity="0.15"/>
              <polygon points="92,50 92,60 100,55" fill="var(--accent)"/>
            </svg>
        </div>
        """
        components.html(idle_svg, height=140)

        # Title and description using CSS variable colors
        st.markdown(
            "<h2 style='text-align:center; color:var(--text-primary, #374151); font-weight:700; "
            "margin-bottom:0.5rem;'>Awaiting Computation</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center; color:var(--text-secondary, #6b7280); font-size:0.95rem; "
            "line-height:1.6; max-width:420px; margin:0 auto;'>"
            "The results engine is idle. Configure your <strong>Test Dataset</strong> "
            "and <strong>Classification Model</strong> in the left pane to initialize analysis."
            "</p>",
            unsafe_allow_html=True
        )

        # Placeholder bars
        st.write("")
        bar_cols = st.columns([3, 2, 4, 1.5, 3.5, 2.5])
        for col in bar_cols:
            col.progress(0)


# ============================================================================
# COMPARISON REPORT
# ============================================================================

def render_comparison_report(model_mapping):
    """Run inference on every available model and render a comparison report."""

    # Feature engineering (compute once, reuse)
    if st.session_state.df_engineered is None:
        with st.spinner("Applying feature engineering..."):
            st.session_state.df_engineered = prepare_features_for_inference(
                st.session_state.df_raw
            )

    df = st.session_state.df_engineered

    # Check for target column
    has_target = 'deposit' in df.columns
    if not has_target:
        st.error(
            "The loaded dataset does not contain a **deposit** target column. "
            "Model comparison requires ground-truth labels."
        )
        return

    X = df.drop('deposit', axis=1)
    y_true = df['deposit']

    st.markdown(
        '<div class="section-header">Model Performance Comparison</div>',
        unsafe_allow_html=True,
    )

    # ------ Run inference on every model ------
    all_metrics = {}
    all_cm = {}

    for display_name, filename in model_mapping.items():
        model_path = os.path.join('trained_models', filename)
        model_data = load_model(model_path)
        if model_data is None:
            st.warning(f"Skipping **{display_name}** — could not load model.")
            continue

        model = model_data['model']
        scaler = model_data.get('scaler')

        if scaler:
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)
        else:
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        all_metrics[display_name] = {
            'Accuracy': metrics['accuracy'],
            'AUC Score': metrics['auc'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'MCC': metrics['mcc'],
        }
        all_cm[display_name] = np.array(metrics['confusion_matrix'])

    if not all_metrics:
        st.error("No models could be loaded for comparison.")
        return

    # ------ Metrics comparison table ------
    df_comparison = pd.DataFrame(all_metrics).T.round(4)
    styled = (
        df_comparison.style
        .background_gradient(cmap='RdYlGn', axis=0)
        .map(lambda _: 'color: #1a1a2e')  # dark text for readability on colored cells
    )
    st.dataframe(styled, use_container_width=True)

    # ------ Confusion matrices (two per row) ------
    st.markdown("---")
    st.markdown(
        '<div class="section-header">Confusion Matrices</div>',
        unsafe_allow_html=True,
    )

    model_names = list(all_cm.keys())
    for i in range(0, len(model_names), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(model_names):
                name = model_names[idx]
                with col:
                    fig = plot_confusion_matrix(all_cm[name], title=name)
                    st.pyplot(fig)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_engineered' not in st.session_state:
        st.session_state.df_engineered = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'classification_run' not in st.session_state:
        st.session_state.classification_run = False
    if 'comparison_run' not in st.session_state:
        st.session_state.comparison_run = False
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'model_select_key' not in st.session_state:
        st.session_state.model_select_key = 0

    # ------------------------------------------------------------------
    # Top Navbar
    # ------------------------------------------------------------------
    st.markdown("""
    <div class="navbar">
        <div class="navbar-brand">
            <span class="accent">Bank Marketing</span> Deposit Prediction
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================================================================
    # SIDEBAR
    # ==================================================================
    with st.sidebar:
        # ------ Step 1: Load Test Dataset ------
        st.markdown(
            '<div class="step-label">DATA SOURCE: '
            '<a href="https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset" '
            'target="_blank" style="color:#4361ee; text-decoration:none;">BANK MARKETING DATASET - KAGGLE</a>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="step-title">Step 1: Load Test Dataset</div>', unsafe_allow_html=True)

        # Import Test Dataset
        import_clicked = st.button("Import Test Dataset", type="primary", use_container_width=True)

        # Download Test Dataset
        test_file_path = 'data/bank_test.csv'
        if os.path.exists(test_file_path):
            with open(test_file_path, 'rb') as f:
                test_csv_bytes = f.read()
            st.download_button(
                label="Download Test Dataset",
                data=test_csv_bytes,
                file_name="bank_test.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Browse and Upload File
        uploaded_file = st.file_uploader(
            "Browse and Upload File",
            type=['csv'],
            label_visibility="collapsed"
        )
        # Show a subtle label for the uploader

        st.markdown("---")

        # ------ Step 2: Model Selection ------
        st.markdown('<div class="step-title">Step 2: Model Selection</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-label">APPROPRIATE CLASSIFICATION MODEL</div>', unsafe_allow_html=True)

        model_mapping = get_available_models()
        model_names = list(model_mapping.keys())

        if model_names:
            selected_display_name = st.selectbox(
                "Select a model...",
                options=model_names,
                index=None,
                placeholder="Select a model...",
                label_visibility="collapsed",
                key=f"model_select_{st.session_state.model_select_key}"
            )
        else:
            st.warning("No trained models found.")
            selected_display_name = None

        st.markdown("---")

        # ------ Step 3: Compare Model Performance Button ------
        st.markdown('<div class="step-title">Step 3: Compare All Models</div>', unsafe_allow_html=True)
        compare_clicked = st.button(
            "Compare Model Performance",
            type="primary",
            use_container_width=True
        )

    # ==================================================================
    # PROCESS SIDEBAR INPUTS
    # ==================================================================

    # Handle Import button
    if import_clicked:
        try:
            df_imported = pd.read_csv('data/bank_test.csv')
            
            # Validate schema
            is_valid, missing = validate_dataset(df_imported)
            if not is_valid:
                st.error(
                    f"**Invalid Dataset**: Missing required columns: {', '.join(missing)}.\n"
                    "Please check the data/bank_test.csv file."
                )
            else:
                st.session_state.df_raw = df_imported
                st.session_state.df_engineered = None
                st.session_state.data_loaded = True
                st.session_state.classification_run = False
                st.session_state.comparison_run = False
                st.session_state.uploaded_file_name = None
                st.session_state.model_select_key += 1
                st.rerun()
                
        except FileNotFoundError:
            st.error("data/bank_test.csv not found. Train a model first: `python train.py --model all --save`")
            st.stop()

    # Handle Upload
    if uploaded_file is not None:
        # Only reset when a genuinely new file is uploaded
        if uploaded_file.name != st.session_state.uploaded_file_name:
            st.session_state.df_raw = load_csv_file(uploaded_file)
            if st.session_state.df_raw is not None:
                st.session_state.df_engineered = None
                st.session_state.data_loaded = True
                st.session_state.classification_run = False
                st.session_state.comparison_run = False
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.model_select_key += 1
                st.rerun()
    elif st.session_state.uploaded_file_name is not None:
        # User clicked X to remove the uploaded file — full reset
        st.session_state.df_raw = None
        st.session_state.df_engineered = None
        st.session_state.data_loaded = False
        st.session_state.classification_run = False
        st.session_state.comparison_run = False
        st.session_state.uploaded_file_name = None
        st.session_state.model_select_key += 1
        st.rerun()

    # Handle Model Selection (auto-classify when data is loaded and model is picked)
    if selected_display_name is not None and not st.session_state.data_loaded:
        st.error("Please load a dataset first (Step 1).")
        st.stop()

    if st.session_state.data_loaded and selected_display_name is not None:
        st.session_state.classification_run = True
        st.session_state.comparison_run = False
    else:
        st.session_state.classification_run = False

    # Handle Compare Model Performance
    if compare_clicked:
        if not st.session_state.data_loaded:
            st.error("Please load a dataset first (Step 1).")
            st.stop()
        st.session_state.df_engineered = None
        st.session_state.comparison_run = True
        st.session_state.classification_run = False
        st.session_state.model_select_key += 1
        st.rerun()

    # ==================================================================
    # MAIN CONTENT AREA
    # ==================================================================

    if st.session_state.comparison_run:
        render_comparison_report(model_mapping)
        st.stop()

    if not st.session_state.classification_run:
        # Show data load confirmation if loaded
        if st.session_state.data_loaded and st.session_state.df_raw is not None:
            st.success(
                f"Dataset loaded: **{len(st.session_state.df_raw)}** samples, "
                f"**{len(st.session_state.df_raw.columns)}** columns"
            )

        # Idle state
        render_idle_state()
        st.stop()

    # ------------------------------------------------------------------
    # CLASSIFICATION RESULTS
    # ------------------------------------------------------------------

    # Apply feature engineering
    if st.session_state.df_engineered is None:
        with st.spinner("Applying feature engineering..."):
            st.session_state.df_engineered = prepare_features_for_inference(st.session_state.df_raw)

    df = st.session_state.df_engineered

    # Show feature engineering info
    fe_info = get_feature_engineering_info(st.session_state.df_raw, df)
    with st.expander("View Feature Engineering Details"):
        st.write(f"**Original Features:** {fe_info['original_features']}")
        st.write(f"**Total Features:** {fe_info['total_features']}")
        for step in fe_info['engineering_steps']:
            st.write(f"- {step}")

    # Load model
    selected_model = model_mapping[selected_display_name]
    model_path = os.path.join('trained_models', selected_model)
    model_data = load_model(model_path)

    if model_data is None:
        st.error("Failed to load model.")
        st.stop()

    st.markdown(f'<div class="model-badge">{selected_display_name}</div>', unsafe_allow_html=True)

    # Check for target column
    has_target = 'deposit' in df.columns

    try:
        # Prepare features
        if has_target:
            X = df.drop('deposit', axis=1)
            y_true = df['deposit']
        else:
            X = df
            y_true = None

        # Predict
        model = model_data['model']
        scaler = model_data.get('scaler')

        if scaler:
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)
        else:
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)

        # ---- Dataset Overview ----
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(df))
        feature_count = len(df.columns) - 1 if has_target else len(df.columns)
        col2.metric("Features", feature_count)
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("View Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)

        # ---- Predictions ----
        st.markdown('<div class="section-header">Model Predictions</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted: No Deposit (0)", int((y_pred == 0).sum()))
        with col2:
            st.metric("Predicted: Deposit (1)", int((y_pred == 1).sum()))

        df_results = df.copy()
        df_results['Predicted'] = y_pred
        df_results['Probability_Deposit'] = y_pred_proba[:, 1]

        with st.expander("View Predictions"):
            st.dataframe(df_results, use_container_width=True)
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

        # ---- Evaluation Metrics (if target exists) ----
        if has_target:
            st.markdown("---")
            st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)

            metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("AUC", f"{metrics['auc']:.4f}")
            col3.metric("Precision", f"{metrics['precision']:.4f}")
            col4.metric("Recall", f"{metrics['recall']:.4f}")
            col5.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            col6.metric("MCC", f"{metrics['mcc']:.4f}")

            # Confusion Matrix
            st.markdown("---")
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)

            cm = np.array(metrics['confusion_matrix'])
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = plot_confusion_matrix(cm)
                st.pyplot(fig)

            with col2:
                st.write("**Matrix Interpretation:**")
                st.write(f"- **True Negatives (TN):** {cm[0][0]}")
                st.write(f"- **False Positives (FP):** {cm[0][1]}")
                st.write(f"- **False Negatives (FN):** {cm[1][0]}")
                st.write(f"- **True Positives (TP):** {cm[1][1]}")

                total = cm.sum()
                correct = cm[0][0] + cm[1][1]
                st.write(f"\n**Performance:**")
                st.write(f"- Correct: {correct}/{total} ({correct/total*100:.1f}%)")
                st.write(f"- Incorrect: {total-correct}/{total} ({(total-correct)/total*100:.1f}%)")

            # Classification Report
            st.markdown("---")
            st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)

            df_report = display_classification_report(y_true, y_pred)
            st.dataframe(
                df_report.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
                use_container_width=True
            )

            with st.expander("How to interpret these metrics"):
                st.write("""
                **Precision:** Of all positive predictions, how many were correct?
                - High precision = Few false positives

                **Recall (Sensitivity):** Of all actual positives, how many did we find?
                - High recall = Few false negatives

                **F1-Score:** Harmonic mean of precision and recall

                **MCC:** Matthews Correlation Coefficient (-1 to 1, 1 = perfect)

                **Support:** Number of actual occurrences of each class
                """)

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.exception(e)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
