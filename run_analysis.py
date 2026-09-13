"""
Complete Student Dropout Prediction Analysis

This script runs the entire analysis pipeline:
1. Loads and preprocesses data
2. Performs EDA and saves visualizations
3. Engineers features
4. Trains multiple models
5. Evaluates and compares models
6. Saves model artifacts and results
7. Generates summary report

Usage:
    python run_analysis.py

Requirements:
    - Place Dropout.xlsx in data/ directory
    - Install requirements: pip install -r requirements.txt

Author: [Your Name]
Date: January 2026
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, accuracy_score
)
import joblib

# Import utility functions
from utils import (
    load_data, clean_column_names, create_binary_target,
    identify_feature_types, detect_outliers_iqr, remove_outliers,
    engineer_features, prepare_features_for_modeling,
    plot_target_distribution, plot_correlation_heatmap,
    evaluate_model, plot_roc_curve, plot_feature_importance_logreg
)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*80)
print("STUDENT DROPOUT PREDICTION - COMPLETE ANALYSIS")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ==============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("="*80)

try:
    df_raw = load_data('data/Dropout.xlsx')
except FileNotFoundError:
    print("\nERROR: data/Dropout.xlsx not found!")
    print("\nPlease place your Dropout.xlsx file in the data/ directory.")
    print("Then run this script again: python run_analysis.py")
    sys.exit(1)

# Clean column names
df = clean_column_names(df_raw)
print("Column names cleaned")

# Create binary target
df = create_binary_target(df)
print("Binary target created")

# Identify feature types
num_features, cat_features = identify_feature_types(df)
print(f"Identified {len(num_features)} numerical and {len(cat_features)} categorical features")

# Handle outliers
outlier_check_features = [
    col for col in [
        'previous_qualification_grade', 'admission_grade', 'age_at_enrollment',
        'curricular_units_1st_sem_grade', 'curricular_units_2nd_sem_grade',
        'unemployment_rate', 'inflation_rate', 'gdp'
    ] if col in df.columns
]

outlier_stats = detect_outliers_iqr(df, outlier_check_features, threshold=3)
df_clean = remove_outliers(df, outlier_stats)

print(f"Removed {len(df) - len(df_clean)} outliers ({((len(df) - len(df_clean))/len(df)*100):.2f}%)")
print(f"Final dataset: {len(df_clean)} rows × {df_clean.shape[1]} columns")

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# ==============================================================================
print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Save target distribution
plt.figure(figsize=(14, 5))
dropout_counts = df_clean['dropout'].value_counts()
colors = ['#2ecc71', '#e74c3c']

plt.subplot(1, 2, 1)
plt.bar(['No Dropout', 'Dropout'], dropout_counts.values, color=colors)
plt.ylabel('Count', fontsize=12)
plt.title('Dropout Distribution (Count)', fontsize=14, fontweight='bold')
for i, v in enumerate(dropout_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold', fontsize=11)

plt.subplot(1, 2, 2)
dropout_pct = df_clean['dropout'].value_counts(normalize=True) * 100
plt.bar(['No Dropout', 'Dropout'], dropout_pct.values, color=colors)
plt.ylabel('Percentage (%)', fontsize=12)
plt.title('Dropout Distribution (Percentage)', fontsize=14, fontweight='bold')
for i, v in enumerate(dropout_pct.values):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: visualizations/01_target_distribution.png")

# Key numerical features analysis
key_numerical = [
    col for col in [
        'age_at_enrollment', 'admission_grade', 'previous_qualification_grade',
        'curricular_units_1st_sem_grade', 'curricular_units_2nd_sem_grade',
        'unemployment_rate', 'inflation_rate', 'gdp'
    ] if col in df_clean.columns
]

# Correlation heatmap
if key_numerical:
    correlation_matrix = df_clean[key_numerical + ['dropout']].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}
    )
    plt.title('Correlation Matrix - Key Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: visualizations/02_correlation_heatmap.png")

# Boxplots by target
interesting_features = [
    col for col in [
        'curricular_units_1st_sem_grade',
        'curricular_units_2nd_sem_grade',
        'admission_grade',
        'age_at_enrollment',
        'unemployment_rate'
    ] if col in df_clean.columns
]

if interesting_features:
    ncols = 3
    nrows = (len(interesting_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes

    for idx, col in enumerate(interesting_features):
        df_plot = df_clean[[col, 'dropout']].copy()
        df_plot['dropout'] = df_plot['dropout'].map({0: 'No Dropout', 1: 'Dropout'})

        sns.boxplot(data=df_plot, x='dropout', y=col, ax=axes[idx], palette=['#2ecc71', '#e74c3c'])
        axes[idx].set_title(col.replace('_', ' ').title(), fontweight='bold', fontsize=11)
        axes[idx].set_xlabel('')

    for idx in range(len(interesting_features), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('visualizations/03_feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: visualizations/03_feature_comparison.png")

# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

df_engineered = engineer_features(df_clean)
new_features = [col for col in df_engineered.columns if col not in df_clean.columns]
print(f"Created {len(new_features)} new features:")
for feat in new_features:
    print(f"  - {feat}")

# ==============================================================================
# 4. MODEL PREPARATION
# ==============================================================================
print("\n" + "="*80)
print("STEP 4: MODEL PREPARATION")
print("="*80)

X, y, feature_names, scale_features = prepare_features_for_modeling(df_engineered)
print(f"Feature matrix: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features to scale: {len(scale_features)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if scale_features:
    X_train_scaled[scale_features] = scaler.fit_transform(X_train[scale_features])
    X_test_scaled[scale_features] = scaler.transform(X_test[scale_features])
    print(f"Scaled {len(scale_features)} features")

# ==============================================================================
# 5. MODEL TRAINING AND EVALUATION
# ==============================================================================
print("\n" + "="*80)
print("STEP 5: MODEL TRAINING AND EVALUATION")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
}

results = {}
model_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    model_predictions[name] = {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")

# Results dataframe
results_df = pd.DataFrame(results).T
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(results_df.round(4))

# Save results
results_df.to_csv('results/model_comparison.csv')
print("\nSaved: results/model_comparison.csv")

# ==============================================================================
# 6. VISUALIZATIONS - MODEL PERFORMANCE
# ==============================================================================
print("\n" + "="*80)
print("STEP 6: GENERATING MODEL PERFORMANCE VISUALIZATIONS")
print("="*80)

# Best model determination
best_model_name = results_df['AUC-ROC'].idxmax()
best_model = models[best_model_name]
best_pred = model_predictions[best_model_name]['y_pred']
best_pred_proba = model_predictions[best_model_name]['y_pred_proba']

print(f"Best model: {best_model_name} (AUC-ROC: {results_df.loc[best_model_name, 'AUC-ROC']:.4f})")

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
cm_df = pd.DataFrame(
    cm,
    index=['Actual: No Dropout', 'Actual: Dropout'],
    columns=['Predicted: No Dropout', 'Predicted: Dropout']
)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, linewidths=2, annot_kws={'size': 14})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/04_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: visualizations/04_confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_pred_proba)
auc_score = roc_auc_score(y_test, best_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'{best_model_name} (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', label='Random Classifier (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Dropout Prediction', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/05_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: visualizations/05_roc_curve.png")

# Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar', ax=axes[0], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)

results_df['AUC-ROC'].plot(kind='barh', ax=axes[1], color='#9b59b6')
axes[1].set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('AUC-ROC Score', fontsize=12)
axes[1].set_xlim([0, 1])
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/06_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: visualizations/06_model_comparison.png")

# Feature Importance (for Logistic Regression)
if 'Logistic Regression' in models:
    logreg = models['Logistic Regression']
    coefficients = logreg.coef_[0]

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)

    top_features = feature_importance.head(20)

    plt.figure(figsize=(10, 8))
    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in top_features['coefficient']]
    plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title('Top 20 Most Important Features (Logistic Regression)', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/07_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: visualizations/07_feature_importance.png")

    # Save feature importance to CSV
    feature_importance.to_csv('results/feature_importance.csv', index=False)
    print("Saved: results/feature_importance.csv")

# ==============================================================================
# 7. SAVE MODEL ARTIFACTS
# ==============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING MODEL ARTIFACTS")
print("="*80)

joblib.dump(best_model, 'models/dropout_prediction_model.pkl')
print(f"Saved: models/dropout_prediction_model.pkl ({best_model_name})")

joblib.dump(scaler, 'models/feature_scaler.pkl')
print("Saved: models/feature_scaler.pkl")

joblib.dump(feature_names, 'models/feature_names.pkl')
print("Saved: models/feature_names.pkl")

joblib.dump(scale_features, 'models/scale_features.pkl')
print("Saved: models/scale_features.pkl")

# ==============================================================================
# 8. GENERATE SUMMARY REPORT
# ==============================================================================
print("\n" + "="*80)
print("STEP 8: GENERATING SUMMARY REPORT")
print("="*80)

summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'total_records': len(df_raw),
        'after_cleaning': len(df_clean),
        'features': len(feature_names),
        'dropout_count': int(df_clean['dropout'].sum()),
        'dropout_percentage': float(df_clean['dropout'].mean() * 100)
    },
    'best_model': {
        'name': best_model_name,
        'accuracy': float(results_df.loc[best_model_name, 'Accuracy']),
        'precision': float(results_df.loc[best_model_name, 'Precision']),
        'recall': float(results_df.loc[best_model_name, 'Recall']),
        'f1_score': float(results_df.loc[best_model_name, 'F1-Score']),
        'auc_roc': float(results_df.loc[best_model_name, 'AUC-ROC'])
    },
    'all_models': results_df.to_dict()
}

with open('results/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Saved: results/analysis_summary.json")

# Create markdown report
report_md = f"""# Student Dropout Prediction - Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary

- **Total Records**: {len(df_raw):,}
- **After Cleaning**: {len(df_clean):,}
- **Features Used**: {len(feature_names)}
- **Dropout Count**: {int(df_clean['dropout'].sum())} ({df_clean['dropout'].mean()*100:.2f}%)
- **No Dropout Count**: {int((df_clean['dropout']==0).sum())} ({(1-df_clean['dropout'].mean())*100:.2f}%)

## Model Performance

### Best Model: {best_model_name}

- **Accuracy**: {results_df.loc[best_model_name, 'Accuracy']:.4f} ({results_df.loc[best_model_name, 'Accuracy']*100:.2f}%)
- **Precision**: {results_df.loc[best_model_name, 'Precision']:.4f} ({results_df.loc[best_model_name, 'Precision']*100:.2f}%)
- **Recall**: {results_df.loc[best_model_name, 'Recall']:.4f} ({results_df.loc[best_model_name, 'Recall']*100:.2f}%)
- **F1-Score**: {results_df.loc[best_model_name, 'F1-Score']:.4f}
- **AUC-ROC**: {results_df.loc[best_model_name, 'AUC-ROC']:.4f}

### All Models Comparison

{results_df.round(4).to_markdown()}

## Key Insights

1. **Academic Performance**: First and second semester grades are strong predictors
2. **Financial Factors**: Scholarship status and debtor status influence dropout risk
3. **Early Detection**: Model enables identification of at-risk students after first semester

## Visualizations

All visualizations have been saved to the `visualizations/` directory:

- `01_target_distribution.png` - Dropout class distribution
- `02_correlation_heatmap.png` - Feature correlation matrix
- `03_feature_comparison.png` - Feature distributions by dropout status
- `04_confusion_matrix.png` - Model confusion matrix
- `05_roc_curve.png` - ROC curve analysis
- `06_model_comparison.png` - Model performance comparison
- `07_feature_importance.png` - Top 20 most important features

## Files Generated

- **Models**: `models/dropout_prediction_model.pkl`
- **Scaler**: `models/feature_scaler.pkl`
- **Results**: `results/model_comparison.csv`, `results/feature_importance.csv`
- **Summary**: `results/analysis_summary.json`

## Next Steps

1. Review visualizations for business insights
2. Update README.md with actual performance metrics
3. Test prediction script with new student data
4. Deploy model for institutional use
"""

with open('results/ANALYSIS_REPORT.md', 'w') as f:
    f.write(report_md)

print("Saved: results/ANALYSIS_REPORT.md")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nDataset: {len(df_clean):,} students analyzed")
print(f"Best Model: {best_model_name}")
print(f"AUC-ROC: {results_df.loc[best_model_name, 'AUC-ROC']:.4f}")
print(f"Visualizations: 7 charts saved to visualizations/")
print(f"Models saved to models/")
print(f"Reports saved to results/")
print(f"\nAll tasks completed successfully!")
print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
