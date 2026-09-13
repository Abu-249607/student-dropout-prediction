# Student Dropout Prediction - Analysis Report

**Generated**: 2026-09-13 17:19:48

## Dataset Summary

- **Total Records**: 4,424
- **After Cleaning**: 3,403
- **Features Used**: 41
- **Dropout Count**: 633 (18.60%)
- **No Dropout Count**: 2770 (81.40%)

## Model Performance

### Best Model: Gradient Boosting

- **Accuracy**: 0.8825 (88.25%)
- **Precision**: 0.7640 (76.40%)
- **Recall**: 0.5354 (53.54%)
- **F1-Score**: 0.6296
- **AUC-ROC**: 0.8688

### All Models Comparison

|                     |   Accuracy |   Precision |   Recall |   F1-Score |   AUC-ROC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |     0.8737 |      0.7356 |   0.5039 |     0.5981 |    0.8486 |
| Decision Tree       |     0.8355 |      0.5591 |   0.5591 |     0.5591 |    0.7355 |
| Random Forest       |     0.8825 |      0.7901 |   0.5039 |     0.6154 |    0.866  |
| Gradient Boosting   |     0.8825 |      0.764  |   0.5354 |     0.6296 |    0.8688 |

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
