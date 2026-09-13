"""
Student Dropout Risk Prediction Script

This script loads a trained model and predicts dropout risk for new students.

Usage:
    python predict.py --student_file new_students.csv
    python predict.py --interactive

Author: [Your Name]
Date: [Current Date]
"""

import argparse
import pandas as pd
import joblib
import sys
from pathlib import Path


def load_model_artifacts(model_dir='models'):
    """
    Load trained model, scaler, and feature names.

    Parameters:
    -----------
    model_dir : str
        Directory containing model artifacts

    Returns:
    --------
    tuple
        (model, scaler, feature_names)
    """
    model_path = Path(model_dir) / 'dropout_prediction_model.pkl'
    scaler_path = Path(model_dir) / 'feature_scaler.pkl'
    features_path = Path(model_dir) / 'feature_names.pkl'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    feature_names = joblib.load(features_path) if features_path.exists() else None

    return model, scaler, feature_names


def predict_single_student(student_data, model, scaler=None, feature_names=None):
    """
    Predict dropout risk for a single student.

    Parameters:
    -----------
    student_data : dict or pd.DataFrame
        Student features
    model : sklearn model
        Trained classification model
    scaler : sklearn scaler, optional
        Fitted scaler for feature normalization
    feature_names : list, optional
        Expected feature names

    Returns:
    --------
    dict
        Prediction results
    """
    # Convert to DataFrame if dict
    if isinstance(student_data, dict):
        student_data = pd.DataFrame([student_data])

    # Ensure all features are present
    if feature_names:
        for feature in feature_names:
            if feature not in student_data.columns:
                student_data[feature] = 0
        student_data = student_data[feature_names]

    # Scale if scaler provided
    if scaler is not None:
        # Identify which features to scale (this should match training)
        scale_features = [
            col for col in student_data.columns if col in [
                'age_at_enrollment', 'admission_grade', 'previous_qualification_grade',
                'curricular_units_1st_sem_grade', 'curricular_units_2nd_sem_grade',
                'unemployment_rate', 'inflation_rate', 'gdp'
            ]
        ]
        if scale_features:
            student_data[scale_features] = scaler.transform(student_data[scale_features])

    # Predict
    prediction = model.predict(student_data)[0]
    probability = model.predict_proba(student_data)[0][1]

    # Determine risk level
    if probability < 0.3:
        risk_level = 'Low Risk'
        recommendation = 'Continue monitoring progress'
    elif probability < 0.6:
        risk_level = 'Medium Risk'
        recommendation = 'Provide academic support and monitor closely'
    else:
        risk_level = 'High Risk'
        recommendation = 'Immediate intervention recommended - contact advisor'

    return {
        'prediction': 'At Risk of Dropout' if prediction == 1 else 'Not At Risk',
        'dropout_probability': probability,
        'dropout_probability_pct': f"{probability:.1%}",
        'risk_level': risk_level,
        'recommendation': recommendation
    }


def predict_batch(student_file, model, scaler=None, feature_names=None, output_file=None):
    """
    Predict dropout risk for multiple students from a file.

    Parameters:
    -----------
    student_file : str
        Path to CSV file with student data
    model : sklearn model
        Trained classification model
    scaler : sklearn scaler, optional
        Fitted scaler
    feature_names : list, optional
        Expected feature names
    output_file : str, optional
        Path to save predictions
    """
    # Load student data
    students_df = pd.read_csv(student_file)
    print(f"Loaded {len(students_df)} students from {student_file}")

    # Predict for each student
    predictions = []
    for idx, row in students_df.iterrows():
        result = predict_single_student(row.to_dict(), model, scaler, feature_names)
        predictions.append(result)

    # Add predictions to dataframe
    pred_df = pd.DataFrame(predictions)
    result_df = pd.concat([students_df, pred_df], axis=1)

    # Save if output file specified
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    # Print summary
    print("\nPrediction Summary:")
    print("="*60)
    print(f"Total students: {len(result_df)}")
    print(f"At Risk: {(result_df['prediction'] == 'At Risk of Dropout').sum()}")
    print(f"Not At Risk: {(result_df['prediction'] == 'Not At Risk').sum()}")
    print("\nRisk Level Distribution:")
    print(result_df['risk_level'].value_counts())

    return result_df


def interactive_mode(model, scaler, feature_names):
    """
    Interactive prediction mode for single student input.
    """
    print("\n" + "="*60)
    print("Interactive Student Dropout Risk Assessment")
    print("="*60)
    print("\nEnter student information (press Enter to use default value of 0):\n")

    student_data = {}

    # Key features to collect
    key_features = {
        'age_at_enrollment': 'Age at enrollment',
        'admission_grade': 'Admission grade (0-200)',
        'previous_qualification_grade': 'Previous qualification grade (0-200)',
        'curricular_units_1st_sem_grade': 'First semester grade (0-20)',
        'curricular_units_2nd_sem_grade': 'Second semester grade (0-20)',
        'scholarship_holder': 'Scholarship holder (1=Yes, 0=No)',
        'debtor': 'Debtor status (1=Yes, 0=No)',
        'tuition_fees_up_to_date': 'Tuition fees up to date (1=Yes, 0=No)',
        'gender': 'Gender (1=Male, 0=Female)',
    }

    for feature, description in key_features.items():
        value = input(f"{description}: ").strip()
        student_data[feature] = float(value) if value else 0

    # Make prediction
    result = predict_single_student(student_data, model, scaler, feature_names)

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Prediction:          {result['prediction']}")
    print(f"Dropout Probability: {result['dropout_probability_pct']}")
    print(f"Risk Level:          {result['risk_level']}")
    print(f"Recommendation:      {result['recommendation']}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Predict student dropout risk')
    parser.add_argument('--student_file', type=str, help='CSV file with student data')
    parser.add_argument('--output_file', type=str, help='Output file for predictions')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory with model artifacts')

    args = parser.parse_args()

    try:
        # Load model
        print("Loading model...")
        model, scaler, feature_names = load_model_artifacts(args.model_dir)
        print("Model loaded successfully\n")

        # Choose mode
        if args.interactive:
            interactive_mode(model, scaler, feature_names)
        elif args.student_file:
            predict_batch(args.student_file, model, scaler, feature_names, args.output_file)
        else:
            print("Error: Please specify --student_file or --interactive mode")
            parser.print_help()
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
