"""
Utility functions for Customer Churn Prediction System
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import os
import logging
from datetime import datetime
from config import DIRS, LOGGING_CONFIG

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    os.makedirs(DIRS['logs'], exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        filename=LOGGING_CONFIG['filename'],
        filemode='a'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def create_directories():
    """Create necessary directories for the project"""
    for dir_name in DIRS.values():
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"Created directory: {dir_name}")

def load_model_artifacts():
    """Load all saved model artifacts"""
    try:
        model = joblib.load(os.path.join(DIRS['models'], 'churn_model.pkl'))
        scaler = joblib.load(os.path.join(DIRS['models'], 'scaler.pkl'))
        feature_columns = joblib.load(os.path.join(DIRS['models'], 'feature_columns.pkl'))
        
        with open(os.path.join(DIRS['models'], 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            
        logger.info("Model artifacts loaded successfully")
        return model, scaler, feature_columns, metadata, True
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None, None, False

def save_prediction_log(customer_data, prediction_result):
    """Log prediction for monitoring and analysis"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'customer_data': customer_data,
        'prediction': prediction_result,
        'model_version': '1.0'
    }
    
    log_file = os.path.join(DIRS['logs'], 'predictions.log')
    
    with open(log_file, 'a') as f:
        f.write(f"{log_entry}\n")
    
    logger.info(f"Prediction logged: {prediction_result}")

def calculate_business_metrics(predictions_df):
    """Calculate business impact metrics from predictions"""
    total_customers = len(predictions_df)
    high_risk_customers = len(predictions_df[predictions_df['risk_level'] == 'High'])
    medium_risk_customers = len(predictions_df[predictions_df['risk_level'] == 'Medium'])
    
    # Estimated monthly revenue
    avg_monthly_charges = predictions_df.get('MonthlyCharges', pd.Series([65])).mean()
    
    metrics = {
        'total_customers': total_customers,
        'high_risk_count': high_risk_customers,
        'medium_risk_count': medium_risk_customers,
        'high_risk_percentage': high_risk_customers / total_customers * 100,
        'estimated_monthly_risk': high_risk_customers * avg_monthly_charges,
        'estimated_annual_risk': high_risk_customers * avg_monthly_charges * 12,
        'potential_savings_20pct': high_risk_customers * 0.2 * avg_monthly_charges * 12
    }
    
    return metrics

def validate_input_data(customer_data):
    """Validate customer input data"""
    required_fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod']
    
    errors = []
    
    for field in required_fields:
        if field not in customer_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate numeric fields
    numeric_fields = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for field in numeric_fields:
        if field in customer_data:
            try:
                float(customer_data[field])
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value for {field}")
    
    return errors

def format_currency(amount):
    """Format currency for display"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format percentage for display"""
    return f"{value:.1%}"
