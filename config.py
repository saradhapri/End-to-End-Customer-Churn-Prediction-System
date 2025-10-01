"""
Configuration file for Customer Churn Prediction System
"""

import os

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'scoring_metric': 'roc_auc'
}

# Feature Engineering
FEATURE_CONFIG = {
    'numerical_features': ['tenure', 'MonthlyCharges', 'TotalCharges'],
    'categorical_features': ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                           'PhoneService', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaperlessBilling', 'PaymentMethod'],
    'target_column': 'Churn',
    'id_column': 'customerID'
}

# Directories
DIRS = {
    'models': 'models',
    'plots': 'plots',
    'data': 'data',
    'logs': 'logs'
}

# Business Metrics
BUSINESS_CONFIG = {
    'avg_customer_lifetime_months': 24,
    'retention_cost_factor': 0.1,
    'acquisition_cost': 200,
    'target_churn_reduction': 0.2
}

# Model Thresholds
THRESHOLDS = {
    'high_risk': 0.7,
    'medium_risk': 0.3,
    'low_risk': 0.0
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'logs/churn_prediction.log'
}
