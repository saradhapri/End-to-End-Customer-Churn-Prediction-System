
# Data validation and quality checks for Customer Churn Prediction

import pandas as pd
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation and quality assessment"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.validation_report = {}
    
    def load_and_validate(self):
        """Load data and perform comprehensive validation"""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False, {}
        
        # Perform all validations
        self.check_data_structure()
        self.check_missing_values()
        self.check_data_types()
        self.check_value_ranges()
        self.check_categorical_values()
        self.check_duplicates()
        self.check_business_logic()
        
        # Generate summary report
        self.generate_summary_report()
        
        return True, self.validation_report
    
    def check_data_structure(self):
        """Check basic data structure"""
        expected_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        
        missing_columns = set(expected_columns) - set(self.df.columns)
        extra_columns = set(self.df.columns) - set(expected_columns)
        
        self.validation_report['structure'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns),
            'structure_valid': len(missing_columns) == 0
        }
    
    def check_missing_values(self):
        """Check for missing values"""
        missing_summary = self.df.isnull().sum()
        columns_with_missing = missing_summary[missing_summary > 0]
        
        self.validation_report['missing_values'] = {
            'total_missing_values': missing_summary.sum(),
            'columns_with_missing': len(columns_with_missing),
            'missing_by_column': columns_with_missing.to_dict()
        }
    
    def check_data_types(self):
        """Check data types and identify potential issues"""
        expected_types = {
            'customerID': 'object',
            'tenure': 'int64',
            'MonthlyCharges': 'float64',
            'TotalCharges': ['object', 'float64'],
            'SeniorCitizen': 'int64',
            'Churn': 'object'
        }
        
        type_issues = {}
        
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if isinstance(expected_type, list):
                    if actual_type not in expected_type:
                        type_issues[col] = {
                            'expected': expected_type,
                            'actual': actual_type
                        }
                else:
                    if actual_type != expected_type:
                        type_issues[col] = {
                            'expected': expected_type,
                            'actual': actual_type
                        }
        
        self.validation_report['data_types'] = {
            'type_issues': type_issues,
            'types_valid': len(type_issues) == 0
        }
    
    def check_value_ranges(self):
        """Check if numeric values are within expected ranges"""
        range_checks = {
            'tenure': (0, 72),
            'MonthlyCharges': (0, 200),
            'SeniorCitizen': (0, 1)
        }
        
        range_violations = {}
        
        for col, (min_val, max_val) in range_checks.items():
            if col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    out_of_range = self.df[
                        (self.df[col] < min_val) | (self.df[col] > max_val)
                    ]
                    
                    if len(out_of_range) > 0:
                        range_violations[col] = {
                            'expected_range': f"{min_val}-{max_val}",
                            'violations': len(out_of_range)
                        }
        
        self.validation_report['value_ranges'] = {
            'range_violations': range_violations,
            'ranges_valid': len(range_violations) == 0
        }
    
    def check_categorical_values(self):
        """Check categorical values for unexpected entries"""
        expected_values = {
            'gender': ['Male', 'Female'],
            'Partner': ['Yes', 'No'],
            'Dependents': ['Yes', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['Yes', 'No']
        }
        
        categorical_issues = {}
        
        for col, expected in expected_values.items():
            if col in self.df.columns:
                unique_values = set(self.df[col].unique())
                unexpected_values = unique_values - set(expected)
                
                if unexpected_values:
                    categorical_issues[col] = {
                        'unexpected_values': list(unexpected_values)
                    }
        
        self.validation_report['categorical_values'] = {
            'categorical_issues': categorical_issues,
            'categoricals_valid': len(categorical_issues) == 0
        }
    
    def check_duplicates(self):
        """Check for duplicate records"""
        duplicate_ids = self.df['customerID'].duplicated().sum() if 'customerID' in self.df.columns else 0
        duplicate_rows = self.df.duplicated().sum()
        
        self.validation_report['duplicates'] = {
            'duplicate_customer_ids': int(duplicate_ids),
            'duplicate_rows': int(duplicate_rows),
            'duplicates_found': duplicate_ids > 0 or duplicate_rows > 0
        }
    
    def check_business_logic(self):
        """Check business logic constraints"""
        business_issues = []
        
        # Check tenure vs total charges consistency
        if all(col in self.df.columns for col in ['TotalCharges', 'MonthlyCharges', 'tenure']):
            total_charges_numeric = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            mask = (self.df['tenure'] > 0) & (total_charges_numeric.notna())
            if mask.sum() > 0:
                expected_min_charges = self.df.loc[mask, 'MonthlyCharges'] * self.df.loc[mask, 'tenure'] * 0.5
                actual_charges = total_charges_numeric.loc[mask]
                
                unreasonable_charges = (actual_charges < expected_min_charges).sum()
                if unreasonable_charges > 0:
                    business_issues.append(f"{unreasonable_charges} customers with unreasonably low TotalCharges")
        
        self.validation_report['business_logic'] = {
            'business_issues': business_issues,
            'business_logic_valid': len(business_issues) == 0
        }
    
    def generate_summary_report(self):
        """Generate overall data quality summary"""
        total_checks = 7
        passed_checks = sum([
            self.validation_report['structure']['structure_valid'],
            len(self.validation_report['missing_values']['missing_by_column']) == 0,
            self.validation_report['data_types']['types_valid'],
            self.validation_report['value_ranges']['ranges_valid'],
            self.validation_report['categorical_values']['categoricals_valid'],
            not self.validation_report['duplicates']['duplicates_found'],
            self.validation_report['business_logic']['business_logic_valid']
        ])
        
        self.validation_report['summary'] = {
            'overall_quality_score': (passed_checks / total_checks) * 100,
            'checks_passed': passed_checks,
            'total_checks': total_checks,
            'data_ready_for_training': passed_checks >= 5
        }
        
        quality_score = self.validation_report['summary']['overall_quality_score']
        logger.info(f"Data quality score: {quality_score:.1f}% ({passed_checks}/{total_checks} checks passed)")


def main():
    """Run data validation"""
    print("Starting Data Validation Process")
    print("=" * 50)
    
    # Initialize validator
    validator = DataValidator("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Run validation
    success, report = validator.load_and_validate()
    
    if success:
        print("\nValidation Results:")
        print(f"   Overall Quality Score: {report['summary']['overall_quality_score']:.1f}%")
        print(f"   Checks Passed: {report['summary']['checks_passed']}/{report['summary']['total_checks']}")
        print(f"   Data Ready for Training: {'Yes' if report['summary']['data_ready_for_training'] else 'No'}")
        
        # Save validation report
        import json
        with open('data_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\nValidation report saved to: data_validation_report.json")
    else:
        print("Data validation failed - check logs for details")


if __name__ == "__main__":
    main()
