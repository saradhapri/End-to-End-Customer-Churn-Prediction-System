
# Enhanced Customer Churn Prediction Training Pipeline
# Advanced ML pipeline with comprehensive evaluation and model optimization


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Advanced ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Metrics and evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)

# Model persistence and utilities
import joblib
import pickle
import os
from datetime import datetime

class ChurnPredictor:
    """Advanced Customer Churn Prediction System"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Load and perform initial data exploration"""
        print("Loading dataset...")
        self.df = pd.read_csv(file_path)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Churn rate: {(self.df['Churn'] == 'Yes').mean():.1%}")
        
        return self.df
    
    def exploratory_analysis(self):
        """Comprehensive EDA with advanced visualizations"""
        print("\nPerforming Exploratory Data Analysis...")
        
        # Create output directory for plots
        os.makedirs('plots', exist_ok=True)
        
        # 1. Target variable distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        churn_counts = self.df['Churn'].value_counts()
        plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                colors=['lightgreen', 'lightcoral'])
        plt.title('Churn Distribution')
        
        # 2. Tenure distribution by churn
        plt.subplot(2, 3, 2)
        sns.histplot(data=self.df, x='tenure', hue='Churn', bins=20, alpha=0.7)
        plt.title('Tenure Distribution by Churn')
        
        # 3. Monthly charges by churn
        plt.subplot(2, 3, 3)
        sns.boxplot(data=self.df, x='Churn', y='MonthlyCharges')
        plt.title('Monthly Charges by Churn')
        
        # 4. Contract type vs churn
        plt.subplot(2, 3, 4)
        contract_churn = pd.crosstab(self.df['Contract'], self.df['Churn'], normalize='index')
        contract_churn.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Contract Type vs Churn Rate')
        plt.xticks(rotation=45)
        
        # 5. Payment method vs churn
        plt.subplot(2, 3, 5)
        payment_churn = pd.crosstab(self.df['PaymentMethod'], self.df['Churn'], normalize='index')
        payment_churn.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Payment Method vs Churn Rate')
        plt.xticks(rotation=45)
        
        # 6. Internet service vs churn
        plt.subplot(2, 3, 6)
        internet_churn = pd.crosstab(self.df['InternetService'], self.df['Churn'], normalize='index')
        internet_churn.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Internet Service vs Churn Rate')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key insights
        print("\nKey EDA Insights:")
        insights = self._generate_eda_insights()
        for insight in insights:
            print(f"   • {insight}")
    
    def _generate_eda_insights(self):
        """Generate data-driven insights from EDA"""
        insights = []
        
        # Churn rate by contract
        contract_churn = self.df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean())
        highest_churn_contract = contract_churn.idxmax()
        insights.append(f"{highest_churn_contract} contracts have highest churn rate ({contract_churn.max():.1%})")
        
        # Payment method analysis
        payment_churn = self.df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x=='Yes').mean())
        highest_churn_payment = payment_churn.idxmax()
        insights.append(f"{highest_churn_payment} has highest churn rate ({payment_churn.max():.1%})")
        
        # Tenure analysis
        churned_avg_tenure = self.df[self.df['Churn']=='Yes']['tenure'].mean()
        retained_avg_tenure = self.df[self.df['Churn']=='No']['tenure'].mean()
        insights.append(f"Churned customers have {churned_avg_tenure:.1f} months avg tenure vs {retained_avg_tenure:.1f} for retained")
        
        return insights
    
    def preprocess_data(self):
        """Advanced data preprocessing with feature engineering"""
        print("\nPreprocessing data...")
        
        # Handle missing values in TotalCharges
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'].fillna(self.df['TotalCharges'].median(), inplace=True)
        
        # Feature Engineering
        print("Engineering new features...")
        
        # 1. Tenure groups
        self.df['TenureGroup'] = pd.cut(self.df['tenure'], 
                                       bins=[0, 12, 24, 48, 72], 
                                       labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
        
        # 2. Average charges per tenure month
        self.df['AvgChargesPerTenure'] = self.df['TotalCharges'] / (self.df['tenure'] + 1)
        
        # 3. Total services count
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        self.df['TotalServices'] = self.df[service_cols].apply(
            lambda x: sum(1 for val in x if val == 'Yes'), axis=1
        )
        
        # 4. High value customer flag
        self.df['HighValueCustomer'] = (self.df['MonthlyCharges'] > self.df['MonthlyCharges'].quantile(0.75)).astype(int)
        
        # 5. Family customer flag
        self.df['FamilyCustomer'] = ((self.df['Partner'] == 'Yes') | (self.df['Dependents'] == 'Yes')).astype(int)
        
        # Drop customerID
        self.df.drop('customerID', axis=1, inplace=True)
        
        # Encode categorical variables
        self.df_encoded = pd.get_dummies(self.df, drop_first=True)
        
        # Features and target
        self.X = self.df_encoded.drop('Churn_Yes', axis=1)
        self.y = self.df_encoded['Churn_Yes']
        
        # Store feature columns for later use
        self.feature_columns = list(self.X.columns)
        
        # Scale numerical features
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargesPerTenure']
        self.X_scaled = self.X.copy()
        self.X_scaled[numerical_features] = self.scaler.fit_transform(self.X[numerical_features])
        
        print(f"Preprocessing completed!")
        print(f"Features: {self.X_scaled.shape[1]}")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")
        
        return self.X_scaled, self.y
    
    def train_models(self):
        """Train multiple ML models with cross-validation"""
        print("\nTraining multiple models...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_score = cross_val_score(model, self.X_train, self.y_train, 
                                     cv=cv, scoring='roc_auc', n_jobs=-1)
            
            # Fit model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1': f1_score(self.y_test, y_pred),
                'ROC_AUC': roc_auc_score(self.y_test, y_pred_proba),
                'CV_Mean': cv_score.mean(),
                'CV_Std': cv_score.std()
            }
            
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"   {name}: ROC-AUC = {metrics['ROC_AUC']:.4f} (±{metrics['CV_Std']:.4f})")
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['ROC_AUC'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name}")
        print(f"ROC-AUC: {self.results[best_model_name]['ROC_AUC']:.4f}")
        
        return self.models, self.results
    
    def optimize_best_model(self):
        """Hyperparameter optimization for the best model"""
        print(f"\nOptimizing {self.best_model_name}...")
        
        # Define parameter grids
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                self.best_model, param_grid, 
                cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Update best model
            self.best_model = grid_search.best_estimator_
            
            print(f"Optimization completed!")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self.best_model
    
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        print("\nGenerating Model Performance Report...")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        print("\nModel Performance Comparison:")
        print(results_df)
        
        # Feature importance analysis
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 10 Feature Importances ({self.best_model_name}):")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(15), y='Feature', x='Importance')
            plt.title(f'Top 15 Feature Importances - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return results_df
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("\nSaving models and preprocessing objects...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, 'models/churn_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save feature columns
        joblib.dump(self.feature_columns, 'models/feature_columns.pkl')
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'data_shape': self.df.shape,
            'feature_count': len(self.feature_columns),
            'churn_rate': self.y.mean()
        }
        
        with open('models/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("All models and objects saved successfully!")
        print("Saved files:")
        print("   • churn_model.pkl - Best trained model")
        print("   • scaler.pkl - Feature scaler")
        print("   • feature_columns.pkl - Feature column names")
        print("   • metadata.pkl - Training metadata")
        
        return True


def main():
    """Main training pipeline"""
    print("Starting Enhanced Customer Churn Prediction Training Pipeline")
    print("=" * 70)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load and explore data
    df = predictor.load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    predictor.exploratory_analysis()
    
    # Preprocess data
    X, y = predictor.preprocess_data()
    
    # Train models
    models, results = predictor.train_models()
    
    # Optimize best model
    best_model = predictor.optimize_best_model()
    
    # Generate comprehensive report
    results_df = predictor.generate_model_report()
    
    # Save everything
    predictor.save_models()
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("Ready for deployment with Streamlit app")
    print("Check 'models/' directory for saved artifacts")
    print("Check 'plots/' directory for visualizations")


if __name__ == "__main__":
    main()
