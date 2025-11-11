"""
Credit Risk Default Prediction & Portfolio Segmentation
Developed for financial risk assessment and portfolio management
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CreditRiskPredictor:
    """
    Probability-of-default classification model for credit risk assessment
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.performance_metrics = {}
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for output files"""
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        print("✅ Created output directories: 'results/', 'models/'")
    
    def generate_credit_data(self):
        """Generate realistic borrower credit history dataset"""
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            # Credit History Factors (Primary)
            'credit_score': np.random.normal(650, 120, n_samples).astype(int),
            'credit_history_length': np.random.randint(12, 360, n_samples),
            'total_credit_lines': np.random.randint(1, 20, n_samples),
            'credit_utilization_ratio': np.random.uniform(0.1, 0.9, n_samples),
            'delinquencies_2y': np.random.poisson(0.5, n_samples),
            'public_records': np.random.poisson(0.1, n_samples),
            
            # Financial Capacity Factors
            'annual_income': np.random.normal(60000, 25000, n_samples).astype(int),
            'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
            'loan_amount': np.random.normal(20000, 10000, n_samples).astype(int),
            'employment_length': np.random.randint(0, 30, n_samples),
            
            # Behavioral & Demographic Factors
            'age': np.random.randint(22, 70, n_samples),
            'number_dependents': np.random.randint(0, 5, n_samples),
            'loan_purpose': np.random.choice([
                'DEBT_CONSOLIDATION', 'CREDIT_CARD', 'HOME_IMPROVEMENT', 
                'MAJOR_PURCHASE', 'MEDICAL', 'VACATION', 'BUSINESS'
            ], n_samples),
            'home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN'], n_samples)
        }
        
        self.df = pd.DataFrame(data)
        
        # Create realistic default probability based on key risk drivers
        default_probability = (
            # Credit History Impact (40%)
            0.20 * (self.df['credit_score'] < 580) +
            0.10 * (self.df['delinquencies_2y'] > 2) +
            0.05 * (self.df['credit_utilization_ratio'] > 0.7) +
            0.05 * (self.df['public_records'] > 0) +
            
            # Financial Capacity Impact (35%)
            0.15 * (self.df['debt_to_income_ratio'] > 0.5) +
            0.10 * (self.df['annual_income'] < 30000) +
            0.05 * (self.df['employment_length'] < 2) +
            0.05 * (self.df['loan_amount'] / self.df['annual_income'] > 0.5) +
            
            # Behavioral Impact (25%)
            0.10 * (self.df['age'] < 25) +
            0.05 * (self.df['number_dependents'] > 3) +
            np.random.normal(0, 0.1, n_samples)
        )
        
        self.df['default_probability'] = np.clip(default_probability, 0, 1)
        self.df['loan_default'] = (self.df['default_probability'] > 0.5).astype(int)
        
        print("✅ Credit Risk Dataset Generated")
        print(f"📊 Portfolio Size: {len(self.df):,} borrowers")
        print(f"🎯 Default Rate: {self.df['loan_default'].mean():.2%}")
        
        return self.df
    
    def perform_eda(self):
        """Comprehensive exploratory data analysis"""
        print("\n" + "="*60)
        print("📊 EXPLORATORY DATA ANALYSIS - Credit Risk Assessment")
        print("="*60)
        
        # Portfolio Overview
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Credit Risk Portfolio Analysis - Key Risk Drivers', fontsize=16, fontweight='bold')
        
        # 1. Default Distribution
        default_counts = self.df['loan_default'].value_counts()
        axes[0,0].pie(default_counts.values, labels=['Non-Default', 'Default'], 
                      autopct='%1.1f%%', colors=['lightgreen', 'salmon'], startangle=90)
        axes[0,0].set_title('Portfolio Default Distribution')
        
        # 2. Credit Score Impact
        sns.boxplot(data=self.df, x='loan_default', y='credit_score', ax=axes[0,1], palette='Set2')
        axes[0,1].set_title('Credit Score Distribution by Default Status')
        axes[0,1].set_xlabel('Default Status')
        axes[0,1].set_ylabel('Credit Score')
        
        # 3. Debt-to-Income Ratio Impact
        sns.histplot(data=self.df, x='debt_to_income_ratio', hue='loan_default', 
                    bins=30, ax=axes[0,2], alpha=0.7, palette='viridis')
        axes[0,2].set_title('Debt-to-Income Ratio Distribution')
        axes[0,2].axvline(x=0.5, color='red', linestyle='--', label='50% DTI Threshold')
        axes[0,2].legend()
        
        # 4. Credit Utilization Impact
        sns.boxplot(data=self.df, x='loan_default', y='credit_utilization_ratio', 
                   ax=axes[1,0], palette='coolwarm')
        axes[1,0].set_title('Credit Utilization Ratio by Default Status')
        axes[1,0].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70% Utilization Threshold')
        
        # 5. Delinquency History
        delinquency_impact = self.df.groupby('delinquencies_2y')['loan_default'].mean()
        axes[1,1].bar(delinquency_impact.index, delinquency_impact.values, color='coral', alpha=0.7)
        axes[1,1].set_title('Default Rate by Delinquency History (2 Years)')
        axes[1,1].set_xlabel('Number of Delinquencies')
        axes[1,1].set_ylabel('Default Rate')
        
        # 6. Income Distribution
        sns.histplot(data=self.df, x='annual_income', hue='loan_default', 
                    bins=30, ax=axes[1,2], alpha=0.6, palette='Set1')
        axes[1,2].set_title('Annual Income Distribution by Default Status')
        axes[1,2].set_xlabel('Annual Income ($)')
        
        plt.tight_layout()
        plt.savefig('results/portfolio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Key Risk Statistics
        print("\n🔍 KEY RISK DRIVERS ANALYSIS:")
        risk_metrics = {
            'Credit Score Gap': self.df[self.df['loan_default'] == 0]['credit_score'].mean() - 
                              self.df[self.df['loan_default'] == 1]['credit_score'].mean(),
            'DTI Ratio Gap': self.df[self.df['loan_default'] == 1]['debt_to_income_ratio'].mean() - 
                           self.df[self.df['loan_default'] == 0]['debt_to_income_ratio'].mean(),
            'Utilization Gap': self.df[self.df['loan_default'] == 1]['credit_utilization_ratio'].mean() - 
                             self.df[self.df['loan_default'] == 0]['credit_utilization_ratio'].mean()
        }
        
        for metric, value in risk_metrics.items():
            print(f"  • {metric}: {value:+.1f}")
    
    def preprocess_data(self):
        """Data preprocessing for model training"""
        print("\n" + "="*60)
        print("🔧 DATA PREPROCESSING FOR MODEL DEVELOPMENT")
        print("="*60)
        
        # Encode categorical variables
        categorical_cols = ['loan_purpose', 'home_ownership']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            self.df[col + '_encoded'] = self.label_encoders[col].fit_transform(self.df[col])
        
        # Feature selection based on domain knowledge
        self.feature_cols = [
            # Credit History Factors
            'credit_score', 'credit_history_length', 'total_credit_lines',
            'credit_utilization_ratio', 'delinquencies_2y', 'public_records',
            
            # Financial Capacity
            'annual_income', 'debt_to_income_ratio', 'loan_amount', 'employment_length',
            
            # Behavioral & Demographic
            'age', 'number_dependents', 'loan_purpose_encoded', 'home_ownership_encoded'
        ]
        
        X = self.df[self.feature_cols]
        y = self.df['loan_default']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_cols = ['annual_income', 'loan_amount', 'credit_score', 'age']
        self.X_train[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        
        print("✅ Data Preprocessing Completed")
        print(f"   Training Samples: {self.X_train.shape[0]:,}")
        print(f"   Test Samples: {self.X_test.shape[0]:,}")
        print(f"   Features: {len(self.feature_cols)} risk drivers")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models for probability-of-default prediction"""
        print("\n" + "="*60)
        print("🤖 PROBABILITY-OF-DEFAULT MODEL TRAINING")
        print("="*60)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        self.model_performance = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Performance metrics
            accuracy = (y_pred == self.y_test).mean()
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            self.model_performance[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✅ {name} Performance:")
            print(f"   • Accuracy: {accuracy:.3f}")
            print(f"   • AUC-ROC: {auc_score:.3f}")
        
        # Select best model
        best_model_name = max(self.model_performance.items(), 
                            key=lambda x: x[1]['auc_score'])[0]
        self.model = self.model_performance[best_model_name]['model']
        
        print(f"\n🏆 SELECTED MODEL: {best_model_name}")
        print(f"📈 Best AUC-ROC: {self.model_performance[best_model_name]['auc_score']:.3f}")
        
        # Save the best model
        joblib.dump(self.model, 'models/best_credit_risk_model.pkl')
        print("💾 Model saved: models/best_credit_risk_model.pkl")
        
        return self.model_performance
    
    def analyze_feature_importance(self):
        """Feature importance analysis to identify key risk drivers"""
        print("\n" + "="*60)
        print("🔍 FEATURE IMPORTANCE ANALYSIS - Key Risk Drivers")
        print("="*60)
        
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # Logistic Regression coefficients
            coefficients = np.abs(self.model.coef_[0])
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': coefficients / coefficients.sum()
            }).sort_values('importance', ascending=False)
        
        # Categorize features
        feature_categories = {
            'Credit History': ['credit_score', 'delinquencies_2y', 'credit_utilization_ratio', 
                             'credit_history_length', 'public_records', 'total_credit_lines'],
            'Financial Capacity': ['debt_to_income_ratio', 'annual_income', 'loan_amount', 'employment_length'],
            'Behavioral/Demographic': ['age', 'number_dependents', 'loan_purpose_encoded', 'home_ownership_encoded']
        }
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Color coding by category
        colors = []
        for feature in self.feature_importance['feature']:
            if feature in feature_categories['Credit History']:
                colors.append('#FF6B6B')  # Red for Credit History
            elif feature in feature_categories['Financial Capacity']:
                colors.append('#4ECDC4')  # Teal for Financial
            else:
                colors.append('#45B7D1')  # Blue for Behavioral
        
        # Plot feature importance
        bars = plt.barh(range(len(self.feature_importance)), 
                       self.feature_importance['importance'], 
                       color=colors, alpha=0.8)
        
        plt.yticks(range(len(self.feature_importance)), self.feature_importance['feature'])
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.title('Credit Risk Drivers - Feature Importance Analysis', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Credit History Factors', alpha=0.8),
            Patch(facecolor='#4ECDC4', label='Financial Capacity', alpha=0.8),
            Patch(facecolor='#45B7D1', label='Behavioral/Demographic', alpha=0.8)
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Add value annotations
        for i, (_, row) in enumerate(self.feature_importance.iterrows()):
            plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key insights
        print("\n🎯 TOP 5 RISK DRIVERS INFLUENCING DEFAULT LIKELIHOOD:")
        for i, (_, row) in enumerate(self.feature_importance.head().iterrows(), 1):
            print(f"  {i}. {row['feature']:25} Importance: {row['importance']:.3f}")
        
        # Category-wise importance
        print("\n📊 FEATURE IMPORTANCE BY CATEGORY:")
        for category, features in feature_categories.items():
            category_importance = self.feature_importance[
                self.feature_importance['feature'].isin(features)
            ]['importance'].sum()
            print(f"  • {category}: {category_importance:.1%}")
        
        return self.feature_importance
    
    def segment_portfolio_risk(self):
        """Portfolio segmentation into risk bands for early-warning insights"""
        print("\n" + "="*60)
        print("📊 PORTFOLIO SEGMENTATION - Risk Band Analysis")
        print("="*60)
        
        # Get probabilities from best model
        best_model_name = max(self.model_performance.items(), 
                            key=lambda x: x[1]['auc_score'])[0]
        y_pred_proba = self.model_performance[best_model_name]['probabilities']
        
        # Define risk bands with business rationale
        risk_bands = {
            'Low Risk': (0.0, 0.1),
            'Moderate Risk': (0.1, 0.3),
            'High Risk': (0.3, 0.6),
            'Very High Risk': (0.6, 1.0)
        }
        
        # Assign risk bands
        risk_assignments = []
        for prob in y_pred_proba:
            for band, (low, high) in risk_bands.items():
                if low <= prob < high:
                    risk_assignments.append(band)
                    break
        
        # Create segmentation results
        segmentation_df = pd.DataFrame({
            'actual_default': self.y_test.values,
            'default_probability': y_pred_proba,
            'risk_band': risk_assignments
        })
        
        # Portfolio segmentation analysis
        portfolio_summary = segmentation_df.groupby('risk_band').agg({
            'actual_default': ['count', 'mean'],
            'default_probability': 'mean'
        }).round(4)
        
        portfolio_summary.columns = ['Number of Borrowers', 'Actual Default Rate', 'Average Default Probability']
        
        print("\n📈 PORTFOLIO RISK SEGMENTATION RESULTS:")
        print(portfolio_summary)
        
        # Enhanced visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Credit Portfolio Risk Segmentation & Early-Warning Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Risk Distribution Pie Chart
        risk_counts = segmentation_df['risk_band'].value_counts()
        colors = ['lightgreen', 'gold', 'orange', 'red']
        axes[0,0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                     colors=colors, startangle=90, textprops={'fontsize': 10})
        axes[0,0].set_title('Portfolio Exposure by Risk Band', fontweight='bold')
        
        # 2. Default Rate by Risk Band
        default_rates = segmentation_df.groupby('risk_band')['actual_default'].mean()
        bars = axes[0,1].bar(default_rates.index, default_rates.values, color=colors, alpha=0.8)
        axes[0,1].set_title('Actual Default Rate by Risk Band', fontweight='bold')
        axes[0,1].set_ylabel('Default Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, default_rates.values):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Probability Distribution by Risk Band
        risk_band_data = []
        band_labels = []
        for band in risk_bands.keys():
            band_probs = segmentation_df[segmentation_df['risk_band'] == band]['default_probability']
            risk_band_data.append(band_probs)
            band_labels.append(band)
        
        box_plot = axes[1,0].boxplot(risk_band_data, labels=band_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1,0].set_title('Default Probability Distribution by Risk Band', fontweight='bold')
        axes[1,0].set_ylabel('Predicted Default Probability')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Early Warning Matrix
        risk_exposure = segmentation_df['risk_band'].value_counts().sort_index()
        risk_severity = segmentation_df.groupby('risk_band')['actual_default'].mean().sort_index()
        
        scatter = axes[1,1].scatter(risk_exposure.values, risk_severity.values * 100, 
                                   s=risk_exposure.values * 2, c=range(len(risk_exposure)), 
                                   cmap='RdYlGn_r', alpha=0.7)
        
        for i, (band, exposure, severity) in enumerate(zip(risk_exposure.index, 
                                                         risk_exposure.values, 
                                                         risk_severity.values)):
            axes[1,1].annotate(band, (exposure, severity * 100), 
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1,1].set_xlabel('Number of Borrowers (Exposure)')
        axes[1,1].set_ylabel('Default Rate (%)')
        axes[1,1].set_title('Risk Exposure vs Severity Matrix', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/risk_segmentation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Risk Management Recommendations
        print("\n🚨 EARLY-WARNING INSIGHTS & RISK MANAGEMENT RECOMMENDATIONS:")
        recommendations = {
            'Low Risk': "✅ Maintain standard underwriting, consider premium reduction",
            'Moderate Risk': "📊 Enhanced monitoring, periodic credit reviews",
            'High Risk': "⚠️ Increase interest rates, reduce credit limits, frequent monitoring",
            'Very High Risk': "🔴 Strict underwriting, require collateral, consider declining"
        }
        
        for band, recommendation in recommendations.items():
            count = len(segmentation_df[segmentation_df['risk_band'] == band])
            exposure_pct = (count / len(segmentation_df)) * 100
            if count > 0:
                actual_default_rate = segmentation_df[segmentation_df['risk_band'] == band]['actual_default'].mean()
                print(f"\n• {band}:")
                print(f"  Exposure: {count} borrowers ({exposure_pct:.1f}% of portfolio)")
                print(f"  Actual Default Rate: {actual_default_rate:.1%}")
                print(f"  Action: {recommendation}")
        
        # Save segmentation results
        segmentation_df.to_csv('results/portfolio_segmentation.csv', index=False)
        print(f"\n💾 Portfolio segmentation saved: results/portfolio_segmentation.csv")
        
        return segmentation_df
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("📋 MODEL PERFORMANCE & BUSINESS IMPACT REPORT")
        print("="*60)
        
        best_model = max(self.model_performance.items(), key=lambda x: x[1]['auc_score'])
        model_name, metrics = best_model
        
        report = f"""
CREDIT RISK DEFAULT PREDICTION - PERFORMANCE REPORT
===================================================

MODEL PERFORMANCE:
• Selected Model: {model_name}
• Accuracy: {metrics['accuracy']:.3f}
• AUC-ROC: {metrics['auc_score']:.3f}
• Portfolio Size: {len(self.df):,} borrowers
• Actual Default Rate: {self.df['loan_default'].mean():.2%}

KEY RISK DRIVERS IDENTIFIED:
"""
        
        # Add top risk drivers
        for i, (_, row) in enumerate(self.feature_importance.head(5).iterrows(), 1):
            report += f"  {i}. {row['feature']} (Importance: {row['importance']:.3f})\n"
        
        report += f"""
BUSINESS IMPACT:
• Enhanced credit risk assessment accuracy
• Early-warning system for portfolio monitoring
• Data-driven decision support for underwriting
• Improved risk-based pricing capabilities

PORTFOLIO SEGMENTATION:
• Risk bands established for targeted monitoring
• Exposure concentration analysis
• Default probability calibration
"""
        
        print(report)
        
        # Save report
        with open('results/model_performance.txt', 'w') as f:
            f.write(report)
        
        print("💾 Performance report saved: results/model_performance.txt")
    
    def run_complete_analysis(self):
        """Execute complete credit risk analysis pipeline"""
        print("🚀 CREDIT RISK DEFAULT PREDICTION & PORTFOLIO SEGMENTATION")
        print("="*70)
        print("Developed for Financial Risk Assessment & Portfolio Management")
        print("="*70)
        
        try:
            # 1. Data Generation & Exploration
            self.generate_credit_data()
            self.perform_eda()
            
            # 2. Model Development
            self.preprocess_data()
            self.train_models()
            
            # 3. Risk Analysis
            self.analyze_feature_importance()
            
            # 4. Portfolio Segmentation
            segmentation_results = self.segment_portfolio_risk()
            
            # 5. Performance Reporting
            self.generate_performance_report()
            
            print("\n" + "="*70)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("📊 Key Deliverables:")
            print("  • Probability-of-default classification model")
            print("  • Feature importance analysis identifying key risk drivers")
            print("  • Portfolio segmentation into risk bands")
            print("  • Early-warning insights for risk monitoring")
            print("  • Enhanced credit risk assessment accuracy")
            print("\n💾 Results saved in 'results/' directory")
            
            return segmentation_results
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    # Initialize and run complete analysis
    risk_predictor = CreditRiskPredictor()
    results = risk_predictor.run_complete_analysis()