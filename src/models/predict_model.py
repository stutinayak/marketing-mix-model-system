# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datetime import datetime, timedelta
import json


class PredictionService:
    """Service for making predictions with the trained marketing mix model."""
    
    def __init__(self, model_path=None):
        """Initialize the prediction service."""
        self.logger = logging.getLogger(__name__)
        
        if model_path is None:
            # Load from default location
            models_dir = Path(__file__).resolve().parents[2] / 'models'
            model_path = models_dir / 'best_marketing_mix_model.joblib'
        
        # Load the trained model and components
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model and related components."""
        try:
            models_dir = model_path.parent
            
            # Load the model
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = models_dir / 'feature_scaler.joblib'
            self.scaler = joblib.load(scaler_path)
            
            # Load feature names
            feature_names_path = models_dir / 'feature_names.json'
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            
            # Load model metadata
            metadata_path = models_dir / 'best_marketing_mix_model_metadata.json'
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.logger.info(f'Model loaded successfully: {self.metadata["model_name"]}')
            
        except Exception as e:
            self.logger.error(f'Error loading model: {e}')
            raise
    
    def prepare_input_data(self, input_data):
        """Prepare input data for prediction."""
        # If input is a dictionary, convert to DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            # Fill missing features with default values
            for feature in missing_features:
                input_data[feature] = 0.0
        
        # Select only the required features in the correct order
        input_data = input_data[self.feature_names]
        
        return input_data
    
    def predict(self, input_data):
        """Make predictions on input data."""
        try:
            # Prepare input data
            X = self.prepare_input_data(input_data)
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f'Error making predictions: {e}')
            raise
    
    def predict_with_confidence(self, input_data, n_bootstrap=100):
        """Make predictions with confidence intervals using bootstrap."""
        try:
            # Prepare input data
            X = self.prepare_input_data(input_data)
            X_scaled = self.scaler.transform(X)
            
            # Bootstrap predictions
            predictions = []
            for _ in range(n_bootstrap):
                # For tree-based models, we can use the estimators
                if hasattr(self.model, 'estimators_'):
                    # Random Forest or Gradient Boosting
                    estimators = self.model.estimators_
                    if isinstance(estimators, np.ndarray):
                        estimators = estimators.flatten()
                    pred = np.mean([estimator.predict(X_scaled) for estimator in estimators], axis=0)
                else:
                    # For other models, use the main model
                    pred = self.model.predict(X_scaled)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate confidence intervals
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            lower_ci = np.percentile(predictions, 2.5, axis=0)
            upper_ci = np.percentile(predictions, 97.5, axis=0)
            
            return {
                'predictions': mean_pred,
                'std': std_pred,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            }
            
        except Exception as e:
            self.logger.error(f'Error making predictions with confidence: {e}')
            raise
    
    def scenario_analysis(self, base_scenario, scenarios):
        """Perform scenario analysis with different marketing spend levels."""
        try:
            results = {}
            
            # Base prediction
            base_prediction = self.predict(base_scenario)
            results['base'] = {
                'scenario': base_scenario,
                'prediction': base_prediction[0],
                'change': 0
            }
            
            # Scenario predictions
            for scenario_name, scenario_data in scenarios.items():
                # Merge base scenario with scenario changes
                scenario_input = base_scenario.copy()
                scenario_input.update(scenario_data)
                
                prediction = self.predict(scenario_input)
                change = prediction[0] - base_prediction[0]
                change_pct = (change / base_prediction[0]) * 100
                
                results[scenario_name] = {
                    'scenario': scenario_input,
                    'prediction': prediction[0],
                    'change': change,
                    'change_pct': change_pct
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f'Error in scenario analysis: {e}')
            raise
    
    def marketing_attribution(self, input_data):
        """Calculate marketing attribution for different channels."""
        try:
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
            else:
                return None
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Filter for marketing spend features
            spend_features = feature_importance[
                feature_importance['feature'].str.contains('_spend')
            ]
            
            # Calculate attribution
            total_importance = spend_features['importance'].sum()
            spend_features['attribution_pct'] = (spend_features['importance'] / total_importance) * 100
            
            return spend_features
            
        except Exception as e:
            self.logger.error(f'Error in marketing attribution: {e}')
            raise


def generate_sample_scenarios():
    """Generate sample scenarios for testing."""
    base_scenario = {
        'tv_spend': 50000,
        'radio_spend': 20000,
        'print_spend': 15000,
        'digital_spend': 30000,
        'competitor_price': 50,
        'our_price': 55,
        'holiday': 0
    }
    
    scenarios = {
        'increase_tv_20pct': {'tv_spend': 60000},
        'increase_digital_30pct': {'digital_spend': 39000},
        'decrease_radio_10pct': {'radio_spend': 18000},
        'price_reduction_5pct': {'our_price': 52.25},
        'holiday_campaign': {'holiday': 1, 'tv_spend': 70000, 'digital_spend': 40000}
    }
    
    return base_scenario, scenarios


def main():
    """Test the prediction service."""
    logger = logging.getLogger(__name__)
    logger.info('Testing prediction service')
    
    try:
        # Initialize prediction service
        prediction_service = PredictionService()
        
        # Generate sample scenarios
        base_scenario, scenarios = generate_sample_scenarios()
        
        # Test basic prediction
        logger.info('Testing basic prediction...')
        prediction = prediction_service.predict(base_scenario)
        logger.info(f'Base prediction: ${prediction[0]:,.2f}')
        
        # Test scenario analysis
        logger.info('\nTesting scenario analysis...')
        scenario_results = prediction_service.scenario_analysis(base_scenario, scenarios)
        
        for scenario_name, result in scenario_results.items():
            if scenario_name != 'base':
                logger.info(f'{scenario_name}: ${result["prediction"]:,.2f} '
                          f'(Change: ${result["change"]:,.2f}, {result["change_pct"]:.1f}%)')
        
        # Test marketing attribution
        logger.info('\nTesting marketing attribution...')
        attribution = prediction_service.marketing_attribution(base_scenario)
        if attribution is not None:
            logger.info('Marketing channel attribution:')
            for _, row in attribution.iterrows():
                logger.info(f'{row["feature"]}: {row["attribution_pct"]:.1f}%')
        
        # Test prediction with confidence intervals
        logger.info('\nTesting prediction with confidence intervals...')
        conf_result = prediction_service.predict_with_confidence(base_scenario)
        logger.info(f'Prediction: ${conf_result["predictions"][0]:,.2f}')
        logger.info(f'Confidence interval: ${conf_result["lower_ci"][0]:,.2f} - ${conf_result["upper_ci"][0]:,.2f}')
        
        logger.info('\nPrediction service test completed successfully')
        
        return prediction_service
        
    except Exception as e:
        logger.error(f'Error testing prediction service: {e}')
        raise


@click.command()
def cli_main():
    """CLI interface for prediction testing."""
    main()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli_main()
