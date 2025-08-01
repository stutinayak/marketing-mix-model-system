# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class MarketingMixModel:
    """Marketing Mix Model wrapper class."""
    
    def __init__(self, model, model_name, feature_names):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        self.feature_importance = None
        
    def fit(self, X, y):
        """Fit the model and calculate feature importance."""
        self.model.fit(X, y)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n=20):
        """Get top N most important features."""
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n)
        return None


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    return metrics, y_pred


def train_models(X_train, y_train, X_test, y_test, feature_names):
    """Train multiple models and evaluate them."""
    logger = logging.getLogger(__name__)
    logger.info('Training multiple marketing mix models...')
    
    # Define models to train
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f'Training {name}...')
        
        # Create model wrapper
        model_wrapper = MarketingMixModel(model, name, feature_names)
        
        # Train model
        model_wrapper.fit(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model_wrapper, X_test, y_test)
        
        # Store results
        results[name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'model': model_wrapper
        }
        
        trained_models[name] = model_wrapper
        
        logger.info(f'{name} - R²: {metrics["r2"]:.4f}, RMSE: {metrics["rmse"]:.2f}')
    
    return results, trained_models


def hyperparameter_tuning(X_train, y_train, feature_names):
    """Perform hyperparameter tuning for the best model."""
    logger = logging.getLogger(__name__)
    logger.info('Performing hyperparameter tuning...')
    
    # Define parameter grids for different models
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'Ridge Regression': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'Lasso Regression': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        }
    }
    
    best_models = {}
    
    for model_name, param_grid in param_grids.items():
        logger.info(f'Tuning {model_name}...')
        
        # Get base model
        if model_name == 'Random Forest':
            base_model = RandomForestRegressor(random_state=42)
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingRegressor(random_state=42)
        elif model_name == 'Ridge Regression':
            base_model = Ridge()
        elif model_name == 'Lasso Regression':
            base_model = Lasso()
        else:
            continue
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Create model wrapper with best parameters
        best_model = MarketingMixModel(
            grid_search.best_estimator_, 
            f"{model_name} (Tuned)", 
            feature_names
        )
        best_model.fit(X_train, y_train)
        
        best_models[model_name] = best_model
        
        logger.info(f'{model_name} best score: {grid_search.best_score_:.4f}')
        logger.info(f'{model_name} best params: {grid_search.best_params_}')
    
    return best_models


def save_model(model, model_dir, model_name):
    """Save the trained model and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model_path = model_dir / f'{model_name}.joblib'
    joblib.dump(model.model, model_path)
    
    # Save feature importance
    if model.feature_importance is not None:
        importance_path = model_dir / f'{model_name}_feature_importance.csv'
        model.feature_importance.to_csv(importance_path, index=False)
    
    # Save model metadata
    metadata = {
        'model_name': model.model_name,
        'feature_count': len(model.feature_names),
        'model_type': type(model.model).__name__
    }
    
    metadata_path = model_dir / f'{model_name}_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Train the marketing mix model."""
    logger = logging.getLogger(__name__)
    logger.info('Training marketing mix model')
    
    try:
        # Load features
        from src.features.build_features import main as build_features
        X_train, X_test, y_train, y_test, feature_names, scaler = build_features()
        
        # Train multiple models
        results, trained_models = train_models(X_train, y_train, X_test, y_test, feature_names)
        
        # Find best model based on R² score
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['r2'])
        best_model = trained_models[best_model_name]
        best_metrics = results[best_model_name]['metrics']
        
        logger.info(f'\nBest model: {best_model_name}')
        logger.info(f'R² Score: {best_metrics["r2"]:.4f}')
        logger.info(f'RMSE: {best_metrics["rmse"]:.2f}')
        logger.info(f'MAE: {best_metrics["mae"]:.2f}')
        logger.info(f'MAPE: {best_metrics["mape"]:.2f}%')
        
        # Perform hyperparameter tuning
        tuned_models = hyperparameter_tuning(X_train, y_train, feature_names)
        
        # Evaluate tuned models
        tuned_results = {}
        for name, model in tuned_models.items():
            metrics, _ = evaluate_model(model, X_test, y_test)
            tuned_results[name] = metrics
            logger.info(f'{name} - R²: {metrics["r2"]:.4f}, RMSE: {metrics["rmse"]:.2f}')
        
        # Find overall best model
        all_models = {**trained_models, **tuned_models}
        all_results = {**results, **tuned_results}
        
        # Handle different result structures
        def get_r2_score(model_name):
            if model_name in results:
                return results[model_name]['metrics']['r2']
            else:
                return tuned_results[model_name]['r2']
        
        overall_best_name = max(all_results.keys(), key=get_r2_score)
        overall_best_model = all_models[overall_best_name]
        if overall_best_name in results:
            overall_best_metrics = results[overall_best_name]['metrics']
        else:
            overall_best_metrics = tuned_results[overall_best_name]
        
        logger.info(f'\nOverall best model: {overall_best_name}')
        logger.info(f'R² Score: {overall_best_metrics["r2"]:.4f}')
        
        # Save models
        models_dir = Path(__file__).resolve().parents[2] / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        save_model(overall_best_model, models_dir, 'best_marketing_mix_model')
        
        # Save scaler
        scaler_path = models_dir / 'feature_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        
        # Save feature names
        feature_names_path = models_dir / 'feature_names.json'
        import json
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Create model service for predictions
        class ModelService:
            def __init__(self, model, scaler, feature_names):
                self.model = model
                self.scaler = scaler
                self.feature_names = feature_names
            
            def predict(self, X):
                """Make predictions with the trained model."""
                if isinstance(X, pd.DataFrame):
                    X = X[self.feature_names]
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
            
            def get_feature_importance(self, top_n=20):
                """Get feature importance."""
                return self.model.get_feature_importance(top_n)
        
        model_service = ModelService(overall_best_model, scaler, feature_names)
        
        # Create comprehensive metrics report
        metrics_report = {
            'best_model': overall_best_name,
            'best_metrics': overall_best_metrics,
            'all_models': all_results,
            'feature_count': len(feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Save metrics report
        metrics_path = models_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_report, f, indent=2, default=str)
        
        logger.info('Model training completed successfully')
        logger.info(f'Models saved to: {models_dir}')
        
        return model_service, metrics_report
        
    except Exception as e:
        logger.error(f'Error in model training: {e}')
        raise


@click.command()
def cli_main():
    """CLI interface for model training."""
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
