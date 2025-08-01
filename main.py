"""Main script to run the complete Marketing Mix Model pipeline."""

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete Marketing Mix Model pipeline."""
    logger.info("=== Marketing Mix Model Pipeline ===\n")
    
    try:
        # Step 1: Data Processing
        logger.info("Step 1: Loading and cleaning data...")
        from src.data.make_dataset import main as process_data
        df_clean, cleaning_report = process_data()
        logger.info(f"✓ Data processing completed. Shape: {df_clean.shape}")
        
        # Step 2: Feature Engineering
        logger.info("\nStep 2: Building features...")
        from src.features.build_features import main as build_features
        X_train, X_test, y_train, y_test, feature_names, scaler = build_features()
        logger.info(f"✓ Feature engineering completed. Training: {X_train.shape}, Test: {X_test.shape}")
        
        # Step 3: Model Training
        logger.info("\nStep 3: Training model...")
        from src.models.train_model import main as train_model
        model_service, metrics = train_model()
        logger.info("✓ Model training completed")
        
        # Step 4: Create Visualizations
        logger.info("\nStep 4: Creating visualizations...")
        from src.visualization.visualize import main as create_visualizations
        create_visualizations()
        logger.info("✓ Visualizations completed")
        
        # Step 5: Demonstrate Predictions
        logger.info("\nStep 5: Testing predictions...")
        from src.models.predict_model import main as test_predictions
        prediction_service = test_predictions()
        logger.info("✓ Predictions tested successfully")
        
        logger.info("\n=== Pipeline completed successfully! ===")
        logger.info("\nNext steps:")
        logger.info("- Run 'python -m src.api.app' to start the REST API")
        logger.info("- Visit http://localhost:8000/docs for interactive API documentation")
        logger.info("- Check reports/figures/ for generated visualizations")
        logger.info("- Check models/ for saved model files")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
