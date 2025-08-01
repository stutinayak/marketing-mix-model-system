# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime

# Import our modules
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.predict_model import PredictionService


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Marketing Mix Model API",
    description="API for making predictions and scenario analysis with the Marketing Mix Model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="src/api/static"), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

# Initialize prediction service
prediction_service = None


class PredictionRequest(BaseModel):
    """Request model for making predictions."""
    tv_spend: float = Field(..., description="TV advertising spend")
    radio_spend: float = Field(..., description="Radio advertising spend")
    print_spend: float = Field(..., description="Print advertising spend")
    digital_spend: float = Field(..., description="Digital advertising spend")
    competitor_price: float = Field(..., description="Competitor's price")
    our_price: float = Field(..., description="Our product price")
    holiday: int = Field(0, description="Holiday indicator (0 or 1)")


class ScenarioRequest(BaseModel):
    """Request model for scenario analysis."""
    base_scenario: PredictionRequest = Field(..., description="Base scenario")
    scenarios: Dict[str, Dict[str, float]] = Field(..., description="Scenarios to analyze")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: float = Field(..., description="Predicted sales")
    confidence_interval: Dict[str, float] = Field(..., description="Confidence interval")
    timestamp: str = Field(..., description="Prediction timestamp")


class ScenarioResponse(BaseModel):
    """Response model for scenario analysis."""
    base_prediction: float = Field(..., description="Base scenario prediction")
    scenarios: Dict[str, Dict[str, Any]] = Field(..., description="Scenario results")
    timestamp: str = Field(..., description="Analysis timestamp")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str = Field(..., description="Model name")
    feature_count: int = Field(..., description="Number of features")
    model_type: str = Field(..., description="Type of model")
    training_date: str = Field(..., description="Model training date")


@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup."""
    global prediction_service
    try:
        prediction_service = PredictionService()
        logger.info("Prediction service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        raise


@app.get("/")
async def root():
    """Serve the dashboard UI."""
    try:
        return FileResponse("src/api/static/index.html")
    except FileNotFoundError:
        return {
            "message": "Marketing Mix Model API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "note": "Dashboard UI not found. Please ensure static files are properly configured."
        }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": "true"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    """Make a sales prediction based on marketing inputs."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert request to dictionary
        input_data = request.model_dump()
        
        # Make prediction
        prediction = prediction_service.predict(input_data)
        
        # Get confidence interval
        confidence_result = prediction_service.predict_with_confidence(input_data)
        
        return PredictionResponse(
            prediction=float(prediction[0]),
            confidence_interval={
                "lower": float(confidence_result["lower_ci"][0]),
                "upper": float(confidence_result["upper_ci"][0]),
                "std": float(confidence_result["std"][0])
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/scenario-analysis", response_model=ScenarioResponse)
async def analyze_scenarios(request: ScenarioRequest):
    """Perform scenario analysis with different marketing spend levels."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert base scenario to dictionary
        base_scenario = request.base_scenario.dict()
        
        # Perform scenario analysis
        scenario_results = prediction_service.scenario_analysis(base_scenario, request.scenarios)
        
        # Format response
        formatted_scenarios = {}
        for scenario_name, result in scenario_results.items():
            if scenario_name != 'base':
                formatted_scenarios[scenario_name] = {
                    "prediction": float(result["prediction"]),
                    "change": float(result["change"]),
                    "change_pct": float(result["change_pct"]),
                    "scenario_input": result["scenario"]
                }
        
        return ScenarioResponse(
            base_prediction=float(scenario_results['base']['prediction']),
            scenarios=formatted_scenarios,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the trained model."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        return ModelInfoResponse(
            model_name=prediction_service.metadata["model_name"],
            feature_count=prediction_service.metadata["feature_count"],
            model_type=prediction_service.metadata["model_type"],
            training_date=datetime.now().isoformat()  # In real app, this would come from model metadata
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/feature-importance", response_model=Dict[str, Any])
async def get_feature_importance(top_n: int = 20):
    """Get feature importance for the model."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        feature_importance = prediction_service.get_feature_importance(top_n)
        
        if feature_importance is None:
            raise HTTPException(status_code=404, detail="Feature importance not available for this model")
        
        # Convert to dictionary format
        importance_dict = {
            "features": feature_importance['feature'].tolist(),
            "importance": feature_importance['importance'].tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        return importance_dict
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


@app.get("/marketing-attribution", response_model=Dict[str, Any])
async def get_marketing_attribution():
    """Get marketing attribution analysis."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Use a sample input for attribution analysis
        sample_input = {
            'tv_spend': 50000,
            'radio_spend': 20000,
            'print_spend': 15000,
            'digital_spend': 30000,
            'competitor_price': 50,
            'our_price': 55,
            'holiday': 0
        }
        
        attribution = prediction_service.marketing_attribution(sample_input)
        
        if attribution is None:
            raise HTTPException(status_code=404, detail="Marketing attribution not available for this model")
        
        # Convert to dictionary format
        attribution_dict = {
            "channels": attribution['feature'].tolist(),
            "attribution_pct": attribution['attribution_pct'].tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        return attribution_dict
        
    except Exception as e:
        logger.error(f"Error getting marketing attribution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get marketing attribution: {str(e)}")


@app.get("/sample-scenarios", response_model=Dict[str, Any])
async def get_sample_scenarios():
    """Get sample scenarios for testing."""
    try:
        from src.models.predict_model import generate_sample_scenarios
        base_scenario, scenarios = generate_sample_scenarios()
        
        return {
            "base_scenario": base_scenario,
            "scenarios": scenarios,
            "description": "Sample scenarios for testing the API"
        }
        
    except Exception as e:
        logger.error(f"Error getting sample scenarios: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sample scenarios: {str(e)}")


@app.get("/api-stats", response_model=Dict[str, Any])
async def get_api_stats():
    """Get API usage statistics."""
    # In a real application, this would track actual usage
    return {
        "total_requests": 0,
        "predictions_made": 0,
        "scenario_analyses": 0,
        "uptime": "0 days",
        "last_request": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 