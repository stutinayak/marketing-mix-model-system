# Marketing Mix Model System

A prototype system for Marketing Mix Modeling (MMM) that demonstrates strong programming, system design, and data processing skills. This project implements a complete MMM solution with data processing, model training, and a RESTful API for sales predictions.

## 🚀 Features

### Core Functionality
- **Data Processing**: Automated ingestion and cleaning of marketing spend and sales data
- **Feature Engineering**: Advanced feature creation including lag features, rolling statistics, and interactions
- **Model Training**: Multiple ML algorithms with hyperparameter tuning and model selection
- **RESTful API**: FastAPI backend for making sales predictions
- **Interactive Dashboard**: Web interface for easy interaction with the model

### Marketing Analytics
- **Sales Prediction**: Forecast sales based on marketing spend inputs
- **Scenario Analysis**: Compare different marketing strategies
- **Feature Importance**: Understand which factors drive sales
- **Marketing Attribution**: Analyze channel effectiveness

## 📁 Project Structure

```
project_name/
├── data/                   # Data files (CSV datasets)
├── models/                 # Trained models and metadata
├── reports/                # Generated reports and visualizations
├── src/                    # Source code
│   ├── api/               # FastAPI backend and dashboard
│   ├── data/              # Data processing scripts
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and prediction
│   └── visualization/     # Visualization scripts
├── main.py                # Main pipeline execution
├── run_dashboard.py       # Dashboard startup script
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project_name
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

## 🚀 Quick Start

### Run the Complete Pipeline

Execute the entire Marketing Mix Model pipeline:

```bash
python main.py
```

This will:
1. Generate and process sample marketing data
2. Create advanced features for modeling
3. Train multiple ML models and select the best one
4. Generate comprehensive visualizations

### Start the Dashboard

Run the interactive dashboard:

```bash
python run_dashboard.py
```

Then open your browser and navigate to: **http://localhost:8000**

The dashboard provides:
- **Sales Prediction**: Input marketing spend data and get sales predictions
- **Scenario Analysis**: Compare different marketing strategies
- **Model Insights**: View feature importance and model information
- **Marketing Attribution**: See how each channel contributes to sales

### Individual Components

You can also run individual components:

```bash
# Data processing
python -m src.data.make_dataset

# Feature engineering
python -m src.features.build_features

# Model training
python -m src.models.train_model

# Test predictions
python -m src.models.predict_model

# Create visualizations
python -m src.visualization.visualize
```

## 📊 API Usage

### Start the API Server

```bash
python -m src.api.app
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tv_spend": 50000,
       "radio_spend": 20000,
       "print_spend": 15000,
       "digital_spend": 30000,
       "competitor_price": 50,
       "our_price": 55,
       "holiday": 0
     }'
```

#### 2. Get Model Information
```bash
curl "http://localhost:8000/model-info"
```

#### 3. Get Feature Importance
```bash
curl "http://localhost:8000/feature-importance?top_n=20"
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

## 📈 Model Performance

The system trains multiple models and automatically selects the best performing one based on:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### Supported Models
- Random Forest
- Gradient Boosting
- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression

## 🏗️ System Architecture

### Data Processing Layer
- **Data Ingestion**: CSV file processing with validation
- **Data Cleaning**: Handling missing values and outliers
- **Feature Engineering**: Creating lag features, rolling statistics, and interactions
- **Data Storage**: Efficient data structures for model training

### Model Service Layer
- **Model Training**: Automated training pipeline with cross-validation
- **Model Selection**: Performance-based model selection
- **Model Persistence**: Saving and loading trained models
- **Prediction Service**: Real-time prediction capabilities

### API Layer
- **FastAPI Backend**: High-performance REST API
- **Request Validation**: Pydantic models for data validation
- **Error Handling**: Comprehensive error responses
- **Documentation**: Automatic API documentation

### Frontend Layer
- **Interactive Dashboard**: Modern web interface
- **Real-time Updates**: Live API integration
- **Data Visualization**: Charts and graphs for insights
- **Responsive Design**: Works on all devices

## 🔧 Extensibility and Future Improvements

### Model Extensibility
- **Modular Design**: Easy to add new model types
- **Plugin Architecture**: Support for custom algorithms
- **Hyperparameter Tuning**: Automated optimization
- **Ensemble Methods**: Combining multiple models

### Feature Engineering
- **Custom Features**: Framework for adding domain-specific features
- **Feature Selection**: Automated feature importance analysis
- **Feature Store**: Centralized feature management
- **Real-time Features**: Streaming feature computation

### API Enhancements
- **Authentication**: User management and security
- **Rate Limiting**: API usage controls
- **Caching**: Performance optimization
- **Monitoring**: Usage analytics and alerts

### Data Pipeline
- **Real-time Data**: Streaming data ingestion
- **Data Versioning**: Track data changes over time
- **Data Quality**: Automated quality checks
- **Scalability**: Handle larger datasets

## 📊 Dataset

The system is designed to work with the following CSV files:
- **sales_data.csv**: Daily sales figures
- **tv_spend.csv**: Daily TV advertising spend
- **radio_spend.csv**: Daily radio advertising spend
- **social_media_spend.csv**: Daily social media advertising spend
- **search_spend.csv**: Daily search engine advertising spend
- **print_spend.csv**: Daily print advertising spend
- **outdoor_spend.csv**: Daily outdoor advertising spend