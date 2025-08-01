# Marketing Mix Model

A comprehensive Marketing Mix Modeling (MMM) solution built with Python, following the Cookiecutter Data Science project structure. This project provides end-to-end capabilities for analyzing marketing effectiveness, making predictions, and optimizing marketing spend allocation.

## 🚀 Features

### Core Functionality
- **Data Processing**: Automated data cleaning and preparation for marketing mix analysis
- **Feature Engineering**: Advanced feature creation including lag features, rolling statistics, and interaction terms
- **Model Training**: Multiple ML algorithms with hyperparameter tuning and model selection
- **Predictions**: Sales forecasting with confidence intervals and scenario analysis
- **Visualizations**: Comprehensive charts and interactive dashboards
- **REST API**: Production-ready API for model serving and integration

### Marketing Analytics
- **Channel Attribution**: Understand the contribution of each marketing channel
- **Scenario Analysis**: Test different marketing spend scenarios
- **ROAS Optimization**: Identify the most effective marketing mix
- **Seasonal Analysis**: Account for seasonal patterns and trends
- **Price Elasticity**: Analyze price sensitivity and competitive positioning

## 📁 Project Structure

```
project_name/
├── data/                   # Data files
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── models/                # Trained models and metadata
├── reports/               # Generated reports and visualizations
│   └── figures/          # Charts and plots
├── src/                   # Source code
│   ├── api/              # REST API implementation
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and prediction
│   └── visualization/    # Visualization scripts
├── main.py               # Main pipeline execution
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
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
5. Test prediction capabilities

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

## 🖥️ Dashboard UI

### Quick Start with Dashboard

The easiest way to interact with the Marketing Mix Model is through the web dashboard:

```bash
# Start the dashboard
python run_dashboard.py
```

Then open your browser and navigate to: **http://localhost:8000**

### Dashboard Features

The dashboard provides an intuitive interface for:

- **📊 Sales Prediction**: Input marketing spend data and get instant sales predictions with confidence intervals
- **📈 Scenario Analysis**: Compare different marketing strategies and see their impact on sales
- **🔍 Model Insights**: View feature importance charts and model performance metrics
- **📊 Marketing Attribution**: Visualize how each marketing channel contributes to sales
- **⚡ Real-time Updates**: All calculations happen in real-time with live API calls

### Dashboard Screenshots

The dashboard includes:
- Modern, responsive design with gradient backgrounds
- Interactive charts powered by Chart.js
- Real-time API status monitoring
- Mobile-friendly interface
- Loading states and error handling

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

#### 2. Scenario Analysis
```bash
curl -X POST "http://localhost:8000/scenario-analysis" \
     -H "Content-Type: application/json" \
     -d '{
       "base_scenario": {
         "tv_spend": 50000,
         "radio_spend": 20000,
         "print_spend": 15000,
         "digital_spend": 30000,
         "competitor_price": 50,
         "our_price": 55,
         "holiday": 0
       },
       "scenarios": {
         "increase_tv_20pct": {"tv_spend": 60000},
         "increase_digital_30pct": {"digital_spend": 39000}
       }
     }'
```

#### 3. Get Model Information
```bash
curl "http://localhost:8000/model-info"
```

#### 4. Get Feature Importance
```bash
curl "http://localhost:8000/feature-importance?top_n=20"
```

#### 5. Get Marketing Attribution
```bash
curl "http://localhost:8000/marketing-attribution"
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

## 📊 Visualizations

The system generates comprehensive visualizations including:

1. **Time Series Analysis**
   - Marketing spend over time
   - Sales trends
   - ROAS analysis
   - Price comparisons

2. **Model Performance**
   - Model comparison charts
   - Feature importance plots
   - Prediction accuracy metrics

3. **Scenario Analysis**
   - Sales predictions by scenario
   - Percentage change analysis
   - Marketing mix optimization

4. **Interactive Dashboard**
   - Plotly-based interactive charts
   - Real-time data exploration
   - Customizable views

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_RANDOM_STATE=42
TEST_SIZE=0.2
CROSS_VALIDATION_FOLDS=5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Data Configuration
DATA_DAYS=365
DATA_SEED=42
```

### Model Parameters

Adjust model parameters in `src/models/train_model.py`:

```python
# Hyperparameter grids
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    # ... other models
}
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/test_models.py
```

### Test API Endpoints
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @test_data.json
```

## 📝 Usage Examples

### Python API Usage

```python
from src.models.predict_model import PredictionService

# Initialize prediction service
service = PredictionService()

# Make a prediction
input_data = {
    'tv_spend': 50000,
    'radio_spend': 20000,
    'print_spend': 15000,
    'digital_spend': 30000,
    'competitor_price': 50,
    'our_price': 55,
    'holiday': 0
}

prediction = service.predict(input_data)
print(f"Predicted sales: ${prediction[0]:,.2f}")

# Scenario analysis
scenarios = {
    'increase_tv': {'tv_spend': 60000},
    'increase_digital': {'digital_spend': 40000}
}

results = service.scenario_analysis(input_data, scenarios)
for scenario, result in results.items():
    if scenario != 'base':
        print(f"{scenario}: {result['change_pct']:.1f}% change")
```

### Jupyter Notebook Integration

```python
# Load data
from src.data.make_dataset import main as process_data
df_clean, report = process_data()

# Train model
from src.models.train_model import main as train_model
model_service, metrics = train_model()

# Create visualizations
from src.visualization.visualize import main as create_viz
create_viz()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

1. Check the [documentation](docs/)
2. Review existing [issues](../../issues)
3. Create a new issue with detailed information

## 🔄 Version History

- **v1.0.0**: Initial release with complete MMM pipeline
- **v1.1.0**: Added REST API and interactive visualizations
- **v1.2.0**: Enhanced feature engineering and model selection

## 🙏 Acknowledgments

- Built with [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- Powered by [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)
- API framework by [FastAPI](https://fastapi.tiangolo.com/)
