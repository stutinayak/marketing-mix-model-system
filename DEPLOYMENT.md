# Deployment Guide for Marketing Mix Model System

## 🚀 Quick Deployment Options

### 1. Local Development (Recommended for Demo)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/marketing-mix-model-system.git
cd marketing-mix-model-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python run_dashboard.py
```

**Access:**
- Dashboard: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 2. Heroku Deployment (Free Tier)

#### Step 1: Create Heroku App
```bash
# Install Heroku CLI
# Create new app
heroku create your-mmm-dashboard

# Add Python buildpack
heroku buildpacks:set heroku/python
```

#### Step 2: Create Procfile
Create `Procfile` in root directory:
```
web: uvicorn src.api.app:app --host=0.0.0.0 --port=$PORT
```

#### Step 3: Deploy
```bash
git add .
git commit -m "Add Heroku deployment files"
git push heroku main

# Open the app
heroku open
```

### 3. Railway Deployment (Alternative to Heroku)

#### Step 1: Connect to Railway
1. Go to [Railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway will auto-detect Python and deploy

#### Step 2: Configure Environment
Add these environment variables in Railway dashboard:
```
PORT=8000
```

### 4. Render Deployment

#### Step 1: Create Render Account
1. Go to [Render.com](https://render.com)
2. Connect your GitHub account

#### Step 2: Create Web Service
1. Click "New Web Service"
2. Connect your repository
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`

### 5. Google Cloud Run

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 2: Deploy to Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/mmm-dashboard

# Deploy to Cloud Run
gcloud run deploy mmm-dashboard \
  --image gcr.io/YOUR_PROJECT/mmm-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📊 Demo Preparation

### 1. Screenshots for Portfolio

Take screenshots of:
- **Dashboard Homepage**: Show the main interface
- **Sales Prediction**: Form with sample data and results
- **Scenario Analysis**: Comparison charts
- **Feature Importance**: Interactive bar chart
- **Marketing Attribution**: Pie chart visualization

### 2. Demo Script

Prepare a 2-3 minute demo covering:

1. **Introduction** (30 seconds)
   - "This is a Marketing Mix Modeling system that predicts sales based on marketing spend"
   - "It uses machine learning to analyze the effectiveness of different marketing channels"

2. **Sales Prediction** (45 seconds)
   - Show the form with sample data
   - Enter realistic marketing spend values
   - Demonstrate the prediction with confidence intervals

3. **Scenario Analysis** (45 seconds)
   - Load sample scenarios
   - Show how different strategies affect sales
   - Highlight the visual comparison

4. **Model Insights** (30 seconds)
   - Show feature importance chart
   - Explain what the model learned
   - Demonstrate marketing attribution

### 3. Live Demo Tips

- **Prepare Sample Data**: Have realistic marketing spend values ready
- **Test All Features**: Ensure everything works before the demo
- **Have Backup Plan**: If live demo fails, show screenshots
- **Explain Business Value**: Focus on ROI and decision-making benefits

## 🎯 Portfolio Integration

### 1. GitHub Repository
- Ensure README is comprehensive
- Add screenshots to README
- Include live demo link if deployed
- Add proper tags/topics

### 2. LinkedIn Post
```
🚀 Just completed a comprehensive Marketing Mix Modeling system!

📊 Features:
• AI-powered sales prediction with confidence intervals
• Interactive dashboard with real-time analytics
• Scenario analysis for marketing strategy optimization
• Marketing attribution analysis
• Production-ready FastAPI backend

🔗 Check it out: [GitHub Link]
#DataScience #MachineLearning #MarketingAnalytics #Python #FastAPI
```

### 3. Personal Website
- Add project to portfolio section
- Include live demo link
- Write detailed case study
- Highlight technical skills used

## 🔧 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process
   kill -9 <PID>
   ```

2. **Model Files Missing**
   ```bash
   # Run the training pipeline first
   python main.py
   ```

3. **Dependencies Issues**
   ```bash
   # Reinstall dependencies
   pip uninstall -r requirements.txt
   pip install -r requirements.txt
   ```

4. **CORS Issues**
   - Check that CORS middleware is properly configured
   - Ensure API and frontend are on same domain/port

### Performance Optimization

1. **For Production**
   - Use Gunicorn with multiple workers
   - Add Redis for caching
   - Implement database for storing predictions
   - Add authentication and rate limiting

2. **For Demo**
   - Keep it simple and fast
   - Use pre-trained models
   - Minimize external dependencies

## 📈 Analytics Setup

### 1. Google Analytics
Add to your deployed dashboard:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### 2. GitHub Analytics
- Track repository views
- Monitor star/fork growth
- Analyze traffic sources

## 🎉 Success Metrics

Track these metrics for your portfolio:
- **Repository Stars**: Aim for 10+ stars
- **Forks**: Indicates interest in the project
- **Demo Views**: Track portfolio website analytics
- **Interview Questions**: Be ready to discuss technical decisions
- **Follow-up Opportunities**: Connect with people who show interest

---

**Ready to showcase your Marketing Mix Model System! 🚀📊** 