# Marketing Mix Model Dashboard Demo

## 🚀 Quick Start

1. **Start the Dashboard**
   ```bash
   python run_dashboard.py
   ```

2. **Open Your Browser**
   Navigate to: http://localhost:8000

3. **Start Exploring!**

## 📊 Dashboard Features

### 1. Sales Prediction Tab
- **Input Marketing Data**: Enter TV, Radio, Print, and Digital spend amounts
- **Set Pricing**: Configure competitor and your product prices
- **Holiday Period**: Toggle holiday indicator
- **Get Instant Results**: See predicted sales with confidence intervals

**Example Input:**
- TV Spend: $50,000
- Radio Spend: $20,000
- Print Spend: $15,000
- Digital Spend: $30,000
- Competitor Price: $50
- Our Price: $55
- Holiday: No

**Expected Output:**
- Predicted Sales: ~$125,000 (varies based on model)
- Confidence Interval: Lower/Upper bounds
- Standard Deviation: Uncertainty measure

### 2. Scenario Analysis Tab
- **Load Sample Scenarios**: Click to load predefined marketing scenarios
- **Run Analysis**: Compare different marketing strategies
- **View Results**: See how each scenario affects sales

**Sample Scenarios Include:**
- Increase TV spend by 20%
- Boost digital marketing by 30%
- Reduce print advertising
- Optimize channel mix

### 3. Model Insights Tab
- **Model Information**: View model type, features, training date
- **Feature Importance**: Interactive bar chart showing top 10 features
- **Real-time Loading**: Data refreshes when tab is opened

### 4. Marketing Attribution Tab
- **Channel Contribution**: Pie chart showing each channel's impact
- **Attribution Summary**: Percentage breakdown by channel
- **Visual Analysis**: Easy-to-understand graphical representation

## 🎯 Use Cases

### Marketing Manager
1. **Budget Planning**: Use Sales Prediction to estimate ROI for different spend levels
2. **Channel Optimization**: Use Scenario Analysis to find the best marketing mix
3. **Performance Review**: Use Marketing Attribution to understand channel effectiveness

### Data Analyst
1. **Model Validation**: Check feature importance and model performance
2. **Sensitivity Analysis**: Test different scenarios and see impact
3. **Reporting**: Generate insights for stakeholders

### Business Executive
1. **Quick Insights**: Get instant sales forecasts for decision making
2. **What-if Analysis**: Test different marketing strategies
3. **ROI Assessment**: Understand marketing channel effectiveness

## 🔧 Technical Features

### Frontend
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Bootstrap 5 with custom gradients and animations
- **Interactive Charts**: Chart.js for data visualization
- **Real-time Updates**: Live API calls with loading states

### Backend Integration
- **RESTful API**: All dashboard actions use the FastAPI endpoints
- **Error Handling**: Graceful error messages and fallbacks
- **Health Monitoring**: Real-time API status indicator
- **CORS Support**: Cross-origin requests enabled

### User Experience
- **Intuitive Navigation**: Tab-based interface for easy switching
- **Form Validation**: Input validation and helpful error messages
- **Loading States**: Visual feedback during API calls
- **Success/Error Messages**: Clear feedback for all actions

## 📱 Mobile Experience

The dashboard is fully responsive and works great on mobile devices:
- Touch-friendly interface
- Optimized layouts for small screens
- Swipe gestures for navigation
- Fast loading times

## 🔍 Troubleshooting

### Common Issues

1. **Dashboard Won't Load**
   - Ensure API server is running: `python run_dashboard.py`
   - Check browser console for errors
   - Verify port 8000 is available

2. **API Connection Failed**
   - Check if the model files exist in `models/` directory
   - Ensure all dependencies are installed
   - Check server logs for error messages

3. **Charts Not Displaying**
   - Ensure JavaScript is enabled
   - Check for ad blockers that might block Chart.js
   - Refresh the page and try again

### Getting Help

- **API Documentation**: Visit http://localhost:8000/docs
- **Health Check**: Visit http://localhost:8000/health
- **Console Logs**: Check browser developer tools for detailed errors

## 🎨 Customization

The dashboard can be easily customized:

### Styling
- Modify `src/api/static/styles.css` for custom colors and themes
- Update gradients and animations
- Change fonts and spacing

### Functionality
- Add new API endpoints in `src/api/app.py`
- Extend JavaScript functions in `src/api/static/app.js`
- Modify HTML structure in `src/api/static/index.html`

### Data Sources
- Connect to different models by updating the API
- Add new visualization types
- Integrate with external data sources

## 🚀 Next Steps

After exploring the dashboard:

1. **Train Your Own Model**: Use the training pipeline to create models with your data
2. **Customize Features**: Add your own marketing channels and features
3. **Deploy to Production**: Set up the API on a production server
4. **Integrate with Tools**: Connect to your existing marketing and analytics tools

---

**Happy Marketing Analytics! 📊✨** 