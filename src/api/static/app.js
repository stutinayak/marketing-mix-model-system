// Marketing Mix Model Dashboard - JavaScript Application

// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINTS = {
    predict: '/predict',
    scenarioAnalysis: '/scenario-analysis',
    modelInfo: '/model-info',
    featureImportance: '/feature-importance',
    marketingAttribution: '/marketing-attribution',
    sampleScenarios: '/sample-scenarios',
    health: '/health'
};

// Global variables
let currentScenarios = null;
let featureImportanceChart = null;
let attributionChart = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    try {
        // Check API health
        await checkApiHealth();
        
        // Load initial data for insights tab
        loadModelInfo();
        loadFeatureImportance();
        loadMarketingAttribution();
        
        // Set up event listeners
        setupEventListeners();
        
        console.log('Dashboard initialized successfully');
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
        showError('Failed to connect to API. Please ensure the server is running.');
    }
}

// API Health Check
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.health}`);
        if (!response.ok) {
            throw new Error('API not responding');
        }
        const data = await response.json();
        updateApiStatus(true);
        return data;
    } catch (error) {
        updateApiStatus(false);
        throw error;
    }
}

function updateApiStatus(isConnected) {
    const statusElement = document.getElementById('apiStatus');
    if (isConnected) {
        statusElement.innerHTML = '<i class="fas fa-circle text-success"></i><span>API Connected</span>';
    } else {
        statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i><span>API Disconnected</span>';
    }
}

// Event Listeners Setup
function setupEventListeners() {
    // Prediction form submission
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Tab change events
    const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', handleTabChange);
    });
}

// Handle Tab Changes
function handleTabChange(event) {
    const targetTab = event.target.getAttribute('data-bs-target');
    
    switch (targetTab) {
        case '#insights':
            // Refresh insights data when tab is shown
            loadModelInfo();
            loadFeatureImportance();
            break;
        case '#attribution':
            // Refresh attribution data when tab is shown
            loadMarketingAttribution();
            break;
    }
}

// Prediction Form Handler
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    showLoadingModal();
    
    try {
        const formData = {
            tv_spend: parseFloat(document.getElementById('tvSpend').value),
            radio_spend: parseFloat(document.getElementById('radioSpend').value),
            print_spend: parseFloat(document.getElementById('printSpend').value),
            digital_spend: parseFloat(document.getElementById('digitalSpend').value),
            competitor_price: parseFloat(document.getElementById('competitorPrice').value),
            our_price: parseFloat(document.getElementById('ourPrice').value),
            holiday: parseInt(document.getElementById('holiday').value)
        };
        
        const prediction = await makePrediction(formData);
        displayPredictionResult(prediction);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Failed to make prediction. Please try again.');
    } finally {
        hideLoadingModal();
    }
}

// API Functions
async function makePrediction(inputData) {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.predict}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

async function runScenarioAnalysisRequest(baseScenario, scenarios) {
    const requestData = {
        base_scenario: baseScenario,
        scenarios: scenarios
    };
    
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.scenarioAnalysis}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

async function getModelInfo() {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.modelInfo}`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}

async function getFeatureImportance() {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.featureImportance}`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}

async function getMarketingAttribution() {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.marketingAttribution}`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}

async function getSampleScenarios() {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.sampleScenarios}`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}

// Display Functions
function displayPredictionResult(prediction) {
    const resultContainer = document.getElementById('predictionResult');
    
    const html = `
        <div class="fade-in">
            <div class="prediction-value">
                $${prediction.prediction.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}
            </div>
            <p class="text-muted mb-3">Predicted Sales</p>
            
            <div class="confidence-interval">
                <h6><i class="fas fa-chart-area me-2"></i>Confidence Interval</h6>
                <div class="row">
                    <div class="col-md-4">
                        <div class="text-center">
                            <div class="metric-value">$${prediction.confidence_interval.lower.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}</div>
                            <div class="metric-label">Lower Bound</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="text-center">
                            <div class="metric-value">$${prediction.confidence_interval.upper.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}</div>
                            <div class="metric-label">Upper Bound</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="text-center">
                            <div class="metric-value">±${prediction.confidence_interval.std.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}</div>
                            <div class="metric-label">Standard Dev</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="mt-3">
                <small class="text-muted">
                    <i class="fas fa-clock me-1"></i>
                    Generated at ${new Date(prediction.timestamp).toLocaleString()}
                </small>
            </div>
        </div>
    `;
    
    resultContainer.innerHTML = html;
}

function displayScenarioResults(scenarioData) {
    const resultsContainer = document.getElementById('scenarioResults');
    
    let html = `
        <div class="fade-in">
            <div class="scenario-card">
                <div class="scenario-name">Base Scenario</div>
                <div class="scenario-metrics">
                    <div class="metric">
                        <div class="metric-value">$${scenarioData.base_prediction.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}</div>
                        <div class="metric-label">Predicted Sales</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add scenario comparisons
    Object.entries(scenarioData.scenarios).forEach(([scenarioName, scenario]) => {
        const changeColor = scenario.change >= 0 ? 'text-success' : 'text-danger';
        const changeIcon = scenario.change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
        
        html += `
            <div class="scenario-card fade-in">
                <div class="scenario-name">${scenarioName}</div>
                <div class="scenario-metrics">
                    <div class="metric">
                        <div class="metric-value">$${scenario.prediction.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}</div>
                        <div class="metric-label">Predicted Sales</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value ${changeColor}">
                            <i class="fas ${changeIcon} me-1"></i>
                            $${Math.abs(scenario.change).toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}
                        </div>
                        <div class="metric-label">Change</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value ${changeColor}">
                            ${scenario.change_pct >= 0 ? '+' : ''}${scenario.change_pct.toFixed(1)}%
                        </div>
                        <div class="metric-label">% Change</div>
                    </div>
                </div>
            </div>
        `;
    });
    
    resultsContainer.innerHTML = html;
}

function displayModelInfo(modelInfo) {
    const container = document.getElementById('modelInfo');
    
    const html = `
        <div class="fade-in">
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label class="form-label text-muted">Model Name</label>
                        <div class="fw-bold">${modelInfo.model_name}</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label class="form-label text-muted">Model Type</label>
                        <div class="fw-bold">${modelInfo.model_type}</div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label class="form-label text-muted">Features</label>
                        <div class="fw-bold">${modelInfo.feature_count}</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label class="form-label text-muted">Training Date</label>
                        <div class="fw-bold">${new Date(modelInfo.training_date).toLocaleDateString()}</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function displayFeatureImportance(featureData) {
    const container = document.getElementById('featureImportance');
    
    // Create chart
    const ctx = document.createElement('canvas');
    ctx.id = 'featureImportanceChart';
    container.innerHTML = '<div class="chart-container"><canvas id="featureImportanceChart"></canvas></div>';
    
    // Destroy existing chart if it exists
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }
    
    // Create new chart
    featureImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureData.features.slice(0, 10), // Top 10 features
            datasets: [{
                label: 'Feature Importance',
                data: featureData.importance.slice(0, 10),
                backgroundColor: 'rgba(13, 110, 253, 0.8)',
                borderColor: 'rgba(13, 110, 253, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Features'
                    }
                }
            }
        }
    });
}

function displayMarketingAttribution(attributionData) {
    const resultsContainer = document.getElementById('attributionResults');
    const summaryContainer = document.getElementById('attributionSummary');
    
    // Create pie chart
    const ctx = document.createElement('canvas');
    ctx.id = 'attributionChart';
    resultsContainer.innerHTML = '<div class="chart-container"><canvas id="attributionChart"></canvas></div>';
    
    // Destroy existing chart if it exists
    if (attributionChart) {
        attributionChart.destroy();
    }
    
    // Create new chart
    attributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: attributionData.channels,
            datasets: [{
                data: attributionData.attribution_pct,
                backgroundColor: [
                    'rgba(13, 110, 253, 0.8)',
                    'rgba(25, 135, 84, 0.8)',
                    'rgba(255, 193, 7, 0.8)',
                    'rgba(220, 53, 69, 0.8)',
                    'rgba(108, 117, 125, 0.8)',
                    'rgba(13, 202, 240, 0.8)'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    
    // Create summary
    let summaryHtml = '<div class="fade-in">';
    attributionData.channels.forEach((channel, index) => {
        const percentage = attributionData.attribution_pct[index];
        summaryHtml += `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span class="fw-bold">${channel}</span>
                <span class="badge bg-primary">${percentage.toFixed(1)}%</span>
            </div>
        `;
    });
    summaryHtml += '</div>';
    
    summaryContainer.innerHTML = summaryHtml;
}

// Scenario Analysis Functions
async function loadSampleScenarios() {
    try {
        showLoadingModal();
        const scenarios = await getSampleScenarios();
        currentScenarios = scenarios;
        displayScenarioBuilder(scenarios);
    } catch (error) {
        console.error('Error loading sample scenarios:', error);
        showError('Failed to load sample scenarios.');
    } finally {
        hideLoadingModal();
    }
}

function displayScenarioBuilder(scenarios) {
    const container = document.getElementById('scenarioBuilder');
    
    let html = `
        <div class="fade-in">
            <h6>Base Scenario</h6>
            <div class="mb-3">
                <small class="text-muted">TV: $${scenarios.base_scenario.tv_spend.toLocaleString()}</small><br>
                <small class="text-muted">Radio: $${scenarios.base_scenario.radio_spend.toLocaleString()}</small><br>
                <small class="text-muted">Print: $${scenarios.base_scenario.print_spend.toLocaleString()}</small><br>
                <small class="text-muted">Digital: $${scenarios.base_scenario.digital_spend.toLocaleString()}</small>
            </div>
            
            <h6>Scenarios to Compare</h6>
    `;
    
    Object.keys(scenarios.scenarios).forEach(scenarioName => {
        html += `<div class="mb-2"><small class="text-muted">• ${scenarioName}</small></div>`;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

async function runScenarioAnalysis() {
    if (!currentScenarios) {
        showError('Please load sample scenarios first.');
        return;
    }
    
    try {
        showLoadingModal();
        const results = await runScenarioAnalysisRequest(
            currentScenarios.base_scenario,
            currentScenarios.scenarios
        );
        displayScenarioResults(results);
    } catch (error) {
        console.error('Error running scenario analysis:', error);
        showError('Failed to run scenario analysis.');
    } finally {
        hideLoadingModal();
    }
}

// Data Loading Functions
async function loadModelInfo() {
    try {
        const modelInfo = await getModelInfo();
        displayModelInfo(modelInfo);
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelInfo').innerHTML = '<div class="error-message">Failed to load model information.</div>';
    }
}

async function loadFeatureImportance() {
    try {
        const featureData = await getFeatureImportance();
        displayFeatureImportance(featureData);
    } catch (error) {
        console.error('Error loading feature importance:', error);
        document.getElementById('featureImportance').innerHTML = '<div class="error-message">Failed to load feature importance.</div>';
    }
}

async function loadMarketingAttribution() {
    try {
        const attributionData = await getMarketingAttribution();
        displayMarketingAttribution(attributionData);
    } catch (error) {
        console.error('Error loading marketing attribution:', error);
        document.getElementById('attributionResults').innerHTML = '<div class="error-message">Failed to load marketing attribution.</div>';
    }
}

// Utility Functions
function showLoadingModal() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

function hideLoadingModal() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) {
        modal.hide();
    }
}

function showError(message) {
    // Create a temporary error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the container
    const container = document.querySelector('.container-fluid');
    container.insertBefore(errorDiv, container.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.remove();
        }
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'alert alert-success alert-dismissible fade show';
    successDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container-fluid');
    container.insertBefore(successDiv, container.firstChild);
    
    setTimeout(() => {
        if (successDiv.parentNode) {
            successDiv.remove();
        }
    }, 3000);
}

// Global functions for onclick handlers
window.loadSampleScenarios = loadSampleScenarios;
window.runScenarioAnalysis = runScenarioAnalysis; 