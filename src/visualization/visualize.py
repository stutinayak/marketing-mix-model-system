# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_time_series_plots(df, save_dir):
    """Create time series plots for marketing data."""
    logger = logging.getLogger(__name__)
    logger.info('Creating time series plots...')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Marketing Mix Time Series Analysis', fontsize=16, fontweight='bold')
    
    # 1. Marketing Spend Over Time
    spend_cols = ['tv_spend', 'radio_spend', 'print_spend', 'digital_spend']
    for col in spend_cols:
        axes[0, 0].plot(df['date'], df[col], label=col.replace('_', ' ').title(), linewidth=2)
    axes[0, 0].set_title('Marketing Spend Over Time')
    axes[0, 0].set_ylabel('Spend ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Sales Over Time
    axes[0, 1].plot(df['date'], df['sales'], color='green', linewidth=2)
    axes[0, 1].set_title('Sales Over Time')
    axes[0, 1].set_ylabel('Sales ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ROAS Over Time
    axes[1, 0].plot(df['date'], df['roas'], color='purple', linewidth=2)
    axes[1, 0].set_title('Return on Ad Spend (ROAS) Over Time')
    axes[1, 0].set_ylabel('ROAS')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Price Comparison
    axes[1, 1].plot(df['date'], df['our_price'], label='Our Price', linewidth=2)
    axes[1, 1].plot(df['date'], df['competitor_price'], label='Competitor Price', linewidth=2)
    axes[1, 1].set_title('Price Comparison Over Time')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Total Spend vs Sales
    axes[2, 0].scatter(df['total_spend'], df['sales'], alpha=0.6, color='blue')
    axes[2, 0].set_title('Total Spend vs Sales')
    axes[2, 0].set_xlabel('Total Spend ($)')
    axes[2, 0].set_ylabel('Sales ($)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Seasonal Pattern
    monthly_sales = df.groupby(df['date'].dt.month)['sales'].mean()
    axes[2, 1].bar(monthly_sales.index, monthly_sales.values, color='orange', alpha=0.7)
    axes[2, 1].set_title('Average Sales by Month')
    axes[2, 1].set_xlabel('Month')
    axes[2, 1].set_ylabel('Average Sales ($)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info('Time series plots saved')


def create_feature_importance_plot(feature_importance, save_dir):
    """Create feature importance visualization."""
    logger = logging.getLogger(__name__)
    logger.info('Creating feature importance plot...')
    
    # Get top 20 features
    top_features = feature_importance.head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_features['importance'])
    
    # Color bars based on feature type
    colors = []
    for feature in top_features['feature']:
        if 'tv_spend' in feature:
            colors.append('#FF6B6B')
        elif 'radio_spend' in feature:
            colors.append('#4ECDC4')
        elif 'print_spend' in feature:
            colors.append('#45B7D1')
        elif 'digital_spend' in feature:
            colors.append('#96CEB4')
        elif 'sales' in feature:
            colors.append('#FFEAA7')
        else:
            colors.append('#DDA0DD')
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info('Feature importance plot saved')


def create_model_performance_plots(metrics_report, save_dir):
    """Create model performance comparison plots."""
    logger = logging.getLogger(__name__)
    logger.info('Creating model performance plots...')
    
    # Extract metrics for plotting
    model_names = []
    r2_scores = []
    rmse_scores = []
    
    for model_name, metrics in metrics_report['all_models'].items():
        model_names.append(model_name)
        r2_scores.append(metrics['r2'])
        rmse_scores.append(metrics['rmse'])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² Score comparison
    bars1 = ax1.bar(model_names, r2_scores, color='skyblue', alpha=0.7)
    ax1.set_title('Model R² Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars1, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE comparison
    bars2 = ax2.bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
    ax2.set_title('Model RMSE Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars2, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info('Model performance plots saved')


def create_scenario_analysis_plot(scenario_results, save_dir):
    """Create scenario analysis visualization."""
    logger = logging.getLogger(__name__)
    logger.info('Creating scenario analysis plot...')
    
    # Prepare data for plotting
    scenarios = []
    predictions = []
    changes = []
    
    for scenario_name, result in scenario_results.items():
        if scenario_name != 'base':
            scenarios.append(scenario_name.replace('_', ' ').title())
            predictions.append(result['prediction'])
            changes.append(result['change_pct'])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sales predictions
    colors = ['green' if change > 0 else 'red' for change in changes]
    bars1 = ax1.bar(scenarios, predictions, color=colors, alpha=0.7)
    ax1.set_title('Sales Predictions by Scenario', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Sales ($)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, pred in zip(bars1, predictions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'${pred:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # Percentage changes
    bars2 = ax2.bar(scenarios, changes, color=colors, alpha=0.7)
    ax2.set_title('Percentage Change from Base Scenario', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Change (%)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, change in zip(bars2, changes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if change > 0 else -1), 
                f'{change:.1f}%', ha='center', va='bottom' if change > 0 else 'top', fontsize=10)
    
    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'scenario_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info('Scenario analysis plot saved')


def create_interactive_dashboard(df, save_dir):
    """Create an interactive Plotly dashboard."""
    logger = logging.getLogger(__name__)
    logger.info('Creating interactive dashboard...')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Marketing Spend Over Time', 'Sales Over Time', 
                       'ROAS Over Time', 'Price Comparison',
                       'Spend vs Sales Correlation', 'Monthly Sales Pattern'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Marketing Spend Over Time
    spend_cols = ['tv_spend', 'radio_spend', 'print_spend', 'digital_spend']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for col, color in zip(spend_cols, colors):
        fig.add_trace(
            go.Scatter(x=df['date'], y=df[col], name=col.replace('_', ' ').title(),
                      line=dict(color=color, width=2)),
            row=1, col=1
        )
    
    # 2. Sales Over Time
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['sales'], name='Sales',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # 3. ROAS Over Time
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['roas'], name='ROAS',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # 4. Price Comparison
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['our_price'], name='Our Price',
                  line=dict(color='blue', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['competitor_price'], name='Competitor Price',
                  line=dict(color='red', width=2)),
        row=2, col=2
    )
    
    # 5. Spend vs Sales Correlation
    fig.add_trace(
        go.Scatter(x=df['total_spend'], y=df['sales'], mode='markers',
                  name='Spend vs Sales', marker=dict(color='blue', opacity=0.6)),
        row=3, col=1
    )
    
    # 6. Monthly Sales Pattern
    monthly_sales = df.groupby(df['date'].dt.month)['sales'].mean()
    fig.add_trace(
        go.Bar(x=monthly_sales.index, y=monthly_sales.values,
               name='Monthly Sales', marker_color='orange'),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Marketing Mix Model Interactive Dashboard",
        showlegend=True,
        height=900,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_xaxes(title_text="Total Spend ($)", row=3, col=1)
    fig.update_xaxes(title_text="Month", row=3, col=2)
    
    fig.update_yaxes(title_text="Spend ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sales ($)", row=1, col=2)
    fig.update_yaxes(title_text="ROAS", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=2)
    fig.update_yaxes(title_text="Sales ($)", row=3, col=1)
    fig.update_yaxes(title_text="Average Sales ($)", row=3, col=2)
    
    # Save interactive dashboard
    fig.write_html(save_dir / 'interactive_dashboard.html')
    
    logger.info('Interactive dashboard saved')


def main():
    """Create all visualizations for the marketing mix model."""
    logger = logging.getLogger(__name__)
    logger.info('Creating visualizations for marketing mix model')
    
    try:
        # Create figures directory
        figures_dir = Path(__file__).resolve().parents[2] / 'reports' / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        from src.data.make_dataset import main as process_data
        df_clean, _ = process_data()
        
        # Create time series plots
        create_time_series_plots(df_clean, figures_dir)
        
        # Load model and create feature importance plot
        try:
            from src.models.train_model import main as train_model
            model_service, metrics_report = train_model()
            
            # Get feature importance
            feature_importance = model_service.get_feature_importance()
            if feature_importance is not None:
                create_feature_importance_plot(feature_importance, figures_dir)
            
            # Create model performance plots
            create_model_performance_plots(metrics_report, figures_dir)
            
        except Exception as e:
            logger.warning(f'Could not create model-based visualizations: {e}')
        
        # Test scenario analysis visualization
        try:
            from src.models.predict_model import main as test_predictions
            prediction_service = test_predictions()
            
            # Generate sample scenarios
            from src.models.predict_model import generate_sample_scenarios
            base_scenario, scenarios = generate_sample_scenarios()
            
            # Get scenario results
            scenario_results = prediction_service.scenario_analysis(base_scenario, scenarios)
            
            # Create scenario analysis plot
            create_scenario_analysis_plot(scenario_results, figures_dir)
            
        except Exception as e:
            logger.warning(f'Could not create scenario analysis visualization: {e}')
        
        # Create interactive dashboard
        create_interactive_dashboard(df_clean, figures_dir)
        
        logger.info('All visualizations created successfully')
        logger.info(f'Visualizations saved to: {figures_dir}')
        
    except Exception as e:
        logger.error(f'Error creating visualizations: {e}')
        raise


@click.command()
def cli_main():
    """CLI interface for visualization creation."""
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
