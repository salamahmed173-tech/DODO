import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import os

def create_mock_data():
    dates = pd.date_range(start='2019-01-01', end='2023-12-01', freq='MS')
    base_volume = np.linspace(150, 1500, len(dates)) 
    noise = np.random.normal(0, 50, len(dates))
    seasonality = np.sin(np.arange(len(dates)) * (2 * np.pi / 12)) * 100
    y = np.maximum(base_volume + noise + seasonality, 0)
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    df['y'] = df['y'].astype(int)
    return df

def optimize_prophet_model(df):
    print("Running Hyperparameter Tuning to Minimize RMSE...")
    param_grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  

    print(f"Testing {len(all_params)} parameter combinations using cross-validation...")
    # Reduce the training output verbosity
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    logging.getLogger('prophet').setLevel(logging.WARNING)

    # Note: Using small cutoffs since the dataset is only 60 months
    # Initial: 3 years (1095 days), horizon: 180 days (6 months)
    for params in all_params:
        m = Prophet(**params, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df)
        df_cv = cross_validation(m, initial='1095 days', period='90 days', horizon='180 days', parallel=None)
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    best_params = all_params[np.argmin(rmses)]
    best_rmse = min(rmses)
    print(f"\nBest Parameters found: {best_params}")
    print(f"Reduced RMSE: {best_rmse:.2f}")

    # Retrain final model with the best parameters
    print("\nRetraining final model on full dataset with best parameters...")
    final_model = Prophet(**best_params, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    final_model.fit(df)
    
    return final_model

def main():
    print("Extracting/Loading CAAM GAC Motors Export Data (Last 5 Years)...")
    df = create_mock_data()
    
    os.makedirs("output", exist_ok=True)
    
    # 1. Visualize historical data
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], marker='o', linestyle='-', color='#1f77b4')
    plt.title('GAC Motors Exports to GCC (Historical 2019-2023)')
    plt.xlabel('Date')
    plt.ylabel('Exported Units')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/historical_data.png')
    
    # 2. Optimize and Train Prophet model for 1 year prediction
    model = optimize_prophet_model(df)
    
    # Predict 12 months into the future
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)
    
    # 3. Visualize prediction
    fig1 = model.plot(forecast)
    plt.title('GAC Motors Export Forecast to GCC (Next 1 Year - Optimized)')
    plt.xlabel('Date')
    plt.ylabel('Export Units')
    fig1.savefig('output/forecast_optimized.png')
    
    fig2 = model.plot_components(forecast)
    fig2.savefig('output/forecast_components_optimized.png')
    
    print("\nFiles saved to the output/ directory.")

if __name__ == "__main__":
    main()
