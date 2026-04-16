import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools

st.set_page_config(page_title="GAC Motors Forecast", page_icon="🚗", layout="wide")

st.title("🚗 GAC Motors GCC Export Forecast")
st.markdown("Predicting exported units for the next year using historical CAAM growth proxies and Facebook Prophet.")

@st.cache_data
def create_mock_data():
    dates = pd.date_range(start='2019-01-01', end='2023-12-01', freq='MS')
    base_volume = np.linspace(150, 1500, len(dates)) 
    noise = np.random.normal(0, 50, len(dates))
    seasonality = np.sin(np.arange(len(dates)) * (2 * np.pi / 12)) * 100
    y = np.maximum(base_volume + noise + seasonality, 0)
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    df['y'] = df['y'].astype(int)
    return df

@st.cache_resource
def optimize_prophet_model(df):
    param_grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'seasonality_mode': ['additive']
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  

    for params in all_params:
        m = Prophet(**params, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df)
        df_cv = cross_validation(m, initial='1095 days', period='90 days', horizon='180 days', parallel=None)
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    best_params = all_params[np.argmin(rmses)]
    
    final_model = Prophet(**best_params, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    final_model.fit(df)
    return final_model, best_params, min(rmses)

def main():
    df = create_mock_data()
    
    st.subheader("Historical Data (2019-2023)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['ds'], df['y'], marker='o', linestyle='-', color='#1f77b4')
    ax.set_title('GAC Motors Exports to GCC (Historical)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Exported Units')
    ax.grid(True)
    st.pyplot(fig)
    
    with st.spinner("Optimizing and Training Prophet model..."):
        model, best_params, best_rmse = optimize_prophet_model(df)
    
    st.success(f"Model Optimization Complete! Best RMSE: {best_rmse:.2f}")
    
    # Prediction
    st.subheader("Predictive Forecast (Next 12 Months)")
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)
    
    fig1 = model.plot(forecast)
    plt.title('GAC Motors Export Forecast to GCC')
    plt.xlabel('Date')
    plt.ylabel('Export Units')
    st.pyplot(fig1)
    
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
