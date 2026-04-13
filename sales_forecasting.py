# Sales Forecasting Using Time Series Models
# Technologies: LSTM, ARIMA, Prophet, TensorFlow, Apache Airflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")

def generate_sample_sales_data(n=365*3):
    """Generate sample daily sales data with trend and seasonality."""
    np.random.seed(42)
    dates = pd.date_range(start='2021-01-01', periods=n, freq='D')
    trend = np.linspace(1000, 2000, n)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(0, 50, n)
    sales = trend + seasonal + noise
    df = pd.DataFrame({'date': dates, 'sales': sales.clip(min=0)})
    return df

def prepare_lstm_data(data, lookback=30):
    """Prepare data for LSTM model."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(lookback=30):
    """Build LSTM model for time series forecasting."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(df):
    """Train LSTM model."""
    print("\nTraining LSTM Model...")
    sales_data = df['sales'].values
    train_size = int(len(sales_data) * 0.8)
    train_data = sales_data[:train_size]
    test_data = sales_data[train_size:]
    
    X_train, y_train, scaler = prepare_lstm_data(train_data, lookback=30)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    model = build_lstm_model(lookback=30)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Predictions
    X_test, y_test, _ = prepare_lstm_data(test_data, lookback=30)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_actual, predictions)
    mse = mean_squared_error(y_test_actual, predictions)
    print(f"LSTM - MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}")
    return predictions, y_test_actual

def train_arima(df):
    """Train ARIMA model."""
    print("\nTraining ARIMA Model...")
    train_size = int(len(df) * 0.8)
    train = df['sales'][:train_size]
    test = df['sales'][train_size:]
    
    model = ARIMA(train, order=(5, 1, 2))
    fitted_model = model.fit()
    
    forecast = fitted_model.forecast(steps=len(test))
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    print(f"ARIMA - MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}")
    return forecast.values, test.values

def train_prophet(df):
    """Train Prophet model."""
    print("\nTraining Prophet Model...")
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size][['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    test_df = df[train_size:][['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(train_df)
    
    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)
    predictions = forecast['yhat'][-len(test_df):].values
    
    mae = mean_absolute_error(test_df['y'], predictions)
    mse = mean_squared_error(test_df['y'], predictions)
    print(f"Prophet - MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}")
    return predictions, test_df['y'].values

def visualize_results(actual, lstm_pred, arima_pred, prophet_pred):
    """Visualize forecasts from all models."""
    plt.figure(figsize=(14, 6))
    plt.plot(actual, label='Actual Sales', color='black', linewidth=2)
    plt.plot(lstm_pred, label='LSTM Forecast', linestyle='--')
    plt.plot(arima_pred, label='ARIMA Forecast', linestyle='--')
    plt.plot(prophet_pred, label='Prophet Forecast', linestyle='--')
    plt.title('Sales Forecasting - Model Comparison')
    plt.xlabel('Time Period')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_comparison.png', dpi=150)
    print("\nForecast comparison plot saved: forecast_comparison.png")
    plt.close()

def main():
    print("Sales Forecasting Using Time Series Models")
    print("="*60)
    
    # Generate sample data
    df = generate_sample_sales_data(n=365*3)
    print(f"Dataset size: {len(df)} days")
    
    # Train models
    lstm_pred, lstm_actual = train_lstm(df)
    arima_pred, arima_actual = train_arima(df)
    prophet_pred, prophet_actual = train_prophet(df)
    
    # Ensure same length for visualization
    min_len = min(len(lstm_actual), len(arima_actual), len(prophet_actual))
    visualize_results(
        arima_actual[-min_len:],
        lstm_pred[-min_len:].flatten(),
        arima_pred[-min_len:],
        prophet_pred[-min_len:]
    )
    
    print("\nForecasting complete! All models trained and evaluated.")

if __name__ == '__main__':
    main()
