import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

def train_models(tickers):
    """
    Train predictive models for a list of financial assets and save the resulting predictions.

    Args:
        tickers (list of str): List of financial asset symbols to train models on.

    Returns:
        tuple: A tuple containing DataFrame of predictions, dictionary of model performance results,
               and DataFrame of the historical data used for training.
    """
    # Fetch historical adjusted close price data from Yahoo Finance
    data = yf.download(tickers, start="2020-01-01", end="2024-06-19")['Adj Close']
    # Forward fill to handle any missing data points
    data.fillna(method='ffill', inplace=True)

    # Initialize dictionaries to store model results and final predictions
    model_results = {}
    predictions = {}

    # Loop through each ticker to preprocess data and train a model
    for ticker in tickers:
        # Define the number of days to predict into the future based on asset type
        future_days = 365 if ticker in ['BTC-USD', 'ETH-USD'] else 252

        # Create a future price column shifted backward by the future_days
        data[f'{ticker}_future'] = data[ticker].shift(-future_days)

        # Generate lagged returns as features
        for lag in [1, 5, 22, 60, 120]:
            data[f'{ticker}_return_{lag}'] = data[ticker].pct_change(periods=lag)

        # Generate simple moving averages as features
        for window in [10, 30, 60, 120]:
            data[f'{ticker}_sma_{window}'] = data[ticker].rolling(window=window).mean()

        # Remove rows with NaN values that could interfere with model training
        data_clean = data.dropna(subset=[col for col in data.columns if ticker in col])
        
        # If data is insufficient after cleaning, skip this ticker
        if data_clean.empty:
            print(f"No data available for training on {ticker} after cleaning. Skipping.")
            continue

        # Prepare feature matrix X and target vector y
        X = data_clean.filter(regex=f'^{ticker}_').drop(columns=[f'{ticker}_future'])
        y = data_clean[f'{ticker}_future']

        # Ensure there's enough data to perform the split
        if len(X) < 10:  # arbitrary minimum count check
            print(f"Not enough data for {ticker} to perform train/test split. Skipping.")
            continue

        # Split data into training and test sets without shuffling for time series integrity
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Split data into training and test sets without shuffling for time series integrity
        tscv = TimeSeriesSplit(n_splits=5)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 4]
        }
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Extract the best model and make a prediction for the last point in the test set
        best_model = grid_search.best_estimator_
        prediction = best_model.predict(X_test)
        predictions[ticker] = prediction[-1]
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        model_results[ticker] = {
            'RMSE': rmse,
            'Best Parameters': grid_search.best_params_
        }
    
    # Convert the predictions dictionary to a DataFrame for easy saving and manipulation
    predictions_df = pd.DataFrame(predictions, index=['predicted_price']).T
    predictions_df.index.name = 'ticker'
    
    return predictions_df, model_results, data

# List of tickers to model
tickers = [
    'BTC-USD', 'ETH-USD', 'GC=F', 'SI=F', 'SPY', 'QQQ', 'SOXX', 'VT', 'VO', 'VWO',
    'NVDA', 'MSFT', 'AAPL', 'GOOG', 'AMZN', 'META', 'TSM', 'TSLA', 'V', 'WMT',
    'XOM', 'ASML', 'PG', 'BAC', 'NFLX', 'KO', 'QCOM', 'AMD', 'PEP', 'SHEL',
    'DIS', 'BABA', 'MCD', 'VZ', 'NKE', 'JPM', 'GE', 'BA', 'SBUX', 'UL', 'PM', 'MO',
    'SHOP', 'UBER', 'SPOT'
]

# Train models and get predictions and historical data
predictions_df, model_results, historical_data = train_models(tickers)

# Save the predictions and historical data for later use
predictions_df.to_csv('predictions.csv')

print("Predictions and historical data saved successfully.")
