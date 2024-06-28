from flask import Flask, render_template, request, redirect, url_for
import re
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# Initialize flask app
app = Flask(__name__)

# Risk scores mapping: Maps user responses to a numerical risk level
risk_scores = {
    "I have extensive experience and regularly trade various financial instruments.": 5,
    "I have moderate experience and make several trades a year.": 4,
    "I have limited experience and have made a few trades.": 3,
    "I have no experience and have never traded.": 2,
    "I am new to investing but eager to learn.": 1,
    "I have no interest in learning about investing.": 0,

    "I am very knowledgeable and keep up-to-date with market trends.": 5,
    "I am knowledgeable and understand the basics.": 4,
    "I have some knowledge but find it complex.": 3,
    "I have limited knowledge and find it confusing.": 2,
    "I have minimal knowledge but am willing to learn.": 1,
    "I have no knowledge and prefer to leave it to professionals.": 0,

    "I would buy more to take advantage of the lower prices.": 5,
    "I would hold my investments and wait for recovery.": 4,
    "I would consider reallocating to less volatile assets.": 3,
    "I would sell some of my investments to reduce exposure.": 2,
    "I would sell all my investments to avoid further losses.": 1,
    "I would panic and never invest again.": 0,

    "Maximizing capital growth aggressively.": 5,
    "Achieving high returns with moderate risk.": 4,
    "Balancing growth and income with controlled risk.": 3,
    "Generating a steady income with minimal risk.": 2,
    "Preserving capital with very low risk.": 1,
    "Keeping money safe without any risk of loss.": 0,

    "Exciting and full of opportunities.": 5,
    "Necessary for achieving high returns.": 4,
    "Acceptable if managed properly.": 3,
    "Worrisome but manageable.": 2,
    "Stressful and risky.": 1,
    "Unacceptable and too risky.": 0,

    "Invest all available funds immediately.": 5,
    "Invest a large portion of my funds.": 4,
    "Invest a moderate amount after careful consideration.": 3,
    "Invest a small portion cautiously.": 2,
    "Avoid the investment but keep an eye on it.": 1,
    "Completely avoid the investment.": 0,

    "I see it as an opportunity to invest more.": 5,
    "I remain calm and stick to my investment plan.": 4,
    "I monitor my investments more closely.": 3,
    "I get concerned and consider making changes.": 2,
    "I get anxious and think about reducing my investments.": 1,
    "I get very anxious and consider exiting the market.": 0,

    "I prioritize short-term gains and am willing to take high risks.": 5,
    "I focus on both short-term gains and long-term growth with balanced risks.": 4,
    "I prioritize long-term growth with moderate short-term risks.": 3,
    "I prefer long-term stability with minimal short-term risks.": 2,
    "I avoid short-term risks entirely and focus on long-term preservation.": 1,
    "I avoid both short-term and long-term risks as much as possible.": 0,

    "Based on thorough research and analysis, taking high risks.": 5,
    "After consulting various sources and considering moderate risks.": 4,
    "After some research, preferring balanced risks.": 3,
    "By following conservative advice, minimizing risks.": 2,
    "Based on professional advice, avoiding risks.": 1,
    "I let professionals handle it entirely, avoiding risks.": 0,

    "Completely comfortable; I am aware of the risks.": 5,
    "Somewhat comfortable; I can tolerate moderate losses.": 4,
    "Neutral; I prefer not to think about losses.": 3,
    "Uncomfortable; I prefer safer investments.": 2,
    "Very uncomfortable; I avoid risky investments entirely.": 1
}

# Define the questions for the investment profile questionnaire
questions =  [
        "How would you describe your overall experience with investing in financial markets?",
        "How would you rate your knowledge of financial products and markets?",
        "If your portfolio lost 20% of its value in a month due to market volatility, how would you react?",
        "What is your primary financial goal for investing?",
        "How do you perceive the risk associated with investing in stocks?",
        "When faced with a financial opportunity with a high potential return but significant risk, what would you do?",
        "How do you react to news about economic downturns or financial crises?",
        "How do you prioritize short-term gains versus long-term growth in your investment strategy?",
        "How do you usually make investment decisions?",
        "How comfortable are you with the possibility of losing some or all of your investment?",
        "What is your investment horizon?"
    ]

# Define the options for the investment profile questionnaire
options = [
        ["I have extensive experience and regularly trade various financial instruments.",
         "I have moderate experience and make several trades a year.",
         "I have limited experience and have made a few trades.",
         "I have no experience and have never traded.",
         "I am new to investing but eager to learn.",
         "I have no interest in learning about investing."],
        
        ["I am very knowledgeable and keep up-to-date with market trends.",
         "I am knowledgeable and understand the basics.",
         "I have some knowledge but find it complex.",
         "I have limited knowledge and find it confusing.",
         "I have minimal knowledge but am willing to learn.",
         "I have no knowledge and prefer to leave it to professionals."],
        
        ["I would buy more to take advantage of the lower prices.",
         "I would hold my investments and wait for recovery.",
         "I would consider reallocating to less volatile assets.",
         "I would sell some of my investments to reduce exposure.",
         "I would sell all my investments to avoid further losses.",
         "I would panic and never invest again."],
        
        ["Maximizing capital growth aggressively.",
         "Achieving high returns with moderate risk.",
         "Balancing growth and income with controlled risk.",
         "Generating a steady income with minimal risk.",
         "Preserving capital with very low risk.",
         "Keeping money safe without any risk of loss."],
        
        ["Exciting and full of opportunities.",
         "Necessary for achieving high returns.",
         "Acceptable if managed properly.",
         "Worrisome but manageable.",
         "Stressful and risky.",
         "Unacceptable and too risky."],
        
        ["Invest all available funds immediately.",
         "Invest a large portion of my funds.",
         "Invest a moderate amount after careful consideration.",
         "Invest a small portion cautiously.",
         "Avoid the investment but keep an eye on it.",
         "Completely avoid the investment."],
        
        ["I see it as an opportunity to invest more.",
         "I remain calm and stick to my investment plan.",
         "I monitor my investments more closely.",
         "I get concerned and consider making changes.",
         "I get anxious and think about reducing my investments.",
         "I get very anxious and consider exiting the market."],
        
        ["I prioritize short-term gains and am willing to take high risks.",
         "I focus on both short-term gains and long-term growth with balanced risks.",
         "I prioritize long-term growth with moderate short-term risks.",
         "I prefer long-term stability with minimal short-term risks.",
         "I avoid short-term risks entirely and focus on long-term preservation.",
         "I avoid both short-term and long-term risks as much as possible."],
        
        ["Based on thorough research and analysis, taking high risks.",
         "After consulting various sources and considering moderate risks.",
         "After some research, preferring balanced risks.",
         "By following conservative advice, minimizing risks.",
         "Based on professional advice, avoiding risks.",
         "I let professionals handle it entirely, avoiding risks."],
        
        ["Completely comfortable; I am aware of the risks.",
         "Somewhat comfortable; I can tolerate moderate losses.",
         "Neutral; I prefer not to think about losses.",
         "Uncomfortable; I prefer safer investments.",
         "Very uncomfortable; I avoid risky investments entirely."],
        
        ["1 year", "5 years", "10 years", "15 or more years"]
    ]



# Fetch financial data with names from Yahoo Finance
def fetch_data_with_names(tickers):
    """
    Downloads and computes average returns and covariance of stock tickers, 
    and retrieves long names for each ticker using yfinance. This is for the long term horizon. 
    
    Args:
        tickers (list): List of stock tickers
    
    Returns:
        tuple: Mean returns, covariance matrix, and dictionary of ticker names
    """
    data = yf.download(tickers, start="2010-01-01", end="2024-06-27")['Adj Close']
    returns = data.pct_change().dropna()
    names = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            names[ticker] = stock.info['longName']
        except Exception as e:
            names[ticker] = "*"  # Fallback if the name can't be retrieved
    return returns.mean() * 252, returns.cov() * 252, names


# Fetch model predictions from CSV file
def fetch_predictions():
    """
    Loads predicted prices that are predicted using the machine learning model 
    (refer to 'models/machine_learning_model.py' for the model that is used)
    from a CSV file into a DataFrame. These predictions are used for the one year horizon 
    portfolio optimization

    Returns:
        DataFrame: Predicted prices indexed by ticker symbols
    """
    return pd.read_csv('data/predictions.csv', index_col='ticker')

# Define the objective function for the optimization
def objective_function(weights, mean_returns, cov_matrix, risk_free_rate=0.05):
    """
    Calculates the negative Sharpe ratio of a portfolio. 
    For the long term horizon. 
    
    Args:
        weights (ndarray): Portfolio weights
        mean_returns (ndarray): Mean returns for each asset
        cov_matrix (DataFrame): Covariance matrix of returns
        risk_free_rate (float): Risk-free rate
    
    Returns:
        float: Negative Sharpe ratio
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

# Optimize the portfolio for one year horizon
def optimize_portfolio_one_year(predictions, data, tickers, bounds):
    """
    Optimizes a portfolio for a one-year investment horizon based on predicted prices.
    
    This function calculates expected returns based on the predicted future prices and
    current prices. It then minimizes the negative Sharpe ratio to determine the optimal
    asset weights that maximize the Sharpe ratio.
    
    Args:
        predictions (DataFrame): DataFrame containing predicted prices for the tickers.
        data (DataFrame): Historical price data for the tickers.
        tickers (list): List of ticker symbols being considered.
        bounds (list of tuple): Bounds for the optimizer (min, max weights per asset).
    
    Returns:
        OptimizationResult: Result object from the scipy.optimize.minimize function.
    """
    # Calculate expected returns based on prediction and last known price
    expected_returns = {ticker: (predictions.loc[ticker, 'predicted_price'] - data[ticker].iloc[-1]) / data[ticker].iloc[-1] for ticker in tickers}
    # Compute daily renturns and annual covariance matrix 
    returns = data[tickers].pct_change().dropna()
    cov_matrix = returns.cov() * 252
    risk_free_rate = 0.05
    
    # Define the negative Sharpe ratio function
    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        p_ret = np.dot(weights, expected_returns) # Portfolio return
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) # Portfolio volatility
        return -(p_ret - risk_free_rate) / p_vol # Negative Sharpe ratio

    # Constraint to ensure all weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    # Starting guess (equally weighted)
    initial_weights = np.array([1.0 / len(tickers)] * len(tickers))
    # List of expected returns for optimization function
    expected_returns_list = [expected_returns[ticker] for ticker in tickers]
    # Perform optimization
    result = minimize(neg_sharpe_ratio, initial_weights, args=(expected_returns_list, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# Optimize the portfolio for longer horizons
def optimize_portfolio_long_term(assets, mean_returns, cov_matrix):
    """
    Optimizes a portfolio for longer investment horizons using historical data.
    
    This function utilizes historical mean returns and covariance of returns to
    maximize the Sharpe ratio, thus finding the optimal weights for portfolio assets.
    
    Args:
        assets (list): List of assets to include in the portfolio.
        mean_returns (array): Annualized mean returns of the assets.
        cov_matrix (DataFrame): Annualized covariance matrix of asset returns.
    
    Returns:
        OptimizationResult: Result object from the scipy.optimize.minimize function.
    """
    # Number of assets
    num_assets = len(assets)
    # Bounds for asset weights in the portfolio
    bounds = tuple((0, 0.15) for _ in range(num_assets))
    # Constraint to ensure all weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    # Starting guess (equally weighted)
    initial_guess = np.ones(num_assets) / num_assets
    # Perform optimization using the objective function defined earlier
    result = minimize(objective_function, initial_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Define routes for the Flask app
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    This route handles the landing (home) page of the application. It serves two main functions:
    1. GET request: Display the home page with the investment questionnaire.
    2. POST request: Process the user's responses submitted through the form.

    Upon form submission (POST request), it calculates the user's risk profile and preferred 
    investment horizon based on their responses, then redirects to the results page with these parameters.

    Returns:
        On GET request: Renders the home page with the questionnaire.
        On POST request: Redirects to the results page with calculated risk level and investment horizon.
    """
    # Handle form submission
    if request.method == 'POST':
        # Collect responses from the form where each question's answer is retrieved from the form data
        responses = {question: request.form.get(question) for question in questions}

        # Calculate risk levels based on the responses. Each answer corresponds to a predefined risk score.
        risk_levels = [risk_scores.get(response, 0) for response in responses.values()]

        # Calculate the average risk score to determine the overall risk profile
        average_score = sum(risk_levels) / len(risk_levels)

        # Determine the risk level based on the average score:
        # If average score < 3, risk level is low (1).
        # If average score is between 3 and 4, risk level is moderate (3).
        # If average score >= 4, risk level is high (5).
        risk_level = 1 if average_score < 3 else (3 if average_score < 4 else 5)

        # Retrieve the investment horizon selected by the user from the form
        investment_horizon = responses["What is your investment horizon?"]

        # Redirect to the results page, passing the determined risk level and investment horizon as URL parameters
        return redirect(url_for('results', risk_level=risk_level, investment_horizon=investment_horizon))
    
    # For a GET request, display the questionnaire
    # `questions_and_options` is a zipped object of questions and their corresponding options used in the form
    questions_and_options = zip(questions, options)
    return render_template('home.html', questions_and_options=questions_and_options)

# Results page route
@app.route('/results')
def results():
    """
    Results page route that displays the optimized investment portfolio. This route handles fetching and processing
    financial data, performing portfolio optimization based on user-selected risk level and investment horizon, and
    finally rendering the results on the web page.

    Retrieves user-selected options via URL parameters and uses them to determine the appropriate asset mix.
    Depending on the investment horizon, it either performs a short-term or long-term optimization.
    """
    # Extract risk level and investment horizon from URL parameters
    risk_level = int(request.args.get('risk_level'))
    investment_horizon_str = request.args.get('investment_horizon')
    investment_horizon = int(re.search(r'\d+', investment_horizon_str).group()) # Parse horizon to integer
    
    # Dictionary mapping risk level and investment horizon to specific portfolios
    # The allocation of assets into different risk levels and investment horizons is primarily based on their historical volatility.
    # The volatility calculations and categorization logic are detailed in the 'models/determining_risk_assets.py' file.
    # This file includes a methodology for assessing asset volatility and categorizing them accordingly into low, medium, or high risk categories.
    # Additionally, some assets are placed into different risk categories than their volatility might suggest to ensure a diverse range of asset classes in each portfolio.
    # This approach helps in achieving a balanced and diversified investment portfolio tailored to different investor profiles and time horizons.
    portfolios = {
        # Key is a tuple (risk_level, investment_horizon), value is list of tickers
        (1, 1): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM'],
        (1, 5): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ'],
        (1, 10): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ'],
        (1, 15): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM'],
        (3, 1): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM', 'XOM', 'V', 'MSFT', 'DIS', 'SBUX', 'GOOG', 'NKE', 'JPM', 'AAPL', 'SHEL', 'GE', 'AMZN', 'TSM'],
        (3, 5): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM', 'XOM', 'V', 'MSFT', 'DIS', 'SBUX'],
        (3, 10): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM', 'XOM', 'V', 'MSFT', 'DIS', 'SBUX', 'GOOG', 'NKE', 'JPM', 'AAPL', 'SHEL'],
        (3, 15): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM', 'XOM', 'V', 'MSFT', 'DIS', 'SBUX', 'GOOG', 'NKE', 'JPM', 'AAPL', 'SHEL', 'GE', 'AMZN', 'TSM'],
        (5, 1): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM', 'XOM', 'V', 'MSFT', 'DIS', 'SBUX', 'GOOG', 'NKE', 'JPM', 'AAPL', 'SHEL', 'GE', 'AMZN', 'TSM', 'ETH-USD', 'QCOM', 'BAC', 'ASML', 'BA', 'META', 'BABA', 'NVDA', 'SPOT', 'NFLX', 'UBER', 'AMD', 'TSLA', 'SHOP'],
        (5, 5): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM', 'XOM', 'V', 'MSFT', 'DIS', 'SBUX', 'GOOG', 'NKE', 'JPM', 'AAPL', 'SHEL', 'GE', 'AMZN', 'TSM', 'ETH-USD', 'QCOM', 'BAC', 'ASML', 'BA'],
        (5, 10): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM', 'XOM', 'V', 'MSFT', 'DIS', 'SBUX', 'GOOG', 'NKE', 'JPM', 'AAPL', 'SHEL', 'GE', 'AMZN', 'TSM', 'ETH-USD', 'QCOM', 'BAC', 'ASML', 'BA', 'META', 'BABA', 'NVDA', 'SPOT', 'NFLX'],
        (5, 15): ['BTC-USD', 'GC=F', 'SI=F', 'VT', 'VB', 'VWO', 'SPY', 'QQQ', 'SOXX', 'PG', 'KO', 'PEP', 'VZ', 'MCD', 'WMT', 'MO', 'UL', 'PM', 'XOM', 'V', 'MSFT', 'DIS', 'SBUX', 'GOOG', 'NKE', 'JPM', 'AAPL', 'SHEL', 'GE', 'AMZN', 'TSM', 'ETH-USD', 'QCOM', 'BAC', 'ASML', 'BA', 'META', 'BABA', 'NVDA', 'SPOT', 'NFLX', 'UBER', 'AMD', 'TSLA', 'SHOP']
    }
    
    
    # Check if the optimization is for a one-year horizon
    if investment_horizon == 1:
        predictions = fetch_predictions() # Load model predictions from a CSV file
        tickers = portfolios.get((risk_level, investment_horizon), []) # Get the list of tickers for the specified risk level and horizon

        if tickers:
            # Download historical price data from Yahoo Finance. used a shorter time period same as used in the model for predictions
            data = yf.download(tickers, start="2020-01-01", end="2024-06-27")['Adj Close']
            # Calculate mean returns, covariance matrix, and retrieve asset names
            mean_returns, cov_matrix, names = fetch_data_with_names(tickers)  
            # Set bounds for portfolio weights (none can exceed 15%)
            bounds = [(0, 0.15) for _ in tickers]
            # Perform optimization to find the best portfolio weights based on the negative Sharpe ratio
            result = optimize_portfolio_one_year(predictions, data, tickers, bounds)
            # Store the optimized weights and corresponding asset names if the optimization was successful
            portfolio_weights = {tickers[i]: (result.x[i] * 100, names[tickers[i]]) for i in range(len(tickers)) if result.success and result.x[i] > 0.00001}
    else:
        # For investment horizons longer than one year, perform a different optimization
        assets = portfolios.get((risk_level, investment_horizon), [])
        if assets:
            # Fetch data and perform long-term optimization
            mean_returns, cov_matrix, names = fetch_data_with_names(assets)  # Fetch names here
            result = optimize_portfolio_long_term(assets, mean_returns, cov_matrix)
            portfolio_weights = {assets[i]: (result.x[i] * 100, names[assets[i]]) for i in range(len(assets)) if result.success and result.x[i] > 0.00001}
        else:
            portfolio_weights = {}
    
    # Render the results page with the optimized portfolio weights and asset names
    return render_template('results.html', portfolio=portfolio_weights)

if __name__ == '__main__':
    app.run(debug=True)
