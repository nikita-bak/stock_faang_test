# trying to optimize a users portfolio using the efficient frontier

from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


def main():
    print("Starting...")
    # Get the stock symbols/ tickers in the protfolio
    # FAANG (Facebook, Amazon, Apple, Netflix, Google

    assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

    # Assign weights to the stocks.
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Get the stock/ portfolio starting date
    stock_start_date = '2013-01-01'

    # Get the stocks ending data (today)
    today = datetime.today().strftime('%Y-%m-%d')

    # Create a data frame to store the adjusted close price of the stocks
    df = pd.DataFrame()

    # Store the adjusted close price of the stock into the df
    for stock in assets:
        df[stock] = web.DataReader(stock, data_source='yahoo', start=stock_start_date, end=today)['Adj Close']

    # Visually show the stock / portfolio
    title = 'Portfolio adj. close price history'
    # Get the stocks
    my_stocks = df
    # Create and plot the graph
    for c in my_stocks.columns.values:
        plt.plot(my_stocks[c], label=c)
    plt.title = title
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adj. Price USD', fontsize=18)
    plt.legend(my_stocks.columns.values, loc='upper left')
    plt.show()

    # Show the daily simple return
    returns = df.pct_change()

    # Create the annualized covariance matrix
    cov_matrix_annual = returns.cov() * 252

    # Calculate the portfolio variance
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))

    # Calculate the portfolio volatility aka standard deviation
    port_volatility = np.sqrt(port_variance)

    # Calculate annual portfolio return
    portfolio_simple_annual_return = np.sum(returns.mean() * weights) * 252

    # Show the expected annual return, volatility (risk) and variance
    percent_variance = str(round(port_variance, 2) * 100) + '%'
    percent_volatility = str(round(port_volatility, 2) * 100) + '%'
    percent_return = str(round(portfolio_simple_annual_return, 2) * 100) + '%'
    print('Expected annual return: ' + percent_return)
    print('Annual volatility / risk: ' + percent_volatility)
    print('Annual variance: ' + percent_variance)

    # Portfolio optimization
    # Calculate the expected returns and the annualised sample covariance matrix of asset returns

    mu = expected_returns.mean_historical_return(df)
    s = risk_models.sample_cov(df)

    # Optimize for maximum sharpe (William Sharpe) ratio
    ef = EfficientFrontier(mu, s)
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    print(ef.portfolio_performance(verbose=True))

    # Get the discrete allocation of each share per stock
    latest_prices = get_latest_prices(df)
    weights = cleaned_weights
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=500)

    allocation, leftover = da.lp_portfolio()
    print('Discrete allocation: ' + str(allocation))
    print('Funds remaining: ${:.2f}'.format(leftover))


if __name__ == '__main__':
    main()
