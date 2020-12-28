#!/usr/bin/env python
# coding: utf-8

# In[454]:


# import libraries, packages, modules, sqlite3
import numpy as np
import math
import talib as ta
import pandas as pd
import requests
import time
import os
import yfinance as yf

from pandas import Series, DataFrame
from pandas_datareader import data
from statistics import mean
from datetime import datetime, timedelta


# visual aids 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
from warnings import filterwarnings
from sklearn import tree
import graphviz

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree



# ignore warnings
filterwarnings('ignore')


# In[455]:


# df = yf.download('AMZN', '2019-1-1','2019-12-27')
# df


# In[456]:


start = datetime.now() - timedelta(days=1097)
end = datetime.now()

# user_stock = input("Enter stock ticker: ")
# stock_df = yf.download(user_stock, start, end)
# stock_df


# In[457]:


keep_going = "Y"
stocks_list = ["^GSPC"]

while keep_going == "Y":
    user_stock = input("Enter stock ticker: ")
    stocks_list.append(user_stock)
    user_answer = input("Do you wish to add more stock tickers (Y/N): ")
    if user_answer == "N":
        break



# stocks_list = stock_list_func()


# In[458]:


print(stocks_list)


# In[459]:


all_stocks = yf.download(tickers = " ".join(stocks_list), period = "3y", group_by = "ticker", threads = True)
print(all_stocks)


# In[460]:


for user_stock in stocks_list:
    print(user_stock)


# # Trading Rules 
# These trading rules will be used by the Decision Tree to identify the best combination of indicators to maximize result.
# - EMA: Interested in when price is above average and when the fastest average is above slowest average
# - ATR(14): Interested in threshold that will trigger signal
# - ADX(14): Interested in threshold that will trigger signal
# - RSI(14): Interested in threshold that will trigger signal
# - MACD: Interested in when MACD signal is above MACD 
# 
# Predictor variables for classification DT and regression DT will be teh same. However, the target variable is different for each because the classification DT output will be categorical while the regression DT output will be continuous. 

# In[461]:


# Technical Indicators 
def technical_indicators_function(user_stock):

    benchmark_sp500 = yf.download("^GSPC", start, end)
    user_stock_df = yf.download(user_stock, start, end)

    user_stock_df['SimpleMA'] = ta.SMA(user_stock_df['Close'], timeperiod = 14)
    user_stock_df['ExponentialMA'] = ta.EMA(user_stock_df['Close'], timeperiod = 14)

    # Exponential Moving Average (EMA) - Weighted moving average to favor more recent price action
    # Short-term EMA for 10 days and 30 days
    user_stock_df['ExponentialMA10D'] = ta.EMA(user_stock_df['Close'].values, timeperiod=10)
    user_stock_df['ExponentialMA30D'] = ta.EMA(user_stock_df['Close'].values, timeperiod=30)

    # Average True Range (ATR) - Measures market volatility by decomposing range of an asset price for a given period
    # Typically derived fro the 14-day Moving Average of a series of true range indicators
    user_stock_df['AverageTrueRange'] = ta.ATR(user_stock_df['High'].values, user_stock_df['Low'].values, user_stock_df['Close'].values, timeperiod = 14)

    # Average Directional Index (ADX) - Determines strength of a trend (whether it is up or down: +DI or -DI)
    # Price is moving up when +DI is above -DI
    # Price is moving down when -DI is above +DI
    # Trend is strong when ADX > 25
    # Trend is weak/price is trendless when ADX < 20
    user_stock_df['AverageDirectionalIndex'] = ta.ADX(user_stock_df['High'].values, user_stock_df['Low'].values, user_stock_df['Close'].values, timeperiod = 14)

    # Relative Strength Index (RSI) - Measures magnitude of recent price changes to evaluate overbought/oversold conditions
    # Popular momentum oscillator
    # Asset is overbought when RSI > 70%
    # Asset is oversold when RSI < 30%
    user_stock_df['RelativeStrengthIndex'] = ta.RSI(user_stock_df['Close'].values, timeperiod = 14)

    # Moving Average Convergence Divergence (MACD) - Calculated by subtracting the 26-period EMA from 12-period EMA
    # Signal line: 9-day EMA of MACD
    # When signal line is plotted on top of MACD line, it functions as a trigger for buy and sell signals
    # When MACD crosses above signal line - buy
    # when MACD crosses below signal line - sell/short
    maConvergenceDivergence, maConvergenceDivergenceSignalLine, maConvergenceDivergenceHistory = ta.MACD(user_stock_df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    user_stock_df['MACD'] = maConvergenceDivergence
    user_stock_df['MACD_Signal_Line'] = maConvergenceDivergenceSignalLine

    user_stock_df['Close>EMA10D'] = np.where(user_stock_df['Close'] > user_stock_df['ExponentialMA10D'], 1, -1)
    user_stock_df['EMA10D>EMA30D'] = np.where(user_stock_df['ExponentialMA10D'] > user_stock_df['ExponentialMA30D'], 1, -1)
    user_stock_df['MACD_Signal_Line>MACD'] = np.where(user_stock_df['MACD_Signal_Line'] > user_stock_df['MACD'], 1, -1)

    user_stock_df['DailyReturn'] = user_stock_df['Close'].pct_change(1).shift(-1)
    benchmark_sp500['DailyReturn'] = benchmark_sp500['Close'].pct_change(1).shift(-1)

    # Comparing Daily Return of S&P 500 (benchmark) to stock
    # if stock_df is not "^GSPC":
    user_stock_df['BechmarkComparison'] = np.where(user_stock_df.DailyReturn > benchmark_sp500.DailyReturn, 1, 0)

    # Target variable for classification DT: Transform lagged return to accommodate categorical output
    # If return is positive, 'Up' is assigned
    # If return is negative, 'Down' is assigned 
    # user_stock_df['target_Classification'] = np.where(user_stock_df.DailyReturn > 0, 1, 0)
    user_stock_df['target_Classification'] = np.where(user_stock_df['DailyReturn'].fillna(user_stock_df['DailyReturn'].mean()) > 0, 'Up', 'Down')

    # Target variable for regresion DT: Use lagged return to forecast the price for following/specified day
    user_stock_df['target_Regression'] = user_stock_df['DailyReturn'].fillna(user_stock_df['DailyReturn'].mean())

    return user_stock_df.fillna(0)



for user_stock in stocks_list:
    user_stock_data = technical_indicators_function(user_stock)
    print(user_stock_data)


# In[462]:


# Dataset of predictor variables (technical indicators calculated) to be used by Classification DT and Regression DT 
predictor_indicators = ['AverageTrueRange', 'AverageDirectionalIndex','RelativeStrengthIndex', 'Close>EMA10D', 'EMA10D>EMA30D', 'MACD_Signal_Line>MACD']

def predictor_indicators_function(user_stock):
    X = user_stock[predictor_indicators].fillna(0)
    return X

for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    print(output_predictor_indicators)


# In[463]:


# Classification Function 
def classification_function(X):
    
    y_classification = output_technical_indicators.target_Classification
    y = y_classification
    X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X, y, test_size=0.3, random_state=432, stratify=y)
    
    print(X_cl_train.shape, y_cl_train.shape)
    print(X_cl_test.shape, y_cl_test.shape)
    
    classification_result = DecisionTreeClassifier(criterion='gini', max_depth=5)
    
    classification_fit = classification_result.fit(X_cl_train, y_cl_train)
    return classification_result

for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_classification_result = classification_function(output_predictor_indicators)
    print(output_classification_result)


# In[464]:


# Predictor Indicators 
X = output_predictor_indicators

def classification_fitting(classification_result):

    y_classification = output_technical_indicators.target_Classification
    y = y_classification
    X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X, y, test_size=0.3, random_state=432, stratify=y)
    
    classification_fit = classification_result.fit(X_cl_train, y_cl_train)
    return classification_fit

for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_classification_result = classification_function(output_predictor_indicators)
    output_classification_fitting = classification_fitting(output_classification_result)
    print(output_classification_fitting)


# In[465]:


# Print Classification Decision Tree Text Format
def tree_txt_result(output_classification_result):
    text_representation = tree.export_text(output_classification_result)

    # with open("decistion_tree.log", "w") as fout:
        # fout.write(text_representation)
        
    return text_representation


for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_classification_result = classification_function(output_predictor_indicators)
    output_text = tree_txt_result(output_classification_result)
     
    with open(user_stock + "_classification_decistion_tree" + ".log", "w") as fout:
        fout.write(output_text)
    
    print(output_text)


# In[466]:


# Print Classification Tree Method 1
def print_tree_function(output_classification_result):

    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(output_classification_result, 
                   feature_names=predictor_indicators,  
                   filled=True)
    
    fig.savefig("decistion_tree.png")


for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_classification_result = classification_function(output_predictor_indicators)
    print(print_tree_function(output_classification_result))


# In[467]:


# Print Classification Decision Tree Method 2 - Graphviz 
for user_stock in stocks_list:
    
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_classification_result = classification_function(output_predictor_indicators)

    dot_data = tree.export_graphviz(output_classification_result, out_file=None, 
                                    max_depth=5,
                                    feature_names=predictor_indicators,
                                    class_names=output_technical_indicators.target_Classification,
                                    filled=True)

    graph = graphviz.Source(dot_data, format="png")
    graph.render(user_stock + "_classification_decision_tree")


# In[468]:


# Classification Decision Tree Prediction & Accuracy Report 
def classification_prediction(output_classification_result):
    y_classification = output_technical_indicators.target_Classification
    y = y_classification
    X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X, y, test_size=0.3, random_state=432, stratify=y)
    y_cl_pred = output_classification_result.predict(X_cl_test)
    accuracy = metrics.accuracy_score(y_cl_test, y_cl_pred)
    
    return y_cl_pred, accuracy

for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_classification_result = classification_function(output_predictor_indicators)
    output_classification_prediction = classification_prediction(output_classification_result)
    print(output_classification_prediction)


# In[493]:


# Regression Decision Tree
def regression_function(X):
    
    y_regression = output_technical_indicators.target_Regression
    y = y_regression
    
    train_length = int(len(output_technical_indicators)*0.70)
    X_rg_train = X[:train_length]
    X_rg_test = X[train_length:]
    y_rg_train = y_regression[:train_length]
    y_rg_test = y_regression[train_length:]

    print(X_rg_train.shape, y_rg_train.shape)
    print(X_rg_test.shape, y_rg_test.shape)
    
    regression_result = DecisionTreeRegressor(max_depth=3, min_samples_leaf = 10)
    
    regression_result.fit(X_rg_train, y_rg_train)
    return regression_result

for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_regression_result = regression_function(output_predictor_indicators)
    print(output_regression_result)


# In[494]:


# Print Regression Decision Tree Text Format

def tree_txt_result(output_classification_result):
    text_representation = tree.export_text(output_regression_result)

    return text_representation


for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_regression_result = regression_function(output_predictor_indicators)
    output_text = tree_txt_result(output_regression_result)
     
    with open(user_stock + "_regression_decistion_tree" + ".log", "w") as fout:
        fout.write(output_text)
    
    print(output_text)


# In[495]:


# Print Regression Decision Tree 
for user_stock in stocks_list:
    
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    output_regression_result = regression_function(output_predictor_indicators)

    dot_data = tree.export_graphviz(output_regression_result, out_file=None, 
                            feature_names=predictor_indicators,
                            class_names=output_technical_indicators.target_Classification,
                            filled=True)

    graph = graphviz.Source(dot_data, format="png")
    graph.render(user_stock + "_regression_decision_tree")


# In[472]:


# the 'Monte Carlo' method runs simulations to predict the future many times
# afterwards, the aggregation of all these simulations is used to establish a value for how risky the stock is

# stock_monte_carlo function has parameters of number of days to run, mean, and standard deviation values
def stock_monte_carlo(start_price, days, drift, volatility):

    price = np.zeros(days)
    price[0] = start_price
    
    # set shock and drift 
    new_shock = np.zeros(days)
    new_drift = np.zeros(days)
    
    # calculate price array for given number of days
    for i in range(1, days):
        # shock formula taken from the Monte Carlo formula
        new_shock[i] = np.random.normal(loc = drift * delta, scale = volatility * np.sqrt(delta))
        
        # drift formula taken from the Monte Carlo formula
        new_drift[i] = drift * delta
        # new price = old price + old price* (shock+drift)
        price[i] = price[i - 1] + (price[i - 1] * (new_drift[i] + new_shock[i]))
    return price

for user_stock in stocks_list:
    
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predictor_indicators = predictor_indicators_function(output_technical_indicators)
    
    days = 60
    delta = 1/days 
    
    drift = output_technical_indicators['DailyReturn'].mean()
    volatility = output_technical_indicators['DailyReturn'].std()

    start_price = output_technical_indicators['Open'].tail(1)
    simulations = np.zeros(100)
    
    runs = 10000

    # create matrix for the final price 
    simulations = np.zeros(runs)

    for run in range(runs):
        simulations[run] = stock_monte_carlo(start_price, days, drift, volatility)[days-1]

    # onepercent is the 1% empirical quantile,  meaning that 99% of the values should fall between here
    onepercent = np.percentile(simulations, 1)
    print("Mean Close Price: $%.2f" % simulations.mean(), "VaR(0.99): $%.2f" % (start_price-onepercent,), "onepercent(0.99): $%.2f" % onepercent)


# In[473]:


# Predict Future Close Price 

def predict_close_price(output_technical_indicators):
    future_days = 60
    
    # Just use 'Close' data 
    user_stock_close_df = output_technical_indicators[['Close']]

    user_stock_close_df['PredictedClosePrice'] = user_stock_close_df[['Close']].shift(-future_days)
    X = np.array(user_stock_close_df.drop(['PredictedClosePrice'], 1))[:-future_days]
    y = np.array(user_stock_close_df['PredictedClosePrice'])[:-future_days]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    reg_decision_tree = DecisionTreeRegressor().fit(X_train, y_train)

    X_future = user_stock_close_df.drop(['PredictedClosePrice'], 1)[:-future_days]
    X_future = X_future.tail(future_days)
    X_future = np.array(X_future)

    reg_decision_tree_close_pred = reg_decision_tree.predict(X_future)
    return reg_decision_tree_close_pred

for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    print(user_stock + ":")
    print(output_predict_close_price)


# In[474]:


# Predict Future High Price 

def predict_high_price(output_technical_indicators):
    future_days = 60
    
    # Just use 'High' data
    user_stock_high_df = output_technical_indicators[['High']]

    user_stock_high_df['PredictedHighPrice'] = user_stock_high_df[['High']].shift(-future_days)
    X = np.array(user_stock_high_df.drop(['PredictedHighPrice'], 1))[:-future_days]
    y = np.array(user_stock_high_df['PredictedHighPrice'])[:-future_days]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    reg_decision_tree = DecisionTreeRegressor().fit(X_train, y_train)

    X_future = user_stock_high_df.drop(['PredictedHighPrice'], 1)[:-future_days]
    X_future = X_future.tail(future_days)
    X_future = np.array(X_future)

    reg_decision_tree_high_pred = reg_decision_tree.predict(X_future)
    return reg_decision_tree_high_pred

for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_high_price = predict_high_price(output_technical_indicators)
    print(user_stock + ":")
    print(output_predict_high_price)


# In[475]:


# Predict Future Low Price 

def predict_low_price(output_technical_indicators):
    future_days = 60
    
    # Just use 'Low' data 
    user_stock_low_df = output_technical_indicators[['Low']]

    user_stock_low_df['PredictedLowPrice'] = user_stock_low_df[['Low']].shift(-future_days)
    X = np.array(user_stock_low_df.drop(['PredictedLowPrice'], 1))[:-future_days]
    y = np.array(user_stock_low_df['PredictedLowPrice'])[:-future_days]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    reg_decision_tree = DecisionTreeRegressor().fit(X_train, y_train)

    X_future = user_stock_low_df.drop(['PredictedLowPrice'], 1)[:-future_days]
    X_future = X_future.tail(future_days)
    X_future = np.array(X_future)

    reg_decision_tree_low_pred = reg_decision_tree.predict(X_future)
    return reg_decision_tree_low_pred

for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_low_price = predict_low_price(output_technical_indicators)
    print(user_stock + ":")
    print(output_predict_low_price)


# In[476]:


# Plot Predicted Close Price
def plot_pred_data(output_predict_close_price):

    future_days = 60
    user_stock_close_df = output_technical_indicators[['Close']]
    user_stock_close_df['PredictedPrice'] = user_stock_close_df[['Close']].shift(-future_days)
    X = np.array(user_stock_close_df.drop(['PredictedPrice'], 1))[:-future_days]
    
    predictions = output_predict_close_price
    valid = user_stock_close_df[X.shape[0]:]
    valid['PredictedPrice'] = predictions
    
    plt.figure(figsize=(16,8))
    plt.title('Decision Tree Regressor')
    plt.xlabel('Days', fontsize = 20)
    plt.ylabel('Close Price', fontsize = 20)
    plt.plot(user_stock_close_df['Close'])
    plt.plot(valid[['Close', 'PredictedPrice']])
    plt.legend(['TrainData', 'Valid', 'Predicted'], loc = 'lower right')

    plt.show()
    plt.savefig('foo.png')
    
for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    plot_pred_data(output_predict_close_price)
    


# In[477]:


# Visual for Predicted Close Price
for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    future_days = 60
    user_stock_close_df = output_technical_indicators[['Close']]
    user_stock_close_df['PredictedPrice'] = user_stock_close_df[['Close']].shift(-future_days)
    X = np.array(user_stock_close_df.drop(['PredictedPrice'], 1))[:-future_days]
    
    predictions = output_predict_close_price
    valid = user_stock_close_df[X.shape[0]:]
    valid['PredictedPrice'] = predictions
    
    plt.figure(figsize=(16,8))
    plt.title('Decision Tree Regressor')
    plt.xlabel('Days', fontsize = 20)
    plt.ylabel('Close Price', fontsize = 20)
    plt.plot(user_stock_close_df['Close'])
    plt.plot(valid[['Close', 'PredictedPrice']])
    plt.legend(['TrainData', 'Valid', 'Predicted'], loc = 'lower right')

    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(user_stock + 'predicted_close_price.png')
    


# In[478]:


user_stock_df1 = pd.DataFrame({'Close': output_predict_close_price, 'High': list(output_predict_high_price), 'Low': list(output_predict_low_price)}, columns=['Close', 'High', 'Low'])
print(user_stock_df1)


# In[479]:


for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    output_predict_high_price = predict_high_price(output_technical_indicators)
    output_predict_low_price = predict_low_price(output_technical_indicators)
    user_stock_df1 = pd.DataFrame({'Close': output_predict_close_price, 'High': list(output_predict_high_price), 'Low': list(output_predict_low_price)}, columns=['Close', 'High', 'Low'])
    print(user_stock_df1)
    


# In[486]:


# Technical Indicators 
def pred_func(user_stock_df1):

    benchmark_sp500 = yf.download("^GSPC", start, end)

    user_stock_df1['SimpleMA'] = ta.SMA(user_stock_df1['Close'], timeperiod = 14)
    user_stock_df1['ExponentialMA'] = ta.EMA(user_stock_df1['Close'], timeperiod = 14)

    # Exponential Moving Average (EMA) - Weighted moving average to favor more recent price action
    # Short-term EMA for 10 days and 30 days
    user_stock_df1['ExponentialMA10D'] = ta.EMA(user_stock_df1['Close'].values, timeperiod=10)
    user_stock_df1['ExponentialMA30D'] = ta.EMA(user_stock_df1['Close'].values, timeperiod=30)

    # Average True Range (ATR) - Measures market volatility by decomposing range of an asset price for a given period
    # Typically derived fro the 14-day Moving Average of a series of true range indicators
    user_stock_df1['AverageTrueRange'] = ta.ATR(user_stock_df1['High'].values, user_stock_df1['Low'].values, user_stock_df1['Close'].values, timeperiod = 14)

    # Average Directional Index (ADX) - Determines strength of a trend (whether it is up or down: +DI or -DI)
    # Price is moving up when +DI is above -DI
    # Price is moving down when -DI is above +DI
    # Trend is strong when ADX > 25
    # Trend is weak/price is trendless when ADX < 20
    user_stock_df1['AverageDirectionalIndex'] = ta.ADX(user_stock_df1['High'].values, user_stock_df1['Low'].values, user_stock_df1['Close'].values, timeperiod = 14)

    # Relative Strength Index (RSI) - Measures magnitude of recent price changes to evaluate overbought/oversold conditions
    # Popular momentum oscillator
    # Asset is overbought when RSI > 70%
    # Asset is oversold when RSI < 30%
    user_stock_df1['RelativeStrengthIndex'] = ta.RSI(user_stock_df1['Close'].values, timeperiod = 14)

    # Moving Average Convergence Divergence (MACD) - Calculated by subtracting the 26-period EMA from 12-period EMA
    # Signal line: 9-day EMA of MACD
    # When signal line is plotted on top of MACD line, it functions as a trigger for buy and sell signals
    # When MACD crosses above signal line - buy
    # when MACD crosses below signal line - sell/short
    maConvergenceDivergence, maConvergenceDivergenceSignalLine, maConvergenceDivergenceHistory = ta.MACD(user_stock_df1['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    user_stock_df1['MACD'] = maConvergenceDivergence
    user_stock_df1['MACD_Signal_Line'] = maConvergenceDivergenceSignalLine

    user_stock_df1['Close>EMA10D'] = np.where(user_stock_df1['Close'] > user_stock_df1['ExponentialMA10D'], 1, -1)
    user_stock_df1['EMA10D>EMA30D'] = np.where(user_stock_df1['ExponentialMA10D'] > user_stock_df1['ExponentialMA30D'], 1, -1)
    user_stock_df1['MACD_Signal_Line>MACD'] = np.where(user_stock_df1['MACD_Signal_Line'] > user_stock_df1['MACD'], 1, -1)

    user_stock_df1['DailyReturn'] = user_stock_df1['Close'].pct_change(1).shift(-1)
    # benchmark_sp500['DailyReturn'] = benchmark_sp500['Close'].pct_change(1).shift(-1)

    # Comparing Daily Return of S&P 500 (benchmark) to stock
    # if stock_df is not "^GSPC":
    # user_stock_df['BechmarkComparison'] = np.where(user_stock_df.DailyReturn > benchmark_sp500.DailyReturn, 1, 0)

    # Target variable for classification DT: Transform lagged return to accommodate categorical output
    # If return is positive, 1 is assigned
    # If return is negative, 0 is assigned 
    # user_stock_df['target_Classification'] = np.where(user_stock_df.DailyReturn > 0, 1, 0)
    user_stock_df1['target_Classification'] = np.where(user_stock_df1['DailyReturn'].fillna(user_stock_df1['DailyReturn'].mean()) > 0, 'Up', 'Down')

    # Target variable for regresion DT: Use lagged return to forecast the price for following/specified day
    user_stock_df1['target_Regression'] = user_stock_df1['DailyReturn'].fillna(user_stock_df1['DailyReturn'].mean())
    
    return user_stock_df1.fillna(0)



# In[488]:


for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    output_predict_high_price = predict_high_price(output_technical_indicators)
    output_predict_low_price = predict_low_price(output_technical_indicators)
    user_stock_df1 = pd.DataFrame({'Close': output_predict_close_price, 'High': list(output_predict_high_price), 'Low': list(output_predict_low_price)}, columns=['Close', 'High', 'Low'])
    output_pred_func = pred_func(user_stock_df1)
    print(output_pred_func)


# In[ ]:


# Dataset of predictor variables (technical indicators calculated) to be used by Classification DT and Regression DT 
predictor_indicators = ['AverageTrueRange', 'AverageDirectionalIndex','RelativeStrengthIndex', 'Close>EMA10D', 'EMA10D>EMA30D', 'MACD_Signal_Line>MACD']

def predictor_indicators_function(user_stock):
    X = user_stock[predictor_indicators].fillna(0)
    return X

outcome_predictor_indicators_function = predictor_indicators_function(user_stock_df1)
print(outcome_predictor_indicators_function)





# In[489]:


for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    output_predict_high_price = predict_high_price(output_technical_indicators)
    output_predict_low_price = predict_low_price(output_technical_indicators)
    user_stock_df1 = pd.DataFrame({'Close': output_predict_close_price, 'High': list(output_predict_high_price), 'Low': list(output_predict_low_price)}, columns=['Close', 'High', 'Low'])
    output_pred_func = pred_func(user_stock_df1)
    outcome_predictor_indicators_function = predictor_indicators_function(user_stock_df1)
    print(outcome_predictor_indicators_function)


# In[428]:


def pred_new_func(outcome_predictor_indicators_function):
    
    # opif = outcome_predictor_indicators_function.T
    # ATR  = opif.iloc[0].tail(1)
    # ADI  = opif.iloc[1].tail(1)
    # RSI  = opif.iloc[2].tail(1)
    # C    = opif.iloc[3].tail(1)
    # EMA  = opif.iloc[4].tail(1)
    #MACD = opif.iloc[5].tail(1)
    
    ATR  = outcome_predictor_indicators_function.iloc[:,0].tail(1)
    ADI  = outcome_predictor_indicators_function.iloc[:,1].tail(1)
    RSI  = outcome_predictor_indicators_function.iloc[:,2].tail(1)
    C    = outcome_predictor_indicators_function.iloc[:,3].tail(1)
    EMA  = outcome_predictor_indicators_function.iloc[:,4].tail(1)
    MACD = outcome_predictor_indicators_function.iloc[:,5].tail(1)
    
    if RSI.any() <= 82.693:
        if ADI.any() <= 47.799:
            if RSI.any() <= 48.245:
                if RSI.any() <= 45.915:
                    if ATR.any() <= 23.837:
                        return str('Up')
                    else:
                        return str('Down')
                elif ADI.any() <= 40.606:
                    return str('Up')
                else:
                    return str('Down') 
            elif RSI.any() > 48.245:
                if RSI.any() <= 56.954:
                    if ATR.any() <= 58.721:
                        return str('Down')
                elif RSI.any() > 56.954:
                    if ADI.any() <= 10.563:
                        return str('Down')
        elif ADI.any() > 47.799:
            if EMA.any() <= 0.0:
                return str('Down')
            elif EMA.any() > 0:
                if ATR.any() <= 20.506:
                    return str('Down')
                elif ATR.any() > 20.506:
                    if RSI.any() <= 33.229:
                        return str('Down')
                    elif RSI.any() > 33.229:
                        return str('Up')
            
            else: 
                return str('Down')
        
    else:
        return str('Down')


# In[500]:


def pred_new_func_regression(outcome_predictor_indicators_function):
    
    # opif = outcome_predictor_indicators_function.T
    # ATR  = opif.iloc[0].tail(1)
    # ADI  = opif.iloc[1].tail(1)
    # RSI  = opif.iloc[2].tail(1)
    # C    = opif.iloc[3].tail(1)
    # EMA  = opif.iloc[4].tail(1)
    #MACD = opif.iloc[5].tail(1)
    
    ATR  = outcome_predictor_indicators_function.iloc[:,0].tail(1)
    ADI  = outcome_predictor_indicators_function.iloc[:,1].tail(1)
    RSI  = outcome_predictor_indicators_function.iloc[:,2].tail(1)
    C    = outcome_predictor_indicators_function.iloc[:,3].tail(1)
    EMA  = outcome_predictor_indicators_function.iloc[:,4].tail(1)
    MACD = outcome_predictor_indicators_function.iloc[:,5].tail(1)
    
    if ATR.any() <= 58.999:
        if ATR.any() <= 51.744:
            if RSI.any() <= 30.181:
                return str('Positive Return')
            else:
                return str('Negative Return')
        else:
            return str('Negative Return')     
    else:
        return str('Negative Return')


# In[501]:


"""
for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    output_predict_high_price = predict_high_price(output_technical_indicators)
    output_predict_low_price = predict_low_price(output_technical_indicators)
    user_stock_df1 = pd.DataFrame({'Close': output_predict_close_price, 'High': list(output_predict_high_price), 'Low': list(output_predict_low_price)}, columns=['Close', 'High', 'Low'])
    output_pred_func = pred_func(user_stock_df1)
    outcome_predictor_indicators_function = predictor_indicators_function(user_stock_df1)
    final_output = pred_new_func(outcome_predictor_indicators_function)
    # print(final_output)
"""


# In[503]:


for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    output_predict_high_price = predict_high_price(output_technical_indicators)
    output_predict_low_price = predict_low_price(output_technical_indicators)
    user_stock_df1 = pd.DataFrame({'Close': output_predict_close_price, 'High': list(output_predict_high_price), 'Low': list(output_predict_low_price)}, columns=['Close', 'High', 'Low'])
    output_pred_func = pred_func(user_stock_df1)
    outcome_predictor_indicators_function = predictor_indicators_function(user_stock_df1)
    final_output_regression = pred_new_func_regression(outcome_predictor_indicators_function)
    print(final_output_regression)


# In[511]:


# Return output to user
for user_stock in stocks_list:
    output_technical_indicators = technical_indicators_function(user_stock)
    output_predict_close_price = predict_close_price(output_technical_indicators)
    output_predict_high_price = predict_high_price(output_technical_indicators)
    output_predict_low_price = predict_low_price(output_technical_indicators)
    user_stock_df1 = pd.DataFrame({'Close': output_predict_close_price, 'High': list(output_predict_high_price), 'Low': list(output_predict_low_price)}, columns=['Close', 'High', 'Low'])
    output_pred_func = pred_func(user_stock_df1)
    outcome_predictor_indicators_function = predictor_indicators_function(user_stock_df1)
    final_output = pred_new_func(outcome_predictor_indicators_function)
    final_output_regression = pred_new_func_regression(outcome_predictor_indicators_function)
    if final_output == 'Up':
        print(user_stock + " will be up.")
        if final_output_regression == 'Positive Return':
            print(user_stock + " return will be positive.'")
        else:
            print(user_stock + " return, however, may be negative.")

    else:
        print(user_stock + " will be down.")
        if final_output_regression == 'Negative Return':
            print(user_stock + " return will be negative.")
        else:
            print(user_stock + " return, however, may be positive.")

        
   


# In[ ]:




