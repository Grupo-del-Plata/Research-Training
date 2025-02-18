# importYFdatas function downloads data from Yahoo Finance with a start date

import pandas as pd 
import yfinance as yf
import fredapi
from fredapi import Fred
fred = Fred(api_key='286d44038c995c89e1fe6cd4b887db8a')

def importFREDdata(ticker):
    data = fred.get_series(ticker)
    data = data.dropna()
    
    return data

def importYFdatas(tickers,startdate,tickers_labels):
    data = pd.DataFrame(columns=tickers)
    
    start = startdate
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start)['Adj Close']
    
    data.columns = tickers_labels
    data = data.round(2)
    data = data.dropna()
    
    return data

# importYFdata function downloads data from Yahoo Finance 

def importYFdata(tickers,tickers_labels):
    data = pd.DataFrame(columns=tickers)

    for ticker in tickers:
        data[ticker] = yf.download(ticker)['Adj Close']
    
    data.columns = tickers_labels
    data = data.round(2)
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    
    return data