import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ccxt
from config import *
import time

def get_price(symbol, api):
    return api.fetch_ticker(symbol)['last']
    
def get_data(symbol, timeframe, api, number_of_days=30):
    """ TimeFrame examples: "1m", "15m", "1h"..."""
    
    # Initialiaze the Exchange
    api.load_markets()
    
    # Define the dates & import the data
    one_week_ago = datetime.utcnow() - timedelta(days=number_of_days)
    since = int(one_week_ago.timestamp() * 1000)
    data = api.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)

    # Format the data
    df_data = pd.DataFrame(data)
    df_data.columns = ["time", "open", "high", "low", "close", "volume"]
    df_data["time"] = pd.to_datetime(df_data["time"], unit='ms')
    df_data = df_data.set_index("time")
    
    return df_data

def buy_market(symbol, symbol_amount, api):
    price = get_price(symbol, api)
    symbol_amount_usdt = symbol_amount * price
  
    # You maybe need to modify the inputs of this function according to your exchange
    order = api.create_market_buy_order(symbol, symbol_amount_usdt)
    return order

def sell_market(symbol, symbol_amount, api):
    order = api.create_market_sell_order(symbol, symbol_amount)
    return order

def get_available_amount(symbol,api):
    balances = api.fetch_balance()
    for dictionnary in balances["info"]:
        if dictionnary["coin"]==symbol:
            amount_symbol = dictionnary["available"]
        else:
            amount_symbol = 0
    return float(amount_symbol)

balances = api.fetch_balance()
balances["info"]

