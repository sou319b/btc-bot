import ccxt
import pandas as pd
import numpy as np
import time
import os

class BitcoinTrader:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.btc_balance = 0
        self.exchange = ccxt.binance()
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.trade_history = []
        
    def get_ohlcv(self, symbol='BTC/USDT', timeframe='1m', limit=100):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def execute_trade(self, action, price, amount):
        if action == 'buy' and self.balance >= amount * price:
            self.btc_balance += amount
            self.balance -= amount * price
            self.trade_history.append({
                'action': 'buy',
                'price': price,
                'amount': amount,
                'timestamp': pd.Timestamp.now()
            })
        elif action == 'sell' and self.btc_balance >= amount:
            self.btc_balance -= amount
            self.balance += amount * price
            self.trade_history.append({
                'action': 'sell',
                'price': price,
                'amount': amount,
                'timestamp': pd.Timestamp.now()
            })
    
    def run_simulation(self):
        print("Starting Bitcoin Trading Simulation...")
        print(f"Initial Balance: {self.balance} USDT")
        
        while True:
            try:
                # Get market data
                data = self.get_ohlcv()
                current_price = data['close'].iloc[-1]
                
                # Calculate RSI
                rsi = self.calculate_rsi(data['close'], self.rsi_period)
                current_rsi = rsi.iloc[-1]
                
                # Trading logic
                if current_rsi < self.rsi_oversold and self.balance > 0:
                    amount = self.balance / current_price * 0.1  # 10% of balance
                    self.execute_trade('buy', current_price, amount)
                    print(f"Bought {amount:.6f} BTC at {current_price:.2f} USDT")
                
                elif current_rsi > self.rsi_overbought and self.btc_balance > 0:
                    amount = self.btc_balance * 0.1  # 10% of holdings
                    self.execute_trade('sell', current_price, amount)
                    print(f"Sold {amount:.6f} BTC at {current_price:.2f} USDT")
                
                # Display current status
                total_value = self.balance + (self.btc_balance * current_price)
                
                # Print status in Japanese
                print(f"\n現在の価格: {current_price:.2f} USDT")
                print(f"RSI: {current_rsi:.2f}")
                print(f"USDT残高: {self.balance:.2f}")
                print(f"BTC保有量: {self.btc_balance:.6f}")
                print(f"総資産額: {total_value:.2f} USDT")
                print("-" * 50)
                
                # Wait for next iteration
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\nSimulation stopped by user")
                break
            except Exception as e:
                print(f"Error occurred: {e}")
                time.sleep(10)

if __name__ == "__main__":
    initial_balance = float(input("Enter initial balance (USDT): "))
    trader = BitcoinTrader(initial_balance)
    trader.run_simulation()
