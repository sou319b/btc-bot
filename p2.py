import time
import random
import numpy as np

class TradingSimulator:
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.btc_balance = 0
        self.position = 0  # 0: 未保持, 1: 買い, -1: 売り
        self.price_history = []
        self.profit_history = []
        
    def generate_price(self):
        # ランダムウォークで価格を生成
        if not self.price_history:
            return 500000  # 初期価格
        last_price = self.price_history[-1]
        change = random.uniform(-0.02, 0.02)  # ±2%の変動
        return last_price * (1 + change)
    
    def calculate_moving_averages(self):
        prices = np.array(self.price_history)
        if len(prices) >= 20:
            short_ma = np.mean(prices[-5:])  # 5期間移動平均
            long_ma = np.mean(prices[-20:])  # 20期間移動平均
            return short_ma, long_ma
        return None, None
    
    def execute_trade(self, current_price):
        short_ma, long_ma = self.calculate_moving_averages()
        
        if short_ma is None or long_ma is None:
            return
            
        # 買いシグナル
        if short_ma > long_ma and self.position <= 0:
            self.btc_balance = self.capital / current_price
            self.capital = 0
            self.position = 1
            print(f"[買いシグナル]")
            print(f"価格: {current_price:.0f}円")
            print(f"購入BTC量: {self.btc_balance:.7f} BTC")
            print("-------------------------")
            
        # 売りシグナル
        elif short_ma < long_ma and self.position >= 0:
            self.capital = self.btc_balance * current_price
            self.btc_balance = 0
            self.position = -1
            profit = self.capital - 10000
            self.profit_history.append(profit)
            print(f"[売りシグナル]")
            print(f"価格: {current_price:.0f}円")
            print(f"現金残高: {self.capital:.0f}円")
            print(f"利益: {profit:.0f}円")
            print("=========================")

    def run(self):
        print("シミュレーション開始")
        print(f"初期資金: {self.capital:.0f}円")
        
        while True:
            current_price = self.generate_price()
            self.price_history.append(current_price)
            
            self.execute_trade(current_price)
            
            time.sleep(2)

if __name__ == "__main__":
    simulator = TradingSimulator(initial_capital=100000)
    simulator.run()
