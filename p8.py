import ccxt
import time
from datetime import datetime
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

class TradingBot:
    def __init__(self, initial_balance):
        self.exchange = ccxt.binance()  # 価格取得用
        self.jpy_balance = initial_balance
        self.btc_balance = 0
        self.fee_rate = 0.001  # 0.1%の取引手数料
        self.price_history = []
        self.ma_short = 5  # 短期移動平均の期間
        self.ma_long = 20  # 長期移動平均の期間
        self.min_order = 0.00000001
        self.max_order = 20
        self.profit_threshold = 0.003  # 0.3%の利益を目標（より積極的な取引）
        self.last_trade_price = None
        self.last_trade_type = None
        self.last_trade_time = None
        self.min_trade_interval = 3  # 最小取引間隔を3秒に短縮
        self.trade_amount_ratio = 0.4  # 取引量を40%に増加
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.training_data = []
        self.window_size = 30  # 予測に使用するデータポイントの数

    def get_btc_price(self):
        ticker = self.exchange.fetch_ticker('BTC/USDT')
        price = ticker['last'] * 150  # USDTからJPYへの概算換算
        self.price_history.append(price)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
        return price

    def calculate_total_assets(self, current_price):
        return self.jpy_balance + (self.btc_balance * current_price)

    def prepare_features(self):
        if len(self.price_history) < self.window_size:
            return None
        
        prices = np.array(self.price_history[-self.window_size:])
        features = []
        
        # テクニカル指標の計算
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        momentum = (prices[-1] / prices[0]) - 1
        ma_short = np.mean(prices[-self.ma_short:])
        ma_long = np.mean(prices[-self.ma_long:])
        
        features.extend([
            volatility,
            momentum,
            ma_short / prices[-1] - 1,  # 短期MAと現在価格の乖離
            ma_long / prices[-1] - 1,   # 長期MAと現在価格の乖離
            self.calculate_rsi(),
        ])
        
        return np.array(features).reshape(1, -1)

    def update_model(self):
        if len(self.price_history) < self.window_size + 1:
            return
        
        X = self.prepare_features()
        if X is None:
            return
        
        # 次の価格変化率を予測対象とする
        y = (self.price_history[-1] / self.price_history[-2]) - 1
        
        self.training_data.append((X[0], y))
        if len(self.training_data) > 1000:  # 直近1000データポイントのみ保持
            self.training_data = self.training_data[-1000:]
        
        if len(self.training_data) >= 100:  # 100データポイント以上で学習
            X_train = np.array([x for x, _ in self.training_data])
            y_train = np.array([y for _, y in self.training_data])
            self.model.fit(X_train, y_train)

    def predict_price_movement(self):
        X = self.prepare_features()
        if X is None or len(self.training_data) < 100:
            return 0
        
        return self.model.predict(X)[0]

    def calculate_ma(self, period):
        if len(self.price_history) < period:
            return None
        return sum(self.price_history[-period:]) / period

    def calculate_rsi(self, period=14):
        if len(self.price_history) < period + 1:
            return 50  # デフォルト値として中立的な50を返す
        
        deltas = np.diff(self.price_history[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def can_trade(self):
        if self.last_trade_time is None:
            return True
        return time.time() - self.last_trade_time >= self.min_trade_interval

    def should_buy(self, current_price):
        if not self.can_trade() or len(self.price_history) < self.ma_long or self.jpy_balance <= 1000:
            return False

        predicted_movement = self.predict_price_movement()
        rsi = self.calculate_rsi()
        ma_short = self.calculate_ma(self.ma_short)
        ma_long = self.calculate_ma(self.ma_long)

        if not ma_short or not ma_long:
            return False

        # 機械学習モデルの予測を考慮
        ml_signal = predicted_movement > self.profit_threshold

        # テクニカル指標による判断
        technical_signals = [
            rsi < 30,  # 過売り
            ma_short > ma_long,  # ゴールデンクロス
            current_price < ma_long * 0.995  # 長期MAより0.5%以上安い
        ]

        # 前回の取引との価格差を確認
        price_condition = True
        if self.last_trade_price and self.last_trade_type == "売り":
            price_condition = current_price < self.last_trade_price * (1 - self.profit_threshold)

        return ml_signal and any(technical_signals) and price_condition

    def should_sell(self, current_price):
        if not self.can_trade() or len(self.price_history) < self.ma_long or self.btc_balance <= self.min_order:
            return False

        predicted_movement = self.predict_price_movement()
        rsi = self.calculate_rsi()
        ma_short = self.calculate_ma(self.ma_short)
        ma_long = self.calculate_ma(self.ma_long)

        if not ma_short or not ma_long:
            return False

        # 機械学習モデルの予測を考慮
        ml_signal = predicted_movement < -self.profit_threshold

        # テクニカル指標による判断
        technical_signals = [
            rsi > 70,  # 過買い
            ma_short < ma_long,  # デッドクロス
            current_price > ma_long * 1.005  # 長期MAより0.5%以上高い
        ]

        # 前回の取引との価格差を確認
        price_condition = True
        if self.last_trade_price and self.last_trade_type == "買い":
            price_condition = current_price > self.last_trade_price * (1 + self.profit_threshold)

        return ml_signal and any(technical_signals) and price_condition

    def execute_trade(self, current_price, is_buy):
        if is_buy:
            amount_in_jpy = self.jpy_balance * self.trade_amount_ratio
            btc_amount = amount_in_jpy / current_price
            
            if btc_amount < self.min_order:
                return False
            if btc_amount > self.max_order:
                btc_amount = self.max_order

            fee = btc_amount * self.fee_rate
            total_btc = btc_amount - fee
            total_jpy = amount_in_jpy

            if total_jpy <= self.jpy_balance:
                self.btc_balance += total_btc
                self.jpy_balance -= total_jpy
                self.last_trade_price = current_price
                self.last_trade_type = "買い"
                self.last_trade_time = time.time()
                self.print_trade_info(current_price, "買い")
                return True

        else:  # sell
            if self.btc_balance < self.min_order:
                return False
            
            amount_to_sell = self.btc_balance * self.trade_amount_ratio
            if amount_to_sell > self.max_order:
                amount_to_sell = self.max_order

            fee = amount_to_sell * self.fee_rate
            total_btc = amount_to_sell
            total_jpy = (amount_to_sell - fee) * current_price

            self.btc_balance -= total_btc
            self.jpy_balance += total_jpy
            self.last_trade_price = current_price
            self.last_trade_type = "売り"
            self.last_trade_time = time.time()
            self.print_trade_info(current_price, "売り")
            return True

        return False

    def print_trade_info(self, current_price, trade_type):
        print(f"\n取引実行: {trade_type}")
        print(f"現在のBTC価格：{current_price:,.0f}円")
        print(f"JPY残高: {self.jpy_balance:,.0f}円")
        print(f"BTC保有量：{self.btc_balance:.8f}BTC")
        total_assets = self.calculate_total_assets(current_price)
        print(f"総資産額：{total_assets:,.0f}円")
        
        if self.last_trade_price:
            profit_rate = ((total_assets - self.jpy_balance) / self.jpy_balance) * 100
            print(f"収益率: {profit_rate:+.2f}%")
        
        predicted_movement = self.predict_price_movement()
        print(f"予測価格変動: {predicted_movement*100:+.2f}%")
        print(f"RSI: {self.calculate_rsi():.1f}")

    def run(self):
        print("機械学習を使用した高頻度取引シミュレーションを開始します...")
        print("初期設定:")
        print(f"- 利益目標: {self.profit_threshold*100:.1f}%")
        print(f"- 取引量: 利用可能額の{self.trade_amount_ratio*100:.0f}%")
        print(f"- 最小取引間隔: {self.min_trade_interval}秒")
        print(f"- 機械学習モデル: RandomForest（特徴量: ボラティリティ、モメンタム、MA乖離、RSI）")
        
        while True:
            try:
                current_price = self.get_btc_price()
                self.update_model()  # モデルの更新
                
                if len(self.price_history) > 1:
                    price_change = ((current_price - self.price_history[-2]) / self.price_history[-2]) * 100
                    
                    if not self.execute_trade(current_price, self.should_buy(current_price)) and \
                       not self.execute_trade(current_price, self.should_sell(current_price)):
                        predicted_movement = self.predict_price_movement()
                        rsi = self.calculate_rsi()
                        print(f"\r価格変動: {price_change:+.2f}% | 予測: {predicted_movement*100:+.2f}% | RSI: {rsi:.1f}", end="")
                
                time.sleep(1)

            except KeyboardInterrupt:
                print("\n\nプログラムを終了します...")
                final_assets = self.calculate_total_assets(current_price)
                initial_assets = self.jpy_balance
                profit_rate = ((final_assets - initial_assets) / initial_assets) * 100
                print(f"最終総資産: {final_assets:,.0f}円")
                print(f"総収益率: {profit_rate:+.2f}%")
                sys.exit()

            except Exception as e:
                print(f"エラーが発生しました: {e}")
                time.sleep(5)

if __name__ == "__main__":
    try:
        initial_balance = float(input("初期資金を入力してください（円）: "))
        bot = TradingBot(initial_balance)
        bot.run()
    except ValueError:
        print("有効な数値を入力してください。")
