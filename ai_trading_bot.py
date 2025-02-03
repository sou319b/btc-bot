import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bitget import Bitget
from dotenv import load_dotenv
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import ta
import time

# 環境変数の読み込み
load_dotenv()
api_key = os.getenv('BITGET_API_KEY')
api_secret = os.getenv('BITGET_API_SECRET')
api_passphrase = os.getenv('BITGET_PASSPHRASE')

class AITradingBot:
    def __init__(self):
        # Bitgetクライアントの設定
        self.client = Bitget(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=api_passphrase,
            use_server_time=True,
            demo=True  # デモ取引用
        )
        self.symbol = 'BTCUSDT_UMCBL'  # Bitgetの先物取引ペア
        self.lookback_period = 100
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        
        # 初期残高の表示
        self.print_balance()
        
    def print_balance(self):
        """残高の表示"""
        try:
            account = self.client.mix_get_accounts(productType='UMCBL')
            for asset in account['data']:
                if asset['marginCoin'] == 'USDT':
                    print(f"取引可能残高: {float(asset['available'])} USDT")
                    print(f"証拠金残高: {float(asset['equity'])} USDT")
        except Exception as e:
            print(f"残高取得エラー: {e}")

    def _build_model(self):
        """LSTMモデルの構築"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.lookback_period, 5)),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def get_historical_data(self):
        """過去のデータを取得"""
        try:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(hours=100)).timestamp() * 1000)
            
            candles = self.client.mix_get_candles(
                symbol=self.symbol,
                granularity='1H',
                startTime=start_time,
                endTime=end_time
            )
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
            
        except Exception as e:
            print(f"データ取得エラー: {e}")
            return None

    def prepare_data(self, df):
        """データの前処理"""
        if df is None or df.empty:
            raise Exception("データが取得できませんでした")
            
        # テクニカル指標の追加
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['ma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        
        # 特徴量の選択
        features = df[['close', 'volume', 'rsi', 'macd', 'ma20']].values
        features = self.scaler.fit_transform(features)
        
        # LSTM用のデータ形式に変換
        X = []
        y = []
        for i in range(len(features) - self.lookback_period):
            X.append(features[i:(i + self.lookback_period)])
            y.append(features[i + self.lookback_period, 0])
        return np.array(X), np.array(y)

    def train_model(self):
        """モデルの訓練"""
        df = self.get_historical_data()
        X, y = self.prepare_data(df)
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    def predict_next_price(self):
        """次の価格を予測"""
        df = self.get_historical_data()
        X, _ = self.prepare_data(df)
        prediction = self.model.predict(X[-1:])
        return self.scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

    def execute_trade(self, predicted_price):
        """取引の実行"""
        try:
            # 現在の価格を取得
            ticker = self.client.mix_get_ticker(symbol=self.symbol)
            current_price = float(ticker['data']['last'])
            
            # 予測価格と現在価格の差に基づいて取引判断
            price_difference = ((predicted_price - current_price) / current_price) * 100
            print(f"現在価格: {current_price:.2f} USDT")
            print(f"予測価格差: {price_difference:.2f}%")
            
            if price_difference > 1:  # 1%以上の上昇予測
                # 買い注文（ロング）
                order = self.client.mix_place_order(
                    symbol=self.symbol,
                    marginCoin='USDT',
                    size='0.001',
                    side='open_long',
                    orderType='market'
                )
                print(f"ロングポジション開始: {order}")
                self.print_balance()
                
            elif price_difference < -1:  # 1%以上の下落予測
                # 売り注文（ショート）
                order = self.client.mix_place_order(
                    symbol=self.symbol,
                    marginCoin='USDT',
                    size='0.001',
                    side='open_short',
                    orderType='market'
                )
                print(f"ショートポジション開始: {order}")
                self.print_balance()
                
        except Exception as e:
            print(f"取引エラー: {e}")

    def run(self):
        """ボットの実行"""
        print("AIトレーディングボット（Bitget版）を開始します...")
        while True:
            try:
                # モデルの訓練
                self.train_model()
                
                # 価格予測
                predicted_price = self.predict_next_price()
                print(f"予測価格: {predicted_price}")
                
                # 取引実行
                self.execute_trade(predicted_price)
                
                # 1時間待機
                time.sleep(3600)
                
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = AITradingBot()
    bot.run() 