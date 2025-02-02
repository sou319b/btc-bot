import requests
import pandas as pd
import numpy as np

# ビットコインの価格データを取得（例: Binance API）
def get_bitcoin_data(symbol="BTC/USDT", timeframe="1h", limit=1000):
    import ccxt
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# データ取得
data = get_bitcoin_data()
print(data.tail())

#--2
from sklearn.preprocessing import MinMaxScaler

# 特徴量とターゲットの準備
def preprocess_data(data):
    # 終値を使用
    prices = data["close"].values.reshape(-1, 1)
    
    # 正規化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # 時系列データの作成
    X, y = [], []
    lookback = 60  # 過去60時間分を使用
    for i in range(lookback, len(scaled_prices)):
        X.append(scaled_prices[i - lookback:i, 0])
        y.append(scaled_prices[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM用に形状変更
    return X, y, scaler

X, y, scaler = preprocess_data(data)

#--3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# LSTMモデルの構築
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # 出力層
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

model = build_model((X.shape[1], 1))

# モデルの訓練
model.fit(X, y, batch_size=32, epochs=10)

#--4
# 予測と売買シグナルの生成
def generate_signals(model, data, scaler):
    last_60_prices = data["close"].values[-60:]
    scaled_last_60 = scaler.transform(last_60_prices.reshape(-1, 1))
    X_test = np.array([scaled_last_60])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    current_price = data["close"].iloc[-1]
    if predicted_price > current_price:
        return "BUY"
    elif predicted_price < current_price:
        return "SELL"
    else:
        return "HOLD"

signal = generate_signals(model, data, scaler)
print(f"取引シグナル: {signal}")

#--5
# シミュレーションによるバックテスト
def backtest(data, signals):
    capital = 10000  # 初期資本
    btc_held = 0
    
    for i in range(len(signals)):
        price = data["close"].iloc[i]
        if signals[i] == "BUY" and capital > 0:
            btc_held += capital / price
            capital = 0
        elif signals[i] == "SELL" and btc_held > 0:
            capital += btc_held * price
            btc_held = 0
    
    final_value = capital + btc_held * data["close"].iloc[-1]
    return final_value

# シグナルリストを生成（簡易版）
signals = ["BUY" if i % 2 == 0 else "SELL" for i in range(len(data))]
final_value = backtest(data, signals)
print(f"最終資産: ${final_value:.2f}")
