import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ビットコインのデータを取得
btc = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01')

# 特徴量を作成
def create_features(df):
    df['SMA_7'] = df['Close'].rolling(window=7).mean()    # 7日移動平均
    df['SMA_30'] = df['Close'].rolling(window=30).mean()  # 30日移動平均
    df['Price_Change'] = df['Close'].pct_change()         # 価格変化率
    df['Volume_Change'] = df['Volume'].pct_change()       # 取引量変化率
    return df

# データの前処理
btc = create_features(btc)
btc = btc.dropna()  # 欠損値を削除

# 特徴量とターゲットを定義
features = ['Open', 'High', 'Low', 'Volume', 'SMA_7', 'SMA_30', 'Price_Change', 'Volume_Change']
X = btc[features]
y = btc['Close']

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# スケーリング
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデルの訓練
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# モデルの評価
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f'訓練データのR2スコア: {train_score:.3f}')
print(f'テストデータのR2スコア: {test_score:.3f}')

# 特徴量の重要度を表示
feature_importance = pd.DataFrame({
    '特徴量': features,
    '重要度': model.feature_importances_
}).sort_values('重要度', ascending=False)

print("\n特徴量の重要度:")
print(feature_importance)

# 予測値を計算
y_pred = model.predict(X_test_scaled)

# データを1次元に変換
y_test_series = pd.Series(y_test).reset_index(drop=True)
y_pred_series = pd.Series(y_pred)

# 実際の価格と予測価格を比較
results = pd.DataFrame({
    '実際の価格': y_test_series,
    '予測価格': y_pred_series
})

# 最初の10件を表示
print("\n価格予測の比較（最初の10件）:")
print(results.head(10))

# 予測誤差の計算
mae = mean_absolute_error(y_test_series, y_pred_series)
rmse = np.sqrt(mean_squared_error(y_test_series, y_pred_series))
mape = np.mean(np.abs((y_test_series - y_pred_series) / y_test_series)) * 100

print(f'\n平均絶対誤差（MAE）: ${mae:,.2f}')
print(f'二乗平均平方根誤差（RMSE）: ${rmse:,.2f}')
print(f'平均絶対パーセント誤差（MAPE）: {mape:.2f}%')

# 予測値と実際の値をプロット
plt.figure(figsize=(12, 6))
plt.plot(y_test_series[:100], label='実際の価格', color='blue')
plt.plot(y_pred_series[:100], label='予測価格', color='red', linestyle='--')
plt.title('ビットコイン価格：予測 vs 実際（最初の100日間）')
plt.xlabel('日数')
plt.ylabel('価格 (USD)')
plt.legend()
plt.grid(True)
plt.show()
