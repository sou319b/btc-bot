"""
テストデータ準備スクリプト
- トレーニングデータとは別の期間のデータを取得
- 前処理とスケーリングの適用
- バックテスト用のデータセット作成
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
import time
import logging
import traceback
from sklearn.preprocessing import StandardScaler
import ta

def setup_logging():
    """ロギングの設定"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/test_data_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def fetch_test_data(session, start_time, end_time):
    """テスト期間のデータ取得"""
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        try:
            klines = session.get_kline(
                category="spot",
                symbol="BTCUSDT",
                interval=15,  # 15分足
                start=current_start * 1000,
                end=min(current_start + 900000, end_time) * 1000,
                limit=1000
            )
            
            if 'result' in klines and 'list' in klines['result'] and klines['result']['list']:
                all_data.extend(klines['result']['list'])
                last_timestamp = int(klines['result']['list'][0][0]) // 1000
                current_start = last_timestamp + 900
            else:
                current_start += 900000
            
            time.sleep(0.1)  # API制限対策
            
        except Exception as e:
            logging.error(f"データ取得エラー: {e}")
            current_start += 900000
    
    return all_data

def prepare_features(df):
    """特徴量の作成（トレーニングデータと同じ特徴量を使用）"""
    # 安全な除算のためのヘルパー関数
    def safe_divide(a, b):
        return np.where(b != 0, a / b, 0)
    
    # 基本的な価格指標
    df['price_range'] = df['high'] - df['low']
    df['body_ratio'] = safe_divide(abs(df['close'] - df['open']), df['price_range'])
    
    # トレンド指標
    windows = [20, 50, 100]
    for window in windows:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        df[f'slope_{window}'] = safe_divide(df[f'sma_{window}'] - df[f'sma_{window}'].shift(window), window)
    
    # ボリューム分析
    df['volume_sma_15'] = df['volume'].rolling(window=15).mean()
    df['volume_ratio'] = safe_divide(df['volume'], df['volume_sma_15'])
    
    # モメンタム指標
    for period in [15, 30]:
        df[f'momentum_{period}'] = df['close'].diff(period)
        df[f'roc_{period}'] = safe_divide(df['close'] - df['close'].shift(period), df['close'].shift(period)) * 100
    
    # ボラティリティ指標（ATR）
    window = 20
    tr = pd.DataFrame()
    tr['hl'] = df['high'] - df['low']
    tr['hc'] = abs(df['high'] - df['close'].shift(1))
    tr['lc'] = abs(df['low'] - df['close'].shift(1))
    tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
    df['atr_20'] = tr['tr'].rolling(window=window).mean()
    
    return df

def scale_features(df, feature_columns, scaler_path=None):
    """特徴量のスケーリング（トレーニングデータのスケーラーを使用）"""
    if scaler_path and os.path.exists(scaler_path):
        import joblib
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

def main():
    logger = setup_logging()
    
    try:
        # Bybit APIクライアントの初期化
        session = HTTP(testnet=True)
        
        # テストデータの期間設定（直近1ヶ月）
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (30 * 24 * 60 * 60)  # 30日前
        
        logger.info("テストデータの取得を開始します...")
        all_data = fetch_test_data(session, start_time, end_time)
        
        if not all_data:
            logger.error("テストデータを取得できませんでした")
            return False
        
        # データフレームの作成
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # データの型変換
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)
        
        # タイムスタンプの変換
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # データのクリーニング
        df = df.sort_values('timestamp')
        df = df.drop_duplicates(subset=['timestamp'])
        
        # 特徴量の作成
        logger.info("特徴量を作成します...")
        df = prepare_features(df)
        
        # 使用する特徴量
        feature_columns = [
            'price_range', 'body_ratio',
            'sma_20', 'sma_50', 'sma_100',
            'ema_20', 'ema_50', 'ema_100',
            'slope_20', 'slope_50', 'slope_100',
            'volume_sma_15', 'volume_ratio',
            'momentum_15', 'momentum_30',
            'roc_15', 'roc_30',
            'atr_20'
        ]
        
        # 特徴量のスケーリング
        logger.info("特徴量をスケーリングします...")
        df = scale_features(df, feature_columns)
        
        # データの保存
        output_file = f"data/test_data_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"テストデータを {output_file} に保存しました")
        logger.info(f"データ期間: {df['timestamp'].min()} から {df['timestamp'].max()}")
        logger.info(f"データ数: {len(df)}件")
        
        return True
        
    except Exception as e:
        logger.error(f"テストデータ準備中にエラーが発生しました: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 