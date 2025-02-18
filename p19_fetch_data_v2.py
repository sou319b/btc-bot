"""
価格データおよび追加市場データ取得スクリプト（拡張版）
OHLCV（始値、高値、安値、終値、取引量）データに加え、
追加の市場データ（オーダーブック、取引指標など）を取得
"""

import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import time
import ta
import requests
import traceback

def setup_logging():
    """ロギングの設定"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/data_fetch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """テクニカル指標の計算（拡張版）"""
    # 基本的なテクニカル指標
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_4h'] = ta.momentum.RSIIndicator(df['close'], window=16).rsi()  # 4時間相当
    
    # ボリンジャーバンド
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()
    df['bollinger_pct'] = (df['close'] - df['bollinger_low']) / (df['bollinger_high'] - df['bollinger_low'])
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # 移動平均線
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
    
    # EMA
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    
    # モメンタム指標
    df['momentum_1'] = df['close'].pct_change(1)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    
    # ボラティリティ指標
    df['volatility_5'] = df['close'].rolling(5).std()
    df['volatility_10'] = df['close'].rolling(10).std()
    df['volatility_ratio'] = df['volatility_5'] / df['volatility_10']
    
    # トレンド強度指標（改善版）
    df['trend_strength'] = (
        (df['sma_20'] > df['sma_50']).astype(int) * 2 +
        (df['sma_50'] > df['sma_200']).astype(int) * 3 +
        (df['close'] > df['sma_20']).astype(int) * 2 +
        (df['macd'] > df['macd_signal']).astype(int) * 2 +
        (df['rsi'] > 50).astype(int)
    ) / 10  # 0-1の範囲に正規化
    
    # ボリューム関連指標
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['relative_volume'] = df['volume'] / df['volume_ma']
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    # キャンドルスティックパターン
    df['doji'] = abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1
    df['price_range_ratio'] = (df['high'] - df['low']) / df['low']
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    # サポート/レジスタンスレベル
    df['support'] = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()
    df['price_to_support'] = (df['close'] - df['support']) / df['support']
    df['price_to_resistance'] = (df['resistance'] - df['close']) / df['close']
    
    return df

def get_orderbook_data(session, symbol):
    """オーダーブックデータの取得（改善版）"""
    try:
        orderbook = session.get_orderbook(
            category="spot",
            symbol=symbol,
            limit=50
        )
        
        if 'result' in orderbook:
            bids = orderbook['result']['b']
            asks = orderbook['result']['a']
            
            # オーダーブックインバランスの計算（改善版）
            bid_volume = sum(float(bid[1]) for bid in bids[:10])  # 上位10件
            ask_volume = sum(float(ask[1]) for ask in asks[:10])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
            else:
                imbalance = 0
            
            # スプレッドの計算
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # 価格圧力の計算
            bid_pressure = sum(float(bid[1]) * float(bid[0]) for bid in bids[:5])
            ask_pressure = sum(float(ask[1]) * float(ask[0]) for ask in asks[:5])
            
            return {
                'orderbook_imbalance': imbalance,
                'spread': spread,
                'bid_depth': bid_volume,
                'ask_depth': ask_volume,
                'bid_pressure': bid_pressure,
                'ask_pressure': ask_pressure
            }
    except Exception as e:
        logging.error(f"オーダーブックデータの取得エラー: {e}")
    
    return None

def get_fear_and_greed_index():
    """Fear & Greed Indexの取得"""
    try:
        response = requests.get('https://api.alternative.me/fng/')
        if response.status_code == 200:
            data = response.json()
            return int(data['data'][0]['value'])
    except Exception as e:
        logging.error(f"Fear & Greed Indexの取得エラー: {e}")
    
    return None

def fetch_historical_data():
    """拡張版データ取得関数（3ヶ月分、15分足）"""
    logger = setup_logging()
    
    try:
        # データ保存用のディレクトリを作成
        os.makedirs("data", exist_ok=True)
        
        # Bybit APIクライアントの初期化
        session = HTTP(testnet=True)
        
        # 現在時刻から3ヶ月前までのタイムスタンプを計算
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (90 * 24 * 60 * 60)  # 3ヶ月前
        
        all_data = []
        current_start = start_time
        
        logger.info("OHLCVデータの取得を開始します...")
        
        while current_start < end_time:
            # 1000件ずつデータを取得（APIの制限に対応）
            klines = session.get_kline(
                category="spot",
                symbol="BTCUSDT",
                interval=15,  # 15分足
                start=current_start * 1000,
                end=min(current_start + 900000, end_time) * 1000,  # 最大15000分のデータ
                limit=1000
            )
            
            if 'result' in klines and 'list' in klines['result'] and klines['result']['list']:
                all_data.extend(klines['result']['list'])
                last_timestamp = int(klines['result']['list'][0][0]) // 1000
                current_start = last_timestamp + 900  # 次の開始時刻（15分後）
                logger.info(f"データ取得中: {datetime.fromtimestamp(current_start)}")
            else:
                logger.warning(f"データが取得できませんでした: {datetime.fromtimestamp(current_start)}")
                current_start += 900000  # エラー時は15000分進める
            
            time.sleep(0.1)  # API制限を考慮して少し待機
        
        if not all_data:
            logger.error("データを取得できませんでした")
            return False
        
        # データをDataFrameに変換
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # データの型変換
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)
        
        # タイムスタンプを日時に変換
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # データの並べ替えとクリーニング
        df = df.sort_values('timestamp')
        df = df.drop_duplicates(subset=['timestamp'])
        
        # 必要なカラムのみを抽出
        price_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # 基本的な検証
        logger.info("データの検証を行います...")
        logger.info(f"欠損値の数:\n{price_data.isnull().sum()}")
        logger.info(f"データの範囲:\n{price_data.describe()}")
        
        # 異常値のチェックと処理
        for col in numeric_columns:
            # 異常値を検出（平均から3標準偏差以上離れている値）
            mean = price_data[col].mean()
            std = price_data[col].std()
            outliers = price_data[abs(price_data[col] - mean) > 3 * std]
            
            if len(outliers) > 0:
                logger.warning(f"{col}列に{len(outliers)}件の異常値が存在します")
                # 異常値を前の値で置換
                price_data.loc[outliers.index, col] = price_data[col].shift(1)
        
        # テクニカル指標の追加
        price_data = calculate_technical_indicators(price_data)
        
        # 2時間ごとにオーダーブックデータを取得
        orderbook_data = []
        current_time = price_data['timestamp'].min()
        while current_time <= price_data['timestamp'].max():
            ob_data = get_orderbook_data(session, "BTCUSDT")
            if ob_data:
                ob_data['timestamp'] = current_time
                orderbook_data.append(ob_data)
            current_time += timedelta(hours=2)
            time.sleep(0.1)
        
        # オーダーブックデータをメインのDataFrameとマージ
        if orderbook_data:
            ob_df = pd.DataFrame(orderbook_data)
            price_data = pd.merge_asof(price_data, ob_df, on='timestamp')
        
        # Fear & Greed Indexの取得（日次データ）
        fear_greed = get_fear_and_greed_index()
        if fear_greed is not None:
            price_data['fear_greed_index'] = fear_greed
        
        # CSVファイルとして保存
        csv_filename = f"data/historical_data_{datetime.now().strftime('%Y%m%d')}.csv"
        price_data.to_csv(csv_filename, index=False)
        
        logger.info(f"過去3ヶ月の15分足データを {csv_filename} に保存しました")
        logger.info(f"取得期間: {price_data['timestamp'].min()} から {price_data['timestamp'].max()}")
        logger.info(f"データ数: {len(price_data)}件")
        
        # データの品質チェック
        logger.info("\nデータ品質レポート:")
        logger.info(f"完全なローの数: {len(price_data.dropna())}/{len(price_data)}")
        logger.info("各特徴量の基本統計量:")
        logger.info(price_data.describe().to_string())
        
        return True
        
    except Exception as e:
        logger.error(f"データ取得エラー: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    fetch_historical_data() 