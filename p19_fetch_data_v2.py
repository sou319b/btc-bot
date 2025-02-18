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
    """テクニカル指標の計算"""
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # ボリンジャーバンド
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    return df

def get_orderbook_data(session, symbol):
    """オーダーブックデータの取得"""
    try:
        orderbook = session.get_orderbook(
            category="spot",
            symbol=symbol,
            limit=50
        )
        
        if 'result' in orderbook:
            bids = orderbook['result']['b']
            asks = orderbook['result']['a']
            
            # オーダーブックインバランスの計算
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # スプレッドの計算
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid
            
            return {
                'orderbook_imbalance': imbalance,
                'spread': spread,
                'bid_depth': bid_volume,
                'ask_depth': ask_volume
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
    """拡張版データ取得関数"""
    logger = setup_logging()
    
    try:
        # データ保存用のディレクトリを作成
        os.makedirs("data", exist_ok=True)
        
        # Bybit APIクライアントの初期化
        session = HTTP(testnet=True)
        
        # 現在時刻から1週間前までのタイムスタンプを計算
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (7 * 24 * 60 * 60)  # 1週間前
        
        all_data = []
        current_start = start_time
        
        logger.info("OHLCVデータの取得を開始します...")
        
        while current_start < end_time:
            # 1000件ずつデータを取得（APIの制限に対応）
            klines = session.get_kline(
                category="spot",
                symbol="BTCUSDT",
                interval=1,  # 1分足
                start=current_start * 1000,
                end=min(current_start + 60000, end_time) * 1000,  # 最大1000分のデータ
                limit=1000
            )
            
            if 'result' in klines and 'list' in klines['result'] and klines['result']['list']:
                all_data.extend(klines['result']['list'])
                last_timestamp = int(klines['result']['list'][0][0]) // 1000
                current_start = last_timestamp + 60  # 次の開始時刻
                logger.info(f"データ取得中: {datetime.fromtimestamp(current_start)}")
            else:
                logger.warning(f"データが取得できませんでした: {datetime.fromtimestamp(current_start)}")
                current_start += 60000  # エラー時は1000分進める
            
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
        
        # 異常値のチェック（例：ゼロまたは負の値）
        for col in numeric_columns:
            invalid_count = len(price_data[price_data[col] <= 0])
            if invalid_count > 0:
                logger.warning(f"{col}列に{invalid_count}件の無効なデータが存在します")
        
        # テクニカル指標の追加
        price_data = calculate_technical_indicators(price_data)
        
        # 30分ごとにオーダーブックデータを取得
        orderbook_data = []
        current_time = price_data['timestamp'].min()
        while current_time <= price_data['timestamp'].max():
            ob_data = get_orderbook_data(session, "BTCUSDT")
            if ob_data:
                ob_data['timestamp'] = current_time
                orderbook_data.append(ob_data)
            current_time += timedelta(minutes=30)
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
        
        logger.info(f"過去1週間の1分足OHLCVデータを {csv_filename} に保存しました")
        logger.info(f"取得期間: {price_data['timestamp'].min()} から {price_data['timestamp'].max()}")
        logger.info(f"データ数: {len(price_data)}件")
        logger.info("\nカラム一覧:")
        for col in price_data.columns:
            logger.info(f"- {col}")
        
        logger.info("追加の市場データを含む拡張データセットを作成しました")
        logger.info("\n新しいカラム一覧:")
        for col in price_data.columns:
            logger.info(f"- {col}")
        
        return True
        
    except Exception as e:
        logger.error(f"データ取得エラー: {e}")
        return False

if __name__ == "__main__":
    fetch_historical_data() 