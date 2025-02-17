"""
価格データ取得スクリプト（改良版）
OHLCV（始値、高値、安値、終値、取引量）データを取得
"""

import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from pybit.unified_trading import HTTP
import time

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

def fetch_historical_data():
    """過去1週間の1分足価格データを取得（OHLCV）"""
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
        
        # CSVファイルとして保存
        csv_filename = f"data/historical_data_{datetime.now().strftime('%Y%m%d')}.csv"
        price_data.to_csv(csv_filename, index=False)
        
        logger.info(f"過去1週間の1分足OHLCVデータを {csv_filename} に保存しました")
        logger.info(f"取得期間: {price_data['timestamp'].min()} から {price_data['timestamp'].max()}")
        logger.info(f"データ数: {len(price_data)}件")
        logger.info("\nカラム一覧:")
        for col in price_data.columns:
            logger.info(f"- {col}")
        
        return True
        
    except Exception as e:
        logger.error(f"データ取得エラー: {e}")
        return False

if __name__ == "__main__":
    fetch_historical_data() 