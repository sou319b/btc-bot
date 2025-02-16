import os
import logging
from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP

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
    """過去1週間の価格データを取得"""
    logger = setup_logging()
    
    try:
        # データ保存用のディレクトリを作成
        os.makedirs("data", exist_ok=True)
        
        # Bybit APIクライアントの初期化
        session = HTTP(testnet=True)
        
        # 現在時刻から1週間前までのタイムスタンプを計算
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (7 * 24 * 60 * 60)  # 1週間前
        
        # データ取得
        klines = session.get_kline(
            category="spot",
            symbol="BTCUSDT",
            interval=5,  # 5分足
            start=start_time * 1000,  # ミリ秒単位
            end=end_time * 1000,
            limit=200
        )
        
        # データをDataFrameに変換
        df = pd.DataFrame(klines['result']['list'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # タイムスタンプを日時に変換
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')
        
        # 必要なカラムのみを抽出
        price_data = df[['timestamp', 'close']].copy()
        price_data['close'] = price_data['close'].astype(float)
        
        # CSVファイルとして保存
        csv_filename = f"data/historical_data_{datetime.now().strftime('%Y%m%d')}.csv"
        price_data.to_csv(csv_filename, index=False)
        logger.info(f"過去1週間のデータを {csv_filename} に保存しました")
        
        return True
        
    except Exception as e:
        logger.error(f"データ取得エラー: {e}")
        return False

if __name__ == "__main__":
    fetch_historical_data() 