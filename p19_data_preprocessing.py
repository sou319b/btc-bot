"""
データ前処理スクリプト
- 欠損値の処理
- 異常値の処理
- 特徴量エンジニアリング
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import logging
from datetime import datetime
import traceback
import ta

def setup_logging():
    """ロギングの設定"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_latest_data():
    """最新のデータファイルを読み込む"""
    data_dir = "data"
    data_files = [f for f in os.listdir(data_dir) if f.startswith("historical_data_")]
    if not data_files:
        raise FileNotFoundError("データファイルが見つかりません")
    
    latest_file = max(data_files)
    return pd.read_csv(os.path.join(data_dir, latest_file))

def handle_missing_values(df):
    """欠損値の処理"""
    # 前方補完と後方補完を組み合わせて欠損値を処理
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # volume=0のデータの処理
    df.loc[df['volume'] == 0, 'volume'] = df['volume'].mean()
    
    return df

def remove_outliers(df, columns, n_std=3):
    """異常値の除去（標準偏差による）"""
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        df[f'{column}_outlier'] = ((df[column] < (mean - n_std * std)) | 
                                  (df[column] > (mean + n_std * std))).astype(int)
        
        # 異常値を平均値で置換
        df.loc[df[f'{column}_outlier'] == 1, column] = mean
    
    return df

def add_features(df):
    """特徴量を追加する関数"""
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
    
    # ボラティリティ指標（ATRの手動計算）
    window = 20
    tr = pd.DataFrame()
    tr['hl'] = df['high'] - df['low']
    tr['hc'] = abs(df['high'] - df['close'].shift(1))
    tr['lc'] = abs(df['low'] - df['close'].shift(1))
    tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
    df['atr_20'] = tr['tr'].rolling(window=window).mean()
    
    # 欠損値を0で埋める
    df = df.fillna(0)
    
    return df

def scale_features(df, feature_columns):
    """特徴量のスケーリング"""
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

def preprocess_data(df):
    """メイン処理関数（中頻度取引向け）"""
    logger = setup_logging()
    
    try:
        logger.info("データの読み込みを開始します...")
        
        # 無限大の値をNaNに置換
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # NaNを含む行を削除
        df = df.dropna()
        
        logger.info("基本的なデータクリーニングを実行します...")
        
        # 異常値の除去（99パーセンタイルを超える値をクリップ）
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'target':  # ターゲット変数は除外
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(q1, q99)
        
        logger.info("特徴量エンジニアリングを実行します...")
        df = add_features(df)
        
        logger.info("予測ターゲットを生成します...")
        df['target'] = np.where(df['close'].shift(-8) > df['close'] * 1.005, 1,  # 上昇
                      np.where(df['close'].shift(-8) < df['close'] * 0.995, -1,  # 下落
                      0))  # 中立
        
        # 使用する特徴量の更新
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
        
        logger.info("特徴量のスケーリングを実行します...")
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        return df, feature_columns
        
    except Exception as e:
        logger.error(f"前処理中にエラーが発生しました: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        # データの読み込み
        input_file = "data/historical_data_20250218.csv"
        logger = setup_logging()
        logger.info(f"データファイル {input_file} を読み込みます...")
        df = pd.read_csv(input_file)
        
        # 前処理の実行
        processed_df, feature_columns = preprocess_data(df)
        
        # 処理済みデータの保存
        output_file = f"data/preprocessed_data_{datetime.now().strftime('%Y%m%d')}.csv"
        processed_df.to_csv(output_file, index=False)
        
        # 処理結果の表示
        logger.info(f"\nデータ処理完了:")
        logger.info(f"入力データ数: {len(df)}")
        logger.info(f"出力データ数: {len(processed_df)}")
        logger.info(f"特徴量数: {len(feature_columns)}")
        logger.info("\nクラス分布:")
        logger.info(processed_df['target'].value_counts(normalize=True))
        
    except Exception as e:
        logger = setup_logging()
        logger.error(f"処理中にエラーが発生しました: {str(e)}")
        logger.error(traceback.format_exc()) 