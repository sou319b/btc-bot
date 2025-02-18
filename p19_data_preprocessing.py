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
    """追加の特徴量を生成"""
    # 価格変動率
    df['returns'] = df['close'].pct_change()
    
    # ボラティリティ（過去10分間）
    df['volatility'] = df['returns'].rolling(window=10).std()
    
    # 移動平均線
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # 出来高加重平均価格（VWAP）
    df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    
    # 価格モメンタム
    df['momentum'] = df['close'] - df['close'].shift(5)
    
    return df

def scale_features(df, feature_columns):
    """特徴量のスケーリング"""
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

def preprocess_data():
    """メイン処理関数（改善版）"""
    logger = setup_logging()
    
    try:
        # データの読み込み
        logger.info("データの読み込みを開始します...")
        df = load_latest_data()
        
        # タイムスタンプをdatetime型に変換
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 基本的なデータクリーニング
        logger.info("基本的なデータクリーニングを実行します...")
        
        # 異常値の除外（より厳密な基準）
        df = df[df['close'] > 0]  # 負の価格を除外
        df = df[df['volume'] > 0]  # 取引量が0以下のデータを除外
        
        # 極端な価格変動の除外
        df['returns'] = df['close'].pct_change()
        df = df[df['returns'].abs() < 0.05]  # 5%以上の価格変動を除外
        
        # 欠損値の処理（より洗練された方法）
        logger.info("欠損値の処理を行います...")
        
        # 時系列データの特性を考慮した補完
        for col in ['open', 'high', 'low', 'close', 'volume']:
            # 線形補間を試みる
            df[col] = df[col].interpolate(method='linear')
            # それでも残る欠損値は前方補完
            df[col] = df[col].fillna(method='ffill')
            # 先頭の欠損値は後方補完
            df[col] = df[col].fillna(method='bfill')
        
        # 特徴量エンジニアリング（改善版）
        logger.info("特徴量エンジニアリングを実行します...")
        
        # 価格関連の特徴量
        df['price_range'] = df['high'] - df['low']
        df['price_range_ratio'] = df['price_range'] / df['close']
        df['body_ratio'] = abs(df['close'] - df['open']) / df['price_range']
        
        # 移動平均線
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # ボリューム関連の特徴量
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma10'] = df['volume'].rolling(window=10).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma5']
        
        # モメンタム特徴量
        df['momentum_1'] = df['close'].pct_change(1)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        # ボラティリティ特徴量
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_10']
        
        # トレンド強度指標
        df['trend_strength'] = abs(df['sma_5'] - df['sma_20']) / df['sma_20']
        
        # 予測ターゲットの生成（改善版）
        logger.info("予測ターゲットを生成します...")
        
        # 将来の価格変化率を計算（複数の時間枠）
        for period in [5, 10, 15]:
            df[f'future_return_{period}'] = df['close'].shift(-period) / df['close'] - 1
        
        # 主要な予測ターゲット
        df['target'] = df['future_return_5'].apply(lambda x: 1 if x > 0.001 else (0 if x < -0.001 else 0.5))
        
        # 特徴量のスケーリング（改善版）
        logger.info("特徴量のスケーリングを実行します...")
        
        # スケーリング対象の特徴量を選択
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'bollinger_high', 'bollinger_low',
            'macd', 'macd_signal', 'atr',
            'returns', 'momentum_1', 'momentum_5', 'momentum_10',
            'volatility_5', 'volatility_10', 'trend_strength',
            'price_range_ratio', 'body_ratio', 'relative_volume'
        ]
        
        # RobustScalerを使用してスケーリング
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # 最終的なデータクリーニング
        logger.info("最終的なデータクリーニングを実行します...")
        
        # 無限大や非数値を含む行を除外
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # 特徴量の相関分析
        correlation_matrix = df[feature_columns].corr()
        high_correlation_pairs = []
        for i in range(len(feature_columns)):
            for j in range(i+1, len(feature_columns)):
                if abs(correlation_matrix.iloc[i,j]) > 0.95:
                    high_correlation_pairs.append((feature_columns[i], feature_columns[j]))
        
        if high_correlation_pairs:
            logger.info("高相関の特徴量ペア:")
            for pair in high_correlation_pairs:
                logger.info(f"{pair[0]} - {pair[1]}: {correlation_matrix.loc[pair[0], pair[1]]:.3f}")
        
        # 前処理済みデータの保存
        output_file = f"data/preprocessed_data_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"前処理済みデータを {output_file} に保存しました")
        
        # データの品質レポート
        logger.info("\nデータの品質レポート:")
        logger.info(f"総行数: {len(df)}")
        logger.info(f"特徴量数: {len(feature_columns)}")
        logger.info("\n各クラスの分布:")
        logger.info(df['target'].value_counts(normalize=True))
        
        # 基本統計量の出力
        logger.info("\n特徴量の基本統計量:")
        logger.info(df[feature_columns].describe().to_string())
        
        return True
        
    except Exception as e:
        logger.error(f"前処理中にエラーが発生しました: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    preprocess_data() 