"""
価格データから特徴量を生成するスクリプト
pandas-taを使用して各種テクニカル指標と市場特性を計算
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import pandas_ta as ta
from datetime import datetime

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            timestamp, open, high, low, close, volumeを含む価格データフレーム
        """
        self.df = df.copy()
        
        # カラム名を小文字に統一
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # timestampをインデックスに設定
        if 'timestamp' in self.df.columns:
            self.df.set_index('timestamp', inplace=True)
        
        # 必要なカラムの存在確認
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"データフレームに以下のカラムが不足しています: {missing_columns}")
    
    def add_basic_features(self) -> pd.DataFrame:
        """基本的な特徴量を追加"""
        # RSIの計算
        self.df['rsi_14'] = ta.rsi(self.df['close'], length=14)
        
        # ボリンジャーバンドの計算
        bb = ta.bbands(self.df['close'], length=20, std=2)
        self.df['bb_lower'] = bb['BBL_20_2.0']
        self.df['bb_middle'] = bb['BBM_20_2.0']
        self.df['bb_upper'] = bb['BBU_20_2.0']
        
        # 移動平均の計算
        self.df['ma_5'] = ta.sma(self.df['close'], length=5)
        self.df['ma_13'] = ta.sma(self.df['close'], length=13)
        self.df['ma_21'] = ta.sma(self.df['close'], length=21)
        
        # ATRの計算
        self.df['atr'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        
        # ボラティリティの計算（ATRを価格で正規化）
        self.df['volatility'] = self.df['atr'] / self.df['close']
        
        return self.df
    
    def add_advanced_features(self) -> pd.DataFrame:
        """より高度な特徴量を追加"""
        # MACDの計算
        macd = ta.macd(self.df['close'], fast=12, slow=26, signal=9)
        self.df['macd'] = macd['MACD_12_26_9']
        self.df['macd_signal'] = macd['MACDs_12_26_9']
        self.df['macd_hist'] = macd['MACDh_12_26_9']
        
        # ストキャスティクスの計算
        stoch = ta.stoch(self.df['high'], self.df['low'], self.df['close'], k=14, d=3, smooth_k=3)
        self.df['stoch_k'] = stoch['STOCHk_14_3_3']
        self.df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # ADXの計算
        adx = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=14)
        self.df['adx'] = adx['ADX_14']
        
        # CCIの計算
        self.df['cci'] = ta.cci(self.df['high'], self.df['low'], self.df['close'], length=14)
        
        # OBVの計算
        self.df['obv'] = ta.obv(self.df['close'], self.df['volume'])
        
        return self.df
    
    def add_price_action_features(self) -> pd.DataFrame:
        """価格アクション関連の特徴量を追加"""
        # ローソク足の特徴
        self.df['body_size'] = abs(self.df['close'] - self.df['open']) / self.df['open']
        self.df['upper_shadow'] = (self.df['high'] - self.df['close'].where(self.df['close'] > self.df['open'], self.df['open'])) / self.df['open']
        self.df['lower_shadow'] = (self.df['close'].where(self.df['close'] < self.df['open'], self.df['open']) - self.df['low']) / self.df['open']
        
        # モメンタム指標
        self.df['price_momentum'] = self.df['close'].pct_change(periods=5)
        self.df['volume_momentum'] = self.df['volume'].pct_change(periods=5)
        
        # 価格の加速度
        self.df['price_acceleration'] = self.df['price_momentum'].diff()
        
        # DMIの計算（修正版）
        dm = ta.dm(self.df['high'], self.df['low'], length=14)
        self.df['trend_strength'] = abs(dm['DMP_14'] - dm['DMN_14'])
        
        return self.df
    
    def add_market_timing_features(self) -> pd.DataFrame:
        """市場タイミング関連の特徴量を追加"""
        # 時間帯と曜日
        self.df['hour_of_day'] = pd.to_datetime(self.df.index).hour
        self.df['day_of_week'] = pd.to_datetime(self.df.index).dayofweek
        
        # ボラティリティレジーム
        volatility = self.df['volatility'].rolling(window=20).std()
        self.df['volatility_regime'] = pd.qcut(volatility, q=3, labels=['low', 'medium', 'high'], duplicates='drop')
        
        # トレンドレジーム
        price_trend = self.df['close'].rolling(window=20).mean()
        self.df['trend_regime'] = np.where(
            self.df['close'] > price_trend * 1.02, 'uptrend',
            np.where(self.df['close'] < price_trend * 0.98, 'downtrend', 'sideways')
        )
        
        # 市場フェーズ
        self.df['market_phase'] = self.df['volatility_regime'].astype(str) + '_' + self.df['trend_regime']
        
        return self.df
    
    def create_target_variables(self, time_frame: int) -> pd.DataFrame:
        """予測目標となる変数を生成"""
        # 価格の方向
        future_returns = self.df['close'].shift(-time_frame) / self.df['close'] - 1
        self.df['price_direction'] = np.where(future_returns > 0, 1, 0)
        
        # トレンドの強さ
        self.df['trend_strength_target'] = pd.qcut(
            self.df['trend_strength'],
            q=3,
            labels=['weak', 'medium', 'strong'],
            duplicates='drop'
        )
        
        # 取引シグナル
        self.df['trading_signal'] = np.where(
            future_returns > 0.001, 1,
            np.where(future_returns < -0.001, -1, 0)
        )
        
        # ボラティリティレベル
        self.df['volatility_level'] = pd.qcut(
            self.df['volatility'],
            q=3,
            labels=['low', 'medium', 'high'],
            duplicates='drop'
        )
        
        # 価格変化率
        self.df['price_change'] = future_returns
        
        # 次の価格
        self.df['next_price'] = self.df['close'].shift(-time_frame)
        
        # 最適ポジションサイズ
        volatility_weight = 1 - self.df['volatility'].rank(pct=True)
        trend_weight = self.df['trend_strength'].rank(pct=True)
        self.df['optimal_position'] = (volatility_weight + trend_weight) / 2
        
        # リスク/リワード比
        potential_gain = (self.df['bb_upper'] - self.df['close']) / self.df['close']
        potential_loss = (self.df['close'] - self.df['bb_lower']) / self.df['close']
        self.df['risk_reward'] = potential_gain / potential_loss
        
        return self.df
    
    def prepare_features(self, time_frame: int) -> pd.DataFrame:
        """すべての特徴量を生成"""
        self.add_basic_features()
        self.add_advanced_features()
        self.add_price_action_features()
        self.add_market_timing_features()
        self.create_target_variables(time_frame)
        
        # 欠損値の処理
        self.df = self.df.dropna()
        
        return self.df

def main():
    # データの読み込み
    df = pd.read_csv('./data/historical_data_20250217.csv')
    
    # 特徴量エンジニアリング
    fe = FeatureEngineering(df)
    df_features = fe.prepare_features(time_frame=5)  # 5分後を予測
    
    # 結果の保存
    df_features.to_csv('./data/features_20250217.csv')
    print(f"特徴量の生成が完了しました。Shape: {df_features.shape}")
    print("\n利用可能な特徴量:")
    print(df_features.columns.tolist())

if __name__ == "__main__":
    main() 