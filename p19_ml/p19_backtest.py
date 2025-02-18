"""
バックテストスクリプト
- 学習済みモデルを使用した取引シミュレーション
- パフォーマンス評価
- 取引ログの記録
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import traceback

class BackTester:
    def __init__(self, 
                 initial_balance: float = 10000.0,  # 初期資金（USDT）
                 base_position_size: float = 0.05,   # 基本ポジションサイズを5%に設定
                 base_stop_loss: float = 0.015,      # 基本ストップロスを1.5%に設定
                 base_take_profit: float = 0.03      # 基本利確を3%に設定
                ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.base_position_size = base_position_size
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.position = None
        self.entry_price = None
        self.trades = []
        self.equity_curve = []
        self.logger = None
    
    def setup_logging(self):
        """ロギングの設定"""
        os.makedirs("logs", exist_ok=True)
        log_filename = f"logs/backtest_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        return self.logger
    
    def load_model_and_data(self):
        """モデルとデータの読み込み"""
        try:
            # モデルの読み込み
            self.logger.info("モデルファイル model_20250218.txt を読み込みます")
            self.model = lgb.Booster(model_file='models/model_20250218.txt')
            
            # データの読み込み
            self.logger.info("データファイル preprocessed_data_20250218.csv を読み込みます")
            df = pd.read_csv('data/preprocessed_data_20250218.csv')
            
            # 異常値の除外
            df = df[df['close'] > 0]  # 負の価格を除外
            df = df[df['volume'] > 0]  # 取引量が0以下のデータを除外
            
            # 極端な価格変動の除外
            df['returns'] = df['close'].pct_change()
            df = df[df['returns'].abs() < 0.1]  # 10%以上の価格変動を除外
            
            return df
            
        except Exception as e:
            self.logger.error(f"データ読み込み中にエラーが発生しました: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量の準備"""
        try:
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'bollinger_high', 'bollinger_low',
                'macd', 'macd_signal', 'atr',
                'returns', 'momentum_1', 'momentum_5', 'momentum_10',
                'volatility_5', 'volatility_10', 'trend_strength',
                'price_range_ratio', 'body_ratio', 'relative_volume'
            ]
            
            # 特徴量の存在確認
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"以下の特徴量が見つかりません: {missing_columns}")
            
            return df[feature_columns]
            
        except Exception as e:
            self.logger.error(f"特徴量準備中にエラーが発生しました: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def execute_trade(self, timestamp: pd.Timestamp, price: float, action: str, reason: str) -> Dict:
        """トレードの実行（エラー処理の改善）"""
        try:
            if price <= 0:
                self.logger.warning(f"無効な価格 ({price}) でのトレード試行をスキップします")
                return {'profit': 0, 'trade_type': 'skip'}
            
            if action == 'buy':
                if self.position is not None:
                    self.logger.warning("既にポジションが存在します")
                    return {'profit': 0, 'trade_type': 'skip'}
                
                position_size_usdt = self.balance * self.position_size
                self.position = position_size_usdt / price
                self.entry_price = price
                self.logger.info(f"買い注文実行: 価格 {price:.2f} USDT, サイズ {position_size_usdt:.2f} USDT")
                
            elif action == 'sell':
                if self.position is None:
                    self.logger.warning("ポジションが存在しません")
                    return {'profit': 0, 'trade_type': 'skip'}
                
                exit_value = self.position * price
                entry_value = self.position * self.entry_price
                profit = exit_value - entry_value
                
                self.balance += profit
                self.position = None
                self.entry_price = None
                
                self.logger.info(f"売り注文実行: 価格 {price:.2f} USDT, 利益 {profit:.2f} USDT")
                
                trade_info = {
                    'timestamp': timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'profit': profit,
                    'balance': self.balance,
                    'reason': reason
                }
                self.trades.append(trade_info)
                self.equity_curve.append(self.balance)
                
                return {'profit': profit, 'trade_type': 'sell'}
                
        except Exception as e:
            self.logger.error(f"トレード実行中にエラーが発生しました: {str(e)}")
            return {'profit': 0, 'trade_type': 'error'}
    
    def calculate_dynamic_parameters(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, float, float]:
        """動的パラメータの計算（リスク管理の改善）"""
        try:
            # ボラティリティの計算（20期間）
            volatility = df['returns'].rolling(window=20).std().iloc[current_idx]
            
            # ボラティリティに基づくポジションサイズの調整
            position_size = self.base_position_size
            if volatility > 0.02:  # ボラティリティが2%を超える場合
                position_size *= 0.7  # ポジションサイズを30%削減
            elif volatility < 0.005:  # ボラティリティが0.5%未満の場合
                position_size *= 1.2  # ポジションサイズを20%増加
                
            # ボラティリティに基づくストップロスの調整
            stop_loss = max(self.base_stop_loss, volatility * 2)  # ボラティリティの2倍か基本値の大きい方
            stop_loss = min(stop_loss, 0.02)  # 最大2%に制限
            
            # テイクプロフィットの調整
            take_profit = stop_loss * 3  # リスク・リワード比を3:1に設定
            
            return position_size, stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"パラメータ計算中にエラーが発生しました: {str(e)}")
            return self.base_position_size, self.base_stop_loss, self.base_take_profit

    def check_technical_confirmation(self, df: pd.DataFrame, current_idx: int) -> bool:
        """テクニカル指標による確認（改善版）"""
        try:
            # RSIによる確認
            current_rsi = df['rsi'].iloc[current_idx]
            rsi_signal = 10 <= current_rsi <= 90
            
            # MACDによる確認
            current_macd = df['macd'].iloc[current_idx]
            current_signal = df['macd_signal'].iloc[current_idx]
            macd_signal = current_macd > current_signal * 0.5  # より柔軟な条件
            
            # ボリンジャーバンドによる確認
            current_price = df['close'].iloc[current_idx]
            bb_upper = df['bollinger_high'].iloc[current_idx]
            bb_lower = df['bollinger_low'].iloc[current_idx]
            bb_signal = bb_lower <= current_price <= bb_upper
            
            # いずれかの条件を満たせばOK
            return rsi_signal or macd_signal or bb_signal
            
        except Exception as e:
            self.logger.error(f"テクニカル確認中にエラーが発生しました: {str(e)}")
            return False

    def normalize_prediction(self, prediction: np.ndarray) -> float:
        """予測値の正規化（マルチクラス対応版）"""
        try:
            # 予測確率の最大値とそのインデックスを取得
            max_prob = np.max(prediction)
            pred_class = np.argmax(prediction)
            
            # クラスに基づいて予測値を変換
            if pred_class == 2:  # 上昇
                normalized = 0.6 + (max_prob * 0.1)  # 0.6-0.7の範囲
            elif pred_class == 0:  # 下落
                normalized = 0.3 - (max_prob * 0.1)  # 0.2-0.3の範囲
            else:  # 中立
                normalized = 0.45 + ((max_prob - 0.5) * 0.1)  # 0.4-0.5の範囲
            
            return float(normalized)
            
        except Exception as e:
            self.logger.error(f"予測値の正規化中にエラーが発生しました: {str(e)}")
            return 0.5

    def calculate_statistics(self) -> Dict:
        """バックテスト結果の統計計算"""
        try:
            if not self.trades:
                self.logger.warning("取引が一つも実行されませんでした")
                return {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'max_drawdown': 0,
                    'final_balance': self.initial_balance,
                    'return': 0
                }
            
            trades_df = pd.DataFrame(self.trades)
            
            # 統計の計算
            total_trades = len(trades_df)
            profitable_trades = len(trades_df[trades_df['profit'] > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            total_profit = trades_df['profit'].sum()
            
            # ドローダウンの計算
            equity_curve = pd.Series([self.initial_balance] + self.equity_curve)
            rolling_max = equity_curve.expanding().max()
            drawdowns = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # 最終結果
            final_balance = self.balance
            total_return = (final_balance - self.initial_balance) / self.initial_balance
            
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'max_drawdown': max_drawdown,
                'final_balance': final_balance,
                'return': total_return
            }
            
        except Exception as e:
            self.logger.error(f"統計計算中にエラーが発生しました: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def run_backtest(self):
        """バックテストの実行（マルチクラス対応版）"""
        logger = self.setup_logging()
        
        try:
            # データとモデルの読み込み
            logger.info("データとモデルを読み込みます...")
            df = self.load_model_and_data()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 特徴量の準備
            logger.info("特徴量を準備します...")
            X = self.prepare_features(df)
            
            logger.info("バックテストを開始します...")
            logger.info(f"初期資金: {self.initial_balance:.2f} USDT")
            
            prediction_counts = {'up': 0, 'down': 0, 'neutral': 0}
            
            # 予測閾値の調整
            UP_THRESHOLD = 0.55    # 上昇予測の閾値
            DOWN_THRESHOLD = 0.45  # 下落予測の閾値
            
            # マルチタイムフレームトレンド
            df['trend_short'] = df['close'].rolling(window=3).mean()   # 超短期
            df['trend_mid'] = df['close'].rolling(window=8).mean()     # 短期
            df['trend_long'] = df['close'].rolling(window=15).mean()   # 中期
            
            # トレンド強度の計算（より柔軟に）
            df['trend_strength'] = (
                (df['trend_short'] > df['trend_mid']).astype(int) * 3 +  # 超短期の重みを増加
                (df['trend_mid'] > df['trend_long']).astype(int)
            )
            
            # ボラティリティの計算
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # 予測値の保存用
            all_predictions = []
            
            consecutive_losses = 0
            max_consecutive_losses = 3
            
            for i in range(len(df)):
                if i < 20:
                    continue
                    
                current_time = df['timestamp'].iloc[i]
                current_price = df['close'].iloc[i]
                
                if current_price <= 0 or np.isnan(current_price):
                    continue
                
                position_size, stop_loss, take_profit = self.calculate_dynamic_parameters(df, i)
                
                if i % 100 == 0:
                    logger.info(f"現在のパラメータ - ポジションサイズ: {position_size:.1%}, "
                              f"ストップロス: {stop_loss:.1%}, 利確: {take_profit:.1%}")
                
                if consecutive_losses >= max_consecutive_losses:
                    if self.position is not None:
                        self.execute_trade(current_time, current_price, 'sell', '連続損失制限')
                        consecutive_losses = 0
                    continue
                
                if self.position is not None:
                    price_change = (current_price - self.entry_price) / self.entry_price
                    
                    if price_change <= -stop_loss:
                        logger.info(f"ストップロス発動: 価格変化 {price_change:.2%}")
                        trade_result = self.execute_trade(current_time, current_price, 'sell', 'ストップロス')
                        if trade_result['profit'] < 0:
                            consecutive_losses += 1
                        continue
                        
                    elif price_change >= take_profit:
                        logger.info(f"利確条件達成: 価格変化 {price_change:.2%}")
                        self.execute_trade(current_time, current_price, 'sell', '利確')
                        consecutive_losses = 0
                        continue
                
                features = X.iloc[i:i+1]
                raw_prediction = self.model.predict(features)
                prediction = self.normalize_prediction(raw_prediction)
                all_predictions.append(prediction)
                
                trend_strength = df['trend_strength'].iloc[i]
                is_uptrend = trend_strength >= 2
                
                current_volatility = df['volatility'].iloc[i]
                is_low_volatility = current_volatility < 0.01
                
                # 予測の分類（より柔軟に）
                if prediction > UP_THRESHOLD:
                    prediction_counts['up'] += 1
                elif prediction < DOWN_THRESHOLD:
                    prediction_counts['down'] += 1
                else:
                    prediction_counts['neutral'] += 1
                
                if i % 100 == 0:
                    logger.info(f"現在の予測分布 - 上昇: {prediction_counts['up']}, "
                              f"下落: {prediction_counts['down']}, "
                              f"中立: {prediction_counts['neutral']}")
                    logger.info(f"現在の予測値: 正規化後={prediction:.4f}")
                
                # エントリー条件（より柔軟に）
                if self.position is None:
                    entry_condition = (
                        prediction > UP_THRESHOLD and
                        (is_uptrend or prediction > (UP_THRESHOLD + 0.02)) and
                        self.check_technical_confirmation(df, i)
                    )
                    
                    if entry_condition:
                        logger.info(f"買いシグナル: 予測値 {prediction:.4f}, トレンド強度 {trend_strength}")
                        
                        # ポジションサイズの調整
                        if is_low_volatility:
                            self.position_size = position_size * 1.2
                        else:
                            self.position_size = position_size
                            
                        self.execute_trade(current_time, current_price, 'buy', '上昇トレンド+予測シグナル')
                
                # イグジット条件（より柔軟に）
                else:
                    exit_condition = (
                        prediction < DOWN_THRESHOLD or
                        trend_strength <= 1 or
                        (prediction < 0.5 and price_change > 0.01)  # 小さな利益確定
                    )
                    
                    if exit_condition:
                        logger.info(f"売りシグナル検出: 予測値 {prediction:.4f}, トレンド強度 {trend_strength}")
                        trade_result = self.execute_trade(current_time, current_price, 'sell', '予測シグナル+トレンド弱化')
                        if trade_result['profit'] < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
            
            # 予測値の統計情報
            predictions_array = np.array(all_predictions)
            logger.info("\n予測値の統計情報:")
            logger.info(f"平均: {np.mean(predictions_array):.4f}")
            logger.info(f"標準偏差: {np.std(predictions_array):.4f}")
            logger.info(f"最小値: {np.min(predictions_array):.4f}")
            logger.info(f"最大値: {np.max(predictions_array):.4f}")
            
            logger.info("\n予測分布の最終結果:")
            logger.info(f"上昇予測: {prediction_counts['up']}")
            logger.info(f"下落予測: {prediction_counts['down']}")
            logger.info(f"中立予測: {prediction_counts['neutral']}")
            
            stats = self.calculate_statistics()
            logger.info("\nバックテスト結果:")
            for key, value in stats.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")
            
            self.plot_results()
            
        except Exception as e:
            logger.error(f"バックテスト中にエラーが発生しました: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def plot_results(self):
        """結果の可視化"""
        try:
            # ディレクトリの作成
            os.makedirs("reports", exist_ok=True)
            
            # エクイティカーブの描画
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_curve)
            plt.title('エクイティカーブ')
            plt.xlabel('取引数')
            plt.ylabel('残高 (USDT)')
            plt.grid(True)
            plt.savefig(f"reports/equity_curve_{datetime.now().strftime('%Y%m%d')}.png")
            plt.close()
            
            # 利益の分布
            if len(self.trades) > 0:
                trades_df = pd.DataFrame(self.trades)
                plt.figure(figsize=(12, 6))
                sns.histplot(trades_df[trades_df['profit'] != 0]['profit'], bins=50)
                plt.title('利益の分布')
                plt.xlabel('利益 (USDT)')
                plt.ylabel('頻度')
                plt.savefig(f"reports/profit_distribution_{datetime.now().strftime('%Y%m%d')}.png")
                plt.close()
            
        except Exception as e:
            self.logger.error(f"結果の可視化中にエラーが発生しました: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    # バックテスターの初期化
    backtester = BackTester(
        initial_balance=10000.0,  # 10,000 USDT
        base_position_size=0.05,    # 5%
        base_stop_loss=0.015,       # 1.5%
        base_take_profit=0.03      # 3%
    )
    
    # バックテストの実行
    backtester.run_backtest()

if __name__ == "__main__":
    main() 