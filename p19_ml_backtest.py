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
import json

class BackTester:
    def __init__(self, 
                 initial_balance: float = 100.0,    # 初期資金（USDT）
                 position_size: float = 0.95,       # ポジションサイズ（資金の95%を使用）
                 stop_loss: float = 0.02,           # 2%のストップロス
                 take_profit: float = 0.04,         # 4%の利確
                 maker_fee: float = 0.001,          # メイカー手数料 0.1%
                 taker_fee: float = 0.001           # テイカー手数料 0.1%
                ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.position = 0.0  # BTCの保有量
        self.entry_price = None
        self.trades = []
        self.equity_curve = [initial_balance]
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
            self.logger.info("モデルを読み込みます...")
            model_path = 'models/model_20250218.txt'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"モデルファイル {model_path} が見つかりません")
            self.model = lgb.Booster(model_file=model_path)
            
            # テストデータの読み込み
            self.logger.info("テストデータを読み込みます...")
            test_data_path = 'data/test_data_20250218.csv'
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"テストデータ {test_data_path} が見つかりません")
            df = pd.read_csv(test_data_path)
            
            # タイムスタンプをdatetime型に変換
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"データ読み込み中にエラーが発生しました: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量の準備"""
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
        
        return df[feature_columns]
    
    def execute_trade(self, timestamp: pd.Timestamp, price: float, action: str, reason: str) -> Dict:
        """トレードの実行（手数料を含む）"""
        try:
            if action == 'buy' and self.position == 0:
                # 購入可能な量を計算（手数料を考慮）
                amount_usdt = self.balance * self.position_size
                fee = amount_usdt * self.taker_fee
                amount_usdt_after_fee = amount_usdt - fee
                self.position = amount_usdt_after_fee / price
                self.entry_price = price
                self.balance -= (amount_usdt_after_fee + fee)
                
                self.logger.info(f"買い注文実行: 価格 {price:.2f} USDT, "
                               f"数量 {self.position:.6f} BTC, "
                               f"手数料 {fee:.4f} USDT")
                
            elif action == 'sell' and self.position > 0:
                # 売却額と手数料の計算
                amount_usdt = self.position * price
                fee = amount_usdt * self.taker_fee
                amount_usdt_after_fee = amount_usdt - fee
                
                # 損益計算
                entry_value = self.position * self.entry_price
                exit_value = amount_usdt_after_fee
                profit = exit_value - entry_value
                
                self.balance += amount_usdt_after_fee
                self.position = 0
                
                trade_info = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': float(self.entry_price),
                    'exit_price': float(price),
                    'position_size': float(self.position),
                    'profit': float(profit),
                    'fees': float(fee),
                    'balance': float(self.balance),
                    'reason': reason
                }
                self.trades.append(trade_info)
                self.equity_curve.append(float(self.balance))
                
                self.logger.info(f"売り注文実行: 価格 {price:.2f} USDT, "
                               f"利益 {profit:.4f} USDT, "
                               f"手数料 {fee:.4f} USDT")
                
                self.entry_price = None
                return {'profit': float(profit), 'trade_type': 'sell'}
            
            return {'profit': 0, 'trade_type': 'none'}
            
        except Exception as e:
            self.logger.error(f"トレード実行中にエラーが発生しました: {e}")
            return {'profit': 0, 'trade_type': 'error'}
    
    def calculate_position_size(self, volatility: float) -> float:
        """ボラティリティに基づくポジションサイズの計算"""
        base_size = self.position_size
        
        # ボラティリティが高い場合、ポジションサイズを減少
        if volatility > 0.02:  # 2%以上のボラティリティ
            return base_size * 0.7
        # ボラティリティが低い場合、ポジションサイズを増加
        elif volatility < 0.01:  # 1%未満のボラティリティ
            return min(base_size * 1.2, 0.95)  # 最大95%まで
        
        return base_size
    
    def calculate_statistics(self) -> Dict:
        """バックテスト結果の統計計算"""
        if not self.trades:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_fees': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_balance': float(self.initial_balance)
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # 基本的な統計
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['profit'] > 0])
        win_rate = float(profitable_trades / total_trades)
        total_profit = float(trades_df['profit'].sum())
        total_fees = float(trades_df['fees'].sum())
        
        # ドローダウンの計算
        equity_curve = pd.Series(self.equity_curve)
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = float(drawdowns.min())
        
        # シャープレシオの計算
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if len(returns) > 0:
            sharpe_ratio = float(np.sqrt(365) * (returns.mean() / returns.std()))
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_fees': total_fees,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': float(self.balance)
        }
    
    def plot_results(self):
        """バックテスト結果の可視化"""
        os.makedirs("reports", exist_ok=True)
        
        # エクイティカーブ
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('エクイティカーブ')
        plt.xlabel('取引数')
        plt.ylabel('残高 (USDT)')
        plt.grid(True)
        plt.savefig(f"reports/equity_curve_{datetime.now().strftime('%Y%m%d')}.png")
        plt.close()
        
        if self.trades:
            # 損益分布
            trades_df = pd.DataFrame(self.trades)
            plt.figure(figsize=(12, 6))
            sns.histplot(trades_df['profit'], bins=50)
            plt.title('損益分布')
            plt.xlabel('損益 (USDT)')
            plt.ylabel('頻度')
            plt.savefig(f"reports/profit_distribution_{datetime.now().strftime('%Y%m%d')}.png")
            plt.close()
            
            # 月間パフォーマンス
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df['month'] = trades_df['timestamp'].dt.to_period('M')
            monthly_returns = trades_df.groupby('month')['profit'].sum()
            
            plt.figure(figsize=(12, 6))
            monthly_returns.plot(kind='bar')
            plt.title('月間パフォーマンス')
            plt.xlabel('月')
            plt.ylabel('損益 (USDT)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"reports/monthly_performance_{datetime.now().strftime('%Y%m%d')}.png")
            plt.close()
    
    def run_backtest(self):
        """バックテストの実行"""
        logger = self.setup_logging()
        
        try:
            # データとモデルの読み込み
            df = self.load_model_and_data()
            
            # 特徴量の準備
            features = self.prepare_features(df)
            
            logger.info(f"バックテストを開始します... 初期資金: {self.initial_balance:.2f} USDT")
            
            consecutive_losses = 0
            max_consecutive_losses = 3
            
            for i in range(len(df)):
                if i < 100:  # ウォームアップ期間
                    continue
                
                current_time = df['timestamp'].iloc[i]
                current_price = float(df['close'].iloc[i])
                
                # ボラティリティの計算
                returns = df['close'].pct_change()
                current_volatility = float(returns.rolling(20).std().iloc[i])
                
                # ポジションサイズの調整
                self.position_size = self.calculate_position_size(current_volatility)
                
                # 既存ポジションの確認
                if self.position > 0:
                    price_change = (current_price - self.entry_price) / self.entry_price
                    
                    # ストップロス
                    if price_change <= -self.stop_loss:
                        trade_result = self.execute_trade(
                            current_time, current_price, 'sell', 'ストップロス'
                        )
                        if trade_result['profit'] < 0:
                            consecutive_losses += 1
                        continue
                    
                    # 利確
                    if price_change >= self.take_profit:
                        self.execute_trade(current_time, current_price, 'sell', '利確')
                        consecutive_losses = 0
                        continue
                
                # 連続損失の制限
                if consecutive_losses >= max_consecutive_losses:
                    if self.position > 0:
                        self.execute_trade(current_time, current_price, 'sell', '連続損失制限')
                        consecutive_losses = 0
                    continue
                
                # モデルによる予測
                current_features = features.iloc[i:i+1]
                prediction_probs = self.model.predict(current_features)
                prediction = np.argmax(prediction_probs)  # クラスの予測（0, 1, 2）
                
                # エントリー条件（上昇予測の場合）
                if self.position == 0 and prediction == 2:  # 2は上昇予測
                    self.execute_trade(current_time, current_price, 'buy', '上昇予測')
                
                # イグジット条件（下落予測の場合）
                elif self.position > 0 and prediction == 0:  # 0は下落予測
                    trade_result = self.execute_trade(
                        current_time, current_price, 'sell', '下落予測'
                    )
                    if trade_result['profit'] < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
            
            # 最終ポジションのクローズ
            if self.position > 0:
                self.execute_trade(
                    df['timestamp'].iloc[-1],
                    float(df['close'].iloc[-1]),
                    'sell',
                    'バックテスト終了'
                )
            
            # 結果の集計と表示
            stats = self.calculate_statistics()
            logger.info("\nバックテスト結果:")
            for key, value in stats.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")
            
            # 結果の可視化
            self.plot_results()
            
            # 結果の保存
            results = {
                'statistics': stats,
                'parameters': {
                    'initial_balance': float(self.initial_balance),
                    'position_size': float(self.position_size),
                    'stop_loss': float(self.stop_loss),
                    'take_profit': float(self.take_profit),
                    'maker_fee': float(self.maker_fee),
                    'taker_fee': float(self.taker_fee)
                },
                'trades': self.trades
            }
            
            with open(f"reports/backtest_results_{datetime.now().strftime('%Y%m%d')}.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"バックテスト中にエラーが発生しました: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def main():
    # バックテスターの初期化
    backtester = BackTester(
        initial_balance=100.0,     # 100 USDT
        position_size=0.95,        # 95%
        stop_loss=0.02,           # 2%
        take_profit=0.04,         # 4%
        maker_fee=0.001,          # 0.1%
        taker_fee=0.001           # 0.1%
    )
    
    # バックテストの実行
    backtester.run_backtest()

if __name__ == "__main__":
    main()
