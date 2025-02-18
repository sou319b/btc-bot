"""
基本機能：
過去の価格データを使用して取引戦略のバックテストを実行
取引シミュレーションの結果を分析
パフォーマンスの評価と記録（取引手数料を含む）
"""
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from p19_strategy import TradingStrategy
import json
from typing import List, Dict
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns  # seabornを直接インポート

class BackTester:
    def __init__(self):
        self.strategy = TradingStrategy()
        self.setup_logging()
        
        # バックテストの設定
        self.initial_balance = 100  # 初期USDT残高
        self.current_balance = self.initial_balance
        self.btc_balance = 0
        self.entry_price = None
        self.buy_count = 0
        self.sell_count = 0
        self.trades = []  # 取引履歴
        
        # 手数料の設定
        self.maker_fee = 0.001  # 0.1% メイカー手数料
        self.taker_fee = 0.001  # 0.1% テイカー手数料
        self.total_fees = 0     # 合計手数料
        
        # 結果保存用のディレクトリを作成
        self.results_dir = f"results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
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

    def load_historical_data(self, filename):
        """保存された価格データを読み込む"""
        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            self.logger.error(f"データ読み込みエラー: {e}")
            return None

    def simulate_trade(self, price_data):
        """取引シミュレーションを実行"""
        results = []
        self.btc_balance = 0
        self.usdt_balance = self.initial_balance
        self.entry_price = 0
        
        # 価格履歴を初期化
        price_history = []
        
        for i in range(len(price_data)):
            current_price = price_data.iloc[i]['close']
            current_time = price_data.index[i]
            
            # 価格履歴を更新
            price_history.append(current_price)
            if len(price_history) > self.strategy.history_size:
                price_history.pop(0)
            
            if len(price_history) < self.strategy.history_size:
                continue
            
            # トレンドスコアとボラティリティを計算
            trend_score, volatility = self.strategy.calculate_trend(price_history)
            
            # RSIを計算
            self.strategy.current_rsi = self.strategy.calculate_rsi(price_history, self.strategy.rsi_period)
            
            # 平均トレンドと直近トレンドを計算
            avg_trend = np.mean(self.strategy.trend_memory) if len(self.strategy.trend_memory) > 0 else 0
            recent_trend = np.mean(self.strategy.trend_memory[-2:]) if len(self.strategy.trend_memory) >= 2 else 0
            
            # ポジションがない場合は買いシグナルをチェック
            if self.btc_balance == 0:
                if self.strategy.should_buy(trend_score, avg_trend, recent_trend, volatility, current_time):
                    trade_amount = self.strategy.calculate_position_size(self.usdt_balance, current_price)
                    if trade_amount > 0:
                        self.execute_buy(current_price, trade_amount, current_time)
            
            # ポジションがある場合は売りシグナルをチェック
            elif self.btc_balance > 0:
                if self.strategy.should_sell(trend_score, avg_trend, recent_trend, volatility, current_time):
                    self.execute_sell(current_price, self.btc_balance, current_time)
                
                # ストップロスと利確をチェック
                elif self.check_stop_loss_take_profit(current_price, current_time):
                    continue
            
            # 結果を記録
            results.append({
                'timestamp': current_time,
                'price': current_price,
                'btc_balance': self.btc_balance,
                'usdt_balance': self.usdt_balance,
                'trend_score': trend_score,
                'volatility': volatility,
                'rsi': self.strategy.current_rsi
            })
            
            # 進捗表示（1000件ごと）
            if i % 1000 == 0:
                progress = (i / len(price_data)) * 100
                self.logger.info(f"進捗: {progress:.1f}% 完了 - 現在の資産: {self.usdt_balance:.2f} USDT")
        
        return pd.DataFrame(results)

    def analyze_trades(self, trades: List[Dict], results_df: pd.DataFrame) -> Dict:
        """取引の詳細分析を実行"""
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['hour'] = trades_df['timestamp'].dt.hour

        # 利益/損失の分布分析
        profitable_trades = trades_df[trades_df['type'] == 'SELL']
        profits = profitable_trades['profit'].values
        profit_trades = len(profits[profits > 0])
        loss_trades = len(profits[profits <= 0])

        # 最大の利益/損失取引
        max_profit_trade = profitable_trades.loc[profitable_trades['profit'].idxmax()] if not profitable_trades.empty else None
        max_loss_trade = profitable_trades.loc[profitable_trades['profit'].idxmin()] if not profitable_trades.empty else None

        # 連続損失の分析
        consecutive_losses = 0
        max_consecutive_losses = 0
        current_streak = 0

        for profit in profits:
            if profit < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        # 保有時間の分析
        holding_times = []
        entry_time = None
        for _, trade in trades_df.iterrows():
            if trade['type'] == 'BUY':
                entry_time = trade['timestamp']
            elif trade['type'] == 'SELL' and entry_time is not None:
                holding_time = (trade['timestamp'] - entry_time).total_seconds() / 3600  # 時間単位
                holding_times.append(holding_time)
                entry_time = None

        # 時間帯分析
        hourly_trades = trades_df.groupby('hour').size()
        hourly_profits = profitable_trades.groupby('hour')['profit'].mean()

        # 取引サイズの分析
        trade_sizes = trades_df[trades_df['type'] == 'BUY']['amount']
        profitable_sizes = profitable_trades[profitable_trades['profit'] > 0]['amount']
        loss_sizes = profitable_trades[profitable_trades['profit'] <= 0]['amount']

        # リスク管理指標の計算
        equity_curve = results_df['total_assets']
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = drawdown.min()

        # リターンの計算
        returns = equity_curve.pct_change().dropna()
        risk_free_rate = 0.02  # 2%と仮定
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0

        # 損益比率
        avg_profit = profits[profits > 0].mean() if len(profits[profits > 0]) > 0 else 0
        avg_loss = abs(profits[profits < 0].mean()) if len(profits[profits < 0]) > 0 else 1
        profit_loss_ratio = avg_profit / avg_loss if avg_loss != 0 else 0

        analysis_results = {
            "取引の詳細分析": {
                "総取引数": len(trades),
                "勝率": (profit_trades / (profit_trades + loss_trades) * 100) if (profit_trades + loss_trades) > 0 else 0,
                "最大利益": {
                    "金額": float(max_profit_trade['profit']) if max_profit_trade is not None else 0,
                    "日時": str(max_profit_trade['timestamp']) if max_profit_trade is not None else None
                },
                "最大損失": {
                    "金額": float(max_loss_trade['profit']) if max_loss_trade is not None else 0,
                    "日時": str(max_loss_trade['timestamp']) if max_loss_trade is not None else None
                },
                "連続損失の最大回数": max_consecutive_losses,
                "平均保有時間": np.mean(holding_times) if holding_times else 0,
                "保有時間の中央値": np.median(holding_times) if holding_times else 0
            },
            "時間帯分析": {
                "取引が多い時間帯": hourly_trades.nlargest(3).to_dict(),
                "収益が高い時間帯": hourly_profits.nlargest(3).to_dict(),
                "損失が大きい時間帯": hourly_profits.nsmallest(3).to_dict()
            },
            "取引サイズ分析": {
                "平均取引サイズ": float(trade_sizes.mean()) if not trade_sizes.empty else 0,
                "収益取引の平均サイズ": float(profitable_sizes.mean()) if not profitable_sizes.empty else 0,
                "損失取引の平均サイズ": float(loss_sizes.mean()) if not loss_sizes.empty else 0
            },
            "リスク管理指標": {
                "最大ドローダウン": float(max_drawdown),
                "シャープレシオ": float(sharpe_ratio),
                "損益比率": float(profit_loss_ratio)
            }
        }

        return analysis_results

    def plot_results(self, results_df):
        """バックテスト結果をプロット"""
        plt.figure(figsize=(15, 10))
        
        # 資産推移のプロット
        plt.subplot(3, 1, 1)
        sns.lineplot(data=results_df, x='timestamp', y='usdt_balance', label='USDT残高')
        plt.title('資産推移')
        plt.xlabel('時間')
        plt.ylabel('USDT')
        
        # 価格推移のプロット
        plt.subplot(3, 1, 2)
        sns.lineplot(data=results_df, x='timestamp', y='price', label='価格')
        plt.title('価格推移')
        plt.xlabel('時間')
        plt.ylabel('USDT')
        
        # トレンドスコアとボラティリティのプロット
        plt.subplot(3, 1, 3)
        sns.lineplot(data=results_df, x='timestamp', y='trend_score', label='トレンドスコア')
        sns.lineplot(data=results_df, x='timestamp', y='volatility', label='ボラティリティ')
        plt.title('指標推移')
        plt.xlabel('時間')
        plt.ylabel('値')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'backtest_results.png'))
        plt.close()

    def save_results(self, results):
        """バックテスト結果を保存"""
        results_df = pd.DataFrame(results)
        
        # 結果をCSVファイルに保存
        results_df.to_csv(os.path.join(self.results_dir, 'backtest_results.csv'), index=False)
        
        # グラフを生成
        self.plot_results(results_df)
        
        # 取引サマリーを作成
        initial_balance = 100.0  # 初期残高
        final_balance = results_df['usdt_balance'].iloc[-1]
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        # 取引統計を計算
        trades = pd.DataFrame(self.trades)
        if len(trades) > 0:
            profitable_trades = len(trades[trades['profit'] > 0])
            total_trades = len(trades)
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # 最大ドローダウンを計算
            peak = results_df['usdt_balance'].expanding(min_periods=1).max()
            drawdown = (results_df['usdt_balance'] - peak) / peak
            max_drawdown = drawdown.min() * 100
            
            # シャープレシオを計算
            daily_returns = results_df['usdt_balance'].pct_change()
            sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
            
            summary = {
                '初期残高': f"{initial_balance:.2f} USDT",
                '最終残高': f"{final_balance:.2f} USDT",
                '総利益': f"{final_balance - initial_balance:.2f} USDT ({total_return:.2f}%)",
                '総取引回数': total_trades,
                '勝率': f"{win_rate:.2f}%",
                '最大ドローダウン': f"{abs(max_drawdown):.2f}%",
                'シャープレシオ': f"{sharpe_ratio:.2f}"
            }
        else:
            summary = {
                '初期残高': f"{initial_balance:.2f} USDT",
                '最終残高': f"{final_balance:.2f} USDT",
                '総利益': f"{final_balance - initial_balance:.2f} USDT ({total_return:.2f}%)",
                '総取引回数': 0,
                '勝率': "0.00%",
                '最大ドローダウン': "0.00%",
                'シャープレシオ': "0.00"
            }
        
        # サマリーをテキストファイルに保存
        with open(os.path.join(self.results_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        return summary

    def calculate_trend_score(self, data):
        """トレンドスコアを計算"""
        if len(data) < 2:
            return 0
        
        # 直近の価格変化を計算
        price_changes = data['close'].pct_change().dropna()
        if len(price_changes) == 0:
            return 0
        
        # 重み付き平均を計算（直近のデータにより高い重みを付ける）
        weights = np.linspace(0.5, 1.0, len(price_changes))
        weighted_changes = price_changes * weights
        trend_score = np.mean(weighted_changes)
        
        return trend_score

    def calculate_volatility(self, data):
        """ボラティリティを計算"""
        if len(data) < 2:
            return 0
        
        # 価格変化率の標準偏差を計算
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0
        
        volatility = returns.std()
        return volatility

    def execute_buy(self, price, amount, timestamp):
        """買い注文を実行"""
        fee = amount * self.taker_fee
        btc_amount = (amount - fee) / price
        
        self.usdt_balance -= amount
        self.btc_balance = btc_amount
        self.entry_price = price
        
        self.logger.info(f"買い注文実行: 価格={price:.2f} USDT, 数量={btc_amount:.6f} BTC, 手数料={fee:.4f} USDT")

    def execute_sell(self, price, amount, timestamp):
        """売り注文を実行"""
        gross_value = amount * price
        fee = gross_value * self.taker_fee
        net_value = gross_value - fee
        
        profit = net_value - (amount * self.entry_price)
        profit_pct = (profit / (amount * self.entry_price)) * 100
        
        self.usdt_balance += net_value
        self.btc_balance = 0
        self.entry_price = 0
        
        self.logger.info(f"売り注文実行: 価格={price:.2f} USDT, 手数料={fee:.4f} USDT, 純利益={profit:.2f} USDT ({profit_pct:.2f}%)")

    def check_stop_loss_take_profit(self, current_price, timestamp):
        """ストップロスと利確をチェック"""
        if self.entry_price == 0:
            return False
            
        price_change = (current_price - self.entry_price) / self.entry_price
        
        # ストップロス
        if price_change <= -self.strategy.stop_loss_pct:
            self.execute_sell(current_price, self.btc_balance, timestamp)
            return True
            
        # 利確
        if price_change >= self.strategy.take_profit_pct:
            self.execute_sell(current_price, self.btc_balance, timestamp)
            return True
            
        return False

def main():
    backtest = BackTester()
    
    # 最新の価格データファイルを探す
    data_dir = "data"
    if not os.path.exists(data_dir):
        backtest.logger.error("データディレクトリが見つかりません。p18_fetch_data.pyを実行してデータを取得してください。")
        return
        
    data_files = [f for f in os.listdir(data_dir) if f.startswith('historical_data_') and f.endswith('.csv')]
    if not data_files:
        backtest.logger.error("価格データが見つかりません。p18_fetch_data.pyを実行してデータを取得してください。")
        return
        
    latest_file = os.path.join(data_dir, max(data_files))
    price_data = backtest.load_historical_data(latest_file)
    
    if price_data is None:
        backtest.logger.error("価格データの読み込みに失敗しました")
        return
    
    # バックテストを実行
    results = backtest.simulate_trade(price_data)
    
    # 結果を保存
    summary = backtest.save_results(results)

if __name__ == "__main__":
    main()
