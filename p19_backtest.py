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

class BackTester:
    def __init__(self):
        self.strategy = TradingStrategy()
        self.setup_logging()
        
        # バックテストの設定
        self.initial_balance = 1000  # 初期USDT残高
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
        price_history = []
        results = []
        
        self.logger.info(f"バックテスト期間: {price_data['timestamp'].min()} から {price_data['timestamp'].max()}")
        self.logger.info(f"データ数: {len(price_data)}件")
        self.logger.info("シミュレーションを開始します...")
        
        for index, row in price_data.iterrows():
            current_price = float(row['close'])
            current_time = row['timestamp']
            
            # 価格履歴を更新
            price_history.append(current_price)
            if len(price_history) > self.strategy.history_size:
                price_history.pop(0)
            
            if len(price_history) < self.strategy.history_size:
                continue
            
            # トレンドとボラティリティを計算
            trend_score, volatility, price_change = self.strategy.calculate_trend(price_history)
            
            # 現在の総資産を計算
            total_assets = self.current_balance + (self.btc_balance * current_price)
            
            # 1時間ごとに進捗を表示
            if index % 60 == 0:
                progress = (index + 1) / len(price_data) * 100
                self.logger.info(f"進捗: {progress:.1f}% 完了 - 現在の資産: {total_assets:.2f} USDT")
            
            # ポジションクローズの判断
            if self.btc_balance > 0 and self.entry_price and self.strategy.should_close_position(current_price, self.entry_price):
                # 売り注文をシミュレート（手数料を含む）
                sell_value = self.btc_balance * current_price
                fee = sell_value * self.taker_fee
                net_value = sell_value - fee
                self.current_balance += net_value
                self.total_fees += fee
                
                profit = net_value - (self.btc_balance * self.entry_price)
                profit_pct = (profit / (self.btc_balance * self.entry_price)) * 100
                
                self.trades.append({
                    'timestamp': current_time,
                    'type': 'SELL',
                    'price': current_price,
                    'amount': self.btc_balance,
                    'fee': fee,
                    'net_value': net_value,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'total_assets': total_assets
                })
                
                self.btc_balance = 0
                self.entry_price = None
                self.sell_count += 1
                
                self.logger.info(f"売り注文実行: 価格={current_price:.2f} USDT, 手数料={fee:.4f} USDT, 純利益={profit:.2f} USDT ({profit_pct:.2f}%)")
            
            # 買いシグナルの判断
            elif self.strategy.should_buy(trend_score, volatility) and self.btc_balance == 0:
                # 最適な取引量を計算
                trade_amount = self.strategy.calculate_optimal_trade_amount(
                    current_price, trend_score, volatility, self.current_balance
                )
                
                if trade_amount >= self.strategy.min_trade_amount:
                    # BTCの数量を計算（手数料を考慮）
                    fee = trade_amount * self.taker_fee
                    available_amount = trade_amount - fee
                    btc_qty = self.strategy.calculate_position_size(current_price, available_amount)
                    total_cost = btc_qty * current_price + fee
                    
                    if total_cost <= self.current_balance:
                        self.btc_balance = btc_qty
                        self.current_balance -= total_cost
                        self.entry_price = current_price
                        self.buy_count += 1
                        self.total_fees += fee
                        
                        self.trades.append({
                            'timestamp': current_time,
                            'type': 'BUY',
                            'price': current_price,
                            'amount': btc_qty,
                            'fee': fee,
                            'cost': total_cost,
                            'total_assets': total_assets
                        })
                        
                        self.logger.info(f"買い注文実行: 価格={current_price:.2f} USDT, 数量={btc_qty:.6f} BTC, 手数料={fee:.4f} USDT")
            
            # 売りシグナルの判断
            elif self.strategy.should_sell(trend_score) and self.btc_balance > 0:
                # 売り注文をシミュレート（手数料を含む）
                sell_value = self.btc_balance * current_price
                fee = sell_value * self.taker_fee
                net_value = sell_value - fee
                self.current_balance += net_value
                self.total_fees += fee
                
                profit = net_value - (self.btc_balance * self.entry_price)
                profit_pct = (profit / (self.btc_balance * self.entry_price)) * 100
                
                self.trades.append({
                    'timestamp': current_time,
                    'type': 'SELL',
                    'price': current_price,
                    'amount': self.btc_balance,
                    'fee': fee,
                    'net_value': net_value,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'total_assets': total_assets
                })
                
                self.btc_balance = 0
                self.entry_price = None
                self.sell_count += 1
                
                self.logger.info(f"売り注文実行: 価格={current_price:.2f} USDT, 手数料={fee:.4f} USDT, 純利益={profit:.2f} USDT ({profit_pct:.2f}%)")
            
            # 結果を記録
            results.append({
                'timestamp': current_time,
                'price': current_price,
                'usdt_balance': self.current_balance,
                'btc_balance': self.btc_balance,
                'total_assets': total_assets,
                'total_fees': self.total_fees,
                'profit_pct': ((total_assets - self.initial_balance) / self.initial_balance) * 100,
                'trend_score': trend_score,
                'volatility': volatility
            })
        
        self.logger.info("シミュレーション完了")
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

    def plot_results(self, results_df: pd.DataFrame):
        """バックテスト結果のグラフを生成"""
        plt.style.use('default')  # seabornの代わりにデフォルトスタイルを使用
        
        # フォントの設定
        plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォントの設定
        
        # 資産推移のプロット
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['timestamp'], results_df['total_assets'], label='総資産')
        plt.title('資産推移')
        plt.xlabel('日時')
        plt.ylabel('USDT')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'equity_curve.png'))
        plt.close()

        # 価格とトレンドスコアの関係
        plt.figure(figsize=(12, 6))
        plt.scatter(results_df['price'], results_df['trend_score'], alpha=0.5)
        plt.title('価格とトレンドスコアの関係')
        plt.xlabel('価格')
        plt.ylabel('トレンドスコア')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'price_trend_correlation.png'))
        plt.close()

        # ボラティリティの分布
        plt.figure(figsize=(12, 6))
        plt.hist(results_df['volatility'], bins=50)
        plt.title('ボラティリティの分布')
        plt.xlabel('ボラティリティ')
        plt.ylabel('頻度')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'volatility_distribution.png'))
        plt.close()

    def save_results(self, results_df):
        """バックテスト結果を保存"""
        # 結果をCSVファイルに保存
        results_filename = os.path.join(self.results_dir, "backtest_results.csv")
        results_df.to_csv(results_filename, index=False)
        
        # 取引履歴をJSONファイルに保存
        trades_filename = os.path.join(self.results_dir, "trades.json")
        with open(trades_filename, 'w', encoding='utf-8') as f:
            json.dump(self.trades, f, indent=2, default=str)
        
        # 詳細な分析を実行
        analysis_results = self.analyze_trades(self.trades, results_df)
        
        # グラフを生成
        self.plot_results(results_df)
        
        # 最終結果を計算
        final_total = results_df['total_assets'].iloc[-1]
        total_profit = final_total - self.initial_balance
        total_profit_pct = (total_profit / self.initial_balance) * 100
        net_profit = total_profit - self.total_fees
        net_profit_pct = (net_profit / self.initial_balance) * 100
        
        summary = {
            'initial_balance': self.initial_balance,
            'final_balance': final_total,
            'total_profit': total_profit,
            'total_profit_pct': total_profit_pct,
            'total_fees': self.total_fees,
            'net_profit': net_profit,
            'net_profit_pct': net_profit_pct,
            'total_trades': len(self.trades),
            'buy_count': self.buy_count,
            'sell_count': self.sell_count,
            'average_fee_per_trade': self.total_fees / len(self.trades) if self.trades else 0,
            'detailed_analysis': analysis_results
        }
        
        # サマリーをJSONファイルに保存
        summary_filename = os.path.join(self.results_dir, "summary.json")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n━━━━━━━━━━ バックテスト結果 ━━━━━━━━━━")
        self.logger.info(f"初期残高　　：{self.initial_balance:,.2f} USDT")
        self.logger.info(f"最終残高　　：{final_total:,.2f} USDT")
        self.logger.info(f"総利益　　　：{total_profit:,.2f} USDT ({total_profit_pct:.2f}%)")
        self.logger.info(f"総手数料　　：{self.total_fees:,.2f} USDT")
        self.logger.info(f"純利益　　　：{net_profit:,.2f} USDT ({net_profit_pct:.2f}%)")
        self.logger.info(f"総取引回数　：{len(self.trades)}回")
        self.logger.info(f"買い取引　　：{self.buy_count}回")
        self.logger.info(f"売り取引　　：{self.sell_count}回")
        self.logger.info(f"平均手数料　：{self.total_fees/len(self.trades) if self.trades else 0:.4f} USDT/取引")
        self.logger.info(f"勝率　　　　：{analysis_results['取引の詳細分析']['勝率']:.2f}%")
        self.logger.info(f"損益比率　　：{analysis_results['リスク管理指標']['損益比率']:.2f}")
        self.logger.info(f"最大ドローダウン：{abs(analysis_results['リスク管理指標']['最大ドローダウン']):.2f}%")
        self.logger.info(f"シャープレシオ：{analysis_results['リスク管理指標']['シャープレシオ']:.2f}")
        self.logger.info(f"結果保存先　：{self.results_dir}")
        self.logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━")
        
        return summary

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
