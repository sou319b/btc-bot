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
from p18_strategy import TradingStrategy
import json

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

    def save_results(self, results_df):
        """バックテスト結果を保存"""
        # 結果をCSVファイルに保存
        results_filename = os.path.join(self.results_dir, "backtest_results.csv")
        results_df.to_csv(results_filename, index=False)
        
        # 取引履歴をJSONファイルに保存
        trades_filename = os.path.join(self.results_dir, "trades.json")
        with open(trades_filename, 'w', encoding='utf-8') as f:
            json.dump(self.trades, f, indent=2, default=str)
        
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
            'average_fee_per_trade': self.total_fees / len(self.trades) if self.trades else 0
        }
        
        # サマリーをJSONファイルに保存
        summary_filename = os.path.join(self.results_dir, "summary.json")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
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
