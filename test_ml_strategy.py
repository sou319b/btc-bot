"""
機械学習取引戦略のテスト
"""

import pandas as pd
import numpy as np
from datetime import datetime
from p19_ml_strategy import MLTradingStrategy

def test_strategy():
    # 戦略のインスタンス化
    strategy = MLTradingStrategy()
    
    # テストデータの読み込み
    df = pd.read_csv('./data/historical_data_20250217.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # テスト結果の保存用
    results = []
    
    # 各時点でのシグナルをテスト
    for i in range(100, len(df)):
        current_time = df.iloc[i]['timestamp']
        current_price = df.iloc[i]['close']
        price_history = df.iloc[i-100:i]['close'].tolist()
        
        # トレンドスコアと変動性の計算
        trend_score, volatility, _ = strategy.calculate_trend(price_history, current_time)
        
        # 買いシグナルのチェック
        should_buy = strategy.should_buy(trend_score, volatility, current_time)
        
        # 売りシグナルのチェック
        should_sell = strategy.should_sell(trend_score, current_time)
        
        # 最適取引量の計算
        optimal_amount = strategy.calculate_optimal_trade_amount(
            current_price, trend_score, volatility, 1000.0, current_time
        )
        
        # 結果の保存
        results.append({
            'timestamp': current_time,
            'price': current_price,
            'trend_score': trend_score,
            'volatility': volatility,
            'buy_signal': should_buy,
            'sell_signal': should_sell,
            'optimal_amount': optimal_amount
        })
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    
    # 結果の表示
    print("\nテスト結果サマリー:")
    print(f"テストした期間: {results_df['timestamp'].min()} から {results_df['timestamp'].max()}")
    print(f"買いシグナル回数: {results_df['buy_signal'].sum()}")
    print(f"売りシグナル回数: {results_df['sell_signal'].sum()}")
    print(f"平均最適取引量: {results_df['optimal_amount'].mean():.2f}")
    
    # 結果の保存
    results_df.to_csv('./results/ml_strategy_test_results.csv', index=False)
    print("\n結果を ./results/ml_strategy_test_results.csv に保存しました")

if __name__ == "__main__":
    test_strategy() 