import ccxt
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# 取引パラメータ
symbol = 'BTC/USDT'
timeframe = '1h'  # 1時間足
amount = 0.001    # 取引量（BTC）
position = None   # 現在のポジション

# Bybitテストネットの設定
exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_TEST_API_KEY'),
    'secret': os.getenv('BYBIT_TEST_SECRET'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'linear',  # 先物取引を使用
        'recvWindow': 60000,    # タイムスタンプの許容範囲を1分に設定
        'adjustForTimeDifference': True,  # 時差調整を有効化
        'createMarketBuyOrderRequiresPrice': False,
        'test': True,  # テストモードを有効化
    },
    'timeout': 30000,  # タイムアウト時間を30秒に設定
})

# テストネットのURLを設定
exchange.urls['api'] = {
    'public': 'https://api-testnet.bybit.com',
    'private': 'https://api-testnet.bybit.com',
}

def initialize_exchange():
    """取引所の初期化を行う"""
    try:
        # マーケットデータの読み込み
        exchange.load_markets()
        print("マーケットデータ読み込み完了")
        
        # アカウント情報の取得
        balance = exchange.fetch_balance()
        print(f"USDTの残高: {balance['USDT']['free']}")
        
        return True
    except Exception as e:
        print(f"初期化エラー: {str(e)}")
        return False

def get_moving_averages(symbol, timeframe):
    """移動平均を計算する"""
    try:
        # OHLCVデータを取得
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=21)
        
        if not ohlcv or len(ohlcv) < 21:
            print("十分なデータが取得できませんでした")
            return None, None
            
        # 終値を抽出
        closes = [x[4] for x in ohlcv]
        
        # 7期間と21期間の移動平均を計算
        ma7 = sum(closes[-7:]) / 7
        ma21 = sum(closes) / 21
        
        return ma7, ma21
    except Exception as e:
        print(f"エラー（移動平均の計算）: {e}")
        return None, None

def execute_trade():
    """取引を実行する"""
    global position
    
    try:
        # 移動平均を取得
        ma7, ma21 = get_moving_averages(symbol, timeframe)
        if ma7 is None or ma21 is None:
            return
        
        # 現在の価格を取得
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        print(f"現在の価格: {current_price}")
        print(f"7期間MA: {ma7:.2f}")
        print(f"21期間MA: {ma21:.2f}")
        
        # 取引ロジック
        if ma7 > ma21 and position != 'long':  # ゴールデンクロス
            # 買い注文
            order = exchange.create_market_buy_order(symbol, amount)
            position = 'long'
            print(f"買い注文実行: {order}")
            
        elif ma7 < ma21 and position == 'long':  # デッドクロス
            # 売り注文
            order = exchange.create_market_sell_order(symbol, amount)
            position = None
            print(f"売り注文実行: {order}")
            
    except Exception as e:
        print(f"エラー（取引実行）: {e}")

def main():
    """メインループ"""
    print("テストネット環境で取引を開始します")
    
    # 取引所の初期化
    if not initialize_exchange():
        print("初期化に失敗しました。プログラムを終了します。")
        return
        
    while True:
        try:
            print(f"\n{datetime.now()} - 取引チェック開始")
            execute_trade()
            
            # 1時間待機
            time.sleep(3600)
            
        except KeyboardInterrupt:
            print("\nプログラムを終了します")
            break
        except Exception as e:
            print(f"エラー（メインループ）: {e}")
            time.sleep(60)  # エラー時は1分待機

if __name__ == "__main__":
    main()
