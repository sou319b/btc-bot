#利益がでるbot
#成功例
#手数料なし

import ccxt
import time
import random

# 初期設定
INITIAL_JPY = 1000000  # 初期資金100万円
BTC_HOLDINGS = 0
TICK_INTERVAL = 3  # 3秒ごとに取引
API = ccxt.bitflyer()

def get_btc_price(retries=3, delay=5):
    """現在のBTC価格を取得"""
    for i in range(retries):
        try:
            ticker = API.fetch_ticker('BTC/JPY')
            return ticker['last']
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"価格取得失敗 ({i+1}/{retries}): {str(e)}")
            time.sleep(delay)

def print_status(price, jpy, btc, total, action):
    """現在の状態を表示し、ファイルに保存"""
    log_entry = f"""
[取引時刻] {time.strftime('%Y-%m-%d %H:%M:%S')}
現在のBTC価格：{price} JPY
JPY残高: {jpy} JPY
BTC保有量：{btc} BTC
総資産額：{total} JPY
買/売：{action}
-------------------------
"""
    print(log_entry.strip())
    
    # ファイルに追記
    try:
        with open('trading_log.txt', 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"ログファイル書き込みエラー: {e}")

def simulate_trading():
    # ログファイル初期化
    try:
        with open('trading_log.txt', 'w', encoding='utf-8') as f:
            f.write(f"シミュレーション開始: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"初期資金: {INITIAL_JPY} JPY\n\n")
    except Exception as e:
        print(f"ログファイル初期化エラー: {e}")
        return

    # API接続テスト
    try:
        print("API接続テスト中...")
        test_price = get_btc_price()
        print(f"初期価格取得成功: {test_price} JPY")
    except Exception as e:
        print(f"API接続エラー: {str(e)}")
        print("bitFlyer APIに接続できませんでした。ネットワーク接続を確認してください。")
        return

    jpy_balance = INITIAL_JPY
    btc_holdings = BTC_HOLDINGS
    best_price = test_price  # 初期価格
    
    while True:
        try:
            base_price = get_btc_price()
            # ランダムな価格変動を追加 (±0.2%)
            current_price = base_price * (1 + (random.random() * 0.004 - 0.002))
            total_assets = jpy_balance + (btc_holdings * current_price)
            
            # 取引ロジック
            price_diff = ((current_price - best_price) / best_price) * 100
            print(f"価格変動: {price_diff:.2f}%")
            
            # 買い条件：現在価格がベスト価格から0.1%以上下落
            if current_price < best_price * 0.999:  # 0.1%下落で買い
                if jpy_balance > 0:
                    # 最大5万円分購入（残高が5万円未満の場合は全額使用）
                    buy_amount = min(jpy_balance, 50000)  # 最大5万円分購入
                    btc_bought = buy_amount / current_price
                    jpy_balance -= buy_amount
                    btc_holdings += btc_bought
                    best_price = current_price
                    print_status(current_price, jpy_balance, btc_holdings, total_assets, f"買 {btc_bought:.8f} BTC")
            
            # 売り条件：現在価格がベスト価格から0.1%以上上昇
            elif current_price > best_price * 1.001:  # 0.1%上昇で売り
                if btc_holdings > 0:
                    # 保有BTCの50%を売却
                    sell_amount = btc_holdings * 0.5  # 50%売却
                    jpy_balance += sell_amount * current_price
                    btc_holdings -= sell_amount
                    best_price = current_price
                    print_status(current_price, jpy_balance, btc_holdings, total_assets, f"売 {sell_amount:.8f} BTC")
            
            time.sleep(TICK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nシミュレーションを終了します")
            final_total = jpy_balance + (btc_holdings * current_price)
            profit = final_total - INITIAL_JPY
            print(f"最終資産額: {final_total} JPY")
            print(f"利益: {profit} JPY")
            
            # 最終結果をログに記録
            try:
                with open('trading_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"\nシミュレーション終了: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"最終資産額: {final_total} JPY\n")
                    f.write(f"利益: {profit} JPY\n")
            except Exception as e:
                print(f"最終結果のログ記録エラー: {e}")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            time.sleep(60)  # エラー時は60秒待機

if __name__ == "__main__":
    simulate_trading()
