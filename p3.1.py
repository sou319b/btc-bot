# 利益がでるbot
# 成功例
# 手数料付き

import ccxt
import time
import random

# 初期設定
INITIAL_JPY = 1000000  # 初期資金100万円
BTC_HOLDINGS = 0
TICK_INTERVAL = 3  # 3秒ごとに取引
FEE_RATE = 0.001  # 0.2% の手数料
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


def print_status(price, jpy_no_fee, btc_no_fee, total_no_fee, jpy_fee, btc_fee, total_fee, action):
    """現在の状態を表示し、ファイルに保存（手数料なしと手数料あり）"""
    log_entry = f"""
[取引時刻] {time.strftime('%Y-%m-%d %H:%M:%S')}
現在のBTC価格：{price:.2f} JPY

【手数料なし】
JPY残高: {jpy_no_fee:.2f} JPY
BTC保有量：{btc_no_fee:.8f} BTC
総資産額：{total_no_fee:.2f} JPY

【手数料あり（手数料率 {FEE_RATE*100:.2f}%）】
JPY残高: {jpy_fee:.2f} JPY
BTC保有量：{btc_fee:.8f} BTC
総資産額：{total_fee:.2f} JPY

取引内容: {action}
-------------------------
"""
    print(log_entry.strip())
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

    # シミュレーション用の変数（手数料なしと手数料あり）
    jpy_no_fee = INITIAL_JPY
    btc_no_fee = BTC_HOLDINGS
    jpy_fee = INITIAL_JPY
    btc_fee = BTC_HOLDINGS
    best_price = test_price  # 初期価格

    while True:
        try:
            base_price = get_btc_price()
            # ランダムな価格変動を追加 (±0.2%)
            current_price = base_price * (1 + (random.random() * 0.004 - 0.002))
            total_no_fee = jpy_no_fee + btc_no_fee * current_price
            total_fee = jpy_fee + btc_fee * current_price
            
            # 価格変動を表示
            price_diff = ((current_price - best_price) / best_price) * 100
            print(f"価格変動: {price_diff:.2f}%")
            
            # 買い条件：現在価格がベスト価格から0.1%以上下落
            if current_price < best_price * 0.999:
                if jpy_no_fee > 0 and jpy_fee > 0:
                    # 最大5万円分購入（残高が5万円未満の場合は全額使用）
                    buy_amount_no_fee = min(jpy_no_fee, 50000)
                    buy_amount_fee = min(jpy_fee, 50000)
                    
                    # 手数料なしの場合
                    btc_bought_no_fee = buy_amount_no_fee / current_price
                    jpy_no_fee -= buy_amount_no_fee
                    btc_no_fee += btc_bought_no_fee
                    
                    # 手数料ありの場合（購入時に手数料を差引）
                    btc_bought_fee = (buy_amount_fee / current_price) * (1 - FEE_RATE)
                    jpy_fee -= buy_amount_fee
                    btc_fee += btc_bought_fee
                    
                    best_price = current_price
                    
                    total_no_fee = jpy_no_fee + btc_no_fee * current_price
                    total_fee = jpy_fee + btc_fee * current_price
                    
                    action = f"買い: 手数料なし {btc_bought_no_fee:.8f} BTC, 手数料あり {btc_bought_fee:.8f} BTC"
                    print_status(current_price, jpy_no_fee, btc_no_fee, total_no_fee, jpy_fee, btc_fee, total_fee, action)
            
            # 売り条件：現在価格がベスト価格から0.1%以上上昇
            elif current_price > best_price * 1.001:
                if btc_no_fee > 0 and btc_fee > 0:
                    # 保有BTCの50%を売却
                    sell_amount_no_fee = btc_no_fee * 0.5
                    sell_amount_fee = btc_fee * 0.5
                    
                    # 手数料なしの場合
                    jpy_no_fee += sell_amount_no_fee * current_price
                    btc_no_fee -= sell_amount_no_fee
                    
                    # 手数料ありの場合（売却時に手数料を差引）
                    jpy_fee += sell_amount_fee * current_price * (1 - FEE_RATE)
                    btc_fee -= sell_amount_fee
                    
                    best_price = current_price
                    
                    total_no_fee = jpy_no_fee + btc_no_fee * current_price
                    total_fee = jpy_fee + btc_fee * current_price
                    
                    action = f"売り: 手数料なし {sell_amount_no_fee:.8f} BTC, 手数料あり {sell_amount_fee:.8f} BTC"
                    print_status(current_price, jpy_no_fee, btc_no_fee, total_no_fee, jpy_fee, btc_fee, total_fee, action)
            
            time.sleep(TICK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nシミュレーションを終了します")
            final_total_no_fee = jpy_no_fee + btc_no_fee * current_price
            final_total_fee = jpy_fee + btc_fee * current_price
            profit_no_fee = final_total_no_fee - INITIAL_JPY
            profit_fee = final_total_fee - INITIAL_JPY
            print(f"最終資産額（手数料なし）: {final_total_no_fee:.2f} JPY, 利益: {profit_no_fee:.2f} JPY")
            print(f"最終資産額（手数料あり）: {final_total_fee:.2f} JPY, 利益: {profit_fee:.2f} JPY")
            
            # 最終結果をログに記録
            try:
                with open('trading_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"\nシミュレーション終了: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"最終資産額（手数料なし）: {final_total_no_fee:.2f} JPY, 利益: {profit_no_fee:.2f} JPY\n")
                    f.write(f"最終資産額（手数料あり）: {final_total_fee:.2f} JPY, 利益: {profit_fee:.2f} JPY\n")
            except Exception as e:
                print(f"最終結果のログ記録エラー: {e}")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            time.sleep(60)


if __name__ == "__main__":
    simulate_trading() 