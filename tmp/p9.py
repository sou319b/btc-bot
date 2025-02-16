# より実践的なビットコイン取引シミュレーションボット
# 手数料と取引制限を考慮

import ccxt
import time
import random

# 初期設定
INITIAL_JPY = 1000000  # 初期資金100万円
BTC_HOLDINGS = 0
TICK_INTERVAL = 3  # 3秒ごとに取引
API = ccxt.bitflyer()

# 取引制限
MIN_ORDER_SIZE = 0.00000001  # 最小発注数量（BTC）
MAX_ORDER_SIZE = 20  # 最大発注数量（BTC）
FEE_RATE = random.uniform(0.001, 0.0015)  # 0.1%～0.15%のランダムな手数料率

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

def calculate_fee(amount_btc):
    """取引手数料を計算"""
    return amount_btc * FEE_RATE

def validate_order_size(amount_btc):
    """注文サイズが制限内かチェック"""
    if amount_btc < MIN_ORDER_SIZE:
        return MIN_ORDER_SIZE
    elif amount_btc > MAX_ORDER_SIZE:
        return MAX_ORDER_SIZE
    return amount_btc

def print_status(price, jpy, btc, total, action, fee_btc=0):
    """現在の状態を表示し、ファイルに保存"""
    fee_jpy = fee_btc * price
    log_entry = f"""
[取引時刻] {time.strftime('%Y-%m-%d %H:%M:%S')}
現在のBTC価格：{price:,.0f} JPY
JPY残高: {jpy:,.0f} JPY
BTC保有量：{btc:.8f} BTC
総資産額：{total:,.0f} JPY
買/売：{action}
取引手数料：{fee_btc:.8f} BTC ({fee_jpy:,.0f} JPY)
手数料率：{FEE_RATE*100:.3f}%
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
            f.write(f"初期資金: {INITIAL_JPY:,.0f} JPY\n")
            f.write(f"取引手数料率: {FEE_RATE*100:.3f}%\n\n")
    except Exception as e:
        print(f"ログファイル初期化エラー: {e}")
        return

    # API接続テスト
    try:
        print("API接続テスト中...")
        test_price = get_btc_price()
        print(f"初期価格取得成功: {test_price:,.0f} JPY")
    except Exception as e:
        print(f"API接続エラー: {str(e)}")
        print("bitFlyer APIに接続できませんでした。ネットワーク接続を確認してください。")
        return

    jpy_balance = INITIAL_JPY
    btc_holdings = BTC_HOLDINGS
    best_price = test_price
    
    while True:
        try:
            base_price = get_btc_price()
            current_price = base_price * (1 + (random.random() * 0.004 - 0.002))
            total_assets = jpy_balance + (btc_holdings * current_price)
            
            price_diff = ((current_price - best_price) / best_price) * 100
            print(f"価格変動: {price_diff:.2f}%")
            
            if current_price < best_price * 0.999:  # 0.1%下落で買い
                if jpy_balance > 0:
                    buy_amount_jpy = min(jpy_balance, 50000)
                    btc_to_buy = buy_amount_jpy / current_price
                    btc_to_buy = validate_order_size(btc_to_buy)
                    
                    # 手数料を考慮
                    fee_btc = calculate_fee(btc_to_buy)
                    total_btc = btc_to_buy - fee_btc
                    
                    if total_btc >= MIN_ORDER_SIZE:
                        actual_cost = btc_to_buy * current_price
                        if actual_cost <= jpy_balance:
                            jpy_balance -= actual_cost
                            btc_holdings += total_btc
                            best_price = current_price
                            print_status(current_price, jpy_balance, btc_holdings, total_assets,
                                       f"買 {btc_to_buy:.8f} BTC", fee_btc)
            
            elif current_price > best_price * 1.001:  # 0.1%上昇で売り
                if btc_holdings > 0:
                    sell_amount = btc_holdings * 0.5
                    sell_amount = validate_order_size(sell_amount)
                    
                    if sell_amount >= MIN_ORDER_SIZE:
                        fee_btc = calculate_fee(sell_amount)
                        actual_sell_amount = sell_amount - fee_btc
                        
                        jpy_gained = actual_sell_amount * current_price
                        jpy_balance += jpy_gained
                        btc_holdings -= sell_amount
                        best_price = current_price
                        print_status(current_price, jpy_balance, btc_holdings, total_assets,
                                   f"売 {sell_amount:.8f} BTC", fee_btc)
            
            time.sleep(TICK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nシミュレーションを終了します")
            final_total = jpy_balance + (btc_holdings * current_price)
            profit = final_total - INITIAL_JPY
            print(f"最終資産額: {final_total:,.0f} JPY")
            print(f"利益: {profit:,.0f} JPY")
            
            try:
                with open('trading_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"\nシミュレーション終了: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"最終資産額: {final_total:,.0f} JPY\n")
                    f.write(f"利益: {profit:,.0f} JPY\n")
            except Exception as e:
                print(f"最終結果のログ記録エラー: {e}")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            time.sleep(60)

if __name__ == "__main__":
    simulate_trading()
