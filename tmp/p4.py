import ccxt
import time
import random

# 初期設定
BTC_HOLDINGS = 0
TICK_INTERVAL = 3  # 3秒ごとに取引
# Mock API for simulation purposes
class MockAPI:
    def fetch_ticker(self, symbol):
        return {'last': 5000000}  # Mock price for BTC/JPY

    def create_order(self, symbol, type, side, amount, price):
        # 手数料率を0.01%～0.15%の範囲でランダムに設定
        fee_rate = random.uniform(0.0001, 0.0015)
        fee = amount * fee_rate  # 手数料（BTC単位）
        
        return {
            'id': 'mock_order_id',
            'filled': amount,
            'cost': amount * price,
            'status': 'closed',
            'price': price,
            'timestamp': time.time(),
            'fee': fee,
            'fee_rate': fee_rate
        }

    def fetch_order(self, order_id):
        return {'status': 'closed'}

    def cancel_order(self, order_id):
        pass

API = MockAPI()

def get_initial_jpy():
    """ユーザーから初期資金を取得"""
    while True:
        try:
            amount = float(input("初期資金をJPYで入力してください（例：10000）："))
            if amount <= 0:
                print("0より大きい値を入力してください")
                continue
            return amount
        except ValueError:
            print("数値を入力してください")

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

def save_trade_history(trade_details):
    """取引履歴をファイルに保存"""
    log_entry = f"""
[取引タイプ] {trade_details['type']}
数量: {trade_details['amount']} BTC
価格: {trade_details['price']} JPY
手数料: {trade_details['fee']:.8f} BTC (手数料率: {trade_details['fee_rate']*100:.4f}%)
タイムスタンプ: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(trade_details['timestamp']))}
-------------------------
"""
    try:
        with open('trading_log.txt', 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"取引履歴の保存エラー: {e}")

    # 初期資金をユーザーから取得
    INITIAL_JPY = get_initial_jpy()
    print(f"\n初期資金: {INITIAL_JPY} JPY でシミュレーションを開始します\n")
    
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
            # 実際の価格変動を模擬
            # ボラティリティを考慮したランダムウォーク
            volatility = 0.002  # 1分間のボラティリティ
            change_percent = random.gauss(0, volatility)
            current_price = base_price * (1 + change_percent)
            total_assets = jpy_balance + (btc_holdings * current_price)
            
            # 取引ロジック
            price_diff = ((current_price - best_price) / best_price) * 100
            print(f"価格変動: {price_diff:.2f}%")
            
            # 買い条件：現在価格がベスト価格から0.05%以上下落
            if current_price < best_price * 0.9995:  # 0.05%下落で買い
                if jpy_balance > 0:
                    # 最大5000円分購入（残高が5000円未満の場合は全額使用）
                    # 取引所の手数料と最小注文数量を考慮
                    fee_rate = 0.0012  # bitFlyerの手数料率（0.12%）
                    min_order_amount = 0.001  # bitFlyerの最小注文数量
                    
                    # 最大5000円分購入（手数料を考慮）
                    buy_amount = min(jpy_balance / (1 + fee_rate), 5000)
                    btc_bought = buy_amount / current_price
                    if btc_bought < min_order_amount:
                        print(f"最小注文数量に満たないためスキップ: 注文可能数量={btc_bought:.8f} BTC, 最小注文数量={min_order_amount} BTC")
                        continue
                        
                    try:
                        # 指値注文を実行
                        order = API.create_order(
                            symbol='BTC/JPY',
                            type='limit',
                            side='buy',
                            amount=btc_bought,
                            price=current_price
                        )
                        
                        # 注文が完全に約定するまで待機
                        start_time = time.time()
                        while order['status'] != 'closed':
                            if time.time() - start_time > 60:  # 60秒でタイムアウト
                                API.cancel_order(order['id'])
                                raise TimeoutError("注文がタイムアウトしました")
                            time.sleep(1)
                            order = API.fetch_order(order['id'])
                            
                        btc_holdings += order['filled']
                        jpy_balance -= (order['cost'] + (order['cost'] * fee_rate))
                        
                        # 取引履歴を保存
                        save_trade_history({
                            'type': 'buy',
                            'amount': order['filled'],
                            'price': order['price'],
                            'timestamp': order['timestamp'],
                            'fee': order['cost'] * fee_rate
                        })
                        
                    except Exception as e:
                        print(f"注文実行エラー: {e}")
                        if 'order' in locals():
                            try:
                                API.cancel_order(order['id'])
                            except:
                                pass
                        continue
                    best_price = current_price
                    print_status(current_price, jpy_balance, btc_holdings, total_assets, f"買 {btc_bought:.8f} BTC")
            
            # 売り条件：現在価格がベスト価格から0.05%以上上昇
            elif current_price > best_price * 1.0005:  # 0.05%上昇で売り
                if btc_holdings > 0:
                    # 保有BTCの50%を売却
                    # 取引所の手数料と最小注文数量を考慮
                    fee_rate = 0.0015  # 0.15%の手数料
                    min_order_amount = 0.0001  # 最小注文数量を0.0001 BTCに緩和
                    
                    # 50%売却（最小注文数量を考慮）
                    sell_amount = btc_holdings * 0.5
                    if sell_amount < min_order_amount:
                        print(f"最小注文数量に満たないためスキップ: 注文可能数量={sell_amount:.8f} BTC, 最小注文数量={min_order_amount} BTC")
                        continue
                        
                    try:
                        # 指値注文を実行
                        order = API.create_order(
                            symbol='BTC/JPY',
                            type='limit',
                            side='sell',
                            amount=sell_amount,
                            price=current_price
                        )
                        
                        # 注文が完全に約定するまで待機
                        start_time = time.time()
                        while order['status'] != 'closed':
                            if time.time() - start_time > 60:  # 60秒でタイムアウト
                                API.cancel_order(order['id'])
                                raise TimeoutError("注文がタイムアウトしました")
                            time.sleep(1)
                            order = API.fetch_order(order['id'])
                            
                        btc_holdings -= order['filled']
                        jpy_balance += (order['cost'] - (order['cost'] * fee_rate))
                        
                        # 取引履歴を保存
                        save_trade_history({
                            'type': 'sell',
                            'amount': order['filled'],
                            'price': order['price'],
                            'timestamp': order['timestamp'],
                            'fee': order['cost'] * fee_rate
                        })
                        
                    except Exception as e:
                        print(f"注文実行エラー: {e}")
                        if 'order' in locals():
                            try:
                                API.cancel_order(order['id'])
                            except:
                                pass
                        continue
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
