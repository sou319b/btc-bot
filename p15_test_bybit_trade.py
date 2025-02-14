import time
from datetime import datetime
from p15_bybit_handler import BybitHandler

def print_status(bybit):
    """現在の状態を表示する関数"""
    current_price = bybit.get_btc_price()
    usdt_balance, btc_holding = bybit.get_wallet_info()
    print("\n=== 現在の状態 ===")
    print(f"現在のBTC価格: {current_price:,.2f} USDT")
    print(f"USDT残高: {usdt_balance:,.2f} USDT")
    print(f"BTC保有量: {btc_holding:.6f} BTC")
    print(f"総資産額: {(usdt_balance + btc_holding * current_price):,.2f} USDT")
    return current_price, usdt_balance, btc_holding

def execute_buy(bybit, amount):
    """買い注文を実行する関数"""
    print(f"\n=== 買い注文実行 ===")
    print(f"発注数量: {amount} BTC")
    order = bybit.place_buy_order(amount)
    if order:
        print("買い注文成功:")
        print(order)
    else:
        print("買い注文失敗")
    
    time.sleep(5)  # 5秒待機
    print_status(bybit)

def execute_sell(bybit, amount):
    """売り注文を実行する関数"""
    print(f"\n=== 売り注文実行 ===")
    print(f"発注数量: {amount} BTC")
    order = bybit.place_sell_order(amount)
    if order:
        print("売り注文成功:")
        print(order)
    else:
        print("売り注文失敗")
    
    time.sleep(5)  # 5秒待機
    print_status(bybit)

def main():
    # Bybitハンドラーの初期化
    bybit = BybitHandler()
    
    # 取引制限の設定
    MIN_ORDER_VALUE = 100  # 最小取引額（USDT）
    MAX_TRADE_AMOUNT = 1.0    # 最大取引量（BTC）
    
    try:
        while True:
            # 現在の状態を表示
            current_price, usdt_balance, btc_holding = print_status(bybit)
            
            # 最小取引量を計算（現在価格から）
            min_trade_amount = round(MIN_ORDER_VALUE / current_price, 6) if current_price else 0.001
            
            # メニューの表示
            print("\n=== 操作メニュー ===")
            print("1: 買い注文")
            print("2: 売り注文")
            print("3: 状態更新")
            print("q: 終了")
            print(f"\n※ 取引制限：")
            print(f"  最小取引額: {MIN_ORDER_VALUE} USDT（約 {min_trade_amount:.6f} BTC）")
            print(f"  最大取引量: {MAX_TRADE_AMOUNT} BTC（約 {MAX_TRADE_AMOUNT * current_price:,.2f} USDT）")
            
            # ユーザー入力の受付
            choice = input("\n操作を選択してください (1/2/3/q): ").strip().lower()
            
            if choice == 'q':
                print("\nプログラムを終了します")
                break
            
            elif choice in ['1', '2']:
                # 取引量の入力
                while True:
                    try:
                        amount = float(input(f"取引量を入力してください（{min_trade_amount}～{MAX_TRADE_AMOUNT} BTC）: ").strip())
                        if amount * current_price < MIN_ORDER_VALUE:
                            print(f"取引額が最小制限（{MIN_ORDER_VALUE} USDT）を下回っています")
                            continue
                        if amount > MAX_TRADE_AMOUNT:
                            print(f"最大取引量（{MAX_TRADE_AMOUNT} BTC）を超えています")
                            continue
                        if amount <= 0:
                            print("取引量は0より大きい値を入力してください")
                            continue
                        
                        # 買い注文時の残高チェック
                        if choice == '1':
                            required_balance = amount * current_price
                            if required_balance > usdt_balance:
                                print(f"USDT残高が不足しています（必要額: {required_balance:,.2f} USDT）")
                                continue
                        # 売り注文時の保有量チェック
                        else:
                            if amount > btc_holding:
                                print(f"BTC保有量が不足しています（必要量: {amount} BTC）")
                                continue
                        
                        break
                    except ValueError:
                        print("有効な数値を入力してください")
                
                # 確認メッセージの表示
                action = "買い" if choice == '1' else "売り"
                confirm = input(f"\n{action}注文を実行します：\n"
                              f"取引量: {amount} BTC\n"
                              f"想定取引額: {amount * current_price:,.2f} USDT\n"
                              f"実行しますか？ (y/n): ").strip().lower()
                
                if confirm == 'y':
                    if choice == '1':
                        execute_buy(bybit, amount)
                    else:
                        execute_sell(bybit, amount)
                else:
                    print("取引をキャンセルしました")
            
            elif choice == '3':
                print("\n状態を更新します...")
                time.sleep(1)
            
            else:
                print("\n無効な選択です。もう一度選択してください。")
            
            # 各操作の後に少し待機
            time.sleep(1)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
    
    print("\nテスト完了")

if __name__ == "__main__":
    main() 