
"""
現物取引を行うプログラム
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
from pybit.unified_trading import HTTP

MIN_USDT_VALUE = 1  # 最小注文金額（USDT）
MIN_BTC_QTY = 0.000100  # 最小BTC取引量

def round_btc(amount):
    """BTCの数量を5桁に丸める"""
    return "{:.5f}".format(float(amount))

def main():
    # 環境変数からAPIキーとシークレットを取得
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        print("環境変数 BYBIT_API_KEY と BYBIT_API_SECRET を設定してください。")
        sys.exit(1)

    # テストネットのセッションを作成
    session = HTTP(testnet=True, api_key=api_key, api_secret=api_secret)

    print("=== Bybitスポット取引 ===")
    order_type_input = input("注文タイプを選択してください (buy/sell): ").strip().lower()
    if order_type_input not in ["buy", "sell"]:
        print("無効な注文タイプです。")
        sys.exit(1)
    side = order_type_input.capitalize()

    try:
        ticker = session.get_tickers(
            category="spot",
            symbol="BTCUSDT"
        )
        current_price = float(ticker["result"]["list"][0]["lastPrice"])
        print(f"現在のBTC価格: {current_price} USDT")
        min_usdt_value = MIN_BTC_QTY * current_price
        print(f"最小取引金額: {min_usdt_value:.2f} USDT（{MIN_BTC_QTY} BTC）")
    except Exception as e:
        print("マーケット情報の取得に失敗しました。", e)
        sys.exit(1)

    try:
        # 注文方法の選択（USDTまたはBTC）
        if side == "Sell":
            order_type = input("注文方法を選択してください (1: USDT金額で指定, 2: BTC数量で指定): ").strip()
            if order_type not in ["1", "2"]:
                print("無効な選択です。")
                sys.exit(1)
        else:
            order_type = "1"  # 買い注文は常にUSDT金額で指定

        if order_type == "1":  # USDT金額で指定
            order_value_input = input("注文金額（USDT）を入力してください: ").strip()
            order_value = float(order_value_input)
            if order_value < min_usdt_value:
                print(f"注文金額が小さすぎます。最小注文金額は {min_usdt_value:.2f} USDT（{MIN_BTC_QTY} BTC相当）です。")
                sys.exit(1)
            
            if side == "Buy":
                qty = str(order_value)  # 買い注文はUSDT金額をそのまま使用
                display_amount = f"{order_value} USDT"
                estimated_btc = order_value / current_price
                print(f"概算取得量: {round_btc(estimated_btc)} BTC")
            else:
                btc_amount = order_value / current_price  # 売り注文は相当するBTC数量を計算
                if btc_amount < MIN_BTC_QTY:
                    print(f"取引量が小さすぎます。最小取引量は {MIN_BTC_QTY} BTCです。")
                    sys.exit(1)
                btc_amount = round_btc(btc_amount)  # 5桁に丸める
                qty = btc_amount
                display_amount = f"{order_value} USDT（{btc_amount} BTC）"
                print(f"売却BTC数量: {btc_amount} BTC")
        else:  # BTC数量で指定（売り注文のみ）
            btc_amount_input = input(f"売却するBTC数量を入力してください（最小: {MIN_BTC_QTY} BTC）: ").strip()
            btc_amount = float(btc_amount_input)
            if btc_amount < MIN_BTC_QTY:
                print(f"取引量が小さすぎます。最小取引量は {MIN_BTC_QTY} BTCです。")
                sys.exit(1)
            btc_amount = round_btc(btc_amount)  # 5桁に丸める
            estimated_value = float(btc_amount) * current_price
            qty = btc_amount
            display_amount = f"{btc_amount} BTC（{estimated_value:.2f} USDT）"
            print(f"概算取得額: {estimated_value:.2f} USDT")
        
    except Exception as e:
        print("無効な入力です。")
        sys.exit(1)

    print(f"注文内容: {side} {display_amount} @ {current_price} USDT/BTC")
    confirm = input("この注文を実行しますか？ (yes/no): ").strip().lower()
    if confirm != "yes":
        print("注文をキャンセルしました。")
        sys.exit(0)

    # スポット注文を作成
    try:
        order_params = {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": side,
            "orderType": "Market",
            "qty": qty,
            "timeInForce": "IOC"  # Market注文にはIOCを指定
        }

        order_response = session.place_order(**order_params)
    
        print("\n注文が完了しました。")
        print("注文結果:", order_response)
        if side == "Buy":
            print("Bybitのウォレットでお持ちのBTC残高を確認してください。")
        else:
            print("Bybitのウォレットでお持ちのUSDT残高を確認してください。")
    except Exception as e:
        print("注文実行中にエラーが発生しました:", e)
        print("注文詳細:", order_params)


if __name__ == '__main__':
    main()
