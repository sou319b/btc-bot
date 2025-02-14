#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pybit.unified_trading import HTTP


def main():
    # 環境変数からAPIキーとシークレットを取得
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        print("環境変数 BYBIT_API_KEY と BYBIT_API_SECRET を設定してください。")
        sys.exit(1)

    # テストネットのセッションを作成
    session = HTTP(api_key=api_key, api_secret=api_secret, endpoint="https://api-testnet.bybit.com")

    print("注文を実行します。")
    order_type_input = input("注文タイプを選択してください (buy/sell): ").strip().lower()
    if order_type_input not in ["buy", "sell"]:
        print("無効な注文タイプです。")
        sys.exit(1)
    side = order_type_input.upper()

    # 取引量（BTCの数量）の入力
    try:
        qty_input = input("取引量（BTCの数量）を入力してください: ").strip()
        qty = float(qty_input)
    except Exception as e:
        print("無効な数量です。")
        sys.exit(1)

    # 市場注文を作成
    try:
        order_response = session.place_order(
            symbol="BTCUSDT",
            side=side,
            orderType="Market",
            qty=str(qty),
            timeInForce="IOC",
            category="spot"
        )
        print("注文結果:", order_response)
    except Exception as e:
        print("注文実行中にエラーが発生しました:", e)


if __name__ == '__main__':
    main()
