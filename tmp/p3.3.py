# 利益がでるbot バックテスト版 (p3.3)
# bitflyerの公開APIを使用して取引履歴からOHLCVバーを構築し、1ヶ月分のバックテストを実施

import ccxt
import time
from datetime import datetime

# 初期設定
INITIAL_JPY = 1000000    # 初期資金100万円
BTC_HOLDINGS = 0
FEE_RATE = 0.002         # 0.2% の手数料
TIMEFRAME = '1h'         # 1時間足
TIMEFRAME_MS = 3600 * 1000  # 1時間をミリ秒に変換

API = ccxt.bitflyer()


def fetch_all_trades(since):
    """bitflyerのfetch_tradesを利用して、指定時刻以降の取引履歴を取得する
    since: 取得開始時刻（ミリ秒）
    ※ fetch_trades は多くの場合最新の取引のみを返すため、
    複数回呼び出して古いデータを集めるようにする
    """
    symbol = 'BTC/JPY'
    trades = []
    last_timestamp = since
    while True:
        try:
            fetched = API.fetch_trades(symbol, last_timestamp)
        except Exception as e:
            print(f"取引履歴取得エラー: {e}")
            break
        if not fetched:
            break
        # 確実に時間順にする
        fetched = sorted(fetched, key=lambda x: x['timestamp'])
        new_trades = []
        for trade in fetched:
            ts = trade['timestamp']
            if ts >= since:
                new_trades.append({
                    'timestamp': ts,
                    'price': float(trade['price']),
                    'amount': float(trade['amount'])
                })
        if not new_trades:
            break
        trades += new_trades
        new_last = new_trades[-1]['timestamp']
        # 終了条件：新たなデータが取得できない場合
        if new_last == last_timestamp:
            break
        last_timestamp = new_last + 1
    return trades


def build_ohlcv_bars(trades, timeframe_ms, since, until):
    """取得した取引データから、指定した時間枠のOHLCVバーを構築する
    trades: 取引のリスト
    timeframe_ms: 時間枠（ミリ秒）
    since: 取得開始時刻
    until: 取得終了時刻
    """
    # 開始時刻のバケット境界を調整
    start = since - (since % timeframe_ms)
    # 各時間枠のバケットを初期化
    candles = {}
    for bucket in range(start, until + timeframe_ms, timeframe_ms):
        candles[bucket] = []
    for trade in trades:
        t = trade['timestamp']
        if t < since or t > until:
            continue
        bucket = t - (t % timeframe_ms)
        if bucket in candles:
            candles[bucket].append(trade)
        else:
            candles[bucket] = [trade]
    ohlcv = []
    sorted_keys = sorted(candles.keys())
    prev_close = None
    for bucket in sorted_keys:
        trades_in_bucket = candles[bucket]
        if trades_in_bucket:
            open_price = trades_in_bucket[0]['price']
            close_price = trades_in_bucket[-1]['price']
            high_price = max(t['price'] for t in trades_in_bucket)
            low_price = min(t['price'] for t in trades_in_bucket)
            volume = sum(t['amount'] for t in trades_in_bucket)
            prev_close = close_price
            ohlcv.append([bucket, open_price, high_price, low_price, close_price, volume])
        else:
            # バケット内に取引がなければ、前回の終値をそのまま使用（取引量は0）
            if prev_close is not None:
                ohlcv.append([bucket, prev_close, prev_close, prev_close, prev_close, 0])
    return ohlcv


def print_status(price, jpy_no_fee, btc_no_fee, total_no_fee, jpy_fee, btc_fee, total_fee, action, timestamp):
    """現在の状態を表示し、ログに記録（手数料なしと手数料あり）"""
    log_entry = f"""
[時刻] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp/1000))}
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
        with open('trading_log_backtest.txt', 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"ログファイル書き込みエラー: {e}")


def simulate_trading_backtest():
    # ログファイル初期化
    try:
        with open('trading_log_backtest.txt', 'w', encoding='utf-8') as f:
            f.write(f"シミュレーション(バックテスト)開始: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"初期資金: {INITIAL_JPY} JPY\n\n")
    except Exception as e:
        print(f"ログファイル初期化エラー: {e}")
        return

    one_month_ms = 30 * 24 * 60 * 60 * 1000
    until = API.milliseconds()
    since = until - one_month_ms

    print("取引履歴を取得中...")
    trades = fetch_all_trades(since)
    if not trades:
        print("取引履歴が取得できませんでした。")
        return
    print(f"取引履歴 {len(trades)} 件を取得しました。")

    # 取引履歴からOHLCVバーを構築
    ohlcv = build_ohlcv_bars(trades, TIMEFRAME_MS, since, until)
    print(f"OHLCVバー {len(ohlcv)} 本を構築しました。")

    # シミュレーション用の変数（手数料なしと手数料あり）
    jpy_no_fee = INITIAL_JPY
    btc_no_fee = BTC_HOLDINGS
    jpy_fee = INITIAL_JPY
    btc_fee = BTC_HOLDINGS

    # シミュレーション開始の基準価格を最初のローソク足の終値で設定
    first_candle = ohlcv[0]
    best_price = first_candle[4]  # 終値

    for candle in ohlcv:
        timestamp, open_, high, low, close, volume = candle
        current_price = close
        total_no_fee = jpy_no_fee + btc_no_fee * current_price
        total_fee = jpy_fee + btc_fee * current_price

        # 価格変動の割合を計算
        price_diff = ((current_price - best_price) / best_price) * 100
        print(f"データ時刻: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp/1000))}  価格変動: {price_diff:.2f}%")

        # 買い条件：現在価格が基準価格から0.1%下落
        if current_price < best_price * 0.999:
            if jpy_no_fee > 0 and jpy_fee > 0:
                # 最大5万円分購入（残高が5万円未満の場合は全額使用）
                buy_amount_no_fee = min(jpy_no_fee, 50000)
                buy_amount_fee = min(jpy_fee, 50000)

                # 手数料なしの場合
                btc_bought_no_fee = buy_amount_no_fee / current_price
                jpy_no_fee -= buy_amount_no_fee
                btc_no_fee += btc_bought_no_fee

                # 手数料ありの場合（購入時に手数料を差し引く）
                btc_bought_fee = (buy_amount_fee / current_price) * (1 - FEE_RATE)
                jpy_fee -= buy_amount_fee
                btc_fee += btc_bought_fee

                best_price = current_price

                total_no_fee = jpy_no_fee + btc_no_fee * current_price
                total_fee = jpy_fee + btc_fee * current_price

                action = f"買い: 手数料なし {btc_bought_no_fee:.8f} BTC, 手数料あり {btc_bought_fee:.8f} BTC"
                print_status(current_price, jpy_no_fee, btc_no_fee, total_no_fee,
                             jpy_fee, btc_fee, total_fee, action, timestamp)

        # 売り条件：現在価格が基準価格から0.1%上昇
        elif current_price > best_price * 1.001:
            if btc_no_fee > 0 and btc_fee > 0:
                # 保有BTCの50%を売却
                sell_amount_no_fee = btc_no_fee * 0.5
                sell_amount_fee = btc_fee * 0.5

                # 手数料なしの場合
                jpy_no_fee += sell_amount_no_fee * current_price
                btc_no_fee -= sell_amount_no_fee

                # 手数料ありの場合（売却時に手数料を差し引く）
                jpy_fee += sell_amount_fee * current_price * (1 - FEE_RATE)
                btc_fee -= sell_amount_fee

                best_price = current_price

                total_no_fee = jpy_no_fee + btc_no_fee * current_price
                total_fee = jpy_fee + btc_fee * current_price

                action = f"売り: 手数料なし {sell_amount_no_fee:.8f} BTC, 手数料あり {sell_amount_fee:.8f} BTC"
                print_status(current_price, jpy_no_fee, btc_no_fee, total_no_fee,
                             jpy_fee, btc_fee, total_fee, action, timestamp)

    final_total_no_fee = jpy_no_fee + btc_no_fee * current_price
    final_total_fee = jpy_fee + btc_fee * current_price
    profit_no_fee = final_total_no_fee - INITIAL_JPY
    profit_fee = final_total_fee - INITIAL_JPY

    print("\nバックテスト終了")
    print(f"最終資産額（手数料なし）: {final_total_no_fee:.2f} JPY, 利益: {profit_no_fee:.2f} JPY")
    print(f"最終資産額（手数料あり）: {final_total_fee:.2f} JPY, 利益: {profit_fee:.2f} JPY")

    try:
        with open('trading_log_backtest.txt', 'a', encoding='utf-8') as f:
            f.write(f"\nバックテスト終了: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最終資産額（手数料なし）: {final_total_no_fee:.2f} JPY, 利益: {profit_no_fee:.2f} JPY\n")
            f.write(f"最終資産額（手数料あり）: {final_total_fee:.2f} JPY, 利益: {profit_fee:.2f} JPY\n")
    except Exception as e:
        print(f"バックテスト結果のログ記録エラー: {e}")


if __name__ == '__main__':
    simulate_trading_backtest() 