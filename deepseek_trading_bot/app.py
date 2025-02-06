import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from trading_bot import TradingBot
from exchange_handler import ExchangeHandler
import config
import time
from datetime import datetime

class DashboardApp:
    def __init__(self):
        self.bot = TradingBot()
        self.exchange = ExchangeHandler()

    def run(self):
        st.set_page_config(page_title="DeepSeek Trading Bot", layout="wide")
        st.title("DeepSeek Trading Bot Dashboard")

        # サイドバー
        self.render_sidebar()

        # メインコンテンツ
        col1, col2 = st.columns([2, 1])

        with col1:
            self.render_price_chart()

        with col2:
            self.render_position_info()
            self.render_balance_info()

        # 取引履歴
        self.render_trade_history()

    def render_sidebar(self):
        """サイドバーの表示"""
        st.sidebar.header("設定")
        
        # ボット制御
        if st.sidebar.button("Start Bot" if not self.bot.running else "Stop Bot"):
            if self.bot.running:
                self.bot.stop()
            else:
                self.bot.start()

        # 取引設定
        st.sidebar.subheader("取引設定")
        new_amount = st.sidebar.number_input(
            "取引量 (BTC)",
            min_value=0.001,
            value=config.TRADE_AMOUNT,
            step=0.001
        )
        if new_amount != config.TRADE_AMOUNT:
            config.TRADE_AMOUNT = new_amount

        # リスク設定
        st.sidebar.subheader("リスク設定")
        new_stop_loss = st.sidebar.number_input(
            "ストップロス (%)",
            min_value=0.1,
            value=config.STOP_LOSS_PERCENT * 100,
            step=0.1
        )
        if new_stop_loss != config.STOP_LOSS_PERCENT * 100:
            config.STOP_LOSS_PERCENT = new_stop_loss / 100

    def render_price_chart(self):
        """価格チャートの表示"""
        st.subheader("BTC/USDT Price Chart")
        
        # データの取得
        ohlcv = self.exchange.fetch_ohlcv(limit=500)
        if not ohlcv:
            st.error("価格データの取得に失敗しました")
            return

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Plotlyチャートの作成
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_position_info(self):
        """ポジション情報の表示"""
        st.subheader("現在のポジション")
        
        position = self.exchange.get_position()
        if position:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("サイズ", f"{float(position['size']):.3f} BTC")
            with col2:
                st.metric("損益", f"{float(position['unrealizedPnl']):.2f} USDT")
        else:
            st.info("現在のポジションはありません")

    def render_balance_info(self):
        """残高情報の表示"""
        st.subheader("口座残高")
        
        balance = self.exchange.get_balance()
        if balance:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("総残高", f"{balance['total']:.2f} USDT")
            with col2:
                st.metric("利用可能", f"{balance['free']:.2f} USDT")
        else:
            st.error("残高情報の取得に失敗しました")

    def render_trade_history(self):
        """取引履歴の表示"""
        st.subheader("取引履歴")
        
        try:
            with open('trading_bot.log', 'r') as f:
                logs = f.readlines()[-20:]  # 最新20件
                
            for log in logs:
                if "注文実行" in log or "ストップロス" in log:
                    st.text(log.strip())
        except Exception as e:
            st.error(f"取引履歴の読み込みに失敗しました: {e}")

if __name__ == "__main__":
    app = DashboardApp()
    app.run()
