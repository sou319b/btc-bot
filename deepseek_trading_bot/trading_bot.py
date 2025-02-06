import time
from typing import Dict, Any
from exchange_handler import ExchangeHandler
from market_analyzer import MarketAnalyzer
import config
import logging

logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingBot:
    def __init__(self):
        self.exchange = ExchangeHandler()
        self.analyzer = MarketAnalyzer()
        self.running = False
        self.position = None
        
    def start(self):
        """取引ボットの開始"""
        self.running = True
        logging.info("取引ボット開始")
        
        while self.running:
            try:
                self.execute_trading_cycle()
                time.sleep(60)  # 1分待機
            except Exception as e:
                logging.error(f"取引サイクルエラー: {e}")
                time.sleep(60)

    def stop(self):
        """取引ボットの停止"""
        self.running = False
        logging.info("取引ボット停止")

    def execute_trading_cycle(self):
        """取引サイクルの実行"""
        # 市場データの取得
        ohlcv = self.exchange.fetch_ohlcv()
        if not ohlcv:
            return

        # データの分析
        df = self.analyzer.prepare_data(ohlcv)
        df = self.analyzer.calculate_indicators(df)
        analysis = self.analyzer.get_ai_analysis(df)

        # ポジション情報の更新
        self.position = self.exchange.get_position()

        # 取引判断
        if self.should_trade(analysis):
            self.execute_trade(analysis)

    def should_trade(self, analysis: Dict[str, Any]) -> bool:
        """取引判断"""
        # 最小信頼度チェック
        if analysis['confidence'] < 0.7:
            return False

        # 残高チェック
        balance = self.exchange.get_balance()
        if not balance or balance['free'] < config.TRADE_AMOUNT * 1000:
            # 1000USDT相当分くらいないとトレードしない、など
            return False

        # ポジション制限チェック（既にポジション持ってる場合は見送り）
        if self.position and abs(float(self.position['size'])) > 0:
            return False

        return True

    def execute_trade(self, analysis: Dict[str, Any]):
        """取引の実行"""
        try:
            # 取引サイドの決定
            side = self.determine_trade_side(analysis['recommendation'])
            if not side:
                return

            # 注文の発行
            order = self.exchange.place_order(
                side=side,
                amount=config.TRADE_AMOUNT
            )

            if order:
                logging.info(f"注文実行: {side}, 数量: {config.TRADE_AMOUNT}")
                
                # ストップロス注文の設定
                self.set_stop_loss(side, order['price'])

        except Exception as e:
            logging.error(f"取引実行エラー: {e}")

    def determine_trade_side(self, recommendation: str) -> str:
        """取引サイドの決定"""
        recommendation = recommendation.lower()
        if 'buy' in recommendation or 'long' in recommendation:
            return 'buy'
        elif 'sell' in recommendation or 'short' in recommendation:
            return 'sell'
        return ''

    def set_stop_loss(self, side: str, entry_price: float):
        """ストップロスの設定"""
        try:
            stop_price = entry_price * (1 - config.STOP_LOSS_PERCENT) if side == 'buy' else \
                         entry_price * (1 + config.STOP_LOSS_PERCENT)

            self.exchange.place_order(
                side='sell' if side == 'buy' else 'buy',
                amount=config.TRADE_AMOUNT,
                price=stop_price,
                params={'stopPrice': stop_price}
            )
            
            logging.info(f"ストップロス設定: {stop_price}")

        except Exception as e:
            logging.error(f"ストップロス設定エラー: {e}")
