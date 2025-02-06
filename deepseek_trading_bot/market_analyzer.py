import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Any
import config

class MarketAnalyzer:
    def __init__(self):
        self.headers = {
            'Authorization': f'Bearer {config.DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }

    def prepare_data(self, ohlcv_data: List[List[Any]]) -> pd.DataFrame:
        """価格データの前処理"""
        df = pd.DataFrame(
            ohlcv_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標の計算"""
        # SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
        df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()

        return df

    def get_ai_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """DeepSeek AIを使用した市場分析"""
        try:
            # 最新の市場データを準備
            latest_data = market_data.tail(1).to_dict('records')[0]
            
            # AIへのプロンプト作成
            prompt = f"""
            以下の市場データを分析し、取引推奨を提供してください：
            価格: {latest_data['close']}
            RSI: {latest_data['rsi']:.2f}
            SMA20: {latest_data['sma_20']:.2f}
            SMA50: {latest_data['sma_50']:.2f}
            BB上限: {latest_data['bb_upper']:.2f}
            BB下限: {latest_data['bb_lower']:.2f}
            """

            # DeepSeek APIリクエスト
            response = requests.post(
                config.DEEPSEEK_API_ENDPOINT,
                headers=self.headers,
                json={
                    'messages': [{'role': 'user', 'content': prompt}],
                    'model': 'deepseek-chat',
                    'temperature': 0.7
                }
            )

            if response.status_code == 200:
                analysis = response.json()
                return {
                    'recommendation': analysis['choices'][0]['message']['content'],
                    'confidence': self.calculate_confidence(latest_data)
                }
            else:
                return {
                    'recommendation': 'AI分析エラー',
                    'confidence': 0.0
                }

        except Exception as e:
            print(f"AI分析エラー: {e}")
            return {
                'recommendation': 'エラー',
                'confidence': 0.0
            }

    def calculate_confidence(self, data: Dict[str, float]) -> float:
        """分析の信頼度計算"""
        confidence = 0.0
        
        # RSIによる判断
        if data['rsi'] < 30 or data['rsi'] > 70:
            confidence += 0.3
        
        # SMAクロスオーバー
        if abs(data['sma_20'] - data['sma_50']) / data['sma_50'] < 0.001:
            confidence += 0.3
            
        # ボリンジャーバンド
        if data['close'] < data['bb_lower'] or data['close'] > data['bb_upper']:
            confidence += 0.4
            
        return min(confidence, 1.0)
