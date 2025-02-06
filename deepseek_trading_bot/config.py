import os
from dotenv import load_dotenv

load_dotenv()

# API設定
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY')
EXCHANGE_SECRET_KEY = os.getenv('EXCHANGE_SECRET_KEY')

print(f"EXCHANGE_API_KEY: {EXCHANGE_API_KEY}")
print(f"EXCHANGE_SECRET_KEY: {EXCHANGE_SECRET_KEY}")


# 取引設定
TRADING_PAIR = 'BTCUSDT'
TRADE_AMOUNT = 0.001
INTERVAL = '1m'

# DeepSeek API設定
DEEPSEEK_API_ENDPOINT = 'https://api.deepseek.ai/v1/chat/completions'

# リスク管理設定
MAX_LEVERAGE = 1
MAX_POSITION_PERCENT = 0.1
STOP_LOSS_PERCENT = 0.01

# その他の設定
TESTNET = True
RECV_WINDOW = 20000
