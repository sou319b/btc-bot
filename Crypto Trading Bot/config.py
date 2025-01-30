import ccxt
api = ccxt.bitget({
    'apiKey':"bg_ba011040842577f37c7a109474010f8e",
    'secret':"f1f1e4cfe2897d314b8ef0039fef9c2470a1f1b71fb2a9bfbc1d776e374261ee",
    "password": "P2cSgZSDtCyp",
    'enableRateLimit': True,
    'timeout': 60000,
    'options': {
        'defaultType': 'spot',
    }
})

api.options['createMarketBuyOrderRequiresPrice'] = False