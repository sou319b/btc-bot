import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from sqlalchemy import create_engine
from typing import Dict, List, Optional

# Enhanced Configuration
CONFIG = {
    "DB": {
        "dbname": "dexscreener",
        "user": "admin",
        "password": "your_password",
        "host": "localhost",
        "port": "5432"
    },
    "FILTERS": {
        "min_liquidity": 5000,  # USD
        "min_age_days": 3,
        "coin_blacklist": [
            "0x123...def",  # Known scam token address
            "SUSPECTCOIN"   # Blacklisted symbol
        ],
        "dev_blacklist": [
            "0x456...abc",  # Known rug developer address
            "0x789...fed"   # Another scam developer
        ],
        "chain_whitelist": ["ethereum", "binance-smart-chain"]
    }
}

class EnhancedDexScreenerBot:
    def __init__(self):
        self.engine = create_engine(
            f'postgresql+psycopg2://{CONFIG["DB"]["user"]}:{CONFIG["DB"]["password"]}'
            f'@{CONFIG["DB"]["host"]}/{CONFIG["DB"]["dbname"]}'
        )
        self._init_db()
        self.model = IsolationForest(n_estimators=100, contamination=0.01)
        self.historical_data = self._load_historical_data()

    def _init_db(self):
        """Initialize database with additional security tables"""
        with self.engine.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blacklist (
                    address VARCHAR(42) PRIMARY KEY,
                    type VARCHAR(20) CHECK (type IN ('coin', 'dev')),
                    reason TEXT,
                    listed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_blacklist_type ON blacklist(type);
            """)
            # Migrate config blacklists to database
            self._seed_initial_blacklists()

    def _seed_initial_blacklists(self):
        """Initialize blacklists from config"""
        with self.engine.connect() as conn:
            # Seed coin blacklist
            for address in CONFIG["FILTERS"]["coin_blacklist"]:
                conn.execute(
                    """INSERT INTO blacklist (address, type)
                       VALUES (%s, 'coin')
                       ON CONFLICT (address) DO NOTHING""",
                    (address,)
                )
            
            # Seed dev blacklist
            for address in CONFIG["FILTERS"]["dev_blacklist"]:
                conn.execute(
                    """INSERT INTO blacklist (address, type)
                       VALUES (%s, 'dev')
                       ON CONFLICT (address) DO NOTHING""",
                    (address,)
                )

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all security and quality filters"""
        # Chain whitelist filter
        df = df[df['chain'].isin(CONFIG["FILTERS"]["chain_whitelist"])]
        
        # Liquidity filter
        df = df[df['liquidity'] >= CONFIG["FILTERS"]["min_liquidity"]]
        
        # Age filter
        min_age = datetime.utcnow() - timedelta(days=CONFIG["FILTERS"]["min_age_days"])
        df = df[pd.to_datetime(df['created_at']) < min_age]
        
        # Database blacklist check
        blacklisted_coins = pd.read_sql(
            "SELECT address FROM blacklist WHERE type = 'coin'",
            self.engine
        )['address'].tolist()
        
        blacklisted_devs = pd.read_sql(
            "SELECT address FROM blacklist WHERE type = 'dev'",
            self.engine
        )['address'].tolist()
        
        # Address and symbol checks
        df = df[
            ~df['pair_address'].isin(blacklisted_coins) &
            ~df['base_token_address'].isin(blacklisted_coins) &
            ~df['creator_address'].isin(blacklisted_devs) &
            ~df['base_token_name'].isin(CONFIG["FILTERS"]["coin_blacklist"])
        ]
        
        return df

    def process_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Enhanced data processing with security fields"""
        df = pd.DataFrame(raw_data)[[
            'pairAddress', 'baseToken', 'quoteToken', 'priceUsd',
            'liquidity', 'volume', 'chainId', 'dexId', 'createdAt'
        ]]
        
        processed = pd.DataFrame({
            'pair_address': df['pairAddress'],
            'base_token_name': df['baseToken'].apply(lambda x: x['name']),
            'base_token_address': df['baseToken'].apply(lambda x: x['address']),
            'quote_token_address': df['quoteToken'].apply(lambda x: x['address']),
            'price': pd.to_numeric(df['priceUsd']),
            'liquidity': pd.to_numeric(df['liquidity']),
            'volume_24h': pd.to_numeric(df['volume']['h24']),
            'chain': df['chainId'],
            'exchange': df['dexId'],
            'created_at': pd.to_datetime(df['createdAt'], unit='ms'),
            'timestamp': datetime.utcnow()
        })
        
        # Apply security filters
        processed = self.apply_filters(processed)
        
        return processed

    def detect_anomalies(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Anomaly detection with blacklist awareness"""
        if not new_data.empty:
            features = new_data[['price', 'liquidity', 'volume_24h']]
            features = np.log1p(features)
            
            self.model.fit(self.historical_data)
            anomalies = self.model.predict(features)
            new_data['anomaly_score'] = self.model.decision_function(features)
            return new_data[anomalies == -1]
        return pd.DataFrame()

    def analyze_market_events(self, anomalous_data: pd.DataFrame):
        """Enhanced analysis with blacklist monitoring"""
        for _, row in anomalous_data.iterrows():
            # Check for blacklist pattern matches
            if self._detect_blacklist_pattern(row):
                self._log_event(row, 'BLACKLIST_PATTERN')
            
            # Existing detection logic
            ...

    def _detect_blacklist_pattern(self, row: pd.Series) -> bool:
        """Detect patterns matching known blacklist characteristics"""
        # Check for new addresses similar to blacklisted ones
        similar_coins = pd.read_sql(f"""
            SELECT COUNT(*) FROM blacklist
            WHERE type = 'coin'
            AND similarity(address, '{row['base_token_address']}') > 0.8
            """, self.engine).scalar()
        
        similar_devs = pd.read_sql(f"""
            SELECT COUNT(*) FROM blacklist
            WHERE type = 'dev'
            AND similarity(address, '{row['creator_address']}') > 0.8
            """, self.engine).scalar()
        
        return similar_coins > 0 or similar_devs > 0

    def add_to_blacklist(self, address: str, list_type: str, reason: str):
        """Programmatically add entries to blacklist"""
        with self.engine.connect() as conn:
            conn.execute(
                """INSERT INTO blacklist (address, type, reason)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (address) DO UPDATE SET reason = EXCLUDED.reason""",
                (address, list_type, reason)
            )

    def run(self):
        """Enhanced main loop with filtering"""
        while True:
            try:
                raw_data = self.fetch_pair_data()
                processed_data = self.process_data(raw_data)
                
                if not processed_data.empty:
                    anomalies = self.detect_anomalies(processed_data)
                    self.analyze_market_events(anomalies)
                    
                    processed_data.to_sql(
                        'pairs', self.engine, 
                        if_exists='append', index=False
                    )
                    
                    self.historical_data = pd.concat(
                        [self.historical_data, processed_data]
                    ).tail(100000)
                
                # Update blacklists periodically
                self._refresh_blacklists()
                time.sleep(60)  # Add sleep between iterations

            except Exception as e:
                print(f"Runtime error: {e}")

    def _refresh_blacklists(self):
        """Refresh blacklists from external sources"""
        # Example: Sync with community-maintained blacklists
        try:
            response = requests.get("https://api.gopluslabs.io/api/v1/token_security/1")
            data = response.json()
            for token in data['tokens']:
                if token['is_honeypot']:
                    self.add_to_blacklist(
                        token['contract_address'], 
                        'coin', 
                        'Automated honeypot detection'
                    )
        except Exception as e:
            print(f"Blacklist refresh failed: {e}")

# Example usage with blacklist management
if __name__ == "__main__":
    bot = EnhancedDexScreenerBot()
    
    # Manually add suspicious entry
    bot.add_to_blacklist(
        "0xNEW...SCAM", 
        "dev", 
        "Suspicious deployment pattern"
    )
    
    bot.run()
    