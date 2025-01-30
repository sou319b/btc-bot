import time
import json
import sqlite3
from datetime import datetime, timedelta
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

# Load configuration from config.json
def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

# Save configuration back to config.json (for dynamic updates)
def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, indent=4, fp=f)

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('token_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_name TEXT,
            token_address TEXT,
            pair_created_at TEXT,
            initial_price REAL,
            current_price REAL,
            liquidity REAL,
            action TEXT,
            detected_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Save token data to SQLite
def save_token_data(token_name, token_address, pair_created_at, initial_price, current_price, liquidity, action):
    conn = sqlite3.connect('token_data.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO tokens (token_name, token_address, pair_created_at, initial_price, current_price, liquidity, action, detected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    ''', (token_name, token_address, pair_created_at, initial_price, current_price, liquidity, action))
    conn.commit()
    conn.close()

# Fetch token data from Pocker Universe API
def fetch_pocker_universe_data(token_address):
    api_url = f"https://api.pockeruniverse.com/v1/token/{token_address}"  # Replace with actual API endpoint
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()  # Assuming the API returns JSON data
        else:
            print(f"Error fetching data from Pocker Universe API: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to Pocker Universe API: {e}")
        return None

# Detect fake volume using Pocker Universe API data
def detect_fake_volume(token_address, config):
    pocker_data = fetch_pocker_universe_data(token_address)
    if not pocker_data:
        return False

    # Example criteria for fake volume detection
    real_volume = pocker_data.get('real_volume', 0)  # Real trading volume
    reported_volume = pocker_data.get('reported_volume', 0)  # Reported trading volume
    volume_ratio = real_volume / reported_volume if reported_volume > 0 else 0

    # If the ratio is below a threshold, consider it fake volume
    fake_volume_threshold = config['filters'].get('fake_volume_ratio_threshold', 0.5)
    if volume_ratio < fake_volume_threshold:
        print(f"Token {token_address} has fake volume (Ratio: {volume_ratio:.2f}).")
        return True
    return False

# Analyze token data for rug pulls or pumps
def analyze_token_data(token_name, token_address, pair_created_at, initial_price, current_price, liquidity, dev_address, config):
    # Check if token is blacklisted
    if token_address.lower() in [addr.lower() for addr in config['coin_blacklist']]:
        print(f"Token {token_name} ({token_address}) is blacklisted.")
        return 'Blacklisted'

    # Check if developer is blacklisted
    if dev_address.lower() in [addr.lower() for addr in config['dev_blacklist']]:
        print(f"Developer of {token_name} ({token_address}) is blacklisted.")
        return 'Blacklisted'

    # Apply filters
    if liquidity < config['filters']['min_liquidity']:
        print(f"Token {token_name} ({token_address}) has insufficient liquidity.")
        return 'Filtered'

    # Check for fake volume
    if detect_fake_volume(token_address, config):
        # Add token to coin blacklist dynamically
        if token_address.lower() not in [addr.lower() for addr in config['coin_blacklist']]:
            config['coin_blacklist'].append(token_address.lower())
            save_config(config)
        return 'Blacklisted'

    # Calculate price change percentage
    price_change = ((current_price - initial_price) / initial_price) * 100 if initial_price > 0 else 0
    if price_change > config['filters']['max_price_change']:
        print(f"Token {token_name} ({token_address}) exceeded max price change threshold.")
        return 'Filtered'

    # Check creation time
    pair_created_time = datetime.strptime(pair_created_at, '%Y-%m-%d %H:%M:%S')
    if datetime.now() - pair_created_time < timedelta(hours=config['filters']['min_creation_time_hours']):
        print(f"Token {token_name} ({token_address}) is too new.")
        return 'Filtered'

    # Detect pump or rug pull
    liquidity_drop_threshold = 0.7  # 70% drop in liquidity
    liquidity_change = liquidity / initial_price if initial_price > 0 else 0

    if price_change > 50:  # Example threshold for pump
        save_token_data(token_name, token_address, pair_created_at, initial_price, current_price, liquidity, 'Pump')
        return 'Pump'
    elif liquidity_change < liquidity_drop_threshold:
        save_token_data(token_name, token_address, pair_created_at, initial_price, current_price, liquidity, 'Rug Pull')
        return 'Rug Pull'
    else:
        save_token_data(token_name, token_address, pair_created_at, initial_price, current_price, liquidity, 'Normal')
        return 'Normal'

# Scrape data from DexScreener
def scrape_dexscreener(config):
    # Set up Selenium WebDriver
    service = Service('/path/to/chromedriver')  # Replace with your chromedriver path
    driver = webdriver.Chrome(service=service)

    # Open DexScreener
    driver.get('https://dexscreener.com')

    while True:
        try:
            # Wait for page to load
            time.sleep(5)

            # Parse the page source with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Extract token data (this will depend on the actual HTML structure of DexScreener)
            tokens = soup.find_all('div', class_='token-item')  # Adjust this selector based on DexScreener's structure

            for token in tokens:
                token_name = token.find('span', class_='token-name').text.strip()
                token_address = token.find('a', class_='token-address')['href'].split('/')[-1]
                pair_created_at = token.find('span', class_='pair-created-at').text.strip()
                initial_price = float(token.find('span', class_='initial-price').text.strip().replace('$', ''))
                current_price = float(token.find('span', class_='current-price').text.strip().replace('$', ''))
                liquidity = float(token.find('span', class_='liquidity').text.strip().replace('$', '').replace(',', ''))
                dev_address = token.find('span', class_='dev-address').text.strip()  # Developer wallet address

                # Analyze token data
                action = analyze_token_data(token_name, token_address, pair_created_at, initial_price, current_price, liquidity, dev_address, config)
                print(f"Token: {token_name}, Action: {action}")

            # Refresh the page periodically
            driver.refresh()

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

    driver.quit()

if __name__ == "__main__":
    config = load_config()
    init_db()
    scrape_dexscreener(config)