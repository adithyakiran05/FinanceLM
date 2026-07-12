# pyrefly: ignore [missing-import]
from sec_edgar_downloader import Downloader
import pandas as pd
import time

import requests

# Company information
company_name = "OpenAI FinanceLM Project"
email = "kiranadithya105@gmail.com"

# Initialize Downloader with company name, email, and download folder
dl = Downloader(company_name, email, "data/raw/sec")

import io

print("Fetching S&P 500 tickers from Wikipedia...")
try:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    table = pd.read_html(io.StringIO(html))
    sp500_tickers = table[0]['Symbol'].tolist()
    # Handle SEC ticker formatting (e.g. BRK.B -> BRK-B)
    companies = [ticker.replace('.', '-') for ticker in sp500_tickers]
except Exception as e:
    print("Failed to fetch S&P 500 tickers due to an error.")
    companies = ["AAPL", "MSFT", "JPM"] # Fallback

forms = ["10-K", "10-Q", "8-K"]
limit_per_form = 5

for ticker in companies:
    for form in forms:
        print(f"Downloading {form} for {ticker}...")
        try:
            dl.get(
                form,
                ticker,
                limit=limit_per_form,
                download_details=True
            )
        except Exception as e:
            print(f"Failed to download {form} for {ticker}: {e}")
            
    # Small sleep to be respectful to the SEC servers
    time.sleep(0.1)