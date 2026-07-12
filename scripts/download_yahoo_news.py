import os
from pathlib import Path
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    print("Please install yfinance: pip install yfinance")
    exit(1)

OUTPUT_DIR = Path("data/raw/news")

def download_yahoo_news():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # We will fetch news for some major S&P 500 components to get market language
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "GS"]
    
    print("Fetching recent Yahoo Finance news articles...")
    
    for ticker in tqdm(tickers, desc="Fetching news"):
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news
            
            for i, item in enumerate(news_items):
                title = item.get("title", "Untitled").replace("/", "-")
                publisher = item.get("publisher", "Unknown")
                link = item.get("link", "")
                
                # In a full pipeline, we would use trafilatura to scrape the `link` text.
                # For now, we save the title and metadata which contains strong market language.
                content = f"Ticker: {ticker}\nTitle: {title}\nPublisher: {publisher}\nLink: {link}\n\n"
                
                # Sanitize filename
                safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                filename = f"{ticker}_news_{i}_{safe_title[:30]}.txt"
                
                out_path = OUTPUT_DIR / filename
                out_path.write_text(content, encoding="utf8", errors="ignore")
                
        except Exception as e:
            print(f"Failed to fetch news for {ticker}: {e}")

if __name__ == "__main__":
    download_yahoo_news()
