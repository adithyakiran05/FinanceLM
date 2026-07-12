import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("data/raw/fed")

def download_fed_reports():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # FOMC Meeting Calendars and Statements
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    print(f"Fetching FOMC meeting statements from {url}...")
    
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Find links to FOMC statements
        links = soup.find_all("a", href=True)
        statement_links = [a["href"] for a in links if "monetarypolicy/fomcminutes" in a["href"] or "fomcprojtabl" in a["href"]]
        
        # Convert relative links to absolute
        base_url = "https://www.federalreserve.gov"
        absolute_links = [base_url + link if link.startswith("/") else link for link in statement_links]
        absolute_links = list(set(absolute_links))
        
        print(f"Found {len(absolute_links)} FOMC document links.")
        
        for link in tqdm(absolute_links[:20], desc="Scraping FOMC documents"):
            try:
                doc_res = requests.get(link, headers=headers)
                if doc_res.status_code == 200:
                    doc_soup = BeautifulSoup(doc_res.text, "html.parser")
                    # Usually the main content is in a div with id 'article'
                    content = doc_soup.find("div", id="article")
                    if content:
                        text = content.get_text(separator="\n\n", strip=True)
                        filename = link.split("/")[-1].replace(".htm", ".txt")
                        out_path = OUTPUT_DIR / filename
                        out_path.write_text(text, encoding="utf8", errors="ignore")
            except Exception as e:
                pass
                
    except Exception as e:
        print(f"Error scraping Fed reports: {e}")

if __name__ == "__main__":
    download_fed_reports()
