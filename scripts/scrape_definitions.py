import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("data/raw/definitions")

def scrape_wikipedia_definitions():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Scrape Wikipedia's Glossary of Economics which is highly robust and won't block us (HTTP 403)
    url = "https://en.wikipedia.org/wiki/Glossary_of_economics"
    print(f"Fetching Economic/Financial definitions from {url}...")
    
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        
        # In Wikipedia glossaries, terms are usually inside <dt> (term) and <dd> (definition)
        terms = soup.find_all("dt")
        descriptions = soup.find_all("dd")
        
        print(f"Found {len(terms)} term definitions to scrape.")
        
        limit = min(len(terms), len(descriptions))
        for i in tqdm(range(limit), desc="Scraping definitions"):
            term_name = terms[i].get_text(strip=True).replace("/", "-")
            definition_text = descriptions[i].get_text(separator="\n", strip=True)
            
            # Sanitize filename
            safe_term = "".join([c for c in term_name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            if safe_term and definition_text:
                out_path = OUTPUT_DIR / f"{safe_term}.txt"
                out_path.write_text(definition_text, encoding="utf8", errors="ignore")
                
        print("Definitions scraping complete!")
            
    except Exception as e:
        print(f"Error scraping definitions: {e}")

if __name__ == "__main__":
    scrape_wikipedia_definitions()
