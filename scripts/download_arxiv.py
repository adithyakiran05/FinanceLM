import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("data/raw/arxiv")

def download_arxiv_qfin():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # We query the official arXiv API for Quantitative Finance (q-fin.*)
    # We'll pull the last 1000 research papers' abstracts
    url = "https://export.arxiv.org/api/query?search_query=cat:q-fin.*&start=0&max_results=1000&sortBy=submittedDate&sortOrder=descending"
    
    print(f"Fetching 1,000 Quantitative Finance abstracts from arXiv API...")
    
    try:
        res = requests.get(url)
        res.raise_for_status()
        
        # Parse the Atom XML feed
        soup = BeautifulSoup(res.text, "xml")
        entries = soup.find_all("entry")
        
        print(f"Found {len(entries)} academic papers.")
        
        for i, entry in enumerate(tqdm(entries, desc="Saving arXiv abstracts")):
            title = entry.find("title").text.strip() if entry.find("title") else "Untitled"
            summary = entry.find("summary").text.strip() if entry.find("summary") else ""
            
            if summary:
                content = f"Title: {title}\n\nAbstract:\n{summary}"
                
                # Sanitize filename
                safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                filename = f"arxiv_{i}_{safe_title[:30]}.txt"
                
                out_path = OUTPUT_DIR / filename
                out_path.write_text(content, encoding="utf8", errors="ignore")
                
        print("Successfully downloaded arXiv quantitative finance abstracts.")
                
    except Exception as e:
        print(f"Error querying arXiv API: {e}")

if __name__ == "__main__":
    download_arxiv_qfin()
