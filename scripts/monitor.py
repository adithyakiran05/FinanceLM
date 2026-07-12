import time
from pathlib import Path
from tqdm import tqdm

def monitor_progress():
    raw_dir = Path("data/raw/sec/sec-edgar-filings")
    
    # Approx: 500 companies * 3 forms * 5 filings each = 7500 expected
    expected_total = 7500 
    
    print("Monitoring SEC downloads (Press Ctrl+C to stop)...")
    
    # Get initial count before starting tqdm so speed/ETA is accurate
    initial_count = sum(1 for _ in raw_dir.glob("*/*/*")) if raw_dir.exists() else 0
    
    with tqdm(total=expected_total, initial=initial_count, unit="filing") as pbar:
        last_count = initial_count
        while True:
            if raw_dir.exists():
                # Count the number of downloaded filings (each gets its own folder)
                current_count = sum(1 for _ in raw_dir.glob("*/*/*"))
                
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
            
            time.sleep(2)

if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
