import requests
import time
import statistics
from collections import defaultdict

BASE_URL = "http://127.0.0.1:5000"

TEST_URLS = {
    "whitelist":    ["https://google.com", "https://roadmap.sh"] * 5,
    "ml_no_gsb":    ["https://sltb.eseat.lk/", "https://worldofpcgames.com/"] * 5,
    "malicious":    ["http://g00gle.com/login", "http://phishing.ru/"] * 5,
}

results = defaultdict(list)

for category, urls in TEST_URLS.items():
    for url in urls:
        start = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/predict", json={"url": url})
        elapsed = (time.perf_counter() - start) * 1000
        results[category].append(elapsed)

for category, latencies in results.items():
    print(f"\n--- {category} ---")
    print(f"count : {len(latencies)}")
    print(f"min   : {min(latencies):.1f} ms")
    print(f"max   : {max(latencies):.1f} ms")
    print(f"mean  : {statistics.mean(latencies):.1f} ms")
    print(f"median: {statistics.median(latencies):.1f} ms")