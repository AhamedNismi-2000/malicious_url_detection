import requests
import time
import statistics

BASE_URL = "http://localhost:5000"   # swap to your Render URL later
# BASE_URL = "https://your-app.onrender.com"

TEST_URLS = [
    "https://google.com",
    "http://g00gle.com/login",
    "https://sltb.eseat.lk/",
    "http://phishing.ru/",
    "https://roadmap.sh",
    "https://worldofpcgames.com/",
] * 5   # repeat to get more samples — 30 requests total

latencies = []

for url in TEST_URLS:
    start = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/predict", json={"url": url})
    elapsed = time.perf_counter() - start
    latencies.append(elapsed)
    print(f"{elapsed*1000:6.1f} ms  [{resp.status_code}]  {url}")

print("\n--- Summary ---")
print(f"count : {len(latencies)}")
print(f"min   : {min(latencies)*1000:.1f} ms")
print(f"max   : {max(latencies)*1000:.1f} ms")
print(f"mean  : {statistics.mean(latencies)*1000:.1f} ms")
print(f"median: {statistics.median(latencies)*1000:.1f} ms")
if len(latencies) > 1:
    print(f"stdev : {statistics.stdev(latencies)*1000:.1f} ms")