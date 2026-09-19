import sys
sys.path.insert(0, 'app')
from model_loader import classifier

urls = [
    # Should hit ML model
    "http://secure-account-verify-2024.net/login",
    "http://banking-update-required.online/confirm",
    "http://your-account-has-been-suspended.com/restore",
    "http://winner-prize-claim-now.site/reward",
    "http://payment-failed-update-now.online/billing",
    "http://document-shared-with-you.online/view",
    "http://security-check-required-2024.net/verify",
    "http://unusual-activity-detected.site/confirm",
    # Should hit leet override
    "http://g00gle.com/login",
    "http://paypa1.tk/secure",
    "http://amaz0n-login.xyz/verify",
    # Should hit suspicious domain label
    "http://phishing.ru/",
    "http://malware.tk/payload",
    # Should hit IP override
    "http://185.220.101.45/steal/credentials",
    # Should be BENIGN (clean HTTP)
    "http://example.com",
    "http://neverssl.com",
    "http://httpbin.org",
    # Should be whitelist
    "https://google.com",
    "https://github.com",
]

print("=" * 75)
print(f"{'SOURCE':<25} {'PREDICTION':<12} {'CONF':>6}  URL")
print("=" * 75)

ml_count       = 0
whitelist_count = 0
gsb_count      = 0
override_count = 0

for url in urls:
    r = classifier.predict_url(url)
    source     = r.get("source", "unknown")
    prediction = r.get("prediction", "UNKNOWN")
    confidence = r.get("confidence", 0)
    domain_age = r.get("domain_age", "")

    # Count layers
    if source == "whitelist":
        whitelist_count += 1
    elif source == "google_safe_browsing":
        gsb_count += 1
    elif domain_age in ("skipped_leet", "skipped_ip",
                        "skipped_brand_impersonation",
                        "skipped_suspicious_domain"):
        override_count += 1
    else:
        ml_count += 1

    # Display
    short_url = url if len(url) <= 45 else url[:42] + "..."
    age_note  = f" [{domain_age}]" if domain_age else ""
    print(f"{source:<25} {prediction:<12} {confidence:>5.1f}%  {short_url}{age_note}")

print("=" * 75)
print(f"\nLayer breakdown:")
print(f"  Whitelist         : {whitelist_count}")
print(f"  Google Safe Browse: {gsb_count}")
print(f"  Rule override     : {override_count}")
print(f"  ML model          : {ml_count}")
print(f"  Total             : {len(urls)}")