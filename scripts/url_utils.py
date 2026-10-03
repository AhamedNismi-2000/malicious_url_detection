#!/usr/bin/env python3
"""
url_utils.py  -  ONE definition of "what a URL looks like" for the whole project
---------------------------------------------------------------------------------
Used by preprocessing.py and split.py now, and by feature_extraction.py and
model_loader.py later, so training and serving always see identical strings.

canonicalize(raw) -> (fields, "")  or  (None, reason)

Canonical form (the string the model should see)
    - scheme removed (it is stored separately: "https", "http" or "" if unknown)
    - leading "www." removed
    - host lower-cased; path / query keep their ORIGINAL case
    - "&amp;" decoded
    - trailing "/" removed when the URL is a bare domain
    - userinfo (user@host) and ports are kept: they are real phishing signals

Fields returned
    url, scheme, host, reg_domain, suffix, group_key, has_path, has_query, is_ip,
    shared_platform (True for github.io / web.app style hosts, roots included)

group_key = the unit that must never be split across train/val/test
    IP host              -> the IP
    tenant on a shared   -> the tenant, e.g. "evil.github.io", "x.web.app"
    platform (PSL private section)
    big first-party site -> the full hostname, e.g. "docs.google.com"
    everything else      -> the registered domain, e.g. "paypal.com"

Self-test:  python scripts/url_utils.py
"""

import ipaddress
import re

import tldextract

MAX_URL_LEN = 2048

_EXT_ICANN = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)
_EXT_PRIV  = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None,
                                   include_psl_private_domains=True)

# Registered domains whose hostnames are independent content owners even though
# the public-suffix list does not say so. Grouping is per hostname for these.
HOSTNAME_GROUPED_DOMAINS = {
    "google.com", "googleapis.com", "googleusercontent.com", "gstatic.com",
    "adobe.com", "dropbox.com", "office.com", "live.com", "sharepoint.com",
    "microsoft.com", "azure.com", "weebly.com", "weeblysite.com",
    "wixsite.com", "wixstudio.com", "wix.com", "squarespace.com",
    "godaddysites.com", "square.site", "pantheonsite.io", "xsph.ru",
    "ipfs.io", "dweb.link", "duckdns.org", "r2.dev", "hsforms.com",
    "jotform.com", "typeform.com", "campaign-archive.com", "mailchimp.com",
    "notion.so", "canva.com", "linktr.ee", "linkin.bio", "ead.me",
    "wordpress.com", "tumblr.com", "medium.com", "webflow.io", "framer.app",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "cutt.ly", "rebrand.ly", "t.ly", "tiny.cc", "shorturl.at",
    "qrco.de", "q-r.to", "l.ead.me",
}

_SCHEME   = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_MD_LINK  = re.compile(r"\((https?://[^)]+)\)", re.IGNORECASE)
_DELIM    = re.compile(r"[/?#\\]")
_HOST_OK  = re.compile(r"[\w.\-]+")
_IPV4     = re.compile(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})")
_NUMERIC  = re.compile(r"\d{6,}|0x[0-9a-f]+")          # obfuscated IPs


def _registered(ext) -> str:
    """tldextract renamed registered_domain in newer releases."""
    if hasattr(ext, "top_domain_under_public_suffix"):
        return ext.top_domain_under_public_suffix
    return ext.registered_domain


def _is_ip_host(host: str) -> bool:
    if ":" in host:
        try:
            ipaddress.IPv6Address(host)
            return True
        except ValueError:
            return False
    m = _IPV4.fullmatch(host)
    if m:
        return all(int(g) <= 255 for g in m.groups())
    return bool(_NUMERIC.fullmatch(host))


def canonicalize(raw) -> tuple:
    """Return (fields_dict, "") or (None, reason_code).
    Reason codes: empty, bad_scheme, too_long, invalid_host, no_public_suffix."""
    if raw is None:
        return None, "empty"
    s = str(raw)
    for _ in range(2):
        s = s.replace("&amp;", "&")
    s = re.sub(r"[\s\x00-\x1f\x7f]+", "", s)
    if not s:
        return None, "empty"

    m = _MD_LINK.search(s)
    if m:
        s = m.group(1)

    scheme = ""
    m = _SCHEME.match(s)
    if m:
        scheme = m.group(1).lower()
        if scheme not in ("http", "https"):
            return None, "bad_scheme"
        s = s[m.end():]
    elif s.startswith("//"):
        s = s[2:]
    if not s:
        return None, "empty"

    cut       = _DELIM.search(s)
    authority = s[:cut.start()] if cut else s
    rest      = s[cut.start():] if cut else ""
    if rest.startswith("\\"):
        rest = "/" + rest[1:]

    userinfo, at, hostport = authority.rpartition("@")
    hostport = hostport.lower()
    port = ""
    if hostport.startswith("["):
        end  = hostport.find("]")
        host = hostport[1:end] if end > 0 else ""
        port = hostport[end + 1:].lstrip(":") if end > 0 else ""
    else:
        host, _, port = hostport.partition(":")
    if port and not port.isdigit():
        port = ""
    host = host.rstrip(".")
    if host.startswith("www.") and host.count(".") >= 2:
        host = host[4:]
    if not host:
        return None, "invalid_host"

    is_ip = _is_ip_host(host)
    if not is_ip:
        if (not _HOST_OK.fullmatch(host) or ".." in host or host.startswith(".")
                or host.startswith("-")):
            return None, "invalid_host"
        ext = _EXT_ICANN(host)
        if not ext.suffix or not ext.domain:
            return None, "no_public_suffix"
        reg, suffix = _registered(ext), ext.suffix
        priv = _EXT_PRIV(host)
        shared_platform = bool(priv.suffix and priv.suffix != ext.suffix)
        if shared_platform:                                       # github.io, web.app...
            group_key = _registered(priv) or host                 # tenant, or the root itself
        elif reg in HOSTNAME_GROUPED_DOMAINS:
            group_key = host
        else:
            group_key = reg
    else:
        reg, suffix, group_key, shared_platform = host, "", host, False

    host_part = f"[{host}]" if ":" in host else host
    canon_auth = (userinfo + "@" if at else "") + host_part + (":" + port if port else "")
    if rest in ("/", ""):
        rest = ""
    url = canon_auth + rest
    if len(url) > MAX_URL_LEN:
        return None, "too_long"

    no_frag = rest.split("#", 1)[0]
    path    = no_frag.split("?", 1)[0]
    return dict(
        url=url, scheme=scheme, host=host, reg_domain=reg, suffix=suffix,
        group_key=group_key, has_path=bool(path.strip("/")),
        has_query="?" in no_frag, is_ip=is_ip, shared_platform=shared_platform,
    ), ""


def near_dup_key(url: str) -> str:
    """Key for near-duplicate detection (case-insensitive, no trailing slash)."""
    return str(url).lower().rstrip("/")


# ───────────────────────── self-test ─────────────────────────
def _self_test() -> None:
    cases = [
        # raw, expected url (None = dropped), expected extras
        ("https://www.Example.com/Path?Q=1", "example.com/Path?Q=1", {"scheme": "https"}),
        ("example.com/",                      "example.com",          {"scheme": "", "has_path": False}),
        ("HTTP://EXAMPLE.COM:8080/a",         "example.com:8080/a",   {"scheme": "http"}),
        ("http://1.2.3.4/x",                  "1.2.3.4/x",            {"is_ip": True, "group_key": "1.2.3.4"}),
        ("http://45.148.120.78:8080/bin.sh",  "45.148.120.78:8080/bin.sh", {"is_ip": True}),
        ("https://[2001:db8::1]/x",           "[2001:db8::1]/x",      {"is_ip": True}),
        ("a.com/p?x=1&amp;y=2",               "a.com/p?x=1&y=2",      {"has_query": True}),
        ("https://foo.github.io/x",           "foo.github.io/x",      {"group_key": "foo.github.io"}),
        ("https://docs.google.com/forms/x",   "docs.google.com/forms/x", {"group_key": "docs.google.com"}),
        ("https://paypal.com/signin",         "paypal.com/signin",    {"group_key": "paypal.com"}),
        ("google.com@evil.tk/login",          "google.com@evil.tk/login", {"reg_domain": "evil.tk"}),
        ("https://rexyune-5ca79.web.app/",    "rexyune-5ca79.web.app", {"group_key": "rexyune-5ca79.web.app"}),
        ("github.io",                         "github.io",            {"group_key": "github.io", "shared_platform": True}),
        ("https://weebly.com",                "weebly.com",           {"group_key": "weebly.com"}),
        ("ftp://x.com/a",                     None, {}),
        ("http://well$fargo-login.online/",   None, {}),
        ("http://localhost/x",                None, {}),
        ("   ",                               None, {}),
        ("https://xn--69-6kcdljc7dbzjr.xn--mk1bu44c", "xn--69-6kcdljc7dbzjr.xn--mk1bu44c", {}),
    ]
    bad = 0
    for raw, want, extra in cases:
        got, reason = canonicalize(raw)
        gurl = got["url"] if got else None
        ok = gurl == want and all(got.get(k) == v for k, v in extra.items()) if got else (want is None)
        bad += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {raw!r:<48} -> {gurl!r} {reason}"
              + (f"  {extra} vs {{{', '.join(f'{k}: {got.get(k)!r}' for k in extra)}}}" if (got and not ok) else ""))
    print(f"\n{len(cases) - bad}/{len(cases)} passed")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    _self_test()