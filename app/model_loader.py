"""
model_loader.py
---------------
Loads model artefacts and exposes three public methods:

  predict_url(url)                   -> dict
  predict_batch(urls)                -> list[dict]
  explain_url(url, num_features=10)  -> dict   (prediction + LIME explanation)

The URLClassifier is a singleton; import `classifier` directly:

  from model_loader import classifier
"""

import json
import os
import re
import sys
import threading
import warnings
from typing import Optional

import joblib
import numpy as np

# ── Locate project root and add scripts/ to path ─────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_APP_DIR, ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from feature_extraction import extract_heuristic_features, preprocess_url_for_nlp

# ── Path constants ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_ROOT, "models")
DATA_DIR   = os.path.join(_ROOT, "data")

# ── Feature names (must match training order) ─────────────────────────────────
HEURISTIC_FEATURES: list[str] = [
    "url_len", "path_len", "num_dots", "path_dots", "num_hyphens",
    "num_underscores", "num_at", "num_qmark", "num_equal", "num_amp",
    "num_percent", "num_digits", "num_letters", "num_subdirs", "num_frag",
    "num_special", "num_repeating", "num_upper", "num_non_ascii",
    "num_slashes", "num_params", "ratio_digits", "ratio_letters",
    "url_entropy", "ip_flag", "subdomain_parts", "has_multi_subdomain",
    "tld_len", "risky_tld", "https_flag", "shortened", "sus_words",
    "brand_mismatch", "puny", "susp_ext", "suspicious_port",
    "max_consonants", "max_vowels", "max_digits",
    "leet_speak_score", "homoglyph_suspicious", "encoding_ratio",
    "punycode_suspicious", "subdomain_spam_score", "visual_brand_similarity",
    "brand_in_domain", "leet_in_domain", "brand_hyphen_suspicious",
]

FEATURE_NAMES: list[str] = (
    HEURISTIC_FEATURES
    + [f"char_{i}" for i in range(300)]
    + [f"word_{i}" for i in range(200)]
)

# Boolean/flag features — LIME treats these as categorical
_CATEGORICAL_FEATURE_NAMES: list[str] = [
    "ip_flag", "has_multi_subdomain", "risky_tld", "https_flag",
    "shortened", "sus_words", "brand_mismatch", "puny", "susp_ext",
    "suspicious_port", "brand_in_domain", "leet_in_domain",
    "brand_hyphen_suspicious",
]

# 54 trusted domains — instant BENIGN without hitting the model
WHITELIST: frozenset[str] = frozenset({
    "163.com",
    "1rx.io",
    "2gis.com",
    "2mdn.net",
    "360.cn",
    "360safe.com",
    "360yield.com",
    "3gppnetwork.org",
    "3lift.com",
    "4dex.io",
    "a-mo.net",
    "a-msedge.net",
    "a2z.com",
    "aaplimg.com",
    "aboutads.info",
    "abovedomains.com",
    "academia.edu",
    "accuweather.com",
    "addtoany.com",
    "adform.net",
    "adgrx.com",
    "adguard-vpn.online",
    "adjust.com",
    "adobe.com",
    "adobe.io",
    "adobe.net",
    "adobedc.net",
    "adobedtm.com",
    "adriver.ru",
    "adsrvr.org",
    "adtrafficquality.google",
    "afafb.com",
    "afilias-nst.info",
    "afilias-nst.org",
    "afternic.com",
    "agkn.com",
    "agoda.com",
    "agora.io",
    "airbnb.com",
    "aiv-delivery.net",
    "akahost.net",
    "akam.net",
    "akamaiedge.net",
    "akamaihd.net",
    "akamaitech.net",
    "alibaba.com",
    "alibabadns.com",
    "alicdn.com",
    "alidns.com",
    "aliexpress.com",
    "alipaydns.com",
    "aliyun.com",
    "aliyuncs.com",
    "aliyuncsslbintl.com",
    "allaboutcookies.org",
    "allawnos.com",
    "allegro.pl",
    "amagi.tv",
    "amazon.ca",
    "amazon.co.jp",
    "amazon.co.uk",
    "amazon.com",
    "amazon.com.au",
    "amazon.com.br",
    "amazon.com.mx",
    "amazon.de",
    "amazon.dev",
    "amazon.es",
    "amazon.fr",
    "amazon.in",
    "amazon.it",
    "amazonalexa.com",
    "amazonaws.com",
    "amazontrust.com",
    "amazonvideo.com",
    "ameblo.jp",
    "amemv.com",
    "ampproject.org",
    "amzn.to",
    "ancestry.com",
    "android.com",
    "anthropic.com",
    "anydesk.com",
    "aol.com",
    "apache.org",
    "apnews.com",
    "app-analytics-services-att.com",
    "app-analytics-services.com",
    "app-measurement.com",
    "appcenter.ms",
    "apple-dns.net",
    "apple.com",
    "applovin.com",
    "appsflyer.com",
    "appsflyersdk.com",
    "appspot.com",
    "arcgis.com",
    "archive.org",
    "arubanetworks.com",
    "arxiv.org",
    "asus.com",
    "atlassian.com",
    "atlassian.net",
    "att.net",
    "autodesk.com",
    "avast.com",
    "avcdn.net",
    "avito.ru",
    "avsxappcaptiveportal.com",
    "aws.amazon.com",
    "aws.dev",
    "awsglobalaccelerator.com",
    "awswaf.com",
    "ax-msedge.net",
    "azure-devices.net",
    "azure-dns.com",
    "azure.com",
    "azure.microsoft.com",
    "azureedge.net",
    "azurefd.net",
    "azurewebsites.net",
    "b-cdn.net",
    "b-msedge.net",
    "baidu.com",
    "bamgrid.com",
    "bandcamp.com",
    "bankofamerica.com",
    "bbc.co.uk",
    "bbc.com",
    "bdydns.com",
    "behance.net",
    "beian.gov.cn",
    "berkeley.edu",
    "bestbuy.com",
    "beyondwickedmapping.org",
    "biblegateway.com",
    "bidmachine.io",
    "bidr.io",
    "bidswitch.net",
    "bild.de",
    "bilibili.com",
    "binance.com",
    "bing.com",
    "bitrix24.ru",
    "blackberry.com",
    "blackhub.team",
    "blogger.com",
    "blogspot.com",
    "bloomberg.com",
    "bluehost.com",
    "bol.com",
    "booking.com",
    "box.com",
    "branch.io",
    "brave.com",
    "braze.com",
    "britannica.com",
    "browser-intake-datadoghq.com",
    "bsky.app",
    "btloader.com",
    "bugsnag.com",
    "bunnyinfra.net",
    "businessinsider.com",
    "bx-msedge.net",
    "bytedns1.com",
    "bytefcdn-oversea.com",
    "bytefcdn-ttpus.com",
    "bytefcdn.com",
    "byteglb.com",
    "byteoversea.net",
    "ca.gov",
    "caixa.gov.br",
    "calendly.com",
    "cambridge.org",
    "canva.com",
    "capcut.com",
    "capcutapi.com",
    "cbsnews.com",
    "cdc.gov",
    "cdn-apple.com",
    "cdn-vk.ru",
    "cdn20.com",
    "cdn77.org",
    "cdnbuild.net",
    "cdngslb.com",
    "cdnhwc1.com",
    "cdnhwc2.com",
    "cdninstagram.com",
    "cdnvideo.ru",
    "change.org",
    "character.ai",
    "chatgpt.com",
    "chaturbate.com",
    "checkpoint.com",
    "chess.com",
    "chinamobile.com",
    "ci-servers.net",
    "ci-servers.org",
    "cisco.com",
    "clarity.ms",
    "claude.ai",
    "clever.com",
    "cloud.google.com",
    "cloud.microsoft",
    "cloudflare-dns.com",
    "cloudflare.com",
    "cloudflare.net",
    "cloudflareinsights.com",
    "cloudinary.com",
    "cloudns.net",
    "cloudsink.net",
    "cmediahub.ru",
    "cnbc.com",
    "cnet.com",
    "cnn.com",
    "columbia.edu",
    "comcast.com",
    "comcast.net",
    "consultant.ru",
    "contentsquare.net",
    "conviva.com",
    "cookiedatabase.org",
    "cookielaw.org",
    "cornell.edu",
    "corriere.it",
    "costco.com",
    "coupang.com",
    "coursera.org",
    "cpanel.net",
    "crashlytics.com",
    "creativecdn.com",
    "creativecommons.org",
    "criteo.net",
    "crpt.ru",
    "crwdcntrl.net",
    "cursor.sh",
    "dailymail.co.uk",
    "dailymotion.com",
    "datadoghq.com",
    "daum.net",
    "dbankcloud.com",
    "dbankcloud.ru",
    "ddnss.de",
    "debian.org",
    "deepl.com",
    "deere.com",
    "dell.com",
    "deloitte.com",
    "demdex.net",
    "deviantart.com",
    "digicert.com",
    "digitalocean.com",
    "digitaloceanspaces.com",
    "discogs.com",
    "discord.com",
    "discord.gg",
    "discord.media",
    "discordapp.com",
    "disneyplus.com",
    "disqus.com",
    "dns-parking.com",
    "dns.google",
    "dnsmadeeasy.com",
    "dnsowl.com",
    "dnspod.net",
    "docker.com",
    "docker.io",
    "docomo.ne.jp",
    "doi.org",
    "domaincontrol.com",
    "dotaplabs.net",
    "dotomi.com",
    "doubleverify.com",
    "douyincdn.com",
    "dreamhost.com",
    "drom.ru",
    "dropbox.com",
    "dropcatch.com",
    "dtkn.ru",
    "dual-s-msedge.net",
    "duckdns.org",
    "duckduckgo.com",
    "duolingo.com",
    "dv.tech",
    "dynatrace.com",
    "dyndns.org",
    "dzen.ru",
    "e2ro.com",
    "ea.com",
    "easebar.com",
    "ebay.co.uk",
    "ebay.com",
    "ebay.de",
    "ecosia.org",
    "edgcdn.net",
    "edgecdn.ru",
    "eeroup.com",
    "elasticbeanstalk.com",
    "elmundo.es",
    "elpais.com",
    "enacdn.net",
    "epa.gov",
    "epicgames.com",
    "eporner.com",
    "erome.com",
    "eset.com",
    "espn.com",
    "etsy.com",
    "eu-1-id5-sync.com",
    "eu.com",
    "europa.eu",
    "eventbrite.com",
    "everesttech.net",
    "example.com",
    "exp-tas.com",
    "expireddomains.com",
    "eye4.cn",
    "ezviz7.com",
    "ezvizlife.com",
    "f5.com",
    "facebook.com",
    "facebook.net",
    "fandom.com",
    "faphouse.com",
    "fast.com",
    "fastly-edge.com",
    "fb.com",
    "fbcdn.net",
    "fbpigeon.com",
    "fbsbx.com",
    "fda.gov",
    "featureassets.org",
    "fidelity.com",
    "figma.com",
    "firefox.com",
    "firetvcaptiveportal.com",
    "fiverr.com",
    "flashtalking.com",
    "flickr.com",
    "flipkart.com",
    "focus.de",
    "fontawesome.com",
    "forbes.com",
    "force.com",
    "forms.gle",
    "forter.com",
    "foxnews.com",
    "free.fr",
    "freepik.com",
    "frontiersin.org",
    "ft.com",
    "fwmrm.net",
    "g.co",
    "g.page",
    "gamepass.com",
    "gandi-ns.fr",
    "gandi.net",
    "garmin.com",
    "gartner.com",
    "gcdn.co",
    "genius.com",
    "geobasket.ru",
    "ggpht.com",
    "giphy.com",
    "github.com",
    "github.io",
    "githubusercontent.com",
    "gitlab.com",
    "globalsign.com",
    "globo.com",
    "gmail.com",
    "gnu.org",
    "go-mpulse.net",
    "go.com",
    "godaddy.com",
    "goodreads.com",
    "google-analytics.com",
    "google.cn",
    "google.co.uk",
    "google.com",
    "google.com.br",
    "google.com.hk",
    "google.de",
    "googleapis.com",
    "googleblog.com",
    "googledomains.com",
    "googletagmanager.com",
    "googletagservices.com",
    "googleusercontent.com",
    "googlevideo.com",
    "googlezip.net",
    "goskope.com",
    "gosuslugi.ru",
    "grammarly.com",
    "grammarly.io",
    "gravatar.com",
    "gstatic.com",
    "gtld-servers.net",
    "gumgum.com",
    "gvt1.com",
    "gvt2.com",
    "gwfb.net",
    "harvard.edu",
    "hath.network",
    "hbr.org",
    "hcaptcha.com",
    "healthline.com",
    "herokuapp.com",
    "herokudns.com",
    "heytapdl.com",
    "heytapmobi.com",
    "heytapmobile.com",
    "hichina.com",
    "hicloud.com",
    "hicloudcam.com",
    "hihonorcloud.com",
    "hilton.com",
    "hm.com",
    "homedepot.com",
    "hostgator.com",
    "hostgator.com.br",
    "hotstar.com",
    "hp.com",
    "hstgr.net",
    "huawei.com",
    "hubspot.com",
    "huffpost.com",
    "hugedomains.com",
    "ibm.com",
    "ibyteimg.com",
    "icloud-content.com",
    "icloud.com",
    "id5-sync.com",
    "ieee.org",
    "ietf.org",
    "iiko.it",
    "ikea.com",
    "imcmdb.net",
    "imdb.com",
    "imgsmail.ru",
    "imgur.com",
    "immedia-semi.com",
    "impervadns.net",
    "imrworldwide.com",
    "indeed.com",
    "independent.co.uk",
    "indiatimes.com",
    "infobae.com",
    "inmobi.com",
    "inner-active.mobi",
    "instagram.com",
    "intel.com",
    "intercom.io",
    "internetwarriors.net",
    "intuit.com",
    "investopedia.com",
    "ioref.io",
    "ip-api.com",
    "ipify.org",
    "ipv4only.arpa",
    "irs.gov",
    "iso.org",
    "issuu.com",
    "it.com",
    "ivi.ru",
    "jd.com",
    "jetbrains.com",
    "jimdo.com",
    "jomodns.com",
    "jotform.com",
    "jquery.com",
    "jsdelivr.net",
    "kaspersky-labs.com",
    "kaspersky.com",
    "keenetic.io",
    "kick.com",
    "kickstarter.com",
    "klaviyo.com",
    "kleinanzeigen.de",
    "kontur.ru",
    "ks-cdn.com",
    "kslawin.com",
    "ksyuncdn.com",
    "kubernetes.io",
    "kueezrtb.com",
    "kunluncan.com",
    "kwai-pro.com",
    "kwai.com",
    "kwai.net",
    "kwaipros.com",
    "kwcdn.com",
    "latimes.com",
    "launchdarkly.com",
    "launchpad.net",
    "lefigaro.fr",
    "leiniao.com",
    "lemonde.fr",
    "lencr.org",
    "lenovo.com",
    "lgtvcommon.com",
    "liadm.com",
    "libp2p.direct",
    "licdn.com",
    "life360.com",
    "liftoff.io",
    "lijit.com",
    "line.me",
    "linkedin.com",
    "linktr.ee",
    "linode.com",
    "list-manage.com",
    "live-video.net",
    "live.com",
    "live.net",
    "livejournal.com",
    "ln-msedge.net",
    "loc.gov",
    "lowes.com",
    "lsrelayaccess.com",
    "macromedia.com",
    "mail.ru",
    "mailchi.mp",
    "mailchimp.com",
    "mailinabox.email",
    "mangosip.ru",
    "markmonitor.com",
    "marriott.com",
    "mayoclinic.org",
    "mcafee.com",
    "mckinsey.com",
    "mdpi.com",
    "me.com",
    "media-amazon.com",
    "mediafire.com",
    "mediatek.com",
    "medium.com",
    "mega.co.nz",
    "meraki.com",
    "mercadolibre.com.ar",
    "mercadolivre.com.br",
    "merriam-webster.com",
    "mhverifier.ru",
    "mi.com",
    "microsoft.com",
    "microsoftonline.com",
    "miit.gov.cn",
    "mikrotik.com",
    "mit.edu",
    "miui.com",
    "miwifi.com",
    "mlb.com",
    "moe.video",
    "moloco.com",
    "moneycontrol.com",
    "mozilla.com",
    "mozilla.net",
    "mozilla.org",
    "msedge.net",
    "msftauth.net",
    "msftconnecttest.com",
    "msftncsi.com",
    "msidentity.com",
    "msn.com",
    "mtgglobals.com",
    "mts.ru",
    "my.com",
    "myfritz.net",
    "myhuaweicloud.com",
    "mynetname.net",
    "myqcloud.com",
    "myshopify.com",
    "myspace.com",
    "mysql.com",
    "mzstatic.com",
    "name-services.com",
    "name.com",
    "namebrightdns.com",
    "nasa.gov",
    "nationalgeographic.com",
    "nature.com",
    "naver.com",
    "nbcnews.com",
    "ndtv.com",
    "nease.net",
    "nel.goog",
    "nelreports.net",
    "netangels.ru",
    "netcraze.io",
    "netease.com",
    "netflix.com",
    "netflix.net",
    "netgear.com",
    "networkadvertising.org",
    "nextcloud.com",
    "nextlgsdp.com",
    "nexusmods.com",
    "nflximg.com",
    "nflxso.net",
    "nflxvideo.net",
    "ngenix.net",
    "nginx.com",
    "nginx.org",
    "nic.direct",
    "nic.io",
    "nic.network",
    "nic.ru",
    "nih.gov",
    "nike.com",
    "nikkei.com",
    "nintendo.com",
    "nintendo.net",
    "nist.gov",
    "nmrodam.com",
    "no-ip.com",
    "noaa.gov",
    "nominetdns.uk",
    "note.com",
    "npmjs.com",
    "npr.org",
    "nstld.com",
    "ntp.org",
    "nvidia.com",
    "nypost.com",
    "nytimes.com",
    "odoo.com",
    "office.com",
    "office.net",
    "office365.com",
    "ok.ru",
    "okcdn.ru",
    "okta.com",
    "omtrdc.net",
    "on.aws",
    "one.one",
    "onelink.me",
    "onesignal.com",
    "onet.pl",
    "onetag-sys.com",
    "onetrust.com",
    "online-metrix.net",
    "onlyfans.com",
    "openai.com",
    "opendns.com",
    "openstreetmap.org",
    "opera-api.com",
    "opera.com",
    "optimizely.com",
    "oracle.com",
    "oraclecloud.com",
    "orderbox-dns.com",
    "otto.de",
    "oup.com",
    "outlook.com",
    "ovh.net",
    "ovscdns.com",
    "ox.ac.uk",
    "oxylabs.io",
    "ozon.ru",
    "ozone.ru",
    "pages.dev",
    "palmplaystore.com",
    "paloaltonetworks.com",
    "pangle.io",
    "patreon.com",
    "paypal.com",
    "pbs.org",
    "pccc.com",
    "people.com",
    "perplexity.ai",
    "pexels.com",
    "php.net",
    "pinimg.com",
    "pinterest.com",
    "pixabay.com",
    "pixiv.net",
    "pki.goog",
    "playrix.com",
    "playstation.com",
    "playstation.net",
    "plesk.com",
    "poki.com",
    "pornhub.com",
    "presage.io",
    "primevideo.com",
    "princeton.edu",
    "privacy-mgmt.com",
    "prnewswire.com",
    "prodregistryv2.org",
    "pushy.io",
    "pv-cdn.net",
    "pvp.net",
    "px-cloud.net",
    "pypi.org",
    "python.org",
    "qlivecdn.com",
    "qq.com",
    "qualtrics.com",
    "quickconnect.to",
    "quizlet.com",
    "quora.com",
    "rackspace.com",
    "rackspace.net",
    "rakuten.co.jp",
    "rakuten.com",
    "rambler.ru",
    "rbxcdn.com",
    "readthedocs.io",
    "recaptcha.net",
    "reddit.com",
    "redhat.com",
    "reg.ru",
    "registrar-servers.com",
    "repubblica.it",
    "researchgate.net",
    "resolver.arpa",
    "reuters.com",
    "richaudience.com",
    "ring.com",
    "ripn.net",
    "rlcdn.com",
    "roblox.com",
    "rocket-cdn.com",
    "roku.com",
    "root-servers.net",
    "rt.ru",
    "run.app",
    "rutube.ru",
    "ryanair.com",
    "rzone.de",
    "safebrowsing.apple",
    "sagepub.com",
    "salesforce.com",
    "samsung.com",
    "samsungacr.com",
    "samsungapps.com",
    "samsungcloud.com",
    "samsungcloudsolution.com",
    "samsungcloudsolution.net",
    "samsungosp.com",
    "samsungqbe.com",
    "sberbank.ru",
    "sc-cdn.net",
    "sc-gw.com",
    "scdn.co",
    "sciencedirect.com",
    "scribd.com",
    "sedo.com",
    "seedtag.com",
    "segment.io",
    "selectel.ru",
    "sendgrid.com",
    "sentry.io",
    "service.gov.uk",
    "seznam.cz",
    "sfx.ms",
    "shalltry.com",
    "share-dns.com",
    "share.google",
    "sharepoint.com",
    "shein.com",
    "shifen.com",
    "shopee.co.id",
    "shopee.com.br",
    "shopeemobile.com",
    "shopify.com",
    "shopifysvc.com",
    "sina.com.cn",
    "skyhigh.cloud",
    "skype.com",
    "slack.com",
    "slideshare.net",
    "smaato.net",
    "smartthings.com",
    "smilewanted.com",
    "snapchat.com",
    "snapkit.com",
    "sohu.com",
    "sophos.com",
    "soundcloud.com",
    "sourceforge.net",
    "spaceweb.pro",
    "speedtest.net",
    "spiegel.de",
    "spo-msedge.net",
    "spotify.com",
    "spotifycdn.com",
    "spov-msedge.net",
    "springer.com",
    "squarespace.com",
    "squarespacedns.com",
    "ssl-images-amazon.com",
    "stackadapt.com",
    "stackoverflow.com",
    "stanford.edu",
    "starlink.com",
    "state.gov",
    "static.microsoft",
    "statista.com",
    "stbid.ru",
    "steamcommunity.com",
    "steampowered.com",
    "steamserver.net",
    "steamstatic.com",
    "stripchat.com",
    "stripe.com",
    "substack.com",
    "supercell.com",
    "supertms.com",
    "surveymonkey.com",
    "svc.ms",
    "synology.com",
    "t-mobile.com",
    "t-msedge.net",
    "t-online.de",
    "t.me",
    "tandfonline.com",
    "taobao.com",
    "tapad.com",
    "target.com",
    "tawk.to",
    "tbcache.com",
    "teads.tv",
    "teamviewer.com",
    "techcrunch.com",
    "ted.com",
    "telecid.ru",
    "telegram.me",
    "telegram.org",
    "telegraph.co.uk",
    "telekom.de",
    "telekom.net",
    "telephony.goog",
    "temu.com",
    "tencent-cloud.net",
    "tencent.com",
    "theatlantic.com",
    "theconversation.com",
    "theguardian.com",
    "themeforest.net",
    "thenai.org",
    "theverge.com",
    "threads.com",
    "tiktok.com",
    "tiktokcdn-eu.com",
    "tiktokcdn-us.com",
    "tiktokcdn.com",
    "tiktokpangle.us",
    "tiktokrow-cdn.com",
    "tiktokv.com",
    "tiktokv.eu",
    "tiktokv.us",
    "tiktokw.us",
    "time.com",
    "timeweb.ru",
    "tm-azurefd.net",
    "tp-link.com",
    "tplinkcloud.com",
    "tplinknbu.com",
    "tradingview.com",
    "trafficmanager.net",
    "trbcdn.net",
    "trendmicro.com",
    "tripadvisor.com",
    "triplinkintl.com",
    "trueconf.net",
    "trustpilot.com",
    "tsyndicate.com",
    "ttdns2.com",
    "ttvnw.net",
    "tumblr.com",
    "turn.com",
    "twilio.com",
    "twimg.com",
    "twitch.tv",
    "twitter.com",
    "typeform.com",
    "typekit.net",
    "uber.com",
    "ubi.com",
    "ubnt.com",
    "ubuntu.com",
    "udemy.com",
    "ui-dns.com",
    "ui.com",
    "uk.com",
    "umich.edu",
    "un.org",
    "unesco.org",
    "unity3d.com",
    "unpkg.com",
    "unsplash.com",
    "uol.com.br",
    "ups.com",
    "usatoday.com",
    "usda.gov",
    "userapi.com",
    "usercontent.goog",
    "usgovcloudapi.net",
    "usps.com",
    "vecdnlb.com",
    "vedcdnlb.com",
    "vedsalb.com",
    "vercel-dns-016.com",
    "vercel-dns-017.com",
    "vercel.app",
    "verisign.com",
    "viber.com",
    "vidaahub.com",
    "vimeo.com",
    "virginm.net",
    "visualstudio.com",
    "vivo.com.cn",
    "vivoglobal.com",
    "vk-analytics.ru",
    "vk.com",
    "vk.ru",
    "vkontakte.ru",
    "vkuser.net",
    "vkuserphoto.ru",
    "volcfcdndvs.com",
    "vungle.com",
    "w3.org",
    "wa.me",
    "wac-msedge.net",
    "walmart.com",
    "washington.edu",
    "washingtonpost.com",
    "wattpad.com",
    "wb.ru",
    "wbbasket.ru",
    "wbx2.com",
    "weather.com",
    "webempresa.eu",
    "webex.com",
    "webmd.com",
    "weebly.com",
    "weforum.org",
    "weibo.com",
    "welt.de",
    "whatsapp.com",
    "whatsapp.net",
    "whecloud.com",
    "whitehouse.gov",
    "who.int",
    "wikimedia.org",
    "wikipedia.org",
    "wildberries.ru",
    "wiley.com",
    "windows.com",
    "windows.net",
    "windowsupdate.com",
    "wired.com",
    "withgoogle.com",
    "wix.com",
    "wixsite.com",
    "wordpress.com",
    "wordpress.org",
    "workers.dev",
    "worldbank.org",
    "worldnic.com",
    "wp.com",
    "wp.pl",
    "wpguardian.com",
    "wpguardian.io",
    "wps.com",
    "wsdvs.com",
    "wsj.com",
    "wswebcdn.com",
    "www.gov.br",
    "www.gov.uk",
    "wyzecam.com",
    "x.com",
    "xboxlive.com",
    "xcal.tv",
    "xerox.com",
    "xhamster.com",
    "xhamster.desi",
    "xhcdn.com",
    "xiaomi.com",
    "xiaomi.net",
    "xnxx.com",
    "xvideos.com",
    "ya.ru",
    "yahoo.co.jp",
    "yahoo.com",
    "yahoodns.net",
    "yandex.com",
    "yandex.com.tr",
    "yandex.net",
    "yandex.ru",
    "yandexcloud.net",
    "yccdn.ru",
    "yellowblue.io",
    "yelp.com",
    "yieldmo.com",
    "youku.com",
    "youronlinechoices.com",
    "youtu.be",
    "youtube-nocookie.com",
    "youtube.com",
    "ys7.com",
    "ytimg.com",
    "yximgs.com",
    "zdnscloud.cn",
    "zendesk.com",
    "zenecn.net",
    "zhihu.com",
    "zillow.com",
    "zoho.com",
    "zoom.com",
    "zoom.us",
})


# ── URLClassifier (singleton) ─────────────────────────────────────────────────

class URLClassifier:
    """Thread-safe singleton. Load artefacts once; serve predictions forever."""

    _instance: Optional["URLClassifier"] = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.model    = joblib.load(os.path.join(MODELS_DIR, "rf_model_latest.joblib"))
        self.vec_char = joblib.load(os.path.join(MODELS_DIR, "vectorizer_char.joblib"))
        self.vec_word = joblib.load(os.path.join(MODELS_DIR, "vectorizer_word.joblib"))
        self.scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

        with open(os.path.join(MODELS_DIR, "threshold.json")) as fh:
            self.threshold = float(json.load(fh).get("threshold", 0.44))

        # LIME explainer — built lazily, cached after first call
        self._explainer: Optional[object] = None
        self._explainer_lock = threading.Lock()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _registered_domain(url: str) -> str:
        """Return 'example.com' from any URL (no external deps)."""
        cleaned = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host    = cleaned.split("/")[0].split(":")[0].split("?")[0].lower()
        parts   = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    def _feature_vector(self, url: str) -> np.ndarray:
        """Build the 548-dim feature vector for *url*."""
        # 48 heuristic features — scaler was fitted on these only
        heuristic = np.array(
            extract_heuristic_features(url), dtype=np.float32
        ).reshape(1, -1)                                            # (1, 48)
        heuristic_scaled = self.scaler.transform(heuristic).flatten()  # (48,)

        # 300 + 200 NLP features — unscaled, exactly as during training
        processed  = preprocess_url_for_nlp(url)
        char_dense = self.vec_char.transform([processed]).toarray().flatten()  # (300,)
        word_dense = self.vec_word.transform([processed]).toarray().flatten()  # (200,)

        # Concatenate in training order: heuristic_scaled + char + word
        return np.concatenate([heuristic_scaled, char_dense, word_dense])  # (548,)

    # ── Public: prediction ────────────────────────────────────────────────────

    def predict_url(self, url: str) -> dict:
        if not url or not isinstance(url, str):
            return {
                "url": url, "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "invalid",
            }

        if self._registered_domain(url) in WHITELIST:
            return {
                "url": url, "prediction": "BENIGN",
                "confidence": 100.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "whitelist",
            }

        try:
            fv    = self._feature_vector(url)
            proba = float(self.model.predict_proba(fv.reshape(1, -1))[0][1])
            label = "MALICIOUS" if proba >= self.threshold else "BENIGN"
            return {
                "url": url, "prediction": label,
                "confidence": round(proba * 100, 2),
                "threshold" : round(self.threshold * 100, 2),
                "source"    : "model",
            }
        except Exception as exc:
            return {
                "url": url, "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "invalid", "error": str(exc),
            }

    def predict_batch(self, urls: list[str]) -> list[dict]:
        return [self.predict_url(u) for u in urls]

    # ── Public: LIME explanation ──────────────────────────────────────────────

    def explain_url(self, url: str, num_features: int = 10) -> dict:
        """
        Predict + explain. Returns standard predict dict plus:
          "explanation": [{"feature": str, "weight": float, "value": float}, ...]
        Whitelist / invalid URLs return an empty explanation list.
        """
        base = self.predict_url(url)
        if base["source"] in ("whitelist", "invalid"):
            return {**base, "explanation": []}

        try:
            explainer = self._get_explainer()
            fv        = self._feature_vector(url)

            exp = explainer.explain_instance(
                data_row     = fv,
                predict_fn   = self._lime_predict_fn,
                num_features = num_features,
                top_labels   = 1,
            )

            raw_list = exp.as_list(label=1)  # label 1 = MALICIOUS

            explanation = []
            for condition_str, weight in raw_list:
                feat_name = _parse_lime_feature(condition_str)
                feat_idx  = FEATURE_NAMES.index(feat_name) \
                            if feat_name in FEATURE_NAMES else -1
                feat_val  = float(fv[feat_idx]) if feat_idx >= 0 else 0.0
                explanation.append({
                    "feature": feat_name,
                    "weight" : round(float(weight), 6),
                    "value"  : round(feat_val, 6),
                })

            explanation.sort(key=lambda x: abs(x["weight"]), reverse=True)
            return {**base, "explanation": explanation}

        except Exception as exc:
            return {**base, "explanation": [], "explain_error": str(exc)}

    # ── LIME internals ────────────────────────────────────────────────────────

    def _get_explainer(self):
        """Build LimeTabularExplainer lazily; cache forever (thread-safe)."""
        if self._explainer is not None:
            return self._explainer

        with self._explainer_lock:
            if self._explainer is not None:  # double-checked
                return self._explainer

            try:
                from lime.lime_tabular import LimeTabularExplainer
            except ImportError as exc:
                raise ImportError(
                    "lime is not installed. Run: pip install lime"
                ) from exc

            bg = self._load_background()

            cat_indices = [
                FEATURE_NAMES.index(n)
                for n in _CATEGORICAL_FEATURE_NAMES
                if n in FEATURE_NAMES
            ]

            self._explainer = LimeTabularExplainer(
                training_data        = bg,
                feature_names        = FEATURE_NAMES,
                class_names          = ["BENIGN", "MALICIOUS"],
                categorical_features = cat_indices,
                mode                 = "classification",
                discretize_continuous= True,
                random_state         = 42,
            )
            return self._explainer

    def _load_background(self, n_samples: int = 5000) -> np.ndarray:
        """Load lime_background.npz; fall back to building from train_urls.csv."""
        bg_path = os.path.join(MODELS_DIR, "lime_background.npz")
        if os.path.exists(bg_path):
            return np.load(bg_path)["X"]

        warnings.warn(
            f"{bg_path} not found — building LIME background from train_urls.csv. "
            "This runs once then saves the result.",
            RuntimeWarning, stacklevel=3,
        )

        import csv, random
        csv_path = os.path.join(_ROOT, "data", "splits", "train_urls.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Cannot find {bg_path} or {csv_path}.\n"
                "Run: python -c \"from model_loader import classifier; "
                "classifier._load_background()\""
            )

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader   = csv.DictReader(fh)
            all_urls = [row["url"] for row in reader if row.get("url")]

        sample = random.sample(all_urls, min(n_samples, len(all_urls)))
        rows   = []
        for u in sample:
            try:
                rows.append(self._feature_vector(u))
            except Exception:
                pass

        bg = np.array(rows, dtype=np.float32)
        np.savez_compressed(bg_path, X=bg)
        return bg

    def _lime_predict_fn(self, X: np.ndarray) -> np.ndarray:
        """predict_proba wrapper for LIME (input is already-scaled vectors)."""
        return self.model.predict_proba(X)


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_lime_feature(condition_str: str) -> str:
    """
    Recover bare feature name from a LIME condition string.
    e.g. 'brand_in_domain=1'  →  'brand_in_domain'
         'url_len > 45.00'    →  'url_len'
    Uses longest-match to handle underscored names correctly.
    """
    for name in sorted(FEATURE_NAMES, key=len, reverse=True):
        if condition_str.startswith(name):
            return name
    return re.split(r"[\s<>=!]", condition_str)[0]


# ── Module-level singleton ────────────────────────────────────────────────────
classifier = URLClassifier()