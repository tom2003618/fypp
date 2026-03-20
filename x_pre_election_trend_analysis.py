# Twitter/X pre-election trend analysis for Canada Federal Elections (2021 vs 2025)

# Scope: Liberal, Conservative, NDP, Bloc Quebecois

# Collection order: local CSV -> official X API (if token) -> snscrape (best effort)

from pathlib import Path

def find_project_root():
    markers = [
        'canada_federal_vote_share_2000_2025.csv',
        'canada_federal_election_province_summary_2000_2021.csv',
        'econ_2021_2025.py',
        '.git',
    ]
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return Path.cwd()
PROJECT_ROOT = find_project_root()
import os
import re
import sys
import time
import urllib.parse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
ELECTIONS = {
    "2021": pd.Timestamp("2021-09-20"),
    "2025": pd.Timestamp("2025-04-28"),
}
WINDOW_DAYS = 90
MAX_TWEETS_PER_PARTY = 3500
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_RETRY_SECONDS = 15
SAVE_COLLECTED_CSV = True
USE_SNSCRAPE_FALLBACK = False
ALLOW_PAID_X_API_CALL = True
CONFIRM_BEFORE_X_API = True
FORCE_REFRESH_X_API = False
RAW_TWEETS_CSV = PROJECT_ROOT / "notebooks" / "data" / "twitter_canada_pre_election_raw.csv"
TEMPLATE_CSV = PROJECT_ROOT / "notebooks" / "data" / "twitter_canada_pre_election_raw.template.csv"
MANUAL_X_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAOug8AEAAAAA4GX0hK0p9ejBqic3SrzAS3JtzOk%3DcJPDhTPffDwDPTXB9csMgt95qGsjzzrVtIkZ46PElhWnzOU02U"
_manual_token = (MANUAL_X_BEARER_TOKEN or "").strip()
if _manual_token and "%3D" in _manual_token:
    _manual_token = urllib.parse.unquote(_manual_token)
X_BEARER_TOKEN = (
    os.getenv("X_BEARER_TOKEN")
    or os.getenv("TWITTER_BEARER_TOKEN")
    or _manual_token
)
if X_BEARER_TOKEN and len(X_BEARER_TOKEN) < 40:
    print(
        "Warning: token looks too short for an X bearer token. "
        "This is likely an API key/client id, not bearer token."
    )
PARTY_TERMS = {
    "Liberal": [
        "liberal",
        "liberal party",
        "liberal party of canada",
        "trudeau",
        "team trudeau",
        "parti liberal",
    ],
    "Conservative": [
        "conservative",
        "conservative party",
        "conservative party of canada",
        "cpc",
        "poilievre",
        "parti conservateur",
    ],
    "NDP": [
        "ndp",
        "new democratic party",
        "jagmeet",
        "jagmeet singh",
        "nouveau parti democratique",
    ],
    "Bloc Quebecois": [
        "bloc",
        "bloc quebecois",
        "blocquebecois",
        "bq",
        "blanchet",
    ],
}
CANADA_TERMS = [
    "canada", "canadian", "ontario", "quebec", "alberta", "british columbia", "bc",
    "manitoba", "saskatchewan", "new brunswick", "nova scotia", "newfoundland", "labrador",
    "prince edward island", "pei", "yukon", "nunavut", "northwest territories", "nwt",
    "toronto", "montreal", "vancouver", "ottawa", "calgary", "edmonton", "winnipeg", "halifax",
]
CANADA_PATTERN = re.compile(r"\b(" + "|".join(re.escape(t) for t in CANADA_TERMS) + r")\b", re.IGNORECASE)
PARTY_PATTERNS = {
    party: re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    for party, terms in PARTY_TERMS.items()
}

def _import_snscrape_with_py312_shim():
    try:
        import snscrape.modules.twitter as sntwitter
        return sntwitter, None
    except Exception as e:
        first_err = f"{type(e).__name__}: {e}"
    # Python 3.12 compatibility shim for snscrape 0.7.x
    try:
        import importlib.machinery
        import importlib.util
        class _LoaderShim:
            def __init__(self, spec):
                self.spec = spec
                self.loader = spec.loader if spec else None
            def load_module(self, fullname):
                if fullname in sys.modules:
                    return sys.modules[fullname]
                mod = importlib.util.module_from_spec(self.spec)
                sys.modules[fullname] = mod
                self.loader.exec_module(mod)
                return mod
        if not hasattr(importlib.machinery.FileFinder, "find_module"):
            def _find_module(self, fullname, path=None):
                spec = self.find_spec(fullname)
                return _LoaderShim(spec) if spec else None
            importlib.machinery.FileFinder.find_module = _find_module
        import snscrape.modules.twitter as sntwitter
        return sntwitter, None
    except Exception as e2:
        return None, f"{first_err}; shim retry failed: {type(e2).__name__}: {e2}"
sntwitter, SNSCRAPE_IMPORT_ERROR = _import_snscrape_with_py312_shim()
HAS_SNSCRAPE = sntwitter is not None

def election_window(election_day, window_days=WINDOW_DAYS):
    since = (election_day - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    until = election_day.strftime("%Y-%m-%d")
    return since, until

def classify_election_year(ts):
    for y, day in ELECTIONS.items():
        if (day - pd.Timedelta(days=WINDOW_DAYS)) <= ts <= day:
            return y
    return np.nan

def _or_query(terms):
    return " OR ".join(f'"{t}"' if " " in t else t for t in terms)

def available_local_election_years(df):
    if df.empty or "election_year" not in df.columns:
        return set()
    years = pd.to_numeric(df["election_year"], errors="coerce").dropna().astype(int).tolist()
    return {str(y) for y in years}

def confirm_x_api_run(requested_years, force_refresh=False):
    if not X_BEARER_TOKEN:
        print("X API token missing; cannot call paid X API.")
        return False
    if not ALLOW_PAID_X_API_CALL:
        print("Skipping paid X API calls. Set ALLOW_PAID_X_API_CALL = True if you want to allow them for this run.")
        return False
    action = (
        f"refresh all configured years {sorted(requested_years)}"
        if force_refresh else f"collect missing years {sorted(requested_years)}"
    )
    if not CONFIRM_BEFORE_X_API:
        print(f"Paid X API call enabled without prompt: {action}")
        return True
    try:
        answer = input(
            f"Paid X API call requested to {action}. This may incur charges. Proceed? [y/N]: "
        ).strip().lower()
    except EOFError:
        answer = ""
    approved = answer in {"y", "yes"}
    if not approved:
        print("X API call cancelled by user.")
    return approved

def collect_with_x_api(requested_years=None):
    rows, errors = [], []
    requested_years = {str(y) for y in (requested_years or ELECTIONS.keys())}
    if not X_BEARER_TOKEN:
        errors.append("X API token missing (set X_BEARER_TOKEN or TWITTER_BEARER_TOKEN).")
        return pd.DataFrame(), errors
    try:
        import requests
    except Exception as e:
        errors.append(f"requests import failed: {type(e).__name__}: {e}")
        return pd.DataFrame(), errors
    endpoint = "https://api.twitter.com/2/tweets/search/all"
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    for election_year, election_day in ELECTIONS.items():
        if str(election_year) not in requested_years:
            continue
        since, until = election_window(election_day)
        start_time = f"{since}T00:00:00Z"
        end_time = f"{until}T00:00:00Z"
        for party, terms in PARTY_TERMS.items():
            query = f"({_or_query(terms)}) ({_or_query(CANADA_TERMS)}) -is:retweet"
            print(f"Collecting (X API) {election_year} - {party}")
            next_token = None
            collected_for_pair = 0
            rate_limit_retries = 0
            while collected_for_pair < MAX_TWEETS_PER_PARTY:
                params = {
                    "query": query,
                    "start_time": start_time,
                    "end_time": end_time,
                    "max_results": 100,
                    "tweet.fields": "created_at,lang,public_metrics,geo",
                    "expansions": "author_id,geo.place_id",
                    "user.fields": "username,name,location,verified,public_metrics",
                    "place.fields": "country,country_code,name,full_name",
                }
                if next_token:
                    params["next_token"] = next_token
                try:
                    resp = requests.get(endpoint, headers=headers, params=params, timeout=60)
                except Exception as e:
                    errors.append(f"x_api {election_year}-{party}: request failed: {type(e).__name__}: {e}")
                    break
                if resp.status_code != 200:
                    detail = ""
                    try:
                        j = resp.json()
                        if isinstance(j, dict):
                            detail = j.get("detail") or j.get("title") or str(j.get("errors", ""))[:180]
                        else:
                            detail = str(j)[:180]
                    except Exception:
                        detail = (resp.text or "")[:180].replace("\n", " ")
                    errors.append(
                        f"x_api {election_year}-{party}: HTTP {resp.status_code} detail={detail}"
                    )
                    # 429: wait and retry same request before giving up this party query.
                    if resp.status_code == 429:
                        rate_limit_retries += 1
                        if rate_limit_retries <= MAX_RATE_LIMIT_RETRIES:
                            wait_s = RATE_LIMIT_RETRY_SECONDS
                            reset_ts = resp.headers.get("x-rate-limit-reset")
                            if reset_ts and str(reset_ts).isdigit():
                                wait_s = max(wait_s, int(reset_ts) - int(time.time()) + 1)
                            wait_s = min(wait_s, 120)
                            print(
                                f"Rate limited on {election_year}-{party}; "
                                f"retry {rate_limit_retries}/{MAX_RATE_LIMIT_RETRIES} in {wait_s}s"
                            )
                            time.sleep(max(wait_s, 1))
                            continue
                        break
                    # Stop early on global auth/entitlement/quota failures.
                    if resp.status_code in {401, 402, 403} or "credits" in detail.lower():
                        return pd.DataFrame(rows), errors
                    # 400 and others: skip this party query and continue with next party.
                    break
                payload = resp.json()
                data = payload.get("data", [])
                includes = payload.get("includes", {})
                users = {u.get("id"): u for u in includes.get("users", [])}
                places = {p.get("id"): p for p in includes.get("places", [])}
                if not data:
                    break
                for tw in data:
                    if collected_for_pair >= MAX_TWEETS_PER_PARTY:
                        break
                    user = users.get(tw.get("author_id"), {})
                    place_name, country = None, None
                    place_id = (tw.get("geo") or {}).get("place_id")
                    if place_id and place_id in places:
                        pl = places[place_id]
                        place_name = pl.get("full_name") or pl.get("name")
                        country = pl.get("country") or pl.get("country_code")
                    pm = tw.get("public_metrics") or {}
                    upm = user.get("public_metrics") or {}
                    rows.append(
                        {
                            "election_year": election_year,
                            "tweet_id": tw.get("id"),
                            "created_at": tw.get("created_at"),
                            "content": tw.get("text"),
                            "lang": tw.get("lang"),
                            "username": user.get("username"),
                            "display_name": user.get("name"),
                            "user_location": user.get("location"),
                            "followers": upm.get("followers_count"),
                            "following": upm.get("following_count"),
                            "statuses": upm.get("tweet_count"),
                            "verified": user.get("verified"),
                            "retweet_count": pm.get("retweet_count"),
                            "like_count": pm.get("like_count"),
                            "reply_count": pm.get("reply_count"),
                            "place_name": place_name,
                            "country": country,
                            "source": "x_api",
                        }
                    )
                    collected_for_pair += 1
                next_token = (payload.get("meta") or {}).get("next_token")
                if not next_token:
                    break
                time.sleep(0.2)
    return pd.DataFrame(rows), errors

def collect_with_snscrape():
    rows, errors = [], []
    canada_query = " OR ".join(CANADA_TERMS)
    for election_year, election_day in ELECTIONS.items():
        since, until = election_window(election_day)
        for party, terms in PARTY_TERMS.items():
            party_query = _or_query(terms)
            query = f"({party_query}) ({canada_query}) since:{since} until:{until}"
            print(f"Collecting (snscrape) {election_year} - {party}: {query}")
            try:
                scraper = sntwitter.TwitterSearchScraper(query)
                for i, tw in enumerate(scraper.get_items()):
                    if i >= MAX_TWEETS_PER_PARTY:
                        break
                    place = getattr(tw, "place", None)
                    rows.append(
                        {
                            "election_year": election_year,
                            "tweet_id": getattr(tw, "id", None),
                            "created_at": getattr(tw, "date", None),
                            "content": getattr(tw, "rawContent", None) or getattr(tw, "content", None),
                            "lang": getattr(tw, "lang", None),
                            "username": getattr(getattr(tw, "user", None), "username", None),
                            "display_name": getattr(getattr(tw, "user", None), "displayname", None),
                            "user_location": getattr(getattr(tw, "user", None), "location", None),
                            "followers": getattr(getattr(tw, "user", None), "followersCount", None),
                            "following": getattr(getattr(tw, "user", None), "friendsCount", None),
                            "statuses": getattr(getattr(tw, "user", None), "statusesCount", None),
                            "verified": getattr(getattr(tw, "user", None), "verified", None),
                            "retweet_count": getattr(tw, "retweetCount", None),
                            "like_count": getattr(tw, "likeCount", None),
                            "reply_count": getattr(tw, "replyCount", None),
                            "place_name": getattr(place, "fullName", None) or getattr(place, "name", None),
                            "country": getattr(place, "country", None),
                            "source": "snscrape",
                        }
                    )
            except Exception as e:
                errors.append(f"snscrape {election_year}-{party}: {type(e).__name__}: {e}")
    return pd.DataFrame(rows), errors

def normalize_columns(df):
    out = df.copy()
    rename = {
        "date": "created_at",
        "datetime": "created_at",
        "text": "content",
        "tweet": "content",
        "body": "content",
        "user": "username",
        "screen_name": "username",
        "location": "user_location",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    required = ["created_at", "content", "username", "user_location", "country", "tweet_id"]
    for c in required:
        if c not in out.columns:
            out[c] = np.nan
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
    out["created_at"] = out["created_at"].dt.tz_convert("America/Toronto").dt.tz_localize(None)
    out["content"] = out["content"].fillna("").astype(str)
    out["username"] = out["username"].fillna("").astype(str)
    out["user_location"] = out["user_location"].fillna("").astype(str)
    if "election_year" not in out.columns:
        out["election_year"] = out["created_at"].apply(classify_election_year)
    else:
        out["election_year"] = out["election_year"].astype(str)
    return out

def clean_text(s):
    s = str(s)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_canadian_row(r):
    country = str(r.get("country", "") or "").strip().lower()
    if country in {"canada", "ca"}:
        return True
    text = " ".join(
        [
            str(r.get("user_location", "") or ""),
            str(r.get("place_name", "") or ""),
            str(r.get("content", "") or ""),
        ]
    )
    return bool(CANADA_PATTERN.search(text))

def likely_bot(r):
    txt = str(r.get("content", "") or "")
    uname = str(r.get("username", "") or "").lower()
    followers = pd.to_numeric(r.get("followers", np.nan), errors="coerce")
    following = pd.to_numeric(r.get("following", np.nan), errors="coerce")
    statuses = pd.to_numeric(r.get("statuses", np.nan), errors="coerce")
    links = len(re.findall(r"http\S+", txt))
    hashtags = len(re.findall(r"#\w+", txt))
    signals = 0
    if "bot" in uname or "newsbot" in uname:
        signals += 1
    if links >= 3:
        signals += 1
    if hashtags >= 8:
        signals += 1
    if pd.notna(followers) and pd.notna(following) and followers <= 5 and following >= 500:
        signals += 1
    if pd.notna(statuses) and pd.notna(followers) and statuses >= 200000 and followers <= 100:
        signals += 1
    return signals >= 2

def detect_parties(text):
    return [party for party, pat in PARTY_PATTERNS.items() if pat.search(text)]

def extract_hashtags(text):
    return [h.lower() for h in re.findall(r"#\w+", text)]

def get_sentiment_scorer():
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            _ = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        def score_fn(text):
            return float(sia.polarity_scores(text).get("compound", 0.0))
        return score_fn, "VADER"
    except Exception:
        pos_words = {
            "good", "great", "support", "strong", "win", "hope", "trust", "better",
            "excellent", "improve", "positive", "leader", "stable", "progress",
        }
        neg_words = {
            "bad", "worse", "corrupt", "hate", "weak", "fail", "lying", "scandal",
            "negative", "angry", "broken", "disaster", "problem", "inflation",
        }
        def score_fn(text):
            tokens = re.findall(r"[a-z']+", str(text).lower())
            if not tokens:
                return 0.0
            p = sum(t in pos_words for t in tokens)
            n = sum(t in neg_words for t in tokens)
            return float((p - n) / max(len(tokens), 1))
        return score_fn, "simple_lexicon"

def sentiment_label(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"
collection_errors = []
raw = pd.DataFrame()
local_years = set()
target_years = {str(y) for y in ELECTIONS.keys()}
if RAW_TWEETS_CSV.exists():
    raw = pd.read_csv(RAW_TWEETS_CSV)
    print(f"Loaded local raw tweets: {RAW_TWEETS_CSV} ({len(raw):,} rows)")
    local_years = available_local_election_years(raw)
    print(f"Local tweet years available: {sorted(local_years) if local_years else 'none'}")
missing_years = sorted(target_years - local_years)
requested_years = []
should_try_x_api = False
if FORCE_REFRESH_X_API:
    requested_years = sorted(target_years)
    should_try_x_api = confirm_x_api_run(requested_years, force_refresh=True)
elif missing_years:
    print(f"Missing X raw data for election years: {missing_years}")
    requested_years = missing_years
    should_try_x_api = confirm_x_api_run(requested_years, force_refresh=False)
if should_try_x_api and requested_years:
    x_df, x_errors = collect_with_x_api(requested_years=requested_years)
    collection_errors.extend(x_errors)
    if not x_df.empty:
        print(f"Collected rows via X API: {len(x_df):,} new rows")
        raw = pd.concat([raw, x_df], ignore_index=True) if not raw.empty else x_df
        if 'tweet_id' in raw.columns:
            raw = raw.drop_duplicates(subset=['tweet_id'], keep='last')
        merged_years = available_local_election_years(raw)
        print(f"Raw tweet years after merge: {sorted(merged_years) if merged_years else 'none'}")
        if x_errors:
            print("X API collected partial data. Sample errors:")
            for e in x_errors[:5]:
                print("  -", e)
    elif x_errors:
        print("X API returned no new rows. Sample errors:")
        for e in x_errors[:5]:
            print("  -", e)
elif requested_years:
    collection_errors.append("X API call skipped by user or disabled by config.")
if raw.empty and USE_SNSCRAPE_FALLBACK:
    if not HAS_SNSCRAPE:
        collection_errors.append(
            "snscrape import failed. "
            f"Python executable: {sys.executable}. Error: {SNSCRAPE_IMPORT_ERROR}"
        )
    else:
        s_df, s_errors = collect_with_snscrape()
        collection_errors.extend(s_errors)
        raw = s_df
        print(f"Collected rows via snscrape: {len(raw):,}")
elif raw.empty and not USE_SNSCRAPE_FALLBACK:
    collection_errors.append(
        "snscrape fallback is disabled (USE_SNSCRAPE_FALLBACK=False)."
    )
if SAVE_COLLECTED_CSV and not raw.empty:
    RAW_TWEETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_TWEETS_CSV, index=False)
    print(f"Saved raw tweets to: {RAW_TWEETS_CSV}")
if raw.empty:
    TEMPLATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not TEMPLATE_CSV.exists():
        pd.DataFrame(
            columns=[
                "tweet_id", "created_at", "content", "username", "user_location", "country",
                "lang", "followers", "following", "statuses", "verified",
                "retweet_count", "like_count", "reply_count", "place_name", "election_year", "source",
            ]
        ).to_csv(TEMPLATE_CSV, index=False)
    err_lines = [
        "No tweets collected.",
        "Likely cause: X CreditsDepleted, missing entitlement for full-archive endpoint "
        "(/2/tweets/search/all) or invalid query/token scope.",
        f"Python executable: {sys.executable}",
        "",
        "What to do next:",
        "1) Top up X API credits for your enrolled account, then retry.",
        "2) Confirm your plan/app includes /2/tweets/search/all (full-archive).",
        f"3) Put dataset at: {RAW_TWEETS_CSV}",
        f"4) Template created at: {TEMPLATE_CSV}",
        "",
        "Collector errors:",
    ]
    err_lines += [f"- {e}" for e in collection_errors[:20]]
    raise RuntimeError("\n".join(err_lines))
df = normalize_columns(raw)
df = df.dropna(subset=["created_at", "content"])
df = df[df["election_year"].isin(ELECTIONS.keys())].copy()
for y, day in ELECTIONS.items():
    start = day - pd.Timedelta(days=WINDOW_DAYS)
    mask = (df["election_year"] == y) & (df["created_at"].between(start, day))
    df.loc[df["election_year"] == y, "_in_window"] = mask
df = df[df["_in_window"] == True].copy()
df = df[df.apply(is_canadian_row, axis=1)].copy()
df["clean_text"] = df["content"].map(clean_text)
df = df[df["clean_text"].str.len() >= 8].copy()
df = df[~df["clean_text"].str.startswith("RT @", na=False)].copy()
if df["tweet_id"].notna().any():
    df = df.drop_duplicates(subset=["tweet_id"])
else:
    df = df.drop_duplicates(subset=["username", "clean_text", "created_at"])
repeat_rank = df.groupby(["username", "clean_text"]).cumcount()
df = df[repeat_rank < 2].copy()
bot_mask = df.apply(likely_bot, axis=1)
df = df[~bot_mask].copy()
df["parties"] = df["clean_text"].apply(detect_parties)
df = df[df["parties"].map(len) > 0].copy()
expanded = df.explode("parties").rename(columns={"parties": "party"}).copy()
expanded["hashtags"] = expanded["content"].apply(extract_hashtags)
score_fn, model_name = get_sentiment_scorer()
expanded["sentiment_score"] = expanded["clean_text"].apply(score_fn)
expanded["sentiment_label"] = expanded["sentiment_score"].apply(sentiment_label)
print(f"Sentiment model: {model_name}")
print(f"Analyzed rows after filtering: {len(expanded):,}")
volume_tbl = (
    expanded.groupby(["election_year", "party"], as_index=False)
    .agg(tweet_volume=("tweet_id", "nunique"), unique_users=("username", "nunique"))
)
sent_counts = (
    expanded.pivot_table(
        index=["election_year", "party"],
        columns="sentiment_label",
        values="clean_text",
        aggfunc="count",
        fill_value=0,
    )
    .reset_index()
)
for col in ["positive", "neutral", "negative"]:
    if col not in sent_counts.columns:
        sent_counts[col] = 0
sent_mean = (
    expanded.groupby(["election_year", "party"], as_index=False)
    .agg(avg_sentiment=("sentiment_score", "mean"))
)
summary = volume_tbl.merge(sent_counts, on=["election_year", "party"], how="left").merge(
    sent_mean, on=["election_year", "party"], how="left"
)
summary["support_index"] = (summary["positive"] - summary["negative"]) / summary["tweet_volume"].clip(lower=1)
hashtag_tbl = (
    expanded.explode("hashtags")
    .dropna(subset=["hashtags"])
    .groupby(["election_year", "party", "hashtags"], as_index=False)
    .size()
    .rename(columns={"size": "count"})
    .sort_values(["election_year", "party", "count"], ascending=[True, True, False])
)
top_hashtags = hashtag_tbl.groupby(["election_year", "party"]).head(10)
expanded["week"] = expanded["created_at"].dt.to_period("W").dt.start_time
weekly = (
    expanded.groupby(["election_year", "party", "week"], as_index=False)
    .agg(tweet_volume=("clean_text", "count"), avg_sentiment=("sentiment_score", "mean"))
)
print("\nTweet volume + sentiment by party:")
print(summary.sort_values(["election_year", "tweet_volume"], ascending=[True, False]).to_string(index=False))
print("\nEstimated political leaning (higher support_index => more positive net sentiment):")
leaning = summary.sort_values(["election_year", "support_index"], ascending=[True, False]).groupby("election_year").head(1)
print(leaning[["election_year", "party", "support_index"]].to_string(index=False))
print("\nTop hashtags by party (top 10):")
if top_hashtags.empty:
    print("No hashtags detected after filtering.")
else:
    print(top_hashtags.to_string(index=False))
party_order = list(PARTY_TERMS.keys())
volume_plot = summary.pivot(index="party", columns="election_year", values="tweet_volume").reindex(party_order)
ax = volume_plot.plot(kind="bar", figsize=(10, 4), rot=20)
ax.set_title("Tweet volume by party (Canada, pre-election window)")
ax.set_ylabel("Tweet count")
ax.set_xlabel("Party")
plt.tight_layout()
plt.show()
sent_plot = summary.pivot(index="party", columns="election_year", values="avg_sentiment").reindex(party_order)
ax = sent_plot.plot(kind="bar", figsize=(10, 4), rot=20)
ax.set_title("Average sentiment score by party (2021 vs 2025)")
ax.set_ylabel("Sentiment score")
ax.set_xlabel("Party")
ax.axhline(0, color="black", linewidth=1)
plt.tight_layout()
plt.show()
for election_year in sorted(ELECTIONS.keys()):
    sub = weekly[weekly["election_year"] == election_year]
    if sub.empty:
        continue
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for party in party_order:
        s = sub[sub["party"] == party].sort_values("week")
        if s.empty:
            continue
        axes[0].plot(s["week"], s["tweet_volume"], marker="o", linewidth=1.5, label=party)
        axes[1].plot(s["week"], s["avg_sentiment"], marker="o", linewidth=1.5, label=party)
    axes[0].set_title(f"{election_year} pre-election trend: tweet volume by week")
    axes[0].set_ylabel("Tweet volume")
    axes[0].legend(ncol=2)
    axes[1].set_title(f"{election_year} pre-election trend: average sentiment by week")
    axes[1].set_ylabel("Sentiment score")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Week")
    plt.tight_layout()
    plt.show()
print("\nDone. This estimates pre-election party preference trends among Canada-related Twitter/X posts.")
