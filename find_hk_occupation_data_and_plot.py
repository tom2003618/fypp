from pathlib import Path
import sys, json, re
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs_demographics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

API = "https://open.canada.ca/data/en/api/3/action/package_search"

def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()

def ckan_search(q, rows=100):
    url = f"{API}?q={urllib.parse.quote(q)}&rows={rows}"
    return json.loads(http_get(url).decode("utf-8"))

def looks_like_occ(s):
    s = str(s).lower()
    return ("noc" in s) or ("occupation" in s) or ("intended occupation" in s)

def try_load_resource(url):
    try:
        if url.lower().endswith(".xlsx"):
            return pd.ExcelFile(url)
        if url.lower().endswith(".csv"):
            return pd.read_csv(url, dtype=str, low_memory=False)
    except Exception:
        return None
    return None

def find_candidate_resources():
    queries = [
        "Permanent Residents occupation Hong Kong",
        "intended occupation Hong Kong",
        "NOC Hong Kong permanent residents",
        "Hong Kong citizenship occupation IRCC",
        "Work permit holders intended occupation Hong Kong",
    ]
    seen = set()
    resources = []
    for q in queries:
        try:
            resp = ckan_search(q, rows=200)
        except Exception as e:
            print(f"CKAN search failed: {e}", file=sys.stderr)
            sys.exit(1)
        pkgs = resp.get("result", {}).get("results", [])
        for pkg in pkgs:
            for r in pkg.get("resources", []):
                u = r.get("url") or ""
                name = (r.get("name") or "").strip()
                if not u:
                    continue
                if not (u.lower().endswith(".xlsx") or u.lower().endswith(".csv")):
                    continue
                key = (u, name)
                if key in seen:
                    continue
                seen.add(key)
                resources.append({"url": u, "name": name, "pkg": pkg.get("title", "")})
    return resources

def detect_hk_occ_table(df):
    cols = [str(c) for c in df.columns]
    text_sample = " ".join(cols[:50]).lower()
    if not (("2021" in text_sample) or ("2025" in text_sample)):
        pass

    has_hk = False
    for c in df.columns[: min(25, len(df.columns))]:
        s = df[c].astype(str).str.lower()
        if s.str.contains("hong kong", regex=False).any():
            has_hk = True
            break
    if not has_hk:
        return None

    occ_cols = [c for c in df.columns if looks_like_occ(c)]
    if not occ_cols:
        occ_cols = [c for c in df.columns[:15] if df[c].astype(str).str.contains(r"\d{4}\s*-\s*", regex=True, na=False).any()]
    if not occ_cols:
        return None

    year_cols = [c for c in df.columns if re.search(r"\b2021\b", str(c)) or re.search(r"\b2025\b", str(c))]
    if not year_cols:
        return None

    return {"hk_found": True, "occ_cols": occ_cols, "year_cols": year_cols}

def extract_top_occ(df, occ_col, country_col, val_2021, val_2025):
    d = df.copy()
    d[country_col] = d[country_col].astype(str).str.strip()
    d = d[d[country_col].str.contains("hong kong", case=False, na=False)]
    d["Y2021"] = pd.to_numeric(d[val_2021], errors="coerce").fillna(0.0)
    d["Y2025"] = pd.to_numeric(d[val_2025], errors="coerce").fillna(0.0)
    d["Occupation"] = d[occ_col].astype(str).str.strip()
    g = d.groupby("Occupation", as_index=False)[["Y2021", "Y2025"]].sum()
    g = g.sort_values("Y2025", ascending=False).head(20)
    return g

def main():
    resources = find_candidate_resources()
    if not resources:
        print("No candidate resources found via CKAN search.", file=sys.stderr)
        sys.exit(1)

    checked = 0
    for r in resources:
        url = r["url"]
        obj = try_load_resource(url)
        checked += 1

        if isinstance(obj, pd.ExcelFile):
            for sh in obj.sheet_names:
                try:
                    df = pd.read_excel(obj, sheet_name=sh, dtype=str)
                except Exception:
                    continue
                info = detect_hk_occ_table(df)
                if not info:
                    continue

                ctry_col = None
                for c in df.columns:
                    s = str(c).lower()
                    if "citizenship" in s or "country" in s or "place of birth" in s:
                        ctry_col = c
                        break
                if ctry_col is None:
                    ctry_col = df.columns[0]

                occ_col = info["occ_cols"][0]

                y2021 = None
                y2025 = None
                for c in df.columns:
                    if re.search(r"\b2021\b", str(c)) and ("total" in str(c).lower() or "annual" in str(c).lower() or "sum" in str(c).lower()):
                        y2021 = c
                    if re.search(r"\b2025\b", str(c)) and ("total" in str(c).lower() or "annual" in str(c).lower() or "sum" in str(c).lower()):
                        y2025 = c
                if y2021 is None:
                    y2021 = next((c for c in df.columns if re.search(r"\b2021\b", str(c))), None)
                if y2025 is None:
                    y2025 = next((c for c in df.columns if re.search(r"\b2025\b", str(c))), None)
                if not y2021 or not y2025:
                    continue

                out = extract_top_occ(df, occ_col, ctry_col, y2021, y2025)
                if out.empty:
                    continue

                csv_path = OUT_DIR / "hk_main_occupations_2021_2025.csv"
                png_path = OUT_DIR / "hk_main_occupations_2021_2025.png"
                out.to_csv(csv_path, index=False, encoding="utf-8")

                x = range(len(out))
                w = 0.4
                plt.figure(figsize=(12, 6))
                plt.bar([i - w/2 for i in x], out["Y2021"].values, width=w, label="2021")
                plt.bar([i + w/2 for i in x], out["Y2025"].values, width=w, label="2025")
                plt.xticks(list(x), out["Occupation"].tolist(), rotation=60, ha="right")
                plt.title("Hong Kong immigrants in Canada — main occupations (2021 vs 2025)")
                plt.ylabel("Count")
                plt.legend()
                plt.tight_layout()
                plt.savefig(png_path, dpi=200)
                plt.close()

                print("FOUND_RESOURCE_URL=" + url)
                print("FOUND_SHEET=" + sh)
                print(csv_path)
                print(png_path)
                return

        if isinstance(obj, pd.DataFrame):
            df = obj
            info = detect_hk_occ_table(df)
            if not info:
                continue
            print("Potential match (CSV) but extractor expects yearly total columns; please share columns head.", file=sys.stderr)
            print(url, file=sys.stderr)
            sys.exit(1)

    print("No public resource found that contains Hong Kong + Occupation + 2021/2025 in the same table.", file=sys.stderr)
    print("Checked resources:", checked, file=sys.stderr)
    for r in resources[:25]:
        print(r["url"], file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
