from pathlib import Path
import sys, json, re
import urllib.request
import urllib.parse
import pandas as pd
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
if not (base / "data").exists() and (base.parent / "data").exists():
    base = base.parent
OUT_DIR = base / "outputs_demographics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API = "https://open.canada.ca/data/en/api/3/action/package_search"
EE_OCC_URL = "https://ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-EE_Admissions-Occ.xlsx"

def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()

def ckan_search(q, rows=100):
    url = f"{API}?q={urllib.parse.quote(q)}&rows={rows}"
    return json.loads(http_get(url).decode("utf-8"))

def parse_num(x):
    s = str(x).strip()
    if s in {"", "nan", "NaN", "None", "--"}:
        return 0.0
    s = s.replace(",", "")
    v = pd.to_numeric(s, errors="coerce")
    return float(v) if pd.notna(v) else 0.0

def parse_ee_matrix_format(ee_raw, years_req):
    if ee_raw.shape[0] < 6 or ee_raw.shape[1] < 20:
        return None

    year_total_cols = {}
    for idx, v in enumerate(ee_raw.iloc[3].tolist()):
        if pd.isna(v):
            continue
        m = re.fullmatch(r"\s*(\d{4})\s*Total\s*", str(v))
        if m:
            year_total_cols[int(m.group(1))] = idx

    if not all(y in year_total_cols for y in years_req):
        return None

    province_total_rows = []
    for i in range(5, ee_raw.shape[0]):
        label = str(ee_raw.iat[i, 0]).strip()
        if not label or label.lower() == "nan":
            continue
        if (
            label.endswith(" Total")
            and label not in {"Total"}
            and "All Canada" not in label
            and "Occupation" not in label
        ):
            province_total_rows.append((i, label[:-6].strip()))

    if not province_total_rows:
        return None

    rows = []
    start = 5
    for idx, province in province_total_rows:
        block = ee_raw.iloc[start:idx]
        for _, r in block.iterrows():
            occ = str(r.iloc[1]).strip()
            if not occ or occ.lower() == "nan":
                continue
            for y in years_req:
                col = year_total_cols[y]
                rows.append(
                    {
                        "Year": int(y),
                        "Province": province,
                        "Occupation": occ,
                        "Admissions": parse_num(r.iloc[col]),
                    }
                )
        start = idx + 1

    if not rows:
        return None

    d = pd.DataFrame(rows)
    d["Admissions"] = pd.to_numeric(d["Admissions"], errors="coerce").fillna(0.0)
    return d

def write_occ_outputs(out):
    csv_path = OUT_DIR / "hk_main_occupations_2021_2025.csv"
    png_path = OUT_DIR / "hk_main_occupations_2021_2025.png"
    out.to_csv(csv_path, index=False, encoding="utf-8")

    y = range(len(out))
    w = 0.4
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.barh([i + w/2 for i in y], out["Y2025"].values, height=w, label="2025")
    ax.barh([i - w/2 for i in y], out["Y2021"].values, height=w, label="2021")
    ax.set_yticks(list(y))
    ax.set_yticklabels(out["Occupation"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Hong Kong immigrants in Canada — main occupations (2021 vs 2025)")
    ax.set_xlabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return csv_path, png_path

def build_proxy_occ_from_ee():
    years_req = [2021, 2025]
    ee_raw = pd.read_excel(EE_OCC_URL, header=None)
    d = parse_ee_matrix_format(ee_raw, years_req)
    if d is None or d.empty:
        return None

    c = d.groupby(["Year", "Occupation"], as_index=False)["Admissions"].sum()
    w = c.pivot(index="Occupation", columns="Year", values="Admissions").fillna(0.0)
    w = w.reset_index()
    for y in years_req:
        if y not in w.columns:
            w[y] = 0.0
    w = w.rename(columns={2021: "Y2021", 2025: "Y2025"})
    w = w.sort_values("Y2025", ascending=False).head(20)
    return w[["Occupation", "Y2021", "Y2025"]]

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
    ckan_ok = False
    for q in queries:
        try:
            resp = ckan_search(q, rows=200)
            ckan_ok = True
        except Exception as e:
            print(f"CKAN search warning: {e}", file=sys.stderr)
            continue
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

    # Fallback URLs if CKAN API is blocked (WAF/HTML response).
    fallback_urls = [EE_OCC_URL]
    for u in fallback_urls:
        key = (u, "fallback")
        if key not in seen:
            seen.add(key)
            resources.append({"url": u, "name": "fallback", "pkg": "direct-url"})

    if not ckan_ok:
        print("CKAN API unavailable; using direct resource fallback.", file=sys.stderr)
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
        proxy = build_proxy_occ_from_ee()
        if proxy is None or proxy.empty:
            print("No candidate resources found and EE proxy build failed.", file=sys.stderr)
            sys.exit(1)
        csv_path, png_path = write_occ_outputs(proxy)
        print("FALLBACK_MODE=EE_PROXY")
        print(csv_path)
        print(png_path)
        return

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

                csv_path, png_path = write_occ_outputs(out)

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

    proxy = build_proxy_occ_from_ee()
    if proxy is not None and not proxy.empty:
        csv_path, png_path = write_occ_outputs(proxy)
        print("FALLBACK_MODE=EE_PROXY")
        print(csv_path)
        print(png_path)
        return

    print("No public resource found that contains Hong Kong + Occupation + 2021/2025 in the same table.", file=sys.stderr)
    print("Checked resources:", checked, file=sys.stderr)
    for r in resources[:25]:
        print(r["url"], file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
