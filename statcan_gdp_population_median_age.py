from pathlib import Path
import sys
import time
import urllib.request
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

PROVINCES = ["Alberta", "British Columbia", "Ontario", "Quebec"]

GDP_ZIP_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/36100711-eng.zip"
POP_ZIP_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip"
PROJECT_MARKERS = (
    ".git",
    "econ_2021_2025.py",
    "canada_federal_vote_share_2000_2025.csv",
)


def col_like(cols, *tokens):
    toks = [t.lower() for t in tokens]
    for c in cols:
        lc = str(c).lower()
        if all(t in lc for t in toks):
            return c
    return None


def parse_year(s):
    a = pd.to_numeric(s, errors="coerce")
    if a.notna().any():
        return a
    b = pd.to_datetime(s, errors="coerce")
    return b.dt.year


def scalar_multiplier(x):
    if x is None:
        return 1.0
    t = str(x).lower().replace(",", "")
    if "1000000" in t or "1 000 000" in t or "million" in t:
        return 1_000_000.0
    if "1000" in t or "1 000" in t or "thousand" in t:
        return 1_000.0
    return 1.0


def pick_value_label(values, contains_any):
    vals = pd.Series(values).dropna().astype(str).unique().tolist()
    for key in contains_any:
        k = key.lower()
        for v in vals:
            if k in v.lower():
                return v
    return None


def find_project_root():
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start


def cache_search_order(project_root):
    notebook_root = project_root / "notebooks"
    candidates = [
        project_root / "data" / "statcan_cache",
        notebook_root / "data" / "statcan_cache",
    ]
    ordered = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def download(url, path, retries=3, timeout=120):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                path.write_bytes(r.read())
            if not zipfile.is_zipfile(path):
                raise RuntimeError(f"Downloaded file is not a valid zip: {path}")
            return path
        except Exception as exc:
            last_error = exc
            if path.exists():
                path.unlink()
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 5))

    raise RuntimeError(
        f"Failed to download StatCan source {url} after {retries} attempts: {last_error}"
    ) from last_error


def ensure_zip(filename, url, cache_dirs):
    for cache_dir in cache_dirs:
        candidate = cache_dir / filename
        if candidate.exists():
            return candidate
    return download(url, cache_dirs[0] / filename)


def load_zip_csv(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        data_names = [n for n in names if "meta" not in n.lower()]
        name = data_names[0] if data_names else names[0]
        with z.open(name) as f:
            return pd.read_csv(f, dtype=str, low_memory=False)


def build_gdp(df):
    ref = "REF_DATE" if "REF_DATE" in df.columns else col_like(df.columns, "ref", "date")
    geo = "GEO" if "GEO" in df.columns else col_like(df.columns, "geo")
    val = "VALUE" if "VALUE" in df.columns else col_like(df.columns, "value")
    sf = "SCALAR_FACTOR" if "SCALAR_FACTOR" in df.columns else col_like(df.columns, "scalar", "factor")
    if not all([ref, geo, val, sf]):
        print("GDP table missing required columns", file=sys.stderr)
        sys.exit(1)

    price_col = col_like(df.columns, "prices")
    naics_col = col_like(df.columns, "north", "industry") or col_like(df.columns, "naics")
    if not price_col or not naics_col:
        print("GDP table missing Prices or NAICS column", file=sys.stderr)
        sys.exit(1)

    current_label = pick_value_label(df[price_col], ["current dollars", "current"])
    all_ind_label = pick_value_label(df[naics_col], ["all industries"])
    if not current_label or not all_ind_label:
        print("GDP table cannot find Current dollars or All industries label", file=sys.stderr)
        sys.exit(1)

    g = df[(df[price_col] == current_label) & (df[naics_col] == all_ind_label)].copy()
    g = g[g[geo].isin(PROVINCES)].copy()
    g["Year"] = parse_year(g[ref])
    g["VALUE_NUM"] = pd.to_numeric(g[val], errors="coerce")
    g = g.dropna(subset=["Year", "VALUE_NUM"])
    g["mult"] = g[sf].apply(scalar_multiplier)
    g["gdp_dollars"] = g["VALUE_NUM"] * g["mult"]
    g = g[[geo, "Year", "gdp_dollars"]].rename(columns={geo: "Province"})
    return g


def build_population_and_median_age(df):
    ref = "REF_DATE" if "REF_DATE" in df.columns else col_like(df.columns, "ref", "date")
    geo = "GEO" if "GEO" in df.columns else col_like(df.columns, "geo")
    val = "VALUE" if "VALUE" in df.columns else col_like(df.columns, "value")
    sf = "SCALAR_FACTOR" if "SCALAR_FACTOR" in df.columns else col_like(df.columns, "scalar", "factor")
    if not all([ref, geo, val, sf]):
        print("Population table missing required columns", file=sys.stderr)
        sys.exit(1)

    sex_col = col_like(df.columns, "sex") or col_like(df.columns, "gender")
    age_col = col_like(df.columns, "age group") or col_like(df.columns, "age")
    if not sex_col or not age_col:
        print("Population table missing Sex/Gender or Age column", file=sys.stderr)
        sys.exit(1)

    total_sex_label = pick_value_label(
        df[sex_col],
        ["both sexes", "total - gender", "total - sex", "total, gender", "total, sex", "total"],
    )
    if not total_sex_label:
        print("Population table cannot find a Total (Both sexes/Total gender/Total sex) label", file=sys.stderr)
        sys.exit(1)

    median_label = pick_value_label(df[age_col], ["median age"])
    all_ages_label = pick_value_label(df[age_col], ["all ages", "total"])
    if not median_label or not all_ages_label:
        print("Population table cannot find Median age or All ages/Total label", file=sys.stderr)
        sys.exit(1)

    d = df[df[geo].isin(PROVINCES) & (df[sex_col] == total_sex_label)].copy()
    d["Year"] = parse_year(d[ref])
    d["VALUE_NUM"] = pd.to_numeric(d[val], errors="coerce")
    d = d.dropna(subset=["Year"])
    d["mult"] = d[sf].apply(scalar_multiplier)
    d["value_scaled"] = d["VALUE_NUM"] * d["mult"]

    pop = d[d[age_col] == all_ages_label][[geo, "Year", "value_scaled"]].rename(
        columns={geo: "Province", "value_scaled": "Population"}
    )
    med = d[d[age_col] == median_label][[geo, "Year", "value_scaled"]].rename(
        columns={geo: "Province", "value_scaled": "Median_age"}
    )

    return pop, med


def main():
    base = find_project_root()
    cache_dirs = cache_search_order(base)
    cache_dirs[0].mkdir(parents=True, exist_ok=True)
    out_dir = base / "outputs_demographics"
    out_dir.mkdir(parents=True, exist_ok=True)

    gdp_zip = ensure_zip("36100711-eng.zip", GDP_ZIP_URL, cache_dirs)
    pop_zip = ensure_zip("17100005-eng.zip", POP_ZIP_URL, cache_dirs)

    gdp_raw = load_zip_csv(gdp_zip)
    pop_raw = load_zip_csv(pop_zip)

    gdp = build_gdp(gdp_raw)
    pop, med = build_population_and_median_age(pop_raw)

    year = int(min(gdp["Year"].max(), pop["Year"].max(), med["Year"].max()))

    gdp_y = gdp[gdp["Year"] == year].copy()
    pop_y = pop[pop["Year"] == year].copy()
    med_y = med[med["Year"] == year].copy()

    df = gdp_y.merge(pop_y, on=["Province", "Year"], how="inner").merge(med_y, on=["Province", "Year"], how="inner")
    if df.empty:
        print("No matching rows after merging GDP/Population/Median age", file=sys.stderr)
        sys.exit(1)

    df["GDP_per_capita"] = df["gdp_dollars"] / df["Population"]
    df = df[["Province", "Year", "Median_age", "Population", "gdp_dollars", "GDP_per_capita"]].copy()
    df = df.sort_values("GDP_per_capita", ascending=False)

    out_csv = out_dir / f"AB_BC_ON_QC_age_econ_statcan_{year}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    plt.figure()
    plt.bar(df["Province"], df["GDP_per_capita"])
    plt.ylabel("CAD per person")
    plt.title(f"GDP per capita - AB, BC, ON, QC ({year})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"gdp_per_capita_AB_BC_ON_QC_{year}.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(df["Province"], df["Median_age"])
    plt.ylabel("Years")
    plt.title(f"Median age - AB, BC, ON, QC ({year})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"median_age_AB_BC_ON_QC_{year}.png", dpi=200)
    plt.close()

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
