from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROVINCES = ["Alberta", "British Columbia", "Ontario", "Quebec"]

PIDS = {
    "gdp": "36100222",
    "unemp": "14100287",
    "wage": "14100223",
    "vac": "14100371",
    "pop": "17100005",
}

URL_FMT = "https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-eng.zip"
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

def to_year(ref_series):
    y = pd.to_numeric(ref_series, errors="coerce")
    if y.notna().any():
        return y
    d = pd.to_datetime(ref_series, errors="coerce")
    return d.dt.year

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

def find_dim_col(df, needle):
    n = needle.lower()
    for c in df.columns:
        s = df[c].dropna().astype(str).str.lower()
        if s.str.contains(n, regex=False).any():
            return c
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

def download_zip(pid, path, retries=3, timeout=120):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = URL_FMT.format(pid=pid)
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
        f"Failed to download Statistics Canada table {pid} after {retries} attempts: {last_error}"
    ) from last_error

def ensure_zip(pid, cache_dirs):
    filename = f"{pid}-eng.zip"
    for cache_dir in cache_dirs:
        candidate = cache_dir / filename
        if candidate.exists():
            return candidate

    primary = cache_dirs[0]
    downloaded = download_zip(pid, primary / filename)

    for cache_dir in cache_dirs[1:]:
        mirror = cache_dir / filename
        if not mirror.exists():
            mirror.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(downloaded, mirror)
            except OSError:
                pass

    return downloaded

def load_zip_csv(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        data_names = [n for n in names if "meta" not in n.lower()]
        name = data_names[0] if data_names else names[0]
        with z.open(name) as f:
            return pd.read_csv(f, dtype=str, low_memory=False)

def build_population(pop_raw):
    ref = "REF_DATE" if "REF_DATE" in pop_raw.columns else col_like(pop_raw.columns, "ref", "date")
    geo = "GEO" if "GEO" in pop_raw.columns else col_like(pop_raw.columns, "geo")
    val = "VALUE" if "VALUE" in pop_raw.columns else col_like(pop_raw.columns, "value")
    sf = "SCALAR_FACTOR" if "SCALAR_FACTOR" in pop_raw.columns else col_like(pop_raw.columns, "scalar", "factor")
    sex_col = col_like(pop_raw.columns, "sex") or col_like(pop_raw.columns, "gender")
    age_col = col_like(pop_raw.columns, "age group") or col_like(pop_raw.columns, "age")
    if not all([ref, geo, val, sf, sex_col, age_col]):
        print("Population table missing required columns", file=sys.stderr)
        sys.exit(1)

    total_sex = pick_value_label(pop_raw[sex_col], ["both sexes", "total - sex", "total - gender", "total"])
    all_ages = pick_value_label(pop_raw[age_col], ["all ages", "total"])
    if not total_sex or not all_ages:
        print("Population table cannot find Total sex/gender or All ages/Total labels", file=sys.stderr)
        sys.exit(1)

    d = pop_raw[(pop_raw[geo].isin(PROVINCES)) & (pop_raw[sex_col] == total_sex) & (pop_raw[age_col] == all_ages)].copy()
    d["Year"] = to_year(d[ref])
    d["VALUE_NUM"] = pd.to_numeric(d[val], errors="coerce")
    d = d.dropna(subset=["Year", "VALUE_NUM"])
    d["mult"] = d[sf].apply(scalar_multiplier)
    d["Population"] = d["VALUE_NUM"] * d["mult"]
    return d[[geo, "Year", "Population"]].rename(columns={geo: "Province"})

def build_gdp(gdp_raw):
    ref = "REF_DATE" if "REF_DATE" in gdp_raw.columns else col_like(gdp_raw.columns, "ref", "date")
    geo = "GEO" if "GEO" in gdp_raw.columns else col_like(gdp_raw.columns, "geo")
    val = "VALUE" if "VALUE" in gdp_raw.columns else col_like(gdp_raw.columns, "value")
    sf = "SCALAR_FACTOR" if "SCALAR_FACTOR" in gdp_raw.columns else col_like(gdp_raw.columns, "scalar", "factor")
    if not all([ref, geo, val, sf]):
        print("GDP table missing required columns", file=sys.stderr)
        sys.exit(1)

    prices_col = find_dim_col(gdp_raw, "current") or col_like(gdp_raw.columns, "prices")
    est_col = find_dim_col(gdp_raw, "gross domestic product") or col_like(gdp_raw.columns, "estimates")
    if not prices_col or not est_col:
        print("GDP table missing Prices/Estimates columns", file=sys.stderr)
        sys.exit(1)

    current = pick_value_label(gdp_raw[prices_col], ["current"])
    gdp_label = pick_value_label(gdp_raw[est_col], ["gross domestic product"])
    if not current or not gdp_label:
        print("GDP table cannot find Current prices or GDP label", file=sys.stderr)
        sys.exit(1)

    d = gdp_raw[(gdp_raw[geo].isin(PROVINCES)) & (gdp_raw[prices_col] == current) & (gdp_raw[est_col] == gdp_label)].copy()
    d["Year"] = to_year(d[ref])
    d["VALUE_NUM"] = pd.to_numeric(d[val], errors="coerce")
    d = d.dropna(subset=["Year", "VALUE_NUM"])
    d["mult"] = d[sf].apply(scalar_multiplier)
    d["GDP"] = d["VALUE_NUM"] * d["mult"]
    return d[[geo, "Year", "GDP"]].rename(columns={geo: "Province"})

def build_unemployment(unemp_raw):
    ref = "REF_DATE" if "REF_DATE" in unemp_raw.columns else col_like(unemp_raw.columns, "ref", "date")
    geo = "GEO" if "GEO" in unemp_raw.columns else col_like(unemp_raw.columns, "geo")
    val = "VALUE" if "VALUE" in unemp_raw.columns else col_like(unemp_raw.columns, "value")
    sf = "SCALAR_FACTOR" if "SCALAR_FACTOR" in unemp_raw.columns else col_like(unemp_raw.columns, "scalar", "factor")
    if not all([ref, geo, val]):
        print("Unemployment table missing required columns", file=sys.stderr)
        sys.exit(1)

    char_col = find_dim_col(unemp_raw, "unemployment rate")
    if not char_col:
        print("Unemployment table cannot find a column containing 'Unemployment rate'", file=sys.stderr)
        sys.exit(1)
    unemp_label = pick_value_label(unemp_raw[char_col], ["unemployment rate"])
    if not unemp_label:
        print("Unemployment table cannot find 'Unemployment rate' label", file=sys.stderr)
        sys.exit(1)

    sex_col = col_like(unemp_raw.columns, "sex") or col_like(unemp_raw.columns, "gender")
    age_col = col_like(unemp_raw.columns, "age group") or col_like(unemp_raw.columns, "age")

    d = unemp_raw[unemp_raw[geo].isin(PROVINCES)].copy()
    d = d[d[char_col] == unemp_label].copy()

    if sex_col:
        total_sex = pick_value_label(d[sex_col], ["both sexes", "total - sex", "total - gender", "total"])
        if total_sex:
            d = d[d[sex_col] == total_sex].copy()

    if age_col:
        age15p = pick_value_label(d[age_col], ["15 years and over"])
        if age15p:
            d = d[d[age_col] == age15p].copy()

    d["Year"] = to_year(d[ref])
    d["VALUE_NUM"] = pd.to_numeric(d[val], errors="coerce")
    d = d.dropna(subset=["Year", "VALUE_NUM"])
    if sf:
        d["mult"] = d[sf].apply(scalar_multiplier)
        d["Unemployment_rate"] = d["VALUE_NUM"] * d["mult"]
    else:
        d["Unemployment_rate"] = d["VALUE_NUM"]
    out = d[[geo, "Year", "Unemployment_rate"]].rename(columns={geo: "Province"})
    return out.groupby(["Province", "Year"], as_index=False)["Unemployment_rate"].mean()

def build_wages(wage_raw):
    ref = "REF_DATE" if "REF_DATE" in wage_raw.columns else col_like(wage_raw.columns, "ref", "date")
    geo = "GEO" if "GEO" in wage_raw.columns else col_like(wage_raw.columns, "geo")
    val = "VALUE" if "VALUE" in wage_raw.columns else col_like(wage_raw.columns, "value")
    sf = "SCALAR_FACTOR" if "SCALAR_FACTOR" in wage_raw.columns else col_like(wage_raw.columns, "scalar", "factor")

    dtype_col = find_dim_col(wage_raw, "average weekly earnings") or col_like(wage_raw.columns, "estimate") or col_like(wage_raw.columns, "data type")
    if not all([ref, geo, val, dtype_col]):
        print("Wage table missing required columns", file=sys.stderr)
        sys.exit(1)

    earn_label = pick_value_label(wage_raw[dtype_col], ["average weekly earnings"])
    if not earn_label:
        print("Wage table cannot find 'Average weekly earnings' label", file=sys.stderr)
        sys.exit(1)

    d = wage_raw[wage_raw[geo].isin(PROVINCES)].copy()
    d = d[d[dtype_col] == earn_label].copy()

    d["Year"] = to_year(d[ref])
    d["VALUE_NUM"] = pd.to_numeric(d[val], errors="coerce")
    d = d.dropna(subset=["Year", "VALUE_NUM"])
    if sf:
        d["mult"] = d[sf].apply(scalar_multiplier)
        d["Avg_weekly_earnings"] = d["VALUE_NUM"] * d["mult"]
    else:
        d["Avg_weekly_earnings"] = d["VALUE_NUM"]

    out = d[[geo, "Year", "Avg_weekly_earnings"]].rename(columns={geo: "Province"})
    return out.groupby(["Province", "Year"], as_index=False)["Avg_weekly_earnings"].mean()

def build_vacancies(vac_raw):
    ref = "REF_DATE" if "REF_DATE" in vac_raw.columns else col_like(vac_raw.columns, "ref", "date")
    geo = "GEO" if "GEO" in vac_raw.columns else col_like(vac_raw.columns, "geo")
    val = "VALUE" if "VALUE" in vac_raw.columns else col_like(vac_raw.columns, "value")
    sf = "SCALAR_FACTOR" if "SCALAR_FACTOR" in vac_raw.columns else col_like(vac_raw.columns, "scalar", "factor")
    if not all([ref, geo, val]):
        print("Vacancy table missing required columns", file=sys.stderr)
        sys.exit(1)

    stat_col = find_dim_col(vac_raw, "job vacancies") or find_dim_col(vac_raw, "job vacancy rate") or col_like(vac_raw.columns, "statistics")
    if not stat_col:
        print("Vacancy table cannot find a Statistics column", file=sys.stderr)
        sys.exit(1)

    vac_label = pick_value_label(vac_raw[stat_col], ["job vacancies"])
    rate_label = pick_value_label(vac_raw[stat_col], ["job vacancy rate"])
    if not vac_label or not rate_label:
        print("Vacancy table cannot find 'Job vacancies' or 'Job vacancy rate' labels", file=sys.stderr)
        sys.exit(1)

    d = vac_raw[vac_raw[geo].isin(PROVINCES)].copy()
    d = d.rename(columns={geo: "Province"})
    d["Year"] = to_year(d[ref])
    d["VALUE_NUM"] = pd.to_numeric(d[val], errors="coerce")
    d = d.dropna(subset=["Year", "VALUE_NUM"])
    if sf:
        d["mult"] = d[sf].apply(scalar_multiplier)
        d["V"] = d["VALUE_NUM"] * d["mult"]
    else:
        d["V"] = d["VALUE_NUM"]

    v = d[d[stat_col] == vac_label][["Province", "Year", "V"]].rename(columns={"V": "Job_vacancies"})
    r = d[d[stat_col] == rate_label][["Province", "Year", "V"]].rename(columns={"V": "Job_vacancy_rate"})

    out = v.merge(r, on=["Province", "Year"], how="outer")

    counts = d[d[stat_col].isin([vac_label, rate_label])].groupby(["Province", "Year"]).size().reset_index(name="vac_rows")
    out = out.merge(counts, on=["Province", "Year"], how="left")
    out["vac_months"] = out["vac_rows"] / 2.0
    out = out.drop(columns=["vac_rows"])

    return out.groupby(["Province", "Year"], as_index=False)[["Job_vacancies", "Job_vacancy_rate", "vac_months"]].mean()


def grouped_bar(ax, df, ycol, title, ylabel, years):
    x = np.arange(len(PROVINCES))
    w = 0.35
    for i, yr in enumerate(years):
        s = df[df["Year"] == yr].set_index("Province").reindex(PROVINCES)
        ax.bar(x + (i - (len(years)-1)/2)*w, s[ycol].values, width=w, label=str(yr))
    ax.set_xticks(x)
    ax.set_xticklabels(PROVINCES, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()

def main():
    base = find_project_root()
    cache_dirs = cache_search_order(base)
    cache_dirs[0].mkdir(parents=True, exist_ok=True)
    out_dir = base / "outputs_demographics"
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = {pid: ensure_zip(pid, cache_dirs) for pid in PIDS.values()}

    pop_raw = load_zip_csv(resolved[PIDS['pop']])
    gdp_raw = load_zip_csv(resolved[PIDS['gdp']])
    unemp_raw = load_zip_csv(resolved[PIDS['unemp']])
    wage_raw = load_zip_csv(resolved[PIDS['wage']])
    vac_raw = load_zip_csv(resolved[PIDS['vac']])

    pop = build_population(pop_raw)
    gdp = build_gdp(gdp_raw)
    unemp = build_unemployment(unemp_raw)
    wages = build_wages(wage_raw)
    vac = build_vacancies(vac_raw)

    years = [2021, 2025]
    gdp_max_year = int(gdp["Year"].max())

    rows = []
    for yr in years:
        gdp_year_used = yr if (gdp["Year"] == yr).any() else gdp_max_year
        g = gdp[gdp["Year"] == gdp_year_used].copy()
        p = pop[pop["Year"] == yr].copy()
        u = unemp[unemp["Year"] == yr].copy()
        w = wages[wages["Year"] == yr].copy()
        v = vac[vac["Year"] == yr].copy()

        m = pd.DataFrame({"Province": PROVINCES}).merge(p, on="Province", how="left")
        m = m.merge(u[["Province", "Unemployment_rate"]], on="Province", how="left")
        m = m.merge(w[["Province", "Avg_weekly_earnings"]], on="Province", how="left")
        m = m.merge(v[["Province", "Job_vacancies", "Job_vacancy_rate", "vac_months"]], on="Province", how="left")
        m = m.merge(g[["Province", "GDP"]], on="Province", how="left")
        m["Year"] = yr
        m["GDP_year_used"] = gdp_year_used
        m["GDP_per_capita"] = m["GDP"] / m["Population"]
        rows.append(m)

    out = pd.concat(rows, ignore_index=True)
    out = out[[
        "Province", "Year", "GDP_year_used",
        "GDP", "GDP_per_capita",
        "Unemployment_rate",
        "Avg_weekly_earnings",
        "Job_vacancies", "Job_vacancy_rate", "vac_months",
        "Population"
    ]].copy()

    out_csv = out_dir / "econ_AB_BC_ON_QC_2021_2025.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    grouped_bar(axs[0,0], out, "GDP_per_capita", "GDP per capita (GDP year may differ for 2025)", "CAD per person", years)
    grouped_bar(axs[0,1], out, "Unemployment_rate", "Unemployment rate (annual avg)", "%", years)
    grouped_bar(axs[1,0], out, "Avg_weekly_earnings", "Average weekly earnings (annual avg)", "CAD per week", years)
    grouped_bar(axs[1,1], out, "Job_vacancy_rate", "Job vacancy rate (annual avg)", "%", years)
    fig.tight_layout()
    fig_path = out_dir / "econ_AB_BC_ON_QC_2021_2025_metrics.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
