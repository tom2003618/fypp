import re
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_project_root(marker: str) -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    return here

PROJECT_ROOT = find_project_root("data/98100351-eng.zip")
VM_ZIP = PROJECT_ROOT / "data/98100351-eng.zip"
POB_ZIP = PROJECT_ROOT / "data/98100349-eng.zip"
OUT_DIR = PROJECT_ROOT / "outputs_demographics"
TARGET_GEO = "Canada"

                                                      
KEEP_AGE = [
    "0 to 14 years",
    "15 to 24 years",
    "25 to 34 years",
    "35 to 44 years",
    "45 to 54 years",
    "55 to 64 years",
    "65 to 74 years",
]

def pick_data_csv_from_zip(zip_path: Path) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv") and "meta" not in n.lower()]
        if not names:
            raise RuntimeError(f"No data CSV found inside {zip_path}.")
        best = max(names, key=lambda n: z.getinfo(n).file_size)
        return z.open(best).read()

def read_statcan_fulltable(zip_path: Path) -> pd.DataFrame:
    b = pick_data_csv_from_zip(zip_path)
    df = pd.read_csv(io.BytesIO(b), dtype=str, encoding="utf-8", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def find_col(df, patterns):
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in df.columns:
            if rx.search(str(c)):
                return c
    return None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

def clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def filter_geo(df: pd.DataFrame, geo_col: str) -> pd.DataFrame:
    g = clean_text(df[geo_col])
    m = g.str.fullmatch(re.escape(TARGET_GEO), case=False, na=False)
    if m.any():
        return df[m].copy()
    m2 = g.str.contains(r"\b" + re.escape(TARGET_GEO) + r"\b", case=False, na=False)
    if m2.any():
        return df[m2].copy()
    return df.copy()

def filter_ref_date_latest(df: pd.DataFrame) -> pd.DataFrame:
    c = find_col(df, [r"^REF_DATE$"])
    if not c:
        return df
    y = pd.to_numeric(clean_text(df[c]).str.extract(r"(\d{4})", expand=False), errors="coerce")
    if y.notna().any():
        latest = int(y.dropna().max())
        return df[y == latest].copy()
    return df

def filter_gender_stats(df: pd.DataFrame) -> pd.DataFrame:
    c_gender = find_col(df, [r"^Gender\b", r"^Sex\b"])
    c_stat = find_col(df, [r"^Statistics\b", r"Statistic"])
    if c_gender:
        g = clean_text(df[c_gender])
        m = g.str.contains("both", case=False, na=False) | g.str.contains("total", case=False, na=False)
        if m.any():
            df = df[m].copy()
    if c_stat:
        st = clean_text(df[c_stat])
        m = (st.str.contains("number", case=False, na=False) | st.str.contains("population", case=False, na=False) | st.str.contains("counts", case=False, na=False)) & (~st.str.contains("percent", case=False, na=False))
        if m.any():
            df = df[m].copy()
    return df

def age_order_key(label: str) -> int:
    s = norm(label)
    if "under" in s:
        return 0
    nums = re.findall(r"\d+", s)
    if nums:
        return int(nums[0])
    return 10**9

def build_pct_by_age(df: pd.DataFrame, age_col: str, value: pd.Series, group_name: str) -> pd.DataFrame:
    tmp = pd.DataFrame({"age_group": clean_text(df[age_col]), "value": value})
    tmp = tmp.dropna(subset=["age_group", "value"]).copy()
    tmp = tmp[~tmp["age_group"].str.contains(r"^Total", case=False, na=False)].copy()
    tmp = tmp[~tmp["age_group"].str.contains(r"\byears\s+and\s+over\b", case=False, na=False)].copy()
    tmp = tmp[tmp["age_group"].isin(KEEP_AGE)].copy()
    tmp = tmp[~tmp["age_group"].str.contains(r"\byears\s+and\s+over\b", case=False, na=False)].copy()
    g = tmp.groupby("age_group", as_index=False)["value"].sum()
    total = g["value"].sum()
    if total and total > 0:
        g["pct"] = g["value"] / total * 100.0
    else:
        g["pct"] = 0.0
    g["group"] = group_name
    g["population"] = g["value"]
    return g[["group", "age_group", "pct", "population"]]

def pick_wide_col(df: pd.DataFrame, must_contain: str, target_regexes) -> str | None:
    cols = [c for c in df.columns if must_contain.lower() in str(c).lower()]
    scored = []
    for c in cols:
        sc = 0
        for rx in target_regexes:
            if re.search(rx, str(c), flags=re.IGNORECASE):
                sc += 10
        if sc > 0:
            scored.append((sc, c))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda x: (x[0], -len(x[1])))
    return scored[0][1]

def extract_black_white(vm: pd.DataFrame) -> pd.DataFrame:
    c_geo = find_col(vm, [r"^GEO$"])
    c_age = find_col(vm, [r"^Age\b"])
    if not c_age:
        raise RuntimeError(f"Age column not found in VM. Columns={list(vm.columns)[:40]}")
    vm = filter_ref_date_latest(vm)
    if c_geo:
        vm = filter_geo(vm, c_geo)
    vm = filter_gender_stats(vm)

    c_black = pick_wide_col(vm, "Visible minority", [r":\s*Black\b", r"Black\["])
    c_white = pick_wide_col(vm, "Visible minority", [r"Not a visible minority", r"Not a visible minority\["])

    if not c_black or not c_white:
        raise RuntimeError(f"Cannot find Black/White wide columns. black={c_black}, white={c_white}")

    black_pct = build_pct_by_age(vm, c_age, to_num(vm[c_black]), "Black")
    white_pct = build_pct_by_age(vm, c_age, to_num(vm[c_white]), "White")
    return pd.concat([black_pct, white_pct], ignore_index=True)

def extract_east_asian_india(pob: pd.DataFrame) -> pd.DataFrame:
    c_geo = find_col(pob, [r"^GEO$"])
    c_age = find_col(pob, [r"^Age\b"])
    c_pob = find_col(pob, [r"^Place of birth\b"])
    if not c_age or not c_pob:
        raise RuntimeError(f"Age/Place-of-birth columns not found in POB. Columns={list(pob.columns)[:60]}")

    pob = filter_ref_date_latest(pob)
    if c_geo:
        pob = filter_geo(pob, c_geo)
    pob = filter_gender_stats(pob)

    c_total = None
    for c in pob.columns:
        if re.search(r"Period of immigration.*Total immigrant population", str(c), flags=re.IGNORECASE):
            c_total = c
            break
    if c_total is None:
        raise RuntimeError(f"Total immigrant population column not found in POB. Columns={list(pob.columns)[:60]}")

    pob[c_age] = clean_text(pob[c_age])
    pob[c_pob] = clean_text(pob[c_pob])
    v = to_num(pob[c_total])

    pob2 = pob.copy()
    pob2["value"] = v
    pob2 = pob2.dropna(subset=[c_age, c_pob, "value"]).copy()

    m_hk = pob2[c_pob].str.contains(r"\bHong Kong\b", case=False, na=False)
    m_cn = pob2[c_pob].str.contains(r"\bChina\b", case=False, na=False) & (~m_hk)
    m_ea = m_hk | m_cn

    m_in = pob2[c_pob].str.contains(r"\bIndia\b", case=False, na=False)

    ea_df = pob2[m_ea].copy()
    in_df = pob2[m_in].copy()

    if ea_df.empty or in_df.empty:
        raise RuntimeError(f"POB filter empty. EastAsian_rows={len(ea_df)} India_rows={len(in_df)}. Sample POB values: {pob2[c_pob].drop_duplicates().head(20).tolist()}")

    ea_pct = build_pct_by_age(ea_df, c_age, ea_df["value"], "East Asian (China+Hong Kong)")
    in_pct = build_pct_by_age(in_df, c_age, in_df["value"], "India")
    return pd.concat([ea_pct, in_pct], ignore_index=True)


def plot_lines(df: pd.DataFrame, out_png: Path) -> None:
    pivot = df.pivot_table(index="age_group", columns="group", values="pct", aggfunc="sum").fillna(0.0)
    ages = [a for a in KEEP_AGE if a in pivot.index]
    if not ages:
        ages = sorted(pivot.index.tolist(), key=age_order_key)
    pivot = pivot.loc[ages]
    ax = pivot.plot(figsize=(12, 6))
    ax.set_xlabel("Age group")
    ax.set_ylabel("Percent within group")
    ax.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    pivot = df.pivot_table(index="age_group", columns="group", values="pct", aggfunc="sum").fillna(0.0)
    ages = [a for a in KEEP_AGE if a in pivot.index]
    if not ages:
        ages = sorted(pivot.index.tolist(), key=age_order_key)
    pivot = pivot.loc[ages]
    groups = ["East Asian (China+Hong Kong)", "India", "Black", "White"]
    for g in groups:
        if g not in pivot.columns:
            pivot[g] = 0.0
    pivot = pivot[groups]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(f"Age distribution by group (percent within group) - {TARGET_GEO}")
    ax.set_xlabel("Group")
    ax.set_ylabel("Age group")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent within group")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_population(df: pd.DataFrame, out_png: Path) -> None:
    pivot = df.pivot_table(index="age_group", columns="group", values="population", aggfunc="sum").fillna(0.0)
    ages = [a for a in KEEP_AGE if a in pivot.index]
    if not ages:
        ages = sorted(pivot.index.tolist(), key=age_order_key)
    pivot = pivot.loc[ages]
    groups = ["East Asian (China+Hong Kong)", "India", "Black", "White"]
    for g in groups:
        if g not in pivot.columns:
            pivot[g] = 0.0
    pivot = pivot[groups]
    x = np.arange(len(pivot.index))
    w = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, grp in enumerate(groups):
        ax.bar(x + i * w, pivot[grp].values, width=w, label=grp)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Population")
    ax.set_title(f"Population by age group and ethnicity - {TARGET_GEO} (2021 Census)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vm = read_statcan_fulltable(VM_ZIP)
    pob = read_statcan_fulltable(POB_ZIP)
    part1 = extract_black_white(vm)
    part2 = extract_east_asian_india(pob)
    all_df = pd.concat([part2, part1], ignore_index=True)
    all_df.to_csv(OUT_DIR / "age_by_group_CANADA.csv", index=False)
    plot_lines(all_df, OUT_DIR / "fig_age_by_group_CANADA.png")
    plot_heatmap(all_df, OUT_DIR / "fig_age_by_group_heatmap_CANADA.png")
    plot_population(all_df, OUT_DIR / "fig_age_by_group_population_CANADA.png")
    print("Saved:", OUT_DIR / "age_by_group_CANADA.csv")
    print("Saved:", OUT_DIR / "fig_age_by_group_CANADA.png")
    print("Saved:", OUT_DIR / "fig_age_by_group_heatmap_CANADA.png")
    print("Saved:", OUT_DIR / "fig_age_by_group_population_CANADA.png")

if __name__ == "__main__":
    main()
