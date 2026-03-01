from pathlib import Path
import re
import sys
import pandas as pd

PROVINCES_4 = ["Alberta", "British Columbia", "Ontario", "Quebec"]
PROV_ALL = {
    "Canada",
    "Newfoundland and Labrador","Prince Edward Island","Nova Scotia","New Brunswick",
    "Quebec","Ontario","Manitoba","Saskatchewan","Alberta","British Columbia",
    "Yukon","Northwest Territories","Nunavut"
}

URL = "https://ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-EE_Admissions-Occ.xlsx"

def make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

def first_sheet_with(xl, token):
    t = token.lower()
    for s in xl.sheet_names:
        if t in s.lower():
            return s
    return xl.sheet_names[0]

def find_row_with_pattern(df, pat, max_rows=80):
    rx = re.compile(pat, re.I)
    n = min(max_rows, len(df))
    for i in range(n):
        row = df.iloc[i].astype(str).tolist()
        if any(rx.search(str(v)) for v in row if v and v != "nan"):
            return i
    return None

def build_columns(raw, year_row, q_row):
    cols = []
    for j in range(raw.shape[1]):
        y = raw.iat[year_row, j]
        q = raw.iat[q_row, j]
        y = "" if pd.isna(y) else str(y).strip()
        q = "" if pd.isna(q) else str(q).strip()
        name = (y + " " + q).strip()
        name = re.sub(r"\s+", " ", name)
        if not name:
            name = f"col{j}"
        cols.append(name)
    return make_unique(cols)

def pick_year_total_col(cols, year):
    y = str(year)
    exact = [c for c in cols if c == f"{y} Total"]
    if exact:
        return exact[0]
    c1 = [c for c in cols if (y in c and "Total" in c and "Q" not in c)]
    if c1:
        return c1[-1]
    c2 = [c for c in cols if (y in c and "Total" in c)]
    return c2[-1] if c2 else None

def series_of(df, c):
    x = df[c]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x

def guess_id_cols(df):
    best = None
    for c in df.columns[:25]:
        s = series_of(df, c).astype(str).str.strip()
        score = s.isin(PROV_ALL).sum()
        if best is None or score > best[0]:
            best = (score, c)
    prov_col = best[1]

    candidates = [c for c in df.columns[:25] if c != prov_col]
    if not candidates:
        print("Cannot detect occupation column", file=sys.stderr)
        sys.exit(1)

    def occ_score(c):
        s = series_of(df, c).astype(str).str.strip()
        nonempty = (s != "") & (s.str.lower() != "nan")
        if nonempty.sum() == 0:
            return -1
        lens = s[nonempty].str.len().mean()
        numeric_ratio = pd.to_numeric(s[nonempty], errors="coerce").notna().mean()
        return lens - 50 * numeric_ratio

    occ_col = max(candidates, key=occ_score)
    return prov_col, occ_col

def main():
    out_dir = Path("outputs_demographics")
    out_dir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(URL)
    sheet = first_sheet_with(xl, "NOC")
    raw = pd.read_excel(xl, sheet_name=sheet, header=None, dtype=str)

    year_row = find_row_with_pattern(raw, r"20\d{2}\s*Total")
    if year_row is None or year_row + 1 >= len(raw):
        print("Cannot locate year/quarter header rows", file=sys.stderr)
        sys.exit(1)
    q_row = year_row + 1

    data_start = None
    for i in range(q_row + 1, min(len(raw), q_row + 800)):
        row = ["" if pd.isna(v) else str(v).strip() for v in raw.iloc[i].tolist()]
        if any(v in PROV_ALL for v in row):
            data_start = i
            break
    if data_start is None:
        data_start = q_row + 2

    cols = build_columns(raw, year_row, q_row)
    df = raw.iloc[data_start:].copy()
    df.columns = cols
    df = df.dropna(how="all")

    prov_col, occ_col = guess_id_cols(df)
    df = df.rename(columns={prov_col: "Province", occ_col: "Occupation"})

    df["Province"] = df["Province"].astype(str).str.strip()
    df["Occupation"] = df["Occupation"].astype(str).str.strip()
    df = df[df["Province"].isin(PROV_ALL)]
    df = df[df["Occupation"].notna() & (df["Occupation"] != "") & (df["Occupation"].str.lower() != "nan")]

    c2021 = pick_year_total_col(df.columns.tolist(), 2021)
    c2025 = pick_year_total_col(df.columns.tolist(), 2025)
    if not c2021 or not c2025:
        print(f"Cannot find year total columns. 2021={c2021}, 2025={c2025}", file=sys.stderr)
        sys.exit(1)

    df["Y2021"] = pd.to_numeric(df[c2021], errors="coerce").fillna(0.0)
    df["Y2025"] = pd.to_numeric(df[c2025], errors="coerce").fillna(0.0)

    tidy = df[["Province", "Occupation", "Y2021", "Y2025"]].copy()

    can = tidy.groupby("Occupation", as_index=False)[["Y2021", "Y2025"]].sum()
    can = can.sort_values("Y2025", ascending=False)
    can.to_csv(out_dir / "ee_noc_occupations_canada_2021_2025.csv", index=False, encoding="utf-8")

    p4 = tidy[tidy["Province"].isin(PROVINCES_4)].copy()
    ab4 = p4.groupby("Occupation", as_index=False)[["Y2021", "Y2025"]].sum()
    ab4 = ab4.sort_values("Y2025", ascending=False)
    ab4.to_csv(out_dir / "ee_noc_occupations_AB_BC_ON_QC_2021_2025.csv", index=False, encoding="utf-8")

    print(out_dir / "ee_noc_occupations_canada_2021_2025.csv")
    print(out_dir / "ee_noc_occupations_AB_BC_ON_QC_2021_2025.csv")

if __name__ == "__main__":
    main()
