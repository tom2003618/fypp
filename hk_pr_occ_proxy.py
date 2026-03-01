from pathlib import Path
import sys
import re
import pandas as pd

PROVINCES = ["Alberta", "British Columbia", "Ontario", "Quebec"]

def find_col(cols, *needles):
    ns = [n.lower() for n in needles]
    for c in cols:
        lc = str(c).lower()
        if all(n in lc for n in ns):
            return c
    return None

def pick_year_col(cols, year):
    y = str(year)
    if y in cols:
        return y
    for c in cols:
        if y in str(c):
            return c
    return None

def main():
    out_dir = Path("outputs_demographics")
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_citz_url = "https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-PR-Citz.xlsx"
    ee_occ_url = "https://ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-EE_Admissions-Occ.xlsx"

    pr = pd.read_excel(pr_citz_url)
    pr.columns = [str(c).strip() for c in pr.columns]
    country_col = pr.columns[0]
    pr = pr.rename(columns={country_col: "Country"})
    hk = pr[pr["Country"].astype(str).str.contains("Hong Kong", case=False, na=False)].copy()
    if hk.empty:
        print("Hong Kong row not found in PR-Citz file", file=sys.stderr)
        sys.exit(1)

    years_req = [2021, 2025]
    year_cols = [c for c in pr.columns if re.fullmatch(r"\d{4}", str(c))]

    picked = {}
    for y in years_req:
        c = pick_year_col(pr.columns, y)
        if c is None:
            if year_cols:
                c = max(year_cols, key=lambda z: int(z))
        picked[y] = c

    hk_out = {"Country": hk["Country"].iloc[0]}
    for y in years_req:
        c = picked[y]
        hk_out[f"Year_{y}_col_used"] = str(c) if c is not None else ""
        hk_out[str(y)] = float(hk[c].iloc[0]) if c is not None else float("nan")

    pd.DataFrame([hk_out]).to_csv(out_dir / "hk_pr_counts_2021_2025.csv", index=False, encoding="utf-8")

    ee = pd.read_excel(ee_occ_url)
    ee.columns = [str(c).strip() for c in ee.columns]

    year_col = find_col(ee.columns, "year") or find_col(ee.columns, "ref", "date")
    prov_col = find_col(ee.columns, "province") or find_col(ee.columns, "territory")
    occ_col = find_col(ee.columns, "occupation") or find_col(ee.columns, "noc")
    val_col = find_col(ee.columns, "value") or find_col(ee.columns, "admissions") or find_col(ee.columns, "count")

    if year_col and prov_col and occ_col:
        if val_col is None:
            num_candidates = []
            for c in ee.columns:
                if c in [year_col, prov_col, occ_col]:
                    continue
                s = pd.to_numeric(ee[c], errors="coerce")
                if s.notna().sum() > 0:
                    num_candidates.append((s.notna().sum(), c))
            if not num_candidates:
                print("Cannot find numeric admissions column in EE file", file=sys.stderr)
                sys.exit(1)
            val_col = sorted(num_candidates, reverse=True)[0][1]

        d = ee[[year_col, prov_col, occ_col, val_col]].copy()
        d["Year"] = pd.to_numeric(d[year_col], errors="coerce")
        d["Admissions"] = pd.to_numeric(d[val_col], errors="coerce").fillna(0)
        d = d.dropna(subset=["Year"])
        d["Year"] = d["Year"].astype(int)

        for y in years_req:
            if (d["Year"] == y).sum() == 0:
                print(f"EE file has no rows for year {y}", file=sys.stderr)

        def top_occ(df, label, path):
            out_rows = []
            for y in years_req:
                s = df[df["Year"] == y].groupby(occ_col, as_index=False)["Admissions"].sum()
                s = s.sort_values("Admissions", ascending=False).head(20)
                s = s.rename(columns={occ_col: "Occupation"})
                s.insert(0, "Year", y)
                s.insert(0, "Scope", label)
                out_rows.append(s)
            out = pd.concat(out_rows, ignore_index=True)
            out.to_csv(path, index=False, encoding="utf-8")

        top_occ(d, "Canada_total_from_EE", out_dir / "ee_top_occupations_canada_2021_2025.csv")

        d4 = d[d[prov_col].isin(PROVINCES)].copy()
        top_occ(d4, "AB_BC_ON_QC_total_from_EE", out_dir / "ee_top_occupations_AB_BC_ON_QC_2021_2025.csv")

        print("Wrote:")
        print(out_dir / "hk_pr_counts_2021_2025.csv")
        print(out_dir / "ee_top_occupations_canada_2021_2025.csv")
        print(out_dir / "ee_top_occupations_AB_BC_ON_QC_2021_2025.csv")
        return

    year_cols_wide = [c for c in ee.columns if re.fullmatch(r"\d{4}", str(c))]
    if year_cols_wide and occ_col:
        base_cols = [c for c in ee.columns if c not in year_cols_wide]
        if prov_col is None:
            for c in base_cols:
                if "province" in str(c).lower() or "territory" in str(c).lower():
                    prov_col = c
                    break

        def melt_and_top(scope_label, filt_prov, out_path):
            m = ee.copy()
            if filt_prov is not None and prov_col is not None:
                m = m[m[prov_col].isin(filt_prov)].copy()
            mm = m.melt(id_vars=[c for c in [prov_col, occ_col] if c is not None], value_vars=year_cols_wide, var_name="Year", value_name="Admissions")
            mm["Year"] = pd.to_numeric(mm["Year"], errors="coerce").astype("Int64")
            mm["Admissions"] = pd.to_numeric(mm["Admissions"], errors="coerce").fillna(0)
            out_rows = []
            for y in years_req:
                s = mm[mm["Year"] == y].groupby(occ_col, as_index=False)["Admissions"].sum()
                s = s.sort_values("Admissions", ascending=False).head(20)
                s = s.rename(columns={occ_col: "Occupation"})
                s.insert(0, "Year", y)
                s.insert(0, "Scope", scope_label)
                out_rows.append(s)
            out = pd.concat(out_rows, ignore_index=True)
            out.to_csv(out_path, index=False, encoding="utf-8")

        melt_and_top("Canada_total_from_EE", None, out_dir / "ee_top_occupations_canada_2021_2025.csv")
        melt_and_top("AB_BC_ON_QC_total_from_EE", PROVINCES, out_dir / "ee_top_occupations_AB_BC_ON_QC_2021_2025.csv")

        print("Wrote:")
        print(out_dir / "hk_pr_counts_2021_2025.csv")
        print(out_dir / "ee_top_occupations_canada_2021_2025.csv")
        print(out_dir / "ee_top_occupations_AB_BC_ON_QC_2021_2025.csv")
        return

    print("EE file format not recognized (need year+province+occupation+value or wide year columns)", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
