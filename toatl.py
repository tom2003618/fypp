import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_URL_2000_2021 = "https://www.sfu.ca/~aheard/elections/1867-present.html"
DATASET_URL_2025 = "https://researchbriefings.files.parliament.uk/documents/CBP-10244/CBP-10244.pdf"

def find_project_root(marker):
    here = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else here
    for candidate in [here, *here.parents, script_dir, *script_dir.parents]:
        if (candidate / marker).exists():
            return candidate
    return here

PROJECT_ROOT = find_project_root("canada_federal_vote_share_2000_2025.csv")

INPUT_CSV = str(PROJECT_ROOT / "canada_federal_vote_share_2000_2025.csv")
OUT_DIR = str(PROJECT_ROOT / "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

PROV_INPUT_CSV = str(PROJECT_ROOT / "canada_province_votes_electors_2000_2025.csv")
KAGGLE_DIR = str(PROJECT_ROOT / "kaggle_canada_election")

def canon_party(p):
    s = str(p).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if "liberal" in s:
        return "Liberal"
    if "conservative" in s or "canadian alliance" in s or "progressive conservative" in s:
        return "Conservative"
    if "new democratic" in s or "(ndp)" in s or s == "ndp":
        return "NDP"
    if "bloc" in s:
        return "Bloc Québécois"
    if "green" in s:
        return "Green"
    if "people" in s:
        return "People's"
    if "other" in s:
        return "Others"
    return "Others"

def detect_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols:
            return cols[k]
    for c in df.columns:
        cl = str(c).lower()
        for k in candidates:
            if k in cl:
                return c
    return None

def build_year_map_from_election_info(path):
    info = pd.read_csv(path, low_memory=False)
    c_eid = detect_col(info, ["election_id"])
    c_evt = detect_col(info, ["event_number"])
    c_date = detect_col(info, ["election_date", "date", "event_date"])
    if c_eid is None:
        raise RuntimeError(f"Election info missing election_id. Columns: {info.columns.tolist()}")
    if c_date is not None:
        dt = pd.to_datetime(info[c_date], errors="coerce")
        info = info.assign(_year=dt.dt.year)
        m = info[[c_eid, "_year"]].dropna().drop_duplicates()
        year_map = dict(zip(m[c_eid].astype(str), m["_year"].astype(int)))
        if year_map:
            return year_map
    if c_evt is not None:
        ev = pd.to_numeric(info[c_evt], errors="coerce")
        info = info.assign(_ev=ev)
        evs = sorted([int(x) for x in info["_ev"].dropna().unique().tolist()])
        known = [2000, 2004, 2006, 2008, 2011, 2015, 2019, 2021]
        year_map = {}
        if len(evs) == len(known):
            emap = dict(zip(evs, known))
            for _, r in info[[c_eid, "_ev"]].dropna().drop_duplicates().iterrows():
                year_map[str(r[c_eid])] = emap.get(int(r["_ev"]), np.nan)
            year_map = {k: int(v) for k, v in year_map.items() if pd.notna(v)}
            if year_map:
                return year_map
    raise RuntimeError(f"Cannot build year map from election info. Columns: {info.columns.tolist()}")

def build_prov_votes_electors_from_kaggle(kdir):
    ei = os.path.join(kdir, "Election Information.CSV")
    results = os.path.join(kdir, "Results.csv")
    poll = os.path.join(kdir, "Poll Details.csv")
    rid = os.path.join(kdir, "Ridings.csv")
    for fp in [ei, results, poll, rid]:
        if not os.path.exists(fp):
            raise RuntimeError(f"Missing required Kaggle file: {fp}")

    year_map = build_year_map_from_election_info(ei)

    ridings = pd.read_csv(rid, low_memory=False)
    rd_ed = detect_col(ridings, ["electoral_district_number", "circonscription"])
    rd_prov = detect_col(ridings, ["canadian_province", "province"])
    if rd_ed is None or rd_prov is None:
        raise RuntimeError(f"Ridings missing ed/province columns. Columns: {ridings.columns.tolist()}")
    rid_map = ridings[[rd_ed, rd_prov]].dropna().drop_duplicates().copy()
    rid_map.columns = ["ed_num", "province"]
    rid_map["ed_num"] = rid_map["ed_num"].astype(str).str.strip()

    poll_df = pd.read_csv(poll, low_memory=False, usecols=lambda c: True)
    p_eid = detect_col(poll_df, ["election_id"])
    p_ed = detect_col(poll_df, ["electoral_district_number", "circonscription"])
    p_elec = detect_col(poll_df, ["electors", "électeurs"])
    if p_eid is None or p_ed is None or p_elec is None:
        raise RuntimeError(f"Poll Details missing required columns. Columns: {poll_df.columns.tolist()}")
    poll_df = poll_df[[p_eid, p_ed, p_elec]].copy()
    poll_df.columns = ["election_id", "ed_num", "electors"]
    poll_df["election_id"] = poll_df["election_id"].astype(str).str.strip()
    poll_df["ed_num"] = poll_df["ed_num"].astype(str).str.strip()
    poll_df["electors"] = pd.to_numeric(poll_df["electors"], errors="coerce")
    poll_df = poll_df.dropna(subset=["electors"])
    electors_by_ed = poll_df.groupby(["election_id", "ed_num"], as_index=False)["electors"].sum()

    res_df = pd.read_csv(results, low_memory=False, usecols=lambda c: True)
    r_eid = detect_col(res_df, ["election_id"])
    r_ed = detect_col(res_df, ["electoral_district_number", "circonscription"])
    r_votes = detect_col(res_df, ["candidate_poll_votes_count", "votes"])
    if r_eid is None or r_ed is None or r_votes is None:
        raise RuntimeError(f"Results missing required columns. Columns: {res_df.columns.tolist()}")
    res_df = res_df[[r_eid, r_ed, r_votes]].copy()
    res_df.columns = ["election_id", "ed_num", "votes"]
    res_df["election_id"] = res_df["election_id"].astype(str).str.strip()
    res_df["ed_num"] = res_df["ed_num"].astype(str).str.strip()
    res_df["votes"] = pd.to_numeric(res_df["votes"], errors="coerce")
    res_df = res_df.dropna(subset=["votes"])
    votes_by_ed = res_df.groupby(["election_id", "ed_num"], as_index=False)["votes"].sum()

    m = votes_by_ed.merge(electors_by_ed, on=["election_id", "ed_num"], how="inner")
    m = m.merge(rid_map, on="ed_num", how="left")
    m["year"] = m["election_id"].map(year_map)
    m = m.dropna(subset=["province", "year"]).copy()
    m["year"] = m["year"].astype(int)

    out = m.groupby(["year", "province"], as_index=False)[["votes", "electors"]].sum()
    out = out.sort_values(["province", "year"])
    return out

def heatmap_single_year(df_clean, year, order):
    wide = df_clean.pivot(index="Year", columns="Party", values="VoteShare").fillna(0)
    wide = wide.reindex(columns=[c for c in order if c in wide.columns])
    if year not in wide.index:
        raise RuntimeError(f"Year {year} not found for heatmap.")
    row = wide.loc[[year]].copy()
    plt.figure(figsize=(12, 3))
    im = plt.imshow(row.values, aspect="auto")
    plt.yticks([0], [str(year)])
    plt.xticks(range(len(row.columns)), row.columns, rotation=45, ha="right")
    plt.xlabel("Party")
    plt.ylabel("Year")
    plt.title(f"{year} vote share heatmap (single-year)")
    plt.colorbar(im, label="Vote share (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"fig3_heatmap_{year}.png"), dpi=160)
    plt.close()

def main():
    df = pd.read_csv(INPUT_CSV)

    c_year = detect_col(df, ["year"])
    c_party = detect_col(df, ["party"])
    c_share = detect_col(df, ["voteshare", "vote_share", "share", "pct", "percent", "percentage"])
    c_source = detect_col(df, ["source", "url"])

    if c_year is None or c_party is None or c_share is None:
        raise RuntimeError(f"Missing required columns. Found: {df.columns.tolist()}")

    df = df[[c_year, c_party, c_share] + ([c_source] if c_source else [])].copy()
    df.columns = ["Year", "Party", "VoteShare"] + (["Source"] if c_source else [])

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["VoteShare"] = pd.to_numeric(df["VoteShare"], errors="coerce")
    df = df.dropna(subset=["Year", "Party", "VoteShare"]).copy()
    df["Year"] = df["Year"].astype(int)

    df["Party_Canon"] = df["Party"].map(canon_party)
    df_clean = df.groupby(["Year", "Party_Canon"], as_index=False)["VoteShare"].sum()
    df_clean = df_clean.rename(columns={"Party_Canon": "Party"})
    df_clean.to_csv(os.path.join(OUT_DIR, "national_vote_share_clean.csv"), index=False)

    check = df_clean.groupby("Year")["VoteShare"].sum().reset_index()
    check["diff_from_100"] = check["VoteShare"] - 100.0
    check.to_csv(os.path.join(OUT_DIR, "national_vote_share_sumcheck.csv"), index=False)

    order = ["Liberal", "Conservative", "NDP", "Bloc Québécois", "Green", "People's", "Others"]
    wide = df_clean.pivot(index="Year", columns="Party", values="VoteShare").fillna(0)
    wide = wide.reindex(columns=[c for c in order if c in wide.columns])

    ax = wide.plot(figsize=(11, 5))
    ax.set_xlabel("Year")
    ax.set_ylabel("Vote share (%)")
    ax.set_title("Canada federal election: national vote share (2000–2025)")
    ax.grid(True)

    election_years = sorted(wide.index.unique().tolist())
    ymin, ymax = ax.get_ylim()
    for y in election_years:
        ax.vlines(x=y, ymin=ymin, ymax=ymax, linestyles="--", linewidth=1.0)
        ax.text(
            y, -0.10, f"{y} election",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            rotation=90,
            clip_on=False
        )
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)
    plt.savefig(os.path.join(OUT_DIR, "fig1_trend.png"), dpi=160)
    plt.close()

    snap = df_clean[df_clean["Year"] == 2025].sort_values("VoteShare", ascending=False).copy()
    plt.figure(figsize=(10, 5))
    plt.bar(snap["Party"], snap["VoteShare"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Party")
    plt.ylabel("Vote share (%)")
    plt.title("2025 national vote share by party")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_bar_2025.png"), dpi=160)
    plt.close()

    heatmap_single_year(df_clean, 2021, order)
    heatmap_single_year(df_clean, 2025, order)

    prov_df = None
    if os.path.exists(PROV_INPUT_CSV):
        prov_df = pd.read_csv(PROV_INPUT_CSV, low_memory=False)
        py = detect_col(prov_df, ["year"])
        pp = detect_col(prov_df, ["province", "prov"])
        pv = detect_col(prov_df, ["votes", "voters", "ballots", "votes_cast"])
        pe = detect_col(prov_df, ["electors", "eligible", "registered", "population"])
        if py is None or pp is None or pv is None or pe is None:
            raise RuntimeError(f"Province CSV missing columns. Found: {prov_df.columns.tolist()}")
        prov_df = prov_df[[py, pp, pv, pe]].copy()
        prov_df.columns = ["year", "province", "votes", "electors"]
        prov_df["year"] = pd.to_numeric(prov_df["year"], errors="coerce").astype("Int64")
        prov_df["votes"] = pd.to_numeric(prov_df["votes"], errors="coerce")
        prov_df["electors"] = pd.to_numeric(prov_df["electors"], errors="coerce")
        prov_df = prov_df.dropna(subset=["year", "province", "votes", "electors"]).copy()
        prov_df["year"] = prov_df["year"].astype(int)
    else:
        if os.path.isdir(KAGGLE_DIR):
            prov_df = build_prov_votes_electors_from_kaggle(KAGGLE_DIR)
            prov_df.to_csv(os.path.join(OUT_DIR, "province_votes_electors_from_kaggle.csv"), index=False)
        else:
            raise RuntimeError(
                f"Need {PROV_INPUT_CSV} or Kaggle folder {KAGGLE_DIR}/ with Results.csv, Poll Details.csv, Ridings.csv, Election Information.CSV"
            )

    prov_df = prov_df.sort_values(["province", "year"]).copy()
    provinces = sorted(prov_df["province"].dropna().unique().tolist())
    n = len(provinces)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(14, 3.2 * rows))
    for i, prov in enumerate(provinces):
        ax = fig.add_subplot(rows, cols, i + 1)
        d = prov_df[prov_df["province"] == prov].sort_values("year").copy()
        x = d["year"].values
        y_votes = (d["votes"].values / 1_000_000.0)
        y_elec = (d["electors"].values / 1_000_000.0)
        ax.plot(x, y_votes, label="Votes (M)")
        ax.plot(x, y_elec, label="Electors (M)")
        ax.set_title(str(prov))
        ax.set_xlabel("Year")
        ax.set_ylabel("Millions")
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig4_votes_vs_electors_by_province.png"), dpi=160)
    plt.close()

    print("Saved:")
    print(os.path.join(OUT_DIR, "fig1_trend.png"))
    print(os.path.join(OUT_DIR, "fig2_bar_2025.png"))
    print(os.path.join(OUT_DIR, "fig3_heatmap_2021.png"))
    print(os.path.join(OUT_DIR, "fig3_heatmap_2025.png"))
    print(os.path.join(OUT_DIR, "fig4_votes_vs_electors_by_province.png"))
    print(os.path.join(OUT_DIR, "national_vote_share_clean.csv"))
    print(os.path.join(OUT_DIR, "national_vote_share_sumcheck.csv"))
    if os.path.exists(os.path.join(OUT_DIR, "province_votes_electors_from_kaggle.csv")):
        print(os.path.join(OUT_DIR, "province_votes_electors_from_kaggle.csv"))
    print("Dataset URLs:")
    print(DATASET_URL_2000_2021)
    print(DATASET_URL_2025)

if __name__ == "__main__":
    main()
