import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_URL_2000_2021 = "https://www.sfu.ca/~aheard/elections/1867-present.html"
DATASET_URL_2025 = "https://researchbriefings.files.parliament.uk/documents/CBP-10244/CBP-10244.pdf"

INPUT_CSV = "canada_federal_vote_share_2000_2025.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

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
        cl = c.lower()
        for k in candidates:
            if k in cl:
                return c
    return None

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
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
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

    hm = wide.copy()
    plt.figure(figsize=(12, 6))
    im = plt.imshow(hm.values, aspect="auto")
    plt.yticks(range(len(hm.index)), hm.index)
    plt.xticks(range(len(hm.columns)), hm.columns, rotation=45, ha="right")
    plt.xlabel("Party")
    plt.ylabel("Year")
    plt.title("Year × Party vote share (heatmap)")
    plt.colorbar(im, label="Vote share (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_heatmap.png"), dpi=160)
    plt.close()

    print("Saved:")
    print(os.path.join(OUT_DIR, "fig1_trend.png"))
    print(os.path.join(OUT_DIR, "fig2_bar_2025.png"))
    print(os.path.join(OUT_DIR, "fig3_heatmap.png"))
    print(os.path.join(OUT_DIR, "national_vote_share_clean.csv"))
    print(os.path.join(OUT_DIR, "national_vote_share_sumcheck.csv"))
    print("Dataset URLs:")
    print(DATASET_URL_2000_2021)
    print(DATASET_URL_2025)

if __name__ == "__main__":
    main()

