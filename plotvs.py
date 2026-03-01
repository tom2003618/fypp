import os
import math
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_CSV = "canada_federal_election_province_summary_2000_2021.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PER_PROVINCE = True
GRID_COLS = 4

def short_province_name(s):
    x = str(s).strip()
    x = x.split("/")[0].strip()
    x = re.sub(r"\s+", " ", x)
    return x

def safe_filename(s):
    x = short_province_name(s)
    x = re.sub(r"[^A-Za-z0-9_-]+", "_", x).strip("_")
    return x

def main():
    df = pd.read_csv(INPUT_CSV)

    need = ["year", "province", "registered_electors", "votes_cast"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

    df = df[need].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["registered_electors"] = pd.to_numeric(df["registered_electors"], errors="coerce")
    df["votes_cast"] = pd.to_numeric(df["votes_cast"], errors="coerce")
    df = df.dropna(subset=["year", "province", "registered_electors", "votes_cast"]).copy()
    df["year"] = df["year"].astype(int)

    df["province_short"] = df["province"].map(short_province_name)
    df = df.sort_values(["province_short", "year"]).copy()

    years = sorted(df["year"].unique().tolist())
    provinces = sorted(df["province_short"].unique().tolist())

    if SAVE_PER_PROVINCE:
        for prov in provinces:
            sub = df[df["province_short"] == prov].copy()
            if sub.empty:
                continue
            plt.figure(figsize=(9, 4))
            x = sub["year"].values
            plt.plot(x, sub["votes_cast"].values / 1e6, marker="o")
            plt.plot(x, sub["registered_electors"].values / 1e6, marker="o")
            plt.xticks(years, rotation=45, ha="right")
            plt.xlabel("Election year")
            plt.ylabel("Millions")
            plt.title(f"{prov}: votes cast vs registered electors")
            plt.grid(True)
            plt.legend(["Votes cast", "Registered electors"], loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"fig4_province_votes_vs_electors_{safe_filename(prov)}.png"), dpi=160)
            plt.close()

    n = len(provinces)
    ncols = GRID_COLS
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 2.8 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    handles = None
    labels = None

    for i, prov in enumerate(provinces):
        ax = axes[i]
        sub = df[df["province_short"] == prov].copy()
        x = sub["year"].values
        l1 = ax.plot(x, sub["votes_cast"].values / 1e6, marker="o")[0]
        l2 = ax.plot(x, sub["registered_electors"].values / 1e6, marker="o")[0]
        ax.set_title(prov, fontsize=10)
        ax.grid(True)
        ax.set_xticks(years)
        if handles is None:
            handles = [l1, l2]
            labels = ["Votes cast", "Registered electors"]

    for j in range(n, len(axes)):
        axes[j].axis("off")

    for ax in axes[:n]:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
            tick.set_fontsize(8)

    fig.supxlabel("Election year")
    fig.supylabel("Millions")
    fig.suptitle("Votes cast vs registered electors by province/territory", y=0.995)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(os.path.join(OUT_DIR, "fig4_province_votes_vs_electors_grid.png"), dpi=160)
    plt.close(fig)

    print(os.path.join(OUT_DIR, "fig4_province_votes_vs_electors_grid.png"))
    if SAVE_PER_PROVINCE:
        print(os.path.join(OUT_DIR, "fig4_province_votes_vs_electors_<province>.png"))

if __name__ == "__main__":
    main()

