import importlib
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import Image, display
except Exception:
    Image = None
    display = None


def _ensure_module(module_name, pip_name=None):
    try:
        importlib.import_module(module_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or module_name])


_ensure_module("requests", "requests")
_ensure_module("lxml", "lxml")
_ensure_module("html5lib", "html5lib")

import requests


def _find_root(marker="canada_federal_vote_share_2000_2025.csv"):
    here = Path.cwd().resolve()
    for candidate in [here] + list(here.parents):
        if (candidate / marker).exists():
            return candidate
    return here


def _detect_col(df, candidates):
    cols = {str(c).lower(): c for c in df.columns}
    for k in candidates:
        if k in cols:
            return cols[k]
    for c in df.columns:
        cl = str(c).lower()
        for k in candidates:
            if k in cl:
                return c
    return None


def _canon_party(name):
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)

    if "liberal" in s:
        return "Liberal"

    conservative_keys = ["conservative", "canadian alliance", "reform", "national government", "unionist"]
    if any(k in s for k in conservative_keys):
        return "Conservative"

    ndp_keys = ["new democratic", " ndp", "(ndp)", "co-operative commonwealth", "cooperative commonwealth", "ccf"]
    if any(k in s for k in ndp_keys) or s == "ndp":
        return "NDP"

    if "bloc" in s:
        return "Bloc Québécois"

    if "green" in s:
        return "Green"

    if "people" in s and "party" in s:
        return "People's"

    if "other" in s or "independent" in s:
        return "Others"

    return "Others"


def _parse_pct(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(" ", " ").strip()
    if not s or s.lower() == "nan":
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if not m:
        return np.nan
    return float(m.group(0))


def _download_sfu_history(url):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    text = resp.content.decode(resp.encoding or "latin-1", errors="ignore")

    tables = pd.read_html(StringIO(text))
    table = None
    for t in tables:
        if any("Popular Vote" in str(c) for c in t.columns):
            table = t.copy()
            break
    if table is None:
        raise RuntimeError("Could not find the SFU popular-vote table on the source page.")

    col_party = next(c for c in table.columns if "Election" in str(c))
    col_vote = next(c for c in table.columns if "Popular Vote" in str(c))

    rows = []
    current_year = None
    for _, r in table[[col_party, col_vote]].iterrows():
        label = str(r[col_party]).replace(" ", " ").strip()
        vote = _parse_pct(r[col_vote])

        if not label or label.lower() == "nan":
            continue

        ymatch = re.match(r"^(18|19|20)\d{2}", label)
        if ymatch:
            current_year = int(ymatch.group(0))
            continue

        low = label.lower()
        # Exclude non-party summary rows such as government/opposition summaries.
        if (
            low.startswith("election")
            or low.startswith("government")
            or low.startswith("opposition")
            or low.startswith("total")
            or "government--" in low
            or "opposition--" in low
            or "minority--" in low
            or "majority--" in low
        ):
            continue

        if current_year is None or pd.isna(vote):
            continue

        if vote < 0 or vote > 100:
            continue

        rows.append({"Year": current_year, "Party": label, "VoteShare": float(vote), "Source": url})

    out = pd.DataFrame(rows)
    out = out[(out["Year"] >= 1867) & (out["Year"] <= 2021)].copy()
    return out


def _fit_hierarchical_rw_gibbs(y_scaled, x, n_iter=12000, burn=4000, thin=4, seed=42):
    rng = np.random.default_rng(seed)

    p_count, t_count = y_scaled.shape

    # Priors for hierarchical drift + party-level innovation variance.
    s_mu2 = 4.0
    a_tau, b_tau = 2.0, 0.5
    a_sig, b_sig = 2.5, 1.0

    drift = np.array([np.sum(x * y_scaled[p]) / np.sum(x * x) for p in range(p_count)], dtype=float)
    sigma2 = np.array([np.var(y_scaled[p] - x * drift[p]) + 1e-3 for p in range(p_count)], dtype=float)

    mu0 = float(np.mean(drift))
    tau2 = float(np.var(drift) + 0.1)

    drift_draws = []
    sigma2_draws = []

    for it in range(n_iter):
        for p in range(p_count):
            y = y_scaled[p]

            prec = np.sum((x * x) / sigma2[p]) + (1.0 / tau2)
            var = 1.0 / prec
            mean = var * (np.sum((x * y) / sigma2[p]) + (mu0 / tau2))
            drift[p] = rng.normal(mean, np.sqrt(var))

            resid = y - (x * drift[p])
            shape = a_sig + (t_count / 2.0)
            scale = b_sig + 0.5 * np.sum(resid * resid)
            sigma2[p] = 1.0 / rng.gamma(shape, 1.0 / scale)

        prec_mu = (p_count / tau2) + (1.0 / s_mu2)
        var_mu = 1.0 / prec_mu
        mean_mu = var_mu * (np.sum(drift) / tau2)
        mu0 = rng.normal(mean_mu, np.sqrt(var_mu))

        shape_tau = a_tau + (p_count / 2.0)
        scale_tau = b_tau + 0.5 * np.sum((drift - mu0) ** 2)
        tau2 = 1.0 / rng.gamma(shape_tau, 1.0 / scale_tau)

        if it >= burn and ((it - burn) % thin == 0):
            drift_draws.append(drift.copy())
            sigma2_draws.append(sigma2.copy())

    return np.asarray(drift_draws), np.asarray(sigma2_draws)


project_root = _find_root()
out_dir = project_root / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

HIST_URL = "https://www.sfu.ca/~aheard/elections/1867-present.html"
hist_cache_csv = out_dir / "sfu_historical_vote_share_1867_2021.csv"

if hist_cache_csv.exists():
    hist_raw = pd.read_csv(hist_cache_csv)
else:
    hist_raw = _download_sfu_history(HIST_URL)
    hist_raw.to_csv(hist_cache_csv, index=False)

# Refresh cache if legacy parse issues are present (e.g., negative or >100 shares).
if ((hist_raw["VoteShare"] < 0) | (hist_raw["VoteShare"] > 100)).any():
    hist_raw = _download_sfu_history(HIST_URL)
    hist_raw.to_csv(hist_cache_csv, index=False)

hist = hist_raw.copy()
hist["Party"] = hist["Party"].map(_canon_party)
hist = hist.groupby(["Year", "Party"], as_index=False)["VoteShare"].sum()

local_csv = project_root / "canada_federal_vote_share_2000_2025.csv"
raw_2025 = pd.read_csv(local_csv)

y_col = _detect_col(raw_2025, ["year"])
p_col = _detect_col(raw_2025, ["party"])
v_col = _detect_col(raw_2025, ["voteshare", "vote_share", "share", "pct", "percent", "percentage"])
if y_col is None or p_col is None or v_col is None:
    raise RuntimeError(f"Missing required columns in {local_csv}. Found: {raw_2025.columns.tolist()}")

holdout = raw_2025[[y_col, p_col, v_col]].copy()
holdout.columns = ["Year", "Party", "VoteShare"]
holdout["Year"] = pd.to_numeric(holdout["Year"], errors="coerce").astype("Int64")
holdout["VoteShare"] = pd.to_numeric(holdout["VoteShare"], errors="coerce")
holdout = holdout.dropna(subset=["Year", "Party", "VoteShare"]).copy()
holdout["Year"] = holdout["Year"].astype(int)
holdout["Party"] = holdout["Party"].map(_canon_party)
holdout_2025 = holdout[holdout["Year"] == 2025].groupby(["Year", "Party"], as_index=False)["VoteShare"].sum()

if holdout_2025.empty:
    raise RuntimeError("No 2025 rows found in local holdout file.")

combined = pd.concat([hist, holdout_2025], ignore_index=True)
combined = combined.groupby(["Year", "Party"], as_index=False)["VoteShare"].sum()

preferred_order = ["Liberal", "Conservative", "NDP", "Bloc Québécois", "Green", "People's", "Others"]
wide = combined.pivot(index="Year", columns="Party", values="VoteShare").fillna(0.0)
parties = [p for p in preferred_order if p in wide.columns] + [p for p in wide.columns if p not in preferred_order]
wide = wide.reindex(columns=parties, fill_value=0.0)

target_year = 2025
if target_year not in wide.index:
    raise RuntimeError("Target year 2025 missing from combined dataset.")

train_years = sorted([int(y) for y in wide.index.tolist() if int(y) < target_year])
if len(train_years) < 12:
    raise RuntimeError("Insufficient pre-2025 elections for Bayesian random-walk fitting.")

train_years_arr = np.asarray(train_years, dtype=float)
dt = np.diff(train_years_arr)
if np.any(dt <= 0):
    raise RuntimeError("Election years are not strictly increasing.")

x = np.sqrt(dt)
y_scaled = []
for party in parties:
    y = wide.loc[train_years, party].astype(float).values
    dy = np.diff(y)
    y_scaled.append(dy / np.sqrt(dt))
y_scaled = np.asarray(y_scaled, dtype=float)

drift_draws, sigma2_draws = _fit_hierarchical_rw_gibbs(
    y_scaled,
    x,
    n_iter=12000,
    burn=4000,
    thin=4,
    seed=42,
)

if len(drift_draws) == 0:
    raise RuntimeError("No posterior draws retained. Increase iterations or reduce burn-in.")

last_train_year = train_years[-1]
dt_pred = float(target_year - last_train_year)
last_share = wide.loc[last_train_year, parties].astype(float).values

rng_pred = np.random.default_rng(2025)
pred_samples = np.zeros((drift_draws.shape[0], len(parties)), dtype=float)
for i in range(drift_draws.shape[0]):
    delta = rng_pred.normal(
        loc=drift_draws[i] * dt_pred,
        scale=np.sqrt(sigma2_draws[i] * dt_pred),
    )
    pred = np.maximum(last_share + delta, 0.0)
    s = pred.sum()
    if s <= 0:
        pred = np.repeat(100.0 / len(parties), len(parties))
    else:
        pred = (pred / s) * 100.0
    pred_samples[i] = pred

pred_mean = pred_samples.mean(axis=0)
pred_lo = np.percentile(pred_samples, 5, axis=0)
pred_hi = np.percentile(pred_samples, 95, axis=0)

actual_2025 = wide.loc[target_year, parties].astype(float).values

compare = pd.DataFrame(
    {
        "Party": parties,
        "PredictedVoteShare": pred_mean,
        "PredictedP05": pred_lo,
        "PredictedP95": pred_hi,
        "ActualVoteShare": actual_2025,
    }
)
compare["Error"] = compare["PredictedVoteShare"] - compare["ActualVoteShare"]
compare["AbsoluteError"] = np.abs(compare["Error"])
compare = compare.sort_values("ActualVoteShare", ascending=False).reset_index(drop=True)

out_csv = out_dir / "prediction_2025_vs_actual.csv"
compare.to_csv(out_csv, index=False)

years_all = sorted([int(y) for y in wide.index.tolist()])
plot_years = [y for y in years_all if 2000 <= y <= target_year]
if len(plot_years) == 0:
    raise RuntimeError("No years in [2000, 2025] found for plotting.")

pred_map = dict(zip(compare["Party"], compare["PredictedVoteShare"]))
lo_map = dict(zip(compare["Party"], compare["PredictedP05"]))
hi_map = dict(zip(compare["Party"], compare["PredictedP95"]))

fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.get_cmap("tab10")

for i, party in enumerate(parties):
    color = cmap(i % 10)
    y_actual_series = wide.loc[plot_years, party].astype(float).values

    ax.plot(
        plot_years,
        y_actual_series,
        color=color,
        linewidth=1.7,
        marker="o",
        markersize=3.5,
        label=party,
    )

    # Label every observed dot with the exact percentage value.
    for yr, val in zip(plot_years, y_actual_series):
        yoff = 5 if (i % 2 == 0) else -9
        ax.annotate(
            f"{val:.2f}%",
            (yr, val),
            textcoords="offset points",
            xytext=(0, yoff),
            ha="center",
            fontsize=6,
            color=color,
            alpha=0.9,
        )

    y_prev = float(wide.loc[last_train_year, party])
    y_pred = float(pred_map[party])
    y_lo = float(lo_map[party])
    y_hi = float(hi_map[party])

    ax.plot([last_train_year, target_year], [y_prev, y_pred], color=color, linestyle=":", linewidth=2.4)
    ax.scatter([target_year], [y_pred], color=color, marker="x", s=56, linewidths=2)
    ax.vlines(target_year, y_lo, y_hi, color=color, alpha=0.55, linewidth=1.3)

    ax.annotate(
        f"pred {y_pred:.2f}%",
        (target_year, y_pred),
        textcoords="offset points",
        xytext=(8, 10 if (i % 2 == 0) else -12),
        ha="left",
        fontsize=6,
        color=color,
        fontweight="bold",
    )

ax.set_xlabel("Year")
ax.set_ylabel("Vote share (%)")
ax.set_title(
    "Canada federal vote share by party: Bayesian hierarchical random walk prediction for 2025\n"
)
ax.set_xticks(plot_years)
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(title="Party", ncol=2, fontsize=8, title_fontsize=9)
plt.tight_layout()
fig.text(
    0.005,
    0.005,
    f"History source URL: {HIST_URL}",
    fontsize=8,
    ha="left",
    va="bottom",
)

out_png = out_dir / "fig5_predicted_vs_actual_2025.png"
plt.savefig(out_png, dpi=170)
plt.close()

mae_2025 = float(np.mean(np.abs(compare["Error"])))
rmse_2025 = float(np.sqrt(np.mean(compare["Error"] ** 2)))

print("Saved:")
print(hist_cache_csv)
print(out_csv)
print(out_png)
print("Model: Bayesian hierarchical random walk (Gibbs sampler)")
print(f"Posterior draws: {len(drift_draws)}")
print(f"2025 MAE: {mae_2025:.2f} percentage points")
print(f"2025 RMSE: {rmse_2025:.2f} percentage points")
print("Source URL used:")
print(HIST_URL)

if Image is not None and display is not None and out_png.exists():
    display(Image(filename=str(out_png)))
