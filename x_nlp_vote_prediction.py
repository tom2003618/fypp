# X/NLP-only vote share prediction: 2021 signal → 2025 prediction → 2029 projection
# Uses the same ridge regression as factor_based_2025_prediction_and_projection.py
# but driven solely by X/social data from x_nlp_party_summary_2021_2025.csv.
#
# Outputs (outputs_demographics/):
#   x_nlp_trend_2021_2025.png          — SupportIndex and XShare: 2021 vs 2025
#   x_nlp_predicted_2021_2025_2029.png — predicted vote shares for all three years
#   x_nlp_predicted_vs_actual_2025.png — predicted vs actual 2025 result

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from IPython.display import Image as IPyImage, display
except Exception:
    IPyImage = None
    display = None


def find_project_root():
    markers = ['.git', 'canada_federal_vote_share_2000_2025.csv', 'econ_2021_2025.py']
    for c in [Path.cwd(), *Path.cwd().parents]:
        if any((c / m).exists() for m in markers):
            return c
    return Path.cwd()


ROOT = find_project_root()
OUT_DIR = ROOT / 'outputs_demographics'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_PARTIES = ['Liberal', 'Conservative', 'NDP', 'Bloc Québécois', 'Green']
PARTY_COLORS = {
    'Liberal':        '#D71920',
    'Conservative':   '#1A4782',
    'NDP':            '#F37021',
    'Bloc Québécois': '#33B2CC',
    'Green':          '#3D9B35',
}
TRAIN_YEAR  = 2021
TARGET_YEAR = 2025
NEXT_YEAR   = 2029

# ── Load inputs ────────────────────────────────────────────────────────────────
x_path     = OUT_DIR / 'x_nlp_party_summary_2021_2025.csv'
votes_path = ROOT / 'outputs' / 'national_vote_share_clean.csv'

if not x_path.exists():
    raise FileNotFoundError(f'Missing: {x_path}  →  run x_nlp_data_recovery.py first.')
if not votes_path.exists():
    raise FileNotFoundError(f'Missing: {votes_path}  →  run election_workflow.py first.')

x_df = pd.read_csv(x_path)
x_df['SupportIndex'] = pd.to_numeric(x_df['SupportIndex'], errors='coerce')
x_df['XShare']       = pd.to_numeric(x_df['XShare'],       errors='coerce')
x_df = x_df[x_df['Party'].isin(MAIN_PARTIES)].copy()

votes = pd.read_csv(votes_path)
votes['VoteShare'] = pd.to_numeric(votes['VoteShare'], errors='coerce')
actual = votes[votes['Year'].isin([TRAIN_YEAR, TARGET_YEAR])].copy()

# ── Fit ridge regression on 2021 data ─────────────────────────────────────────
# Model: log(normalised_main_share) ~ 1 + SupportIndex + log(XShare)
# (same formulation as predict_from_x_nlp_factor in factor_based_...)
actual_2021 = actual[actual['Year'] == TRAIN_YEAR][['Party', 'VoteShare']].copy()
train = (
    x_df[x_df['Year'] == TRAIN_YEAR][['Party', 'SupportIndex', 'XShare']]
    .merge(actual_2021, on='Party', how='inner')
)
train['VoteShareMain'] = train['VoteShare'] / train['VoteShare'].sum()

X_mat = np.column_stack([
    np.ones(len(train)),
    train['SupportIndex'].to_numpy(),
    np.log(train['XShare'].clip(lower=1e-6).to_numpy()),
])
y_vec = np.log(train['VoteShareMain'].clip(lower=1e-6).to_numpy())
ridge   = 0.2
penalty = np.eye(X_mat.shape[1]);  penalty[0, 0] = 0.0
beta = np.linalg.solve(X_mat.T @ X_mat + ridge * penalty, X_mat.T @ y_vec)

others_row = actual[(actual['Year'] == TRAIN_YEAR) & (actual['Party'] == 'Others')]
others_baseline = float(others_row['VoteShare'].iloc[0]) if not others_row.empty else 0.0
main_total = max(0.0, 100.0 - others_baseline)


def predict_year(yr):
    frame = x_df[x_df['Year'] == yr][['Party', 'SupportIndex', 'XShare']].dropna().copy()
    frame = frame.set_index('Party').reindex(MAIN_PARTIES).reset_index()
    if frame[['SupportIndex', 'XShare']].isna().any().any():
        return None
    Z = np.column_stack([
        np.ones(len(frame)),
        frame['SupportIndex'].to_numpy(),
        np.log(frame['XShare'].clip(lower=1e-6).to_numpy()),
    ])
    logits     = Z @ beta
    main_share = np.exp(logits - logits.max())
    main_share /= main_share.sum()
    return pd.DataFrame({'Party': MAIN_PARTIES, 'PredictedVoteShare': main_share * main_total})


predictions = {yr: p for yr in [TRAIN_YEAR, TARGET_YEAR, NEXT_YEAR]
               if (p := predict_year(yr)) is not None}

# ─────────────────────────────────────────────────────────────────────────────
# PNG 1 — X/social signal trend: SupportIndex and XShare, 2021 vs 2025
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
COLORS = [PARTY_COLORS[p] for p in MAIN_PARTIES]
x_pos  = np.arange(len(MAIN_PARTIES))
w = 0.35

for ax, col, title, ylabel in [
    (axes[0], 'SupportIndex',
     'Support Index (Positive% − Negative%)', 'Support Index'),
    (axes[1], 'XShare',
     'X Mention Share per Party', 'Proportion of X mentions'),
]:
    d21 = x_df[x_df['Year'] == TRAIN_YEAR ].set_index('Party')[col].reindex(MAIN_PARTIES)
    d25 = x_df[x_df['Year'] == TARGET_YEAR].set_index('Party')[col].reindex(MAIN_PARTIES)
    ax.bar(x_pos - w/2, d21.values, w, label='2021 (actual tweets)',
           color=COLORS, alpha=0.55, edgecolor='black', linewidth=0.4)
    ax.bar(x_pos + w/2, d25.values, w, label='2025 (actual tweets)',
           color=COLORS, alpha=1.0, edgecolor='black', linewidth=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MAIN_PARTIES, rotation=20, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if col == 'SupportIndex':
        ax.axhline(0, color='black', linewidth=0.8)

fig.suptitle('X / Social Media Signal: 2021 vs 2025', fontsize=13, fontweight='bold')
plt.tight_layout()
png1 = OUT_DIR / 'x_nlp_trend_2021_2025.png'
plt.savefig(png1, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {png1}')

# ─────────────────────────────────────────────────────────────────────────────
# PNG 2 — Predicted vote shares: 2021 / 2025 / 2029
# ─────────────────────────────────────────────────────────────────────────────
available = [yr for yr in [TRAIN_YEAR, TARGET_YEAR, NEXT_YEAR] if yr in predictions]
n  = len(available)
w  = 0.22
x_pos = np.arange(len(MAIN_PARTIES))
year_style = {
    TRAIN_YEAR:  dict(alpha=0.50, hatch='',   linestyle='-'),
    TARGET_YEAR: dict(alpha=0.85, hatch='',   linestyle='-'),
    NEXT_YEAR:   dict(alpha=1.00, hatch='//', linestyle='-'),
}
year_label = {TRAIN_YEAR: '2021 (in-sample)', TARGET_YEAR: '2025 (predicted)', NEXT_YEAR: '2029 (projected)'}

fig, ax = plt.subplots(figsize=(12, 5))
for i, yr in enumerate(available):
    offset = (i - (n - 1) / 2) * w
    vals   = predictions[yr].set_index('Party').reindex(MAIN_PARTIES)['PredictedVoteShare'].fillna(0)
    st     = year_style.get(yr, dict(alpha=0.8, hatch=''))
    ax.bar(x_pos + offset, vals.values, w,
           label=year_label[yr],
           color=COLORS, edgecolor='black', linewidth=0.4, **st)

ax.set_xticks(x_pos)
ax.set_xticklabels(MAIN_PARTIES, rotation=20, ha='right')
ax.set_ylabel('Predicted vote share (%)')
ax.set_title('X/NLP-Based Vote Share Prediction: 2021 → 2025 → 2029',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.text(0.01, 0.97,
        f'Model: ridge regression on SupportIndex + log(XShare)\n'
        f'Fitted on 2021 actual results · Others fixed at {others_baseline:.1f}%',
        transform=ax.transAxes, va='top', fontsize=7.5, color='dimgray',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', edgecolor='lightgray'))
plt.tight_layout()
png2 = OUT_DIR / 'x_nlp_predicted_2021_2025_2029.png'
plt.savefig(png2, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {png2}')

# ─────────────────────────────────────────────────────────────────────────────
# PNG 3 — Predicted 2025 vs actual 2025
# ─────────────────────────────────────────────────────────────────────────────
actual_2025 = actual[actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']].copy()
if TARGET_YEAR not in predictions:
    print('No 2025 predictions — skipping comparison chart.')
    png3 = None
else:
    comp = (
        predictions[TARGET_YEAR]
        .merge(actual_2025, on='Party', how='left')
        .set_index('Party').reindex(MAIN_PARTIES).reset_index()
    )
    comp['Error'] = comp['PredictedVoteShare'] - comp['VoteShare']
    mae  = comp['Error'].abs().mean()
    rmse = np.sqrt((comp['Error'] ** 2).mean())

    x_pos = np.arange(len(comp))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x_pos - w/2, comp['PredictedVoteShare'].values, w,
           label='Predicted (X/NLP)', color=COLORS, alpha=0.75,
           edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + w/2, comp['VoteShare'].fillna(0).values, w,
           label='Actual 2025', color=COLORS, alpha=1.0,
           hatch='//', edgecolor='black', linewidth=0.5)

    for i, row in comp.iterrows():
        if pd.notna(row['VoteShare']):
            top = max(row['PredictedVoteShare'], row['VoteShare']) + 0.5
            ax.annotate(f'{row["Error"]:+.1f}%', xy=(i, top),
                        ha='center', fontsize=8,
                        color='green' if abs(row['Error']) < 2 else 'crimson')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(comp['Party'].values, rotation=20, ha='right')
    ax.set_ylabel('Vote share (%)')
    ax.set_title('X/NLP Prediction vs Actual Result — 2025',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.text(0.98, 0.97, f'MAE = {mae:.2f}%   RMSE = {rmse:.2f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
    plt.tight_layout()
    png3 = OUT_DIR / 'x_nlp_predicted_vs_actual_2025.png'
    plt.savefig(png3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {png3}')

# ── Display in notebook ────────────────────────────────────────────────────────
if 'ipykernel' in sys.modules and IPyImage is not None and display is not None:
    for p in [png1, png2, png3]:
        if p is not None and Path(p).exists():
            display(IPyImage(str(p)))
