from pathlib import Path
import zipfile
import urllib.request
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import Image as IPyImage, display
except Exception:
    IPyImage = None
    display = None

PROJECT_MARKERS = (
    '.git',
    'canada_federal_vote_share_2000_2025.csv',
    'econ_2021_2025.py',
)
PROVINCES = ['Alberta', 'British Columbia', 'Ontario', 'Quebec']
PARTIES = ['Liberal', 'Conservative', 'NDP', 'Bloc Québécois', 'Green', 'Others']
TRAIN_YEAR = 2021
TARGET_YEAR = 2025
NEXT_YEAR = 2029

ELECTION_SOURCE_URLS = {
    2021: 'https://www.elections.ca/content.aspx?section=res&dir=rep/off/44gedata&document=byprovtbl&lang=e',
    2025: 'https://www.elections.ca/content.aspx?section=res&dir=rep/off/45gedata&document=byprovtbl&lang=e',
}
STATCAN_SOURCE_URLS = {
    'population': 'https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip',
    'education': 'https://www150.statcan.gc.ca/n1/tbl/csv/37100130-eng.zip',
}
EDUCATION_LEVELS = [
    ('Below upper secondary', 'Below upper secondary'),
    ('Upper secondary', 'Upper secondary'),
    ('Post-secondary non-tertiary', 'Post-secondary non-tertiary education'),
    ('Short-cycle tertiary', 'Short-cycle tertiary'),
    ("Bachelor's level", "Bachelor's level"),
    ("Master's or Doctoral", "Master's or Doctoral level"),
    ('Tertiary education', 'Tertiary education'),
]


def find_project_root():
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start


def col_like(cols, *tokens):
    tokens = [t.lower() for t in tokens]
    for col in cols:
        low = str(col).lower()
        if all(tok in low for tok in tokens):
            return col
    return None


def pick_value_label(values, contains_any):
    vals = pd.Series(values).dropna().astype(str).unique().tolist()
    for key in contains_any:
        k = key.lower()
        for val in vals:
            if k in val.lower():
                return val
    return None


def scalar_multiplier(x):
    if x is None:
        return 1.0
    text = str(x).lower().replace(',', '')
    if '1000000' in text or '1 000 000' in text or 'million' in text:
        return 1_000_000.0
    if '1000' in text or '1 000' in text or 'thousand' in text:
        return 1_000.0
    return 1.0


def to_year(series):
    y = pd.to_numeric(series, errors='coerce')
    if y.notna().any():
        return y
    d = pd.to_datetime(series, errors='coerce')
    return d.dt.year


def download_if_missing(url, path):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())
    return path


def load_zip_csv(path, url=None):
    if not path.exists():
        if url is None:
            raise FileNotFoundError(path)
        download_if_missing(url, path)
    with zipfile.ZipFile(path, 'r') as zf:
        names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        data_names = [n for n in names if 'meta' not in n.lower()]
        target = data_names[0] if data_names else names[0]
        with zf.open(target) as fh:
            return pd.read_csv(fh, dtype=str, low_memory=False)


def project_linear(df, group_cols, year_col, value_col, target_year):
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        g = grp[[year_col, value_col]].dropna().copy()
        g[year_col] = pd.to_numeric(g[year_col], errors='coerce')
        g[value_col] = pd.to_numeric(g[value_col], errors='coerce')
        g = g.dropna().sort_values(year_col)
        if g.empty:
            pred = np.nan
        elif len(g) == 1:
            pred = float(g[value_col].iloc[-1])
        else:
            slope, intercept = np.polyfit(g[year_col], g[value_col], 1)
            pred = float(slope * target_year + intercept)
        row = {year_col: target_year, value_col: pred}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, key in zip(group_cols, keys):
            row[col] = key
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_provincial_vote_share_cache(root, out_dir):
    cache_path = out_dir / 'elections_canada_ab_bc_on_qc_vote_share_2021_2025.csv'
    if cache_path.exists():
        return pd.read_csv(cache_path)

    manual = [
        (2021, 'Alberta', 15.48, 55.42, 18.22, 0.00, 2.28),
        (2021, 'British Columbia', 26.63, 33.74, 29.50, 0.00, 8.19),
        (2021, 'Ontario', 39.50, 35.06, 17.20, 0.00, 2.29),
        (2021, 'Quebec', 33.65, 18.59, 7.90, 32.16, 2.06),
        (2025, 'Alberta', 27.26, 64.12, 0.00, 0.00, 2.55),
        (2025, 'British Columbia', 40.79, 32.54, 12.76, 0.00, 12.38),
        (2025, 'Ontario', 47.75, 44.90, 6.27, 0.00, 0.87),
        (2025, 'Quebec', 44.23, 23.70, 0.66, 27.67, 0.88),
    ]
    rows = []
    for year, province, liberal, conservative, ndp, bloc, green in manual:
        vals = {
            'Liberal': liberal,
            'Conservative': conservative,
            'NDP': ndp,
            'Bloc Québécois': bloc,
            'Green': green,
        }
        vals['Others'] = round(100.0 - sum(vals.values()), 2)
        for party, share in vals.items():
            rows.append({
                'Year': year,
                'Province': province,
                'Party': party,
                'VoteShare': share,
                'SourceURL': ELECTION_SOURCE_URLS[year],
                'SourceNote': 'Main-party percentages transcribed from official Elections Canada province result pages; Others is residual to 100%.',
            })
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def load_population_series(cache_dir):
    pop_raw = load_zip_csv(cache_dir / '17100005-eng.zip', STATCAN_SOURCE_URLS['population'])
    ref = 'REF_DATE' if 'REF_DATE' in pop_raw.columns else col_like(pop_raw.columns, 'ref', 'date')
    geo = 'GEO' if 'GEO' in pop_raw.columns else col_like(pop_raw.columns, 'geo')
    val = 'VALUE' if 'VALUE' in pop_raw.columns else col_like(pop_raw.columns, 'value')
    sf = 'SCALAR_FACTOR' if 'SCALAR_FACTOR' in pop_raw.columns else col_like(pop_raw.columns, 'scalar', 'factor')
    sex_col = col_like(pop_raw.columns, 'sex') or col_like(pop_raw.columns, 'gender')
    age_col = col_like(pop_raw.columns, 'age group') or col_like(pop_raw.columns, 'age')

    total_sex = pick_value_label(pop_raw[sex_col], ['both sexes', 'total - gender', 'total - sex', 'total'])
    all_ages = pick_value_label(pop_raw[age_col], ['all ages', 'total'])
    geos = PROVINCES + ['Canada']

    d = pop_raw[pop_raw[geo].isin(geos) & (pop_raw[sex_col] == total_sex)].copy()
    d['Year'] = to_year(d[ref])
    d['VALUE_NUM'] = pd.to_numeric(d[val], errors='coerce')
    d['mult'] = d[sf].apply(scalar_multiplier)
    d['value_scaled'] = d['VALUE_NUM'] * d['mult']

    pop = (
        d[d[age_col] == all_ages]
        .groupby([geo, 'Year'], as_index=False)['value_scaled']
        .mean()
        .rename(columns={geo: 'Province', 'value_scaled': 'Population'})
    )
    pop['Population'] = pd.to_numeric(pop['Population'], errors='coerce')
    return pop.dropna(subset=['Population'])


def load_education_level_series(cache_dir, education_level):
    edu_raw = load_zip_csv(cache_dir / '37100130-eng.zip', STATCAN_SOURCE_URLS['education'])
    d = edu_raw[
        edu_raw['GEO'].isin(PROVINCES)
        & (edu_raw['Gender'] == 'Total - Gender')
        & (edu_raw['Age group'] == 'Total, 25 to 64 years')
        & (edu_raw['Education attainment level'] == education_level)
    ].copy()
    d['Year'] = pd.to_numeric(d['REF_DATE'], errors='coerce')
    d['FactorValue'] = pd.to_numeric(d['VALUE'], errors='coerce')
    d = d[['GEO', 'Year', 'FactorValue']].rename(columns={'GEO': 'Province'})
    return d.dropna(subset=['Year', 'FactorValue'])


def build_residual_region_baseline(votes, national_2021, pop_2021, canada_population):
    actual_four = (
        votes[votes['Year'] == TRAIN_YEAR]
        .merge(pop_2021[['Province', 'Population']], on='Province', how='left')
        .assign(weight=lambda d: d['Population'] / d['Population'].sum())
        .groupby('Party', as_index=False)
        .apply(lambda g: pd.Series({'Share': np.average(g['VoteShare'], weights=g['weight'])}), include_groups=False)
        .reset_index()
    )
    actual_four = actual_four[['Party', 'Share']]
    coverage = pop_2021['Population'].sum() / canada_population.loc[canada_population['Year'] == TRAIN_YEAR, 'Population'].iloc[0]
    base = national_2021.merge(actual_four, on='Party', how='left').fillna({'Share': 0.0})
    base['ResidualShare'] = np.where(
        coverage < 1,
        (base['VoteShare'] - coverage * base['Share']) / (1 - coverage),
        base['VoteShare'],
    )
    return base[['Party', 'ResidualShare']], coverage


def predict_from_factor(name, factor_df, votes, pop_all, national_actual, canada_population):
    factor_df = factor_df.copy()
    pop_four = pop_all[pop_all['Province'].isin(PROVINCES)].copy()
    train_factor = factor_df[factor_df['Year'] == TRAIN_YEAR][['Province', 'FactorValue']].copy()
    if train_factor['Province'].nunique() < len(PROVINCES):
        return None

    factor_mean = train_factor['FactorValue'].mean()
    factor_std = train_factor['FactorValue'].std(ddof=0)
    scale = factor_std if factor_std and not pd.isna(factor_std) else 1.0

    coeffs = []
    train = votes[votes['Year'] == TRAIN_YEAR].merge(train_factor, on='Province', how='left')
    train['x_std'] = (train['FactorValue'] - factor_mean) / scale
    for party in PARTIES:
        s = train[train['Party'] == party].copy()
        if s['x_std'].nunique() <= 1:
            slope = 0.0
            intercept = float(s['VoteShare'].mean())
        else:
            slope, intercept = np.polyfit(s['x_std'], s['VoteShare'], 1)
        coeffs.append({'Party': party, 'Slope': float(slope), 'Intercept': float(intercept)})
    coeffs = pd.DataFrame(coeffs)

    residual_baseline, _ = build_residual_region_baseline(
        votes,
        national_actual[national_actual['Year'] == TRAIN_YEAR][['Party', 'VoteShare']],
        pop_four[pop_four['Year'] == TRAIN_YEAR][['Province', 'Population']],
        canada_population,
    )

    national_predictions = []
    for year in [TRAIN_YEAR, TARGET_YEAR, NEXT_YEAR]:
        year_factor = factor_df[factor_df['Year'] == year][['Province', 'FactorValue']].copy()
        if len(year_factor) != len(PROVINCES):
            continue
        year_factor['x_std'] = (year_factor['FactorValue'] - factor_mean) / scale
        pred = year_factor.assign(key=1).merge(coeffs.assign(key=1), on='key').drop(columns='key')
        pred['PredictedShare'] = pred['Intercept'] + pred['Slope'] * pred['x_std']
        pred['PredictedShare'] = pred['PredictedShare'].clip(lower=0.0)
        totals = pred.groupby('Province')['PredictedShare'].transform('sum').replace(0, np.nan)
        pred['PredictedShare'] = np.where(totals.notna(), pred['PredictedShare'] / totals * 100.0, 100.0 / len(PARTIES))

        pop_y = pop_all[(pop_all['Province'].isin(PROVINCES)) & (pop_all['Year'] == year)][['Province', 'Population']].copy()
        canada_y = canada_population.loc[canada_population['Year'] == year, 'Population']
        if pop_y.empty or canada_y.empty:
            continue
        coverage = pop_y['Population'].sum() / canada_y.iloc[0]
        pred = pred.merge(pop_y, on='Province', how='left')
        pred['weight'] = pred['Population'] / pred['Population'].sum()
        national_main = pred.groupby('Party', as_index=False).apply(
            lambda g: pd.Series({'PredictedMainShare': np.average(g['PredictedShare'], weights=g['weight'])}),
            include_groups=False,
        ).reset_index()
        national_main = national_main[['Party', 'PredictedMainShare']]

        merged = national_main.merge(residual_baseline, on='Party', how='left').fillna({'ResidualShare': 0.0})
        merged['PredictedVoteShare'] = coverage * merged['PredictedMainShare'] + (1 - coverage) * merged['ResidualShare']
        merged['PredictedVoteShare'] = merged['PredictedVoteShare'].clip(lower=0.0)
        total = merged['PredictedVoteShare'].sum()
        if total > 0:
            merged['PredictedVoteShare'] = merged['PredictedVoteShare'] / total * 100.0
        merged['Year'] = year
        merged['Factor'] = name
        national_predictions.append(merged[['Year', 'Party', 'PredictedVoteShare', 'Factor']])

    if not national_predictions:
        return None

    national_predictions = pd.concat(national_predictions, ignore_index=True)
    pred_2025 = national_predictions[national_predictions['Year'] == TARGET_YEAR][['Party', 'PredictedVoteShare']]
    actual_2025 = national_actual[national_actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']]
    comp_2025 = pred_2025.merge(actual_2025, on='Party', how='left')
    comp_2025['AbsError'] = (comp_2025['PredictedVoteShare'] - comp_2025['VoteShare']).abs()
    comp_2025['SqError'] = (comp_2025['PredictedVoteShare'] - comp_2025['VoteShare']) ** 2
    return {
        'factor': name,
        'national_predictions': national_predictions,
        'fit_mae': float(comp_2025['AbsError'].mean()),
        'fit_rmse': float(np.sqrt(comp_2025['SqError'].mean())),
    }


ROOT = find_project_root()
OUT_DIR = ROOT / 'outputs_demographics'
CACHE_DIR = ROOT / 'data' / 'statcan_cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

votes = ensure_provincial_vote_share_cache(ROOT, OUT_DIR)
votes['VoteShare'] = pd.to_numeric(votes['VoteShare'], errors='coerce')
votes = votes[votes['Party'].isin(PARTIES)].copy()

national_actual = pd.read_csv(ROOT / 'outputs' / 'national_vote_share_clean.csv')
national_actual = national_actual[national_actual['Year'].isin([TRAIN_YEAR, TARGET_YEAR])].copy()
national_actual['VoteShare'] = pd.to_numeric(national_actual['VoteShare'], errors='coerce')
national_actual = national_actual[national_actual['Party'].isin(PARTIES)].copy()

population_all = load_population_series(CACHE_DIR)
canada_population = population_all[population_all['Province'] == 'Canada'].copy()
if TARGET_YEAR not in set(population_all['Year']):
    pop_proj = project_linear(
        population_all[population_all['Province'].isin(PROVINCES) & population_all['Year'].between(2021, 2024)],
        ['Province'],
        'Year',
        'Population',
        TARGET_YEAR,
    )
    can_proj = project_linear(
        canada_population[canada_population['Year'].between(2021, 2024)],
        ['Province'],
        'Year',
        'Population',
        TARGET_YEAR,
    )
    population_all = pd.concat([population_all, pop_proj, can_proj], ignore_index=True)
    canada_population = population_all[population_all['Province'] == 'Canada'].copy()

# Also project population to 2029 if missing
if NEXT_YEAR not in set(population_all['Year']):
    pop_proj_2029 = project_linear(
        population_all[population_all['Province'].isin(PROVINCES) & population_all['Year'].between(2021, 2025)],
        ['Province'],
        'Year',
        'Population',
        NEXT_YEAR,
    )
    can_proj_2029 = project_linear(
        canada_population[canada_population['Year'].between(2021, 2025)],
        ['Province'],
        'Year',
        'Population',
        NEXT_YEAR,
    )
    population_all = pd.concat([population_all, pop_proj_2029, can_proj_2029], ignore_index=True)
    canada_population = population_all[population_all['Province'] == 'Canada'].copy()

results = []
for display_label, raw_level in EDUCATION_LEVELS:
    factor_df = load_education_level_series(CACHE_DIR, raw_level)
    if TARGET_YEAR not in set(factor_df['Year']):
        factor_2025 = project_linear(
            factor_df[factor_df['Year'].between(2021, 2024)],
            ['Province'],
            'Year',
            'FactorValue',
            TARGET_YEAR,
        )
        factor_df = pd.concat([factor_df, factor_2025], ignore_index=True)
    if NEXT_YEAR not in set(factor_df['Year']):
        factor_2029 = project_linear(
            factor_df[factor_df['Year'].between(2021, 2025)],
            ['Province'],
            'Year',
            'FactorValue',
            NEXT_YEAR,
        )
        factor_df = pd.concat([factor_df, factor_2029], ignore_index=True)
    result = predict_from_factor(
        f'Education: {display_label}',
        factor_df,
        votes,
        population_all,
        national_actual,
        canada_population,
    )
    if result:
        result['education_level'] = raw_level
        results.append(result)

summary = pd.DataFrame([
    {
        'Factor': r['factor'],
        'EducationLevel': r['education_level'],
        'FitMAE_2025': r['fit_mae'],
        'FitRMSE_2025': r['fit_rmse'],
    }
    for r in results
]).sort_values(['FitMAE_2025', 'FitRMSE_2025']).reset_index(drop=True)
summary.to_csv(OUT_DIR / 'education_attainment_fit_summary_2025.csv', index=False)

all_preds = pd.concat([r['national_predictions'] for r in results], ignore_index=True)
pred_2025 = all_preds[all_preds['Year'] == TARGET_YEAR].merge(
    national_actual[national_actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']],
    on='Party',
    how='left',
).rename(columns={'VoteShare': 'ActualVoteShare'})
pred_2025.to_csv(OUT_DIR / 'education_attainment_predictions_2025.csv', index=False)

# Save 2029 projections
pred_2029 = all_preds[all_preds['Year'] == NEXT_YEAR].copy()
if not pred_2029.empty:
    pred_2029.to_csv(OUT_DIR / 'education_attainment_predictions_2029.csv', index=False)
    print(f"Saved: {OUT_DIR / 'education_attainment_predictions_2029.csv'}")

best_factor = summary.iloc[0]['Factor']
best_pred = pred_2025[pred_2025['Factor'] == best_factor].copy().set_index('Party').reindex(PARTIES).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
plot_summary = summary.sort_values('FitMAE_2025', ascending=True)
axes[0].barh(plot_summary['Factor'], plot_summary['FitMAE_2025'], color='#4C78A8')
axes[0].set_xlabel('2025 MAE (percentage points)')
axes[0].set_title('Education definitions ranked by 2025 fit')
axes[0].grid(axis='x', linestyle='--', alpha=0.3)
axes[0].invert_yaxis()
for idx, row in plot_summary.reset_index(drop=True).iterrows():
    axes[0].text(row['FitMAE_2025'] + 0.05, idx, f"{row['FitMAE_2025']:.2f}", va='center', fontsize=8)

x = np.arange(len(PARTIES))
width = 0.38
axes[1].bar(x - width / 2, best_pred['ActualVoteShare'], width=width, label='Actual 2025', color='#2E7BB4')
axes[1].bar(x + width / 2, best_pred['PredictedVoteShare'], width=width, label=best_factor, color='#F28E2B')
axes[1].set_xticks(x)
axes[1].set_xticklabels(PARTIES, rotation=20, ha='right')
axes[1].set_ylabel('Vote share (%)')
axes[1].set_title('Best education-only prediction vs actual 2025')
axes[1].legend(fontsize=8)
axes[1].grid(axis='y', linestyle='--', alpha=0.3)

fig.suptitle('Education-attainment models for predicting the 2025 Canada election')
fig.text(
    0.01,
    0.01,
    'Sources: Elections Canada province results (2021/2025) and Statistics Canada table 37100130. 2025 education shares are linearly extrapolated from 2021-2024.',
    ha='left',
    va='bottom',
    fontsize=8,
    color='dimgray',
)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.84, wspace=0.38)
chart_path = OUT_DIR / 'education_attainment_prediction_2025.png'
plt.savefig(chart_path, dpi=200, bbox_inches='tight')
plt.close(fig)

if 'ipykernel' in sys.modules and IPyImage is not None and display is not None:
    display(IPyImage(filename=str(chart_path)))

print('Source URLs:')
print(f"- Elections Canada 2021 province results: {ELECTION_SOURCE_URLS[2021]}")
print(f"- Elections Canada 2025 province results: {ELECTION_SOURCE_URLS[2025]}")
print(f"- StatCan education attainment: {STATCAN_SOURCE_URLS['education']}")
print(f"- StatCan population (for weighting): {STATCAN_SOURCE_URLS['population']}")
print('')
print('Education-attainment fit summary:')
print(summary.round(3).to_string(index=False))
print('')
print(f'Best education definition by 2025 MAE: {best_factor}')
print('')
print('2025 education-based predictions vs actual:')
print(
    pred_2025
    .pivot_table(index='Party', columns='Factor', values='PredictedVoteShare')
    .merge(national_actual[national_actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']].set_index('Party'), left_index=True, right_index=True, how='left')
    .rename(columns={'VoteShare': 'Actual 2025'})
    .round(2)
    .to_string()
)
print('')
print(f'Saved chart: {chart_path}')
print(f"Saved summary CSV: {OUT_DIR / 'education_attainment_fit_summary_2025.csv'}")
print(f"Saved prediction CSV: {OUT_DIR / 'education_attainment_predictions_2025.csv'}")

