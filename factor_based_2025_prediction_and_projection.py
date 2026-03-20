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

def find_project_root():
    markers = [
        'canada_federal_vote_share_2000_2025.csv',
        'canada_federal_election_province_summary_2000_2021.csv',
        'econ_2021_2025.py',
        '.git',
    ]
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return Path.cwd()


ROOT = find_project_root()
OUT_DIR = ROOT / 'outputs_demographics'
CACHE_DIR = ROOT / 'data' / 'statcan_cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
    'population_age': 'https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip',
    'economy_composite_input': 'https://www150.statcan.gc.ca/n1/tbl/csv/36100222-eng.zip ; https://www150.statcan.gc.ca/n1/tbl/csv/14100287-eng.zip ; https://www150.statcan.gc.ca/n1/tbl/csv/14100223-eng.zip ; https://www150.statcan.gc.ca/n1/tbl/csv/14100371-eng.zip',
}


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
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=90) as resp:
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


def ensure_provincial_vote_share_cache():
    cache_path = OUT_DIR / 'elections_canada_ab_bc_on_qc_vote_share_2021_2025.csv'
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
                'SourceNote': 'Main-party percentages transcribed from official Elections Canada province result pages; Others is residual to 100%.'
            })
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def load_population_age_series():
    pop_raw = load_zip_csv(CACHE_DIR / '17100005-eng.zip', STATCAN_SOURCE_URLS['population_age'])
    ref = 'REF_DATE' if 'REF_DATE' in pop_raw.columns else col_like(pop_raw.columns, 'ref', 'date')
    geo = 'GEO' if 'GEO' in pop_raw.columns else col_like(pop_raw.columns, 'geo')
    val = 'VALUE' if 'VALUE' in pop_raw.columns else col_like(pop_raw.columns, 'value')
    sf = 'SCALAR_FACTOR' if 'SCALAR_FACTOR' in pop_raw.columns else col_like(pop_raw.columns, 'scalar', 'factor')
    sex_col = col_like(pop_raw.columns, 'sex') or col_like(pop_raw.columns, 'gender')
    age_col = col_like(pop_raw.columns, 'age group') or col_like(pop_raw.columns, 'age')

    total_sex = pick_value_label(pop_raw[sex_col], ['both sexes', 'total - gender', 'total - sex', 'total'])
    all_ages = pick_value_label(pop_raw[age_col], ['all ages', 'total'])
    median_age = pick_value_label(pop_raw[age_col], ['median age'])
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
    med = (
        d[d[age_col] == median_age]
        .groupby([geo, 'Year'], as_index=False)['value_scaled']
        .mean()
        .rename(columns={geo: 'Province', 'value_scaled': 'Median_age'})
    )
    return pop, med


def load_economy_factor():
    econ_path = ROOT / 'outputs_demographics' / 'econ_AB_BC_ON_QC_2021_2025.csv'
    econ = pd.read_csv(econ_path)
    metrics = ['GDP_per_capita', 'Avg_weekly_earnings', 'Job_vacancy_rate', 'Unemployment_rate']
    for col in metrics:
        econ[col] = pd.to_numeric(econ[col], errors='coerce')

    out = []
    for year, grp in econ.groupby('Year'):
        g = grp[['Province', 'Year'] + metrics].copy()
        for col in ['GDP_per_capita', 'Avg_weekly_earnings', 'Job_vacancy_rate']:
            std = g[col].std(ddof=0)
            g[f'z_{col}'] = 0.0 if std == 0 or pd.isna(std) else (g[col] - g[col].mean()) / std
        std = g['Unemployment_rate'].std(ddof=0)
        g['z_Unemployment_rate'] = 0.0 if std == 0 or pd.isna(std) else (g['Unemployment_rate'] - g['Unemployment_rate'].mean()) / std
        g['FactorValue'] = g['z_GDP_per_capita'] + g['z_Avg_weekly_earnings'] + g['z_Job_vacancy_rate'] - g['z_Unemployment_rate']
        out.append(g[['Province', 'Year', 'FactorValue']])
    econ_factor = pd.concat(out, ignore_index=True)

    next_rows = []
    for province, grp in econ_factor.groupby('Province'):
        g = grp.sort_values('Year')
        slope, intercept = np.polyfit(g['Year'], g['FactorValue'], 1)
        next_rows.append({'Province': province, 'Year': NEXT_YEAR, 'FactorValue': float(slope * NEXT_YEAR + intercept)})
    econ_factor = pd.concat([econ_factor, pd.DataFrame(next_rows)], ignore_index=True)
    return econ_factor


def build_residual_region_baseline(votes, national_2021, pop_year):
    actual_four = (
        votes[votes['Year'] == TRAIN_YEAR]
        .merge(pop_year[['Province', 'Population']], on='Province', how='left')
        .assign(weight=lambda d: d['Population'] / d['Population'].sum())
        .groupby('Party', as_index=False)
        .apply(lambda g: pd.Series({'Share': np.average(g['VoteShare'], weights=g['weight'])}), include_groups=False)
        .reset_index()
    )
    actual_four = actual_four[['Party', 'Share']]
    coverage = pop_year['Population'].sum() / canada_population.loc[canada_population['Year'] == TRAIN_YEAR, 'Population'].iloc[0]
    base = national_2021.merge(actual_four, on='Party', how='left').fillna({'Share': 0.0})
    base['ResidualShare'] = np.where(
        coverage < 1,
        (base['VoteShare'] - coverage * base['Share']) / (1 - coverage),
        base['VoteShare'],
    )
    return base[['Party', 'ResidualShare']], coverage


def predict_from_factor(name, factor_df, pop_all, national_actual):
    factor_df = factor_df.copy()
    pop_four = pop_all[pop_all['Province'].isin(PROVINCES)].copy()
    train_factor = factor_df[factor_df['Year'] == TRAIN_YEAR][['Province', 'FactorValue']].copy()
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

    residual_baseline, coverage_2021 = build_residual_region_baseline(
        votes,
        national_actual[national_actual['Year'] == TRAIN_YEAR][['Party', 'VoteShare']],
        pop_four[pop_four['Year'] == TRAIN_YEAR][['Province', 'Population']]
    )

    province_predictions = []
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
        pred['Year'] = year
        province_predictions.append(pred[['Year', 'Province', 'Party', 'PredictedShare']])

        pop_y = pop_all[(pop_all['Province'].isin(PROVINCES)) & (pop_all['Year'] == year)][['Province', 'Population']].copy()
        canada_y = canada_population.loc[canada_population['Year'] == year, 'Population']
        if pop_y.empty or canada_y.empty:
            continue
        canada_pop = float(canada_y.iloc[0])
        coverage_y = pop_y['Population'].sum() / canada_pop
        agg = pred.merge(pop_y, on='Province', how='left')
        agg['NationalWeight'] = agg['Population'] / canada_pop
        national = agg.groupby('Party', as_index=False).apply(
            lambda g: pd.Series({'WeightedFourProvinceShare': g['NationalWeight'].mul(g['PredictedShare']).sum()}),
            include_groups=False,
        ).reset_index()[['Party', 'WeightedFourProvinceShare']]
        national = national.merge(residual_baseline, on='Party', how='left').fillna({'ResidualShare': 0.0})
        national['PredictedVoteShare'] = national['WeightedFourProvinceShare'] + (1 - coverage_y) * national['ResidualShare']
        national['PredictedVoteShare'] = national['PredictedVoteShare'].clip(lower=0.0)
        total = national['PredictedVoteShare'].sum()
        if total > 0:
            national['PredictedVoteShare'] = national['PredictedVoteShare'] / total * 100.0
        national['Year'] = year
        national['Factor'] = name
        national['CoverageShare'] = coverage_y
        national_predictions.append(national[['Factor', 'Year', 'Party', 'PredictedVoteShare', 'CoverageShare']])

    if not national_predictions:
        return None

    province_predictions = pd.concat(province_predictions, ignore_index=True) if province_predictions else pd.DataFrame()
    national_predictions = pd.concat(national_predictions, ignore_index=True)

    pred_2025 = national_predictions[national_predictions['Year'] == TARGET_YEAR][['Party', 'PredictedVoteShare']]
    actual_2025 = national_actual[national_actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']]
    comp_2025 = pred_2025.merge(actual_2025, on='Party', how='left')
    comp_2025['AbsError'] = (comp_2025['PredictedVoteShare'] - comp_2025['VoteShare']).abs()
    comp_2025['SqError'] = (comp_2025['PredictedVoteShare'] - comp_2025['VoteShare']) ** 2

    pred_2021 = national_predictions[national_predictions['Year'] == TRAIN_YEAR][['Party', 'PredictedVoteShare']].rename(columns={'PredictedVoteShare': 'Predicted2021'})
    impact = pred_2025.merge(pred_2021, on='Party', how='left')
    impact_score = (impact['PredictedVoteShare'] - impact['Predicted2021']).abs().mean()

    return {
        'factor': name,
        'province_predictions': province_predictions,
        'national_predictions': national_predictions,
        'fit_mae': float(comp_2025['AbsError'].mean()),
        'fit_rmse': float(np.sqrt(comp_2025['SqError'].mean())),
        'impact_score': float(impact_score),
        'coverage_2025': float(national_predictions.loc[national_predictions['Year'] == TARGET_YEAR, 'CoverageShare'].iloc[0]),
    }


votes = ensure_provincial_vote_share_cache()
votes['VoteShare'] = pd.to_numeric(votes['VoteShare'], errors='coerce')

national_actual = pd.read_csv(ROOT / 'outputs' / 'national_vote_share_clean.csv')
national_actual = national_actual[national_actual['Year'].isin([TRAIN_YEAR, TARGET_YEAR])].copy()
national_actual['VoteShare'] = pd.to_numeric(national_actual['VoteShare'], errors='coerce')
national_actual = national_actual[national_actual['Party'].isin(PARTIES)].copy()

population_all, _ = load_population_age_series()
population_all['Population'] = pd.to_numeric(population_all['Population'], errors='coerce')
canada_population = population_all[population_all['Province'] == 'Canada'].copy()
population_four = population_all[population_all['Province'].isin(PROVINCES)].copy()

# Add 2029 projected populations for weighting.
pop_proj = project_linear(population_four[population_four['Year'].between(2021, 2025)], ['Province'], 'Year', 'Population', NEXT_YEAR)
can_proj = project_linear(canada_population[canada_population['Year'].between(2021, 2025)], ['Province'], 'Year', 'Population', NEXT_YEAR)
population_all = pd.concat([population_all, pop_proj, can_proj], ignore_index=True)
canada_population = population_all[population_all['Province'] == 'Canada'].copy()


# Rebase economy factor from existing synchronized 2021/2025 file and linearly project to 2029.
economy_factor = load_economy_factor()

def predict_from_x_nlp_factor(x_summary_path, national_actual):
    if not x_summary_path.exists():
        return None, 'missing_x_nlp_summary_csv'

    x_df = pd.read_csv(x_summary_path)
    main_parties = ['Liberal', 'Conservative', 'NDP', 'Bloc Québécois', 'Green']
    needed_years = {TRAIN_YEAR, TARGET_YEAR, NEXT_YEAR}
    x_df = x_df[x_df['Party'].isin(main_parties) & x_df['Year'].isin(needed_years)].copy()
    if x_df.empty:
        return None, 'empty_x_nlp_summary_csv'

    for col in ['SupportIndex', 'XShare']:
        x_df[col] = pd.to_numeric(x_df[col], errors='coerce')

    year_counts = x_df.groupby('Year')['Party'].nunique().to_dict()
    if any(year_counts.get(year, 0) < len(main_parties) for year in needed_years):
        return None, 'insufficient_x_party_rows'

    actual_main_2021 = national_actual[(national_actual['Year'] == TRAIN_YEAR) & (national_actual['Party'].isin(main_parties))][['Party', 'VoteShare']].copy()
    if len(actual_main_2021) < len(main_parties):
        return None, 'missing_actual_2021_for_x_factor'

    train = x_df[x_df['Year'] == TRAIN_YEAR][['Party', 'SupportIndex', 'XShare']].merge(actual_main_2021, on='Party', how='inner')
    if len(train) < len(main_parties):
        return None, 'missing_x_train_rows'

    train['VoteShareMain'] = train['VoteShare'] / train['VoteShare'].sum()
    X = np.column_stack([
        np.ones(len(train)),
        train['SupportIndex'].to_numpy(),
        np.log(train['XShare'].clip(lower=1e-6)).to_numpy(),
    ])
    y = np.log(train['VoteShareMain'].clip(lower=1e-6)).to_numpy()
    ridge = 0.2
    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + ridge * penalty, X.T @ y)

    others_baseline = national_actual[(national_actual['Year'] == TRAIN_YEAR) & (national_actual['Party'] == 'Others')]['VoteShare']
    others_baseline = float(others_baseline.iloc[0]) if not others_baseline.empty else 0.0

    national_predictions = []
    for year in [TRAIN_YEAR, TARGET_YEAR, NEXT_YEAR]:
        frame = x_df[x_df['Year'] == year][['Party', 'SupportIndex', 'XShare']].copy()
        frame = frame.dropna(subset=['SupportIndex', 'XShare'])
        frame = frame.set_index('Party').reindex(main_parties).reset_index()
        if frame['SupportIndex'].isna().any() or frame['XShare'].isna().any():
            continue
        Z = np.column_stack([
            np.ones(len(frame)),
            frame['SupportIndex'].to_numpy(),
            np.log(frame['XShare'].clip(lower=1e-6)).to_numpy(),
        ])
        logits = Z @ beta
        main_share = np.exp(logits - logits.max())
        main_share = main_share / main_share.sum()
        main_total = max(0.0, 100.0 - others_baseline)
        pred_main = main_share * main_total
        national = pd.DataFrame({
            'Factor': 'Political leaning (X/NLP)',
            'Year': year,
            'Party': main_parties,
            'PredictedVoteShare': pred_main,
            'CoverageShare': 1.0,
        })
        national = pd.concat([
            national,
            pd.DataFrame({
                'Factor': ['Political leaning (X/NLP)'],
                'Year': [year],
                'Party': ['Others'],
                'PredictedVoteShare': [others_baseline],
                'CoverageShare': [1.0],
            }),
        ], ignore_index=True)
        national_predictions.append(national)

    if not national_predictions:
        return None, 'x_predictions_not_generated'

    national_predictions = pd.concat(national_predictions, ignore_index=True)
    pred_2025 = national_predictions[national_predictions['Year'] == TARGET_YEAR][['Party', 'PredictedVoteShare']]
    actual_2025 = national_actual[national_actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']]
    comp_2025 = pred_2025.merge(actual_2025, on='Party', how='left')
    comp_2025['AbsError'] = (comp_2025['PredictedVoteShare'] - comp_2025['VoteShare']).abs()
    comp_2025['SqError'] = (comp_2025['PredictedVoteShare'] - comp_2025['VoteShare']) ** 2

    pred_2021 = national_predictions[national_predictions['Year'] == TRAIN_YEAR][['Party', 'PredictedVoteShare']].rename(columns={'PredictedVoteShare': 'Predicted2021'})
    impact = pred_2025.merge(pred_2021, on='Party', how='left')
    impact_score = (impact['PredictedVoteShare'] - impact['Predicted2021']).abs().mean()

    return {
        'factor': 'Political leaning (X/NLP)',
        'province_predictions': pd.DataFrame(),
        'national_predictions': national_predictions,
        'fit_mae': float(comp_2025['AbsError'].mean()),
        'fit_rmse': float(np.sqrt(comp_2025['SqError'].mean())),
        'impact_score': float(impact_score),
        'coverage_2025': 1.0,
    }, 'ranked'


factor_series = {
    'Economy (AB/BC/ON/QC weighted)': economy_factor,
}

factor_results = []
for factor_name, factor_df in factor_series.items():
    result = predict_from_factor(factor_name, factor_df, population_all, national_actual)
    if result:
        factor_results.append(result)

x_summary_path = OUT_DIR / 'x_nlp_party_summary_2021_2025.csv'
x_result, x_status = predict_from_x_nlp_factor(x_summary_path, national_actual)
if x_result:
    factor_results.append(x_result)

summary_rows = [
    {
        'Factor': r['factor'],
        'ImpactScore': r['impact_score'],
        'FitMAE_2025': r['fit_mae'],
        'FitRMSE_2025': r['fit_rmse'],
        'CoverageShare_2025': r['coverage_2025'],
        'Status': 'ranked',
    }
    for r in factor_results
]

if not x_result:
    summary_rows.append({
        'Factor': 'Political leaning (X/NLP)',
        'ImpactScore': np.nan,
        'FitMAE_2025': np.nan,
        'FitRMSE_2025': np.nan,
        'CoverageShare_2025': np.nan,
        'Status': x_status,
    })

factor_summary = pd.DataFrame(summary_rows)
status_order = {'ranked': 0}
factor_summary['StatusOrder'] = factor_summary['Status'].map(status_order).fillna(1)
factor_summary = factor_summary.sort_values(['StatusOrder', 'FitMAE_2025', 'ImpactScore'], na_position='last').drop(columns='StatusOrder')
ranked = factor_summary[factor_summary['Status'] == 'ranked'].copy()
most_impact_factor = ranked.sort_values('ImpactScore', ascending=False).iloc[0]['Factor']
best_fit_factor = ranked.sort_values('FitMAE_2025').iloc[0]['Factor']

all_national_preds = pd.concat([r['national_predictions'] for r in factor_results], ignore_index=True)
pred_2025 = all_national_preds[all_national_preds['Year'] == TARGET_YEAR].copy()
pred_2025 = pred_2025.merge(national_actual[national_actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']], on='Party', how='left')
pred_2025 = pred_2025.rename(columns={'VoteShare': 'ActualVoteShare'})

next_factors = []
for label in [most_impact_factor, best_fit_factor]:
    next_pred = all_national_preds[(all_national_preds['Factor'] == label) & (all_national_preds['Year'] == NEXT_YEAR)].copy()
    next_pred['Scenario'] = label
    next_factors.append(next_pred)

next_predictions = pd.concat(next_factors, ignore_index=True)
if most_impact_factor != best_fit_factor:
    ensemble = (
        next_predictions.groupby('Party', as_index=False)['PredictedVoteShare']
        .mean()
        .assign(Factor='Average of top-impact and best-fit factors', Year=NEXT_YEAR, CoverageShare=np.nan, Scenario='Average of top-impact and best-fit factors')
    )
    next_predictions = pd.concat([next_predictions, ensemble], ignore_index=True)

factor_summary.to_csv(OUT_DIR / 'factor_fit_impact_summary.csv', index=False)
pred_2025.to_csv(OUT_DIR / 'factor_based_2025_predictions.csv', index=False)
next_predictions.to_csv(OUT_DIR / 'factor_next_election_predictions.csv', index=False)


comparison_chart = (
    pred_2025
    .pivot_table(index='Party', columns='Factor', values='PredictedVoteShare')
    .merge(
        national_actual[national_actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']].set_index('Party'),
        left_index=True,
        right_index=True,
        how='left',
    )
    .rename(columns={'VoteShare': 'Actual 2025'})
)
plot_cols = ['Actual 2025'] + [c for c in comparison_chart.columns if c != 'Actual 2025']
comparison_chart = comparison_chart[plot_cols].reindex(PARTIES)
ax = comparison_chart.plot(kind='bar', figsize=(12, 6), width=0.82)
ax.set_title('2025 factor-based predictions vs actual result')
ax.set_xlabel('Party')
ax.set_ylabel('Vote share (%)')
ax.set_ylim(0, max(50, float(np.nanmax(comparison_chart.values)) + 5))
ax.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=20, ha='right')
plt.legend(title='Series', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
comparison_chart_path = OUT_DIR / 'factor_2025_predictions_vs_actual.png'
plt.savefig(comparison_chart_path, dpi=200, bbox_inches='tight')
plt.close()
if 'ipykernel' in sys.modules and IPyImage is not None and display is not None:
    display(IPyImage(filename=str(comparison_chart_path)))

coverage_2025 = ranked['CoverageShare_2025'].dropna().iloc[0]
print('Source URLs:')
print(f"- Elections Canada 2021 province results: {ELECTION_SOURCE_URLS[2021]}")
print(f"- Elections Canada 2025 province results: {ELECTION_SOURCE_URLS[2025]}")
print(f"- StatCan population (for weighting): {STATCAN_SOURCE_URLS['population_age']}")
print(f"- Economy inputs already cached from: {STATCAN_SOURCE_URLS['economy_composite_input']}")
print('- X/NLP proxy summary: outputs_demographics/x_nlp_party_summary_2021_2025.csv')
print('- 2025 X/NLP public proxy source: https://mozaikanalytics.com/blog/how-to/2025-canada-election-retrospective')
print('')
print(f'Four-province population coverage used for weighting (2025): {coverage_2025:.2%}')
print('')
print('Factor comparison summary:')
print(factor_summary.to_string(index=False))
print('')
print(f'Most impactful factor: {most_impact_factor}')
print(f'Best 2025 fit factor: {best_fit_factor}')
print('')
print('2025 factor-based predictions vs actual:')
print(
    pred_2025
    .pivot_table(index='Party', columns='Factor', values='PredictedVoteShare')
    .merge(national_actual[national_actual['Year'] == TARGET_YEAR][['Party', 'VoteShare']].set_index('Party'), left_index=True, right_index=True, how='left')
    .rename(columns={'VoteShare': 'Actual 2025'})
    .round(2)
    .to_string()
)
print('')
print(f'Next-cycle baseline year used for projection: {NEXT_YEAR}')
print(next_predictions.pivot_table(index='Party', columns='Scenario', values='PredictedVoteShare').round(2).to_string())
