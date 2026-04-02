from pathlib import Path
import io
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
RESEARCH_2019_URL = 'https://researchco.ca/wp-content/uploads/2019/10/Tables_PoliticsCAN_20Oct2019.pdf'
RESEARCH_2021_URL = 'https://researchco.ca/wp-content/uploads/2021/09/Tables_Politics_CAN_14Sep2021.pdf'
TURNOUT_2015_URL = 'https://www.elections.ca/content.aspx?dir=rec/eval/pes2015/vtsa&document=table1&lang=e&section=res'
TURNOUT_2019_URL = 'https://www.elections.ca/content.aspx?dir=rec/eval/pes2019/vtsa&document=index&lang=e&section=res'
TURNOUT_2021_URL = 'https://www.elections.ca/content.aspx?dir=rec/eval/pes2021/evt&document=p5&lang=e&section=res'
STATCAN_POP_URL = 'https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip'
HIST_YEARS = [2011, 2015, 2019, 2021]
TARGET_YEAR = 2025
NEXT_YEAR = 2029
AGE_BANDS = ['18-34', '35-54', '55+']
PARTIES = ['Liberal', 'Conservative', 'NDP', 'Bloc Québécois', 'Green', 'Others']
SEVEN_AGE_GROUPS = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
SEVEN_TO_BROAD = {
    '18-24': '18-34',
    '25-34': '18-34',
    '35-44': '35-54',
    '45-54': '35-54',
    '55-64': '55+',
    '65-74': '55+',
    '75+': '55+',
}
POP_LABELS = {
    '18-24': ['18 to 24 years'],
    '25-34': ['25 to 29 years', '30 to 34 years'],
    '35-44': ['35 to 39 years', '40 to 44 years'],
    '45-54': ['45 to 49 years', '50 to 54 years'],
    '55-64': ['55 to 59 years', '60 to 64 years'],
    '65-74': ['65 to 69 years', '70 to 74 years'],
    '75+': ['75 to 79 years', '80 to 84 years', '85 to 89 years', '90 years and older'],
}


def find_project_root():
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start


def download_if_missing(url, path, retries=3, timeout=120):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                path.write_bytes(resp.read())
            return path
        except Exception as exc:
            last_error = exc
            if path.exists():
                path.unlink()
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 5))
    raise RuntimeError(f'Failed to download {url}: {last_error}') from last_error


def normalize_age_label(label):
    text = str(label).strip()
    text = text.replace('\xa0', ' ')
    text = text.replace('–', '-')
    text = text.replace('&ndash;', '-')
    text = re.sub(r'Footnote\s*\d+', '', text, flags=re.I)
    text = text.replace(' to ', '-')
    text = text.replace(' years and over', '+')
    text = text.replace(' years', '').replace(' year', '')
    text = text.replace(' and older', '+')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def flatten_columns(df):
    if not isinstance(df.columns, pd.MultiIndex):
        return [str(c).strip() for c in df.columns]
    flat = []
    for col in df.columns:
        parts = [str(x).strip() for x in col if str(x).strip() and str(x).strip() != 'nan']
        flat.append(' | '.join(parts))
    return flat


def load_turnout_history(cache_dir):
    rows = []

    table_2015 = pd.read_html(TURNOUT_2015_URL)[0]
    table_2015.columns = flatten_columns(table_2015)
    c_age = next(c for c in table_2015.columns if c.startswith('Age Group'))
    c_2011 = next(c for c in table_2015.columns if c.startswith('2011') and 'Estimate' in c and 'Lower' not in c and 'Upper' not in c)
    c_2015 = next(c for c in table_2015.columns if c.startswith('2015') and 'Estimate' in c and 'Lower' not in c and 'Upper' not in c)
    t15 = table_2015.copy()
    t15['AgeGroup'] = t15[c_age].astype(str).map(normalize_age_label)
    t15 = t15[t15['AgeGroup'].isin(SEVEN_AGE_GROUPS)]
    for year, col in [(2011, c_2011), (2015, c_2015)]:
        tmp = t15[['AgeGroup', col]].copy().rename(columns={col: 'TurnoutPct'})
        tmp['TurnoutPct'] = pd.to_numeric(tmp['TurnoutPct'], errors='coerce')
        tmp['Year'] = year
        rows.append(tmp)

    turnout_2019_path = cache_dir / 'elections_canada_turnout_age_2019.html'
    download_if_missing(TURNOUT_2019_URL, turnout_2019_path)
    table_2019 = pd.read_html(io.StringIO(turnout_2019_path.read_bytes().decode('latin1')))[0]
    table_2019.columns = flatten_columns(table_2019)
    c_geo = next(c for c in table_2019.columns if 'Prov./Terr.' in c)
    c_sex = next(c for c in table_2019.columns if 'Sex' in c)
    c_age = next(c for c in table_2019.columns if 'Age group' in c)
    c_2019 = next(c for c in table_2019.columns if c.startswith('2019 general election') and 'estimate' in c.lower() and 'lower' not in c.lower() and 'upper' not in c.lower())
    t19 = table_2019[(table_2019[c_geo] == 'Canada') & (table_2019[c_sex] == 'Both sexes')].copy()
    t19['AgeGroup'] = t19[c_age].astype(str).map(normalize_age_label)
    t19 = t19[t19['AgeGroup'].isin(SEVEN_AGE_GROUPS)]
    t19 = t19[['AgeGroup', c_2019]].copy().rename(columns={c_2019: 'TurnoutPct'})
    t19['TurnoutPct'] = pd.to_numeric(t19['TurnoutPct'], errors='coerce')
    t19['Year'] = 2019
    rows.append(t19)

    turnout_2021_path = cache_dir / 'elections_canada_turnout_age_2021.html'
    download_if_missing(TURNOUT_2021_URL, turnout_2021_path)
    table_2021 = pd.read_html(io.StringIO(turnout_2021_path.read_bytes().decode('latin1')))[0]
    table_2021['AgeGroup'] = table_2021['Age Group'].astype(str).map(normalize_age_label)
    t21 = table_2021[table_2021['AgeGroup'].isin(SEVEN_AGE_GROUPS)][['AgeGroup', 'Turnout']].copy().rename(columns={'Turnout': 'TurnoutPct'})
    t21['TurnoutPct'] = pd.to_numeric(t21['TurnoutPct'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    t21['Year'] = 2021
    rows.append(t21)

    turnout = pd.concat(rows, ignore_index=True)
    return turnout[['Year', 'AgeGroup', 'TurnoutPct']].sort_values(['Year', 'AgeGroup']).reset_index(drop=True)


def load_population_age_groups(cache_path):
    with zipfile.ZipFile(cache_path, 'r') as zf:
        name = [n for n in zf.namelist() if n.lower().endswith('.csv') and 'meta' not in n.lower()][0]
        with zf.open(name) as fh:
            pop = pd.read_csv(fh, dtype=str, low_memory=False)
    pop = pop[(pop['GEO'] == 'Canada') & (pop['Gender'].astype(str).str.contains('Both|Total', case=False, na=False))].copy()
    pop['Year'] = pd.to_numeric(pop['REF_DATE'].astype(str).str.extract(r'(\d{4})', expand=False), errors='coerce')
    pop['VALUE_NUM'] = pd.to_numeric(pop['VALUE'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    pop = pop[pop['Year'].isin(HIST_YEARS + [TARGET_YEAR, NEXT_YEAR])]
    rows = []
    for age_group, labels in POP_LABELS.items():
        tmp = pop[pop['Age group'].isin(labels)].groupby('Year', as_index=False)['VALUE_NUM'].sum()
        tmp['AgeGroup'] = age_group
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    return out[['Year', 'AgeGroup', 'VALUE_NUM']].rename(columns={'VALUE_NUM': 'Population'})


def broad_age_weights(pop_age, turnout):
    hist = turnout.copy()
    for proj_year in [TARGET_YEAR, NEXT_YEAR]:
        for age_group in SEVEN_AGE_GROUPS:
            g = hist[hist['AgeGroup'] == age_group].sort_values('Year')
            slope, intercept = np.polyfit(g['Year'], g['TurnoutPct'], 1)
            pred = float(np.clip(slope * proj_year + intercept, 0.0, 100.0))
            hist = pd.concat([hist, pd.DataFrame({'Year': [proj_year], 'AgeGroup': [age_group], 'TurnoutPct': [pred]})], ignore_index=True)
    merged = pop_age.merge(hist, on=['Year', 'AgeGroup'], how='inner')
    merged['EstimatedVoters'] = merged['Population'] * merged['TurnoutPct'] / 100.0
    merged['AgeBand'] = merged['AgeGroup'].map(SEVEN_TO_BROAD)
    weights = merged.groupby(['Year', 'AgeBand'], as_index=False)['EstimatedVoters'].sum()
    weights['Weight'] = weights.groupby('Year')['EstimatedVoters'].transform(lambda s: s / s.sum())
    return weights, hist.sort_values(['Year', 'AgeGroup']).reset_index(drop=True)


def load_poll_pref(pdf_cache):
    download_if_missing(RESEARCH_2019_URL, pdf_cache / 'Tables_PoliticsCAN_20Oct2019.pdf')
    download_if_missing(RESEARCH_2021_URL, pdf_cache / 'Tables_Politics_CAN_14Sep2021.pdf')
    ROOT = find_project_root()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from researchco_age_support import parse_multiple_age_tables

    pref = parse_multiple_age_tables({
        2019: pdf_cache / 'Tables_PoliticsCAN_20Oct2019.pdf',
        2021: pdf_cache / 'Tables_Politics_CAN_14Sep2021.pdf',
    }).copy()
    pref['AgeBand'] = pd.Categorical(pref['AgeBand'], categories=AGE_BANDS, ordered=True)
    pref['Party'] = pd.Categorical(pref['Party'], categories=PARTIES, ordered=True)
    pref = pref.sort_values(['Year', 'AgeBand', 'Party']).reset_index(drop=True)
    pref['SupportPct'] = pref.groupby(['Year', 'AgeBand'])['SupportPct'].transform(lambda s: s / s.sum() * 100.0)
    return pref


def load_national_shares(path):
    df = pd.read_csv(path)
    df = df[df['Year'].isin(HIST_YEARS + [TARGET_YEAR])].copy()
    df['Party'] = df['Party'].replace({
        "People's": 'Others',
        'People’s': 'Others',
        "People's Party": 'Others',
        'People’s Party': 'Others',
    })
    df = df[df['Party'].isin(PARTIES)].copy()
    df = df.groupby(['Year', 'Party'], as_index=False)['VoteShare'].sum()
    df['VoteShare'] = pd.to_numeric(df['VoteShare'], errors='coerce')
    df['VoteShare'] = df.groupby('Year')['VoteShare'].transform(lambda s: s / s.sum() * 100.0)
    return df.sort_values(['Year', 'Party']).reset_index(drop=True)


def pivot_matrix(df, year, value_col='SupportPct'):
    return (
        df[df['Year'] == year]
        .pivot(index='AgeBand', columns='Party', values=value_col)
        .reindex(index=AGE_BANDS, columns=PARTIES)
    )


def estimate_age_effects(pref, weights):
    beta_list = []
    for year in [2019, 2021]:
        mat = pivot_matrix(pref, year).fillna(0.0) / 100.0
        w = weights[weights['Year'] == year].set_index('AgeBand').reindex(AGE_BANDS)['Weight'].to_numpy()
        national = w @ mat.to_numpy()
        beta = np.log(np.clip(mat.to_numpy(), 1e-6, 1.0)) - np.log(np.clip(national, 1e-6, 1.0))
        beta_list.append(beta)
    beta = np.mean(beta_list, axis=0)
    beta = np.clip(beta, -2.5, 2.5)
    return pd.DataFrame(beta, index=AGE_BANDS, columns=PARTIES)


def fit_ipf_support(target_party_share, age_weights, beta_df, max_iter=3000, tol=1e-10):
    target = np.asarray(target_party_share, dtype=float)
    target = target / target.sum()
    w = np.asarray(age_weights, dtype=float)
    w = w / w.sum()
    s = np.exp(beta_df.reindex(index=AGE_BANDS, columns=PARTIES).to_numpy()) * target[None, :]
    s = np.clip(s, 1e-8, None)
    for _ in range(max_iter):
        s = s / s.sum(axis=1, keepdims=True)
        current = (w[:, None] * s).sum(axis=0)
        s *= (target / np.clip(current, 1e-12, None))[None, :]
        row_err = np.abs(s.sum(axis=1) - 1.0).max()
        col_err = np.abs((w[:, None] * s).sum(axis=0) - target).max()
        if max(row_err, col_err) < tol:
            break
    s = s / s.sum(axis=1, keepdims=True)
    return pd.DataFrame(s, index=AGE_BANDS, columns=PARTIES)


def project_national_2025(national):
    rows = []
    for party in PARTIES:
        s = national[national['Party'] == party].sort_values('Year')
        years = s['Year'].to_numpy(dtype=float)
        vals = np.clip(s['VoteShare'].to_numpy(dtype=float) / 100.0, 1e-6, 1.0)
        slope, intercept = np.polyfit(years, np.log(vals), 1)
        rows.append((party, float(np.exp(slope * TARGET_YEAR + intercept))))
    out = pd.DataFrame(rows, columns=['Party', 'PredShare'])
    out['PredShare'] = out['PredShare'] / out['PredShare'].sum() * 100.0
    return out


def project_national_year(national, year):
    rows = []
    for party in PARTIES:
        s = national[national['Party'] == party].sort_values('Year')
        years = s['Year'].to_numpy(dtype=float)
        vals = np.clip(s['VoteShare'].to_numpy(dtype=float) / 100.0, 1e-6, 1.0)
        slope, intercept = np.polyfit(years, np.log(vals), 1)
        rows.append((party, float(np.exp(slope * year + intercept))))
    out = pd.DataFrame(rows, columns=['Party', 'PredShare'])
    out['PredShare'] = out['PredShare'] / out['PredShare'].sum() * 100.0
    return out


def draw_heatmap(ax, matrix, title, vmax):
    im = ax.imshow(matrix.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=20, ha='right')
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel('Party')
    for i, age_band in enumerate(matrix.index):
        row = matrix.loc[age_band]
        row_max = row.max()
        for j, party in enumerate(matrix.columns):
            val = row[party]
            color = 'white' if val >= vmax * 0.55 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=8.5)
            if np.isclose(val, row_max):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=2))
    return im


ROOT = find_project_root()
OUT_DIR = ROOT / 'outputs_demographics'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_CACHE = ROOT / 'data' / 'external_age_sources'
PDF_CACHE.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / 'data' / 'statcan_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

download_if_missing(STATCAN_POP_URL, CACHE_DIR / '17100005-eng.zip')
pref_obs = load_poll_pref(PDF_CACHE)
national = load_national_shares(ROOT / 'outputs' / 'national_vote_share_clean.csv')
pop_age = load_population_age_groups(CACHE_DIR / '17100005-eng.zip')

# Project population by age group to 2029 if missing
if NEXT_YEAR not in set(pop_age['Year']):
    proj_rows = []
    for ag in pop_age['AgeGroup'].unique():
        g = pop_age[pop_age['AgeGroup'] == ag].sort_values('Year')
        recent = g[g['Year'].between(2019, 2025)]
        if len(recent) >= 2:
            slope, intercept = np.polyfit(recent['Year'], recent['Population'], 1)
            proj_rows.append({'Year': NEXT_YEAR, 'AgeGroup': ag,
                              'Population': max(0, float(slope * NEXT_YEAR + intercept))})
    if proj_rows:
        pop_age = pd.concat([pop_age, pd.DataFrame(proj_rows)], ignore_index=True)

turnout = load_turnout_history(PDF_CACHE)
weights, turnout_with_projection = broad_age_weights(pop_age, turnout)

beta = estimate_age_effects(pref_obs, weights)
nat_pred_2025 = project_national_2025(national)
weights_2025 = weights[weights['Year'] == TARGET_YEAR].set_index('AgeBand').reindex(AGE_BANDS)['Weight']
model_2025 = fit_ipf_support(
    nat_pred_2025.set_index('Party').reindex(PARTIES)['PredShare'].to_numpy() / 100.0,
    weights_2025.to_numpy(),
    beta,
) * 100.0

# 2029 projection
nat_pred_2029 = project_national_year(national, NEXT_YEAR)
weights_2029 = weights[weights['Year'] == NEXT_YEAR].set_index('AgeBand').reindex(AGE_BANDS)['Weight']
if weights_2029.notna().all() and not weights_2029.empty:
    model_2029 = fit_ipf_support(
        nat_pred_2029.set_index('Party').reindex(PARTIES)['PredShare'].to_numpy() / 100.0,
        weights_2029.to_numpy(),
        beta,
    ) * 100.0
else:
    model_2029 = None

model_rows = []
for year in HIST_YEARS + [TARGET_YEAR, NEXT_YEAR]:
    if year == TARGET_YEAR:
        mat = model_2025.copy()
        data_type = 'predicted_2025'
    elif year == NEXT_YEAR:
        if model_2029 is None:
            continue
        mat = model_2029.copy()
        data_type = 'projected_2029'
    else:
        nat = national[national['Year'] == year].set_index('Party').reindex(PARTIES)['VoteShare'].to_numpy() / 100.0
        w = weights[weights['Year'] == year].set_index('AgeBand').reindex(AGE_BANDS)['Weight'].to_numpy()
        mat = fit_ipf_support(nat, w, beta) * 100.0
        data_type = 'model_implied'
    for age_band in AGE_BANDS:
        for party in PARTIES:
            model_rows.append({
                'Year': year,
                'AgeBand': age_band,
                'Party': party,
                'SupportPct': float(mat.loc[age_band, party]),
                'DataType': data_type,
            })
model_df = pd.DataFrame(model_rows)
model_df.to_csv(OUT_DIR / 'voter_generation_party_preference_model_2011_2025.csv', index=False)

display_df = pd.concat([
    pref_obs.assign(DataType='observed_poll')[['Year', 'AgeBand', 'Party', 'SupportPct', 'DataType']],
    pd.DataFrame(model_rows).query('Year == @TARGET_YEAR')[['Year', 'AgeBand', 'Party', 'SupportPct', 'DataType']],
], ignore_index=True)
display_df.to_csv(OUT_DIR / 'voter_generation_party_preference_display_2019_2025.csv', index=False)

age_weight_df = weights.copy()
age_weight_df.to_csv(OUT_DIR / 'voter_generation_age_weight_history_2011_2025.csv', index=False)

summary_rows = []
for age_band in AGE_BANDS:
    top2 = (
        model_2025.loc[age_band]
        .sort_values(ascending=False)
        .head(2)
    )
    summary_rows.append({
        'Year': TARGET_YEAR,
        'AgeBand': age_band,
        'TopParty': top2.index[0],
        'TopPartyPct': float(top2.iloc[0]),
        'SecondParty': top2.index[1],
        'SecondPartyPct': float(top2.iloc[1]),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / 'voter_generation_party_top_choice_model_2025.csv', index=False)

youth_senior_gap = pd.DataFrame({
    'Party': PARTIES,
    '18-34': model_2025.loc['18-34', PARTIES].to_numpy(),
    '55+': model_2025.loc['55+', PARTIES].to_numpy(),
})
youth_senior_gap['YouthMinusSenior'] = youth_senior_gap['18-34'] - youth_senior_gap['55+']

agg_2025 = pd.Series(0.0, index=PARTIES)
for age_band in AGE_BANDS:
    agg_2025 += weights_2025.loc[age_band] * model_2025.loc[age_band]
actual_2025 = national[national['Year'] == TARGET_YEAR].set_index('Party').reindex(PARTIES)['VoteShare']
comp_2025 = pd.DataFrame({'PredictedNational2025': agg_2025, 'Actual2025': actual_2025})
comp_2025['AbsError'] = (comp_2025['PredictedNational2025'] - comp_2025['Actual2025']).abs()
mae_2025 = float(comp_2025['AbsError'].mean())

fig = plt.figure(figsize=(18.2, 5.8))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
cax = fig.add_subplot(gs[0, 3])
m2019 = pivot_matrix(pref_obs, 2019)
m2021 = pivot_matrix(pref_obs, 2021)
vmax = max(float(m2019.max().max()), float(m2021.max().max()), float(model_2025.max().max()))
im = draw_heatmap(ax1, m2019, '2019 observed age profile', vmax)
draw_heatmap(ax2, m2021, '2021 observed age profile', vmax)
draw_heatmap(ax3, model_2025, '2025 predicted from 2011-2021 elections', vmax)
ax1.set_ylabel('Age group')
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Support within age group (%)')
fig.suptitle('Canada voter-generation party preference\nObserved 2019/2021 age profiles and 2025 prediction from recent federal-election history', y=0.98)
fig.text(
    0.01,
    0.01,
    'Sources: Research Co age cross-tabs (2019/2021); Elections Canada turnout-by-age tables (2011/2015/2019/2021); Statistics Canada population-by-age table 17100005.',
    ha='left',
    va='bottom',
    fontsize=8,
    color='dimgray',
)
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.20, top=0.8)
chart_path = OUT_DIR / 'fig_voter_generation_party_preference_model_2025.png'
plt.savefig(chart_path, dpi=200, bbox_inches='tight')
plt.close(fig)

comparison_fig, comparison_ax = plt.subplots(figsize=(10.2, 5.2))
plot_df = comp_2025.reset_index().rename(columns={'index': 'Party'})
x = np.arange(len(PARTIES))
bar_width = 0.38
comparison_ax.bar(x - bar_width / 2, plot_df['PredictedNational2025'], width=bar_width, label='Predicted 2025 from age model', color='#dd8452')
comparison_ax.bar(x + bar_width / 2, plot_df['Actual2025'], width=bar_width, label='Actual 2025', color='#4c72b0')
comparison_ax.set_xticks(x)
comparison_ax.set_xticklabels(PARTIES, rotation=20, ha='right')
comparison_ax.set_ylabel('Vote share (%)')
comparison_ax.set_xlabel('Party')
comparison_ax.set_title(f'2025 national result comparison: age-model prediction vs actual\nMAE = {mae_2025:.2f} percentage points')
comparison_ax.grid(axis='y', linestyle='--', alpha=0.35)
comparison_ax.legend(frameon=False)
comparison_ax.set_ylim(0, max(float(plot_df[['PredictedNational2025', 'Actual2025']].to_numpy().max()) + 8.0, 12.0))
for offset, col in [(-bar_width / 2, 'PredictedNational2025'), (bar_width / 2, 'Actual2025')]:
    for xi, val in zip(x, plot_df[col]):
        comparison_ax.text(xi + offset, float(val) + 0.5, f'{val:.2f}%', ha='center', va='bottom', fontsize=8)
comparison_fig.subplots_adjust(bottom=0.22, top=0.84, left=0.08, right=0.98)
comparison_chart_path = OUT_DIR / 'fig_voter_generation_prediction_vs_actual_2025.png'
comparison_fig.savefig(comparison_chart_path, dpi=200, bbox_inches='tight')
plt.close(comparison_fig)

if 'ipykernel' in sys.modules and IPyImage is not None and display is not None:
    display(IPyImage(filename=str(chart_path)))
    display(IPyImage(filename=str(comparison_chart_path)))

print('Source URLs:')
print(f'- Research Co age cross-tabs (2019): {RESEARCH_2019_URL}')
print(f'- Research Co age cross-tabs (2021): {RESEARCH_2021_URL}')
print(f'- Elections Canada turnout by age (2011/2015): {TURNOUT_2015_URL}')
print(f'- Elections Canada turnout by age (2019): {TURNOUT_2019_URL}')
print(f'- Elections Canada turnout by age (2021): {TURNOUT_2021_URL}')
print(f'- Statistics Canada population by age: {STATCAN_POP_URL}')
print('- National vote-share input: outputs/national_vote_share_clean.csv')
print('')
print('Electorate age weights used by election year:')
print(weights.pivot(index='Year', columns='AgeBand', values='Weight').round(4).to_string())
print('')
print('Projected 2025 national vote shares from the last four federal elections:')
print(nat_pred_2025.round(2).to_string(index=False))
print('')
print('Predicted 2025 party preference by age group:')
print(model_2025.round(2).to_string())
print('')
print('Top two parties within each predicted 2025 age band:')
print(summary.round(2).to_string(index=False))
print('')
print('Youth minus senior support gap for predicted 2025 (18-34 minus 55+):')
print(youth_senior_gap.round(2).to_string(index=False))
print('')
print('2025 national aggregation check from the age model:')
print(comp_2025.round(2).to_string())
print(f'2025 national MAE from this age model: {mae_2025:.2f}')
print('')
print(f'Saved chart: {chart_path}')
print(f'Saved comparison chart: {comparison_chart_path}')
print(f"Saved model CSV: {OUT_DIR / 'voter_generation_party_preference_model_2011_2025.csv'}")
print(f"Saved display CSV: {OUT_DIR / 'voter_generation_party_preference_display_2019_2025.csv'}")
print(f"Saved age-weight CSV: {OUT_DIR / 'voter_generation_age_weight_history_2011_2025.csv'}")
print(f"Saved top-choice CSV: {OUT_DIR / 'voter_generation_party_top_choice_model_2025.csv'}")

