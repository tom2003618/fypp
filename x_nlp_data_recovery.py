from pathlib import Path
import re
import numpy as np
import pandas as pd


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
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_TWEETS_CSV = ROOT / 'notebooks' / 'data' / 'twitter_canada_pre_election_raw.csv'
OUT_CSV = OUT_DIR / 'x_nlp_party_summary_2021_2025.csv'

PARTY_TERMS = {
    'Liberal': ['liberal', 'liberal party', 'liberal party of canada', 'trudeau', 'team trudeau', 'parti liberal'],
    'Conservative': ['conservative', 'conservative party', 'conservative party of canada', 'cpc', 'poilievre', 'parti conservateur'],
    'NDP': ['ndp', 'new democratic party', 'jagmeet', 'jagmeet singh', 'nouveau parti democratique'],
    'Bloc Québécois': ['bloc', 'bloc quebecois', 'blocquebecois', 'bq', 'blanchet'],
    'Green': ['green', 'green party', 'green party of canada', 'elizabeth may', 'greens'],
}
PARTY_PATTERNS = {
    party: re.compile(r"\b(" + '|'.join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    for party, terms in PARTY_TERMS.items()
}

import torch
from transformers import pipeline as _hf_pipeline

MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'


def preprocess_for_roberta(text):
    """Anonymise @mentions and URLs as the model expects."""
    text = str(text)
    text = re.sub(r'http\S+', 'http', text)
    text = re.sub(r'@\w+', '@user', text)
    return re.sub(r'\s+', ' ', text).strip()


print(f'Loading {MODEL_NAME} ...')
_device = 0 if torch.cuda.is_available() else -1
_pipe = _hf_pipeline(
    'text-classification',
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=_device,
    truncation=True,
    max_length=128,
    top_k=None,
)


def score_batch(texts, batch_size=32):
    """Return list of {positive, neutral, negative} probability dicts."""
    results = []
    for i in range(0, len(texts), batch_size):
        for out in _pipe(texts[i:i + batch_size]):
            results.append({d['label'].lower(): d['score'] for d in out})
    return results


def project_linear(df, value_col, target_year):
    rows = []
    for party, grp in df.groupby('Party'):
        g = grp[['Year', value_col]].dropna().sort_values('Year')
        if len(g) == 1:
            pred = float(g[value_col].iloc[-1])
        else:
            slope, intercept = np.polyfit(g['Year'], g[value_col], 1)
            pred = float(slope * target_year + intercept)
        rows.append({'Party': party, 'Year': target_year, value_col: pred})
    return pd.DataFrame(rows)


raw = pd.read_csv(RAW_TWEETS_CSV)
raw['roberta_text'] = raw['content'].map(preprocess_for_roberta)
raw['party_list'] = raw['content'].map(
    lambda text: [party for party, pattern in PARTY_PATTERNS.items() if pattern.search(text)]
)
expanded = raw.explode('party_list').dropna(subset=['party_list']).rename(columns={'party_list': 'Party'})

# Score all unique texts across both years in one pass
unique_texts = expanded['roberta_text'].unique().tolist()
print(f'Running Twitter-RoBERTa on {len(unique_texts):,} unique texts ...')
scores = score_batch(unique_texts)
text_scores = dict(zip(unique_texts, scores))

expanded['pos_score'] = expanded['roberta_text'].map(lambda t: text_scores[t].get('positive', 0.0))
expanded['neg_score'] = expanded['roberta_text'].map(lambda t: text_scores[t].get('negative', 0.0))
expanded['sentiment_label'] = expanded['roberta_text'].map(
    lambda t: max(text_scores[t], key=text_scores[t].get)
)

# Aggregate per year × party — works for any election years present in the CSV
year_summaries = []
for yr, grp in expanded.groupby('election_year'):
    yr = int(yr)
    vol  = grp.groupby('Party', as_index=False).agg(TweetVolume=('tweet_id', 'nunique'))
    prob = grp.groupby('Party', as_index=False).agg(
        PositivePct=('pos_score', 'mean'),
        NegativePct=('neg_score', 'mean'),
    )
    s = vol.merge(prob, on='Party', how='left')
    s['SupportIndex']           = s['PositivePct'] - s['NegativePct']
    s['XShare']                 = s['TweetVolume'] / s['TweetVolume'].sum()
    s['Year']                   = yr
    s['TotalMentions']          = np.nan
    s['EstimatedXMentions']     = s['TweetVolume']
    s['XChannelShareWithinParty'] = 1.0
    s['SourceType']             = f'x_api_{yr}'
    s['SourceURL']              = str(RAW_TWEETS_CSV)
    s['Notes']                  = (
        f'Computed from X API tweets ({yr} pre-election window) '
        f'using {MODEL_NAME} sentiment model.'
    )
    year_summaries.append(s)

summary = pd.concat(year_summaries, ignore_index=True)

proj_support = project_linear(summary[summary['Year'].isin([2021, 2025])], 'SupportIndex', 2029)
proj_xshare = project_linear(summary[summary['Year'].isin([2021, 2025])], 'XShare', 2029)
summary_2029 = proj_support.merge(proj_xshare, on=['Party', 'Year'], how='inner')
summary_2029['XShare'] = summary_2029['XShare'].clip(lower=0.005)
summary_2029['XShare'] = summary_2029['XShare'] / summary_2029['XShare'].sum()
summary_2029['SupportIndex'] = summary_2029['SupportIndex'].clip(-1.0, 1.0)
summary_2029['TweetVolume'] = np.nan
summary_2029['TotalMentions'] = np.nan
summary_2029['EstimatedXMentions'] = np.nan
summary_2029['PositivePct'] = np.nan
summary_2029['NegativePct'] = np.nan
summary_2029['XChannelShareWithinParty'] = np.nan
summary_2029['SourceType'] = 'linear_projection_2029'
summary_2029['SourceURL'] = str(RAW_TWEETS_CSV)
summary_2029['Notes'] = 'SupportIndex and XShare linearly projected from 2021 and 2025 values.'
summary = pd.concat([summary, summary_2029[summary.columns]], ignore_index=True)

summary.to_csv(OUT_CSV, index=False)

print(f'Saved: {OUT_CSV}')
print('')
print(summary.sort_values(['Year', 'Party']).round(4).to_string(index=False))
