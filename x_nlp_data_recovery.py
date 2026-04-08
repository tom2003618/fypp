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
expanded['neu_score'] = expanded['roberta_text'].map(lambda t: text_scores[t].get('neutral', 0.0))
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
        NeutralPct=('neu_score', 'mean'),
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
summary_2029['NeutralPct'] = np.nan
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

# ── Per-tweet sentiment CSV + PNG tables (2021 / 2025) ───────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def _strip_emoji(text):
    """Replace characters that DejaVu Sans cannot render with empty string."""
    out = []
    for c in str(text):
        cp = ord(c)
        # Keep Basic Latin, Latin Extended, and common punctuation
        if cp < 0x2000 and c.isprintable():
            out.append(c)
        elif 0x2000 <= cp < 0x2100:  # General punctuation (dashes, quotes)
            out.append(c)
        # Skip emoji, CJK, regional indicators, etc.
    return ''.join(out)

tweet_detail = expanded[['election_year', 'Party', 'content',
                         'pos_score', 'neu_score', 'neg_score',
                         'sentiment_label']].copy()
tweet_detail.columns = ['Year', 'Party', 'Content', 'Positive', 'Neutral',
                        'Negative', 'Dominant']
tweet_detail['Year'] = tweet_detail['Year'].astype(int)
tweet_detail = tweet_detail.sort_values(['Year', 'Party']).reset_index(drop=True)

# Round for display
for c in ['Positive', 'Neutral', 'Negative']:
    tweet_detail[c] = tweet_detail[c].round(4)

tweet_csv_path = OUT_DIR / 'x_nlp_per_tweet_sentiment.csv'
tweet_detail.to_csv(tweet_csv_path, index=False)
print(f'\nSaved per-tweet CSV: {tweet_csv_path}  ({len(tweet_detail)} rows)')

# Generate PNG tables — one per year, all pages
for yr in sorted(tweet_detail['Year'].unique()):
    sub = tweet_detail[tweet_detail['Year'] == yr].reset_index(drop=True)
    # Truncate, strip emoji, escape $ for matplotlib
    sub['Content'] = (sub['Content'].map(_strip_emoji).str[:80]
                      .str.replace('\n', ' ', regex=False)
                      .str.replace('$', '\\$', regex=False))
    n_rows = len(sub)

    page_size = 50
    n_pages = (n_rows + page_size - 1) // page_size

    for page in range(n_pages):
        start = page * page_size
        end = min(start + page_size, n_rows)
        chunk = sub.iloc[start:end]

        fig_h = max(4, 0.32 * len(chunk) + 2)
        fig, ax = plt.subplots(figsize=(22, fig_h))
        ax.axis('off')

        col_labels = ['Year', 'Party', 'Content', 'Positive', 'Neutral', 'Negative', 'Dominant']
        cell_data = []
        cell_colors = []
        for _, r in chunk.iterrows():
            dom = r['Dominant']
            row_data = [str(r['Year']), r['Party'], r['Content'],
                        f"{r['Positive']:.2%}", f"{r['Neutral']:.2%}",
                        f"{r['Negative']:.2%}", dom.capitalize()]
            dom_color = ('#E8F5E9' if dom == 'positive'
                         else '#FFEBEE' if dom == 'negative' else '#FFF8E1')
            row_colors = ['white', 'white', 'white',
                          '#E8F5E9', '#FFF8E1', '#FFEBEE', dom_color]
            cell_data.append(row_data)
            cell_colors.append(row_colors)

        tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                       cellColours=cell_colors,
                       colColours=['#E3F2FD'] * len(col_labels),
                       loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.auto_set_column_width(list(range(len(col_labels))))
        tbl.scale(1.0, 1.35)

        page_label = f' (page {page+1}/{n_pages})' if n_pages > 1 else ''
        ax.set_title(f'{yr} Pre-Election Tweets \u2014 Per-Sentence RoBERTa Sentiment{page_label}\n'
                     f'Rows {start+1}\u2013{end} of {n_rows}',
                     fontsize=11, fontweight='bold', pad=15)
        fig.tight_layout()
        suffix = f'_p{page+1}' if n_pages > 1 else ''
        png_path = OUT_DIR / f'x_nlp_per_tweet_{yr}{suffix}.png'
        fig.savefig(png_path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {png_path}')
