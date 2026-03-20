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

ARTICLE_URL = 'https://mozaikanalytics.com/blog/how-to/2025-canada-election-retrospective'
CHART5_URL = 'https://mozaikanalytics.com/wp-content/uploads/2025/06/Chart-5.png'
CHART6_URL = 'https://mozaikanalytics.com/wp-content/uploads/2025/06/Chart-6.png'
CHART7_URL = 'https://mozaikanalytics.com/wp-content/uploads/2025/06/Chart-7.png'

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

POS_WORDS = {
    'good', 'great', 'support', 'strong', 'win', 'hope', 'trust', 'better',
    'excellent', 'improve', 'positive', 'leader', 'stable', 'progress',
}
NEG_WORDS = {
    'bad', 'worse', 'corrupt', 'hate', 'weak', 'fail', 'lying', 'scandal',
    'negative', 'angry', 'broken', 'disaster', 'problem', 'inflation',
}


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def score_text(text):
    tokens = re.findall(r"[a-z']+", str(text).lower())
    if not tokens:
        return 0.0
    pos = sum(t in POS_WORDS for t in tokens)
    neg = sum(t in NEG_WORDS for t in tokens)
    return float((pos - neg) / max(len(tokens), 1))


def sentiment_label(score):
    if score >= 0.05:
        return 'positive'
    if score <= -0.05:
        return 'negative'
    return 'neutral'


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
raw['clean_text'] = raw['content'].map(clean_text)
raw['party_list'] = raw['clean_text'].map(
    lambda text: [party for party, pattern in PARTY_PATTERNS.items() if pattern.search(text)]
)
expanded = raw.explode('party_list').dropna(subset=['party_list']).rename(columns={'party_list': 'Party'})
expanded['sentiment_score'] = expanded['clean_text'].map(score_text)
expanded['sentiment_label'] = expanded['sentiment_score'].map(sentiment_label)

volume_2021 = expanded.groupby('Party', as_index=False).agg(TweetVolume=('tweet_id', 'nunique'))
counts_2021 = expanded.pivot_table(index='Party', columns='sentiment_label', values='tweet_id', aggfunc='count', fill_value=0).reset_index()
summary_2021 = volume_2021.merge(counts_2021, on='Party', how='left')
for col in ['positive', 'negative', 'neutral']:
    if col not in summary_2021.columns:
        summary_2021[col] = 0
summary_2021['PositivePct'] = summary_2021['positive'] / summary_2021['TweetVolume'].clip(lower=1)
summary_2021['NegativePct'] = summary_2021['negative'] / summary_2021['TweetVolume'].clip(lower=1)
summary_2021['SupportIndex'] = summary_2021['PositivePct'] - summary_2021['NegativePct']
summary_2021['XShare'] = summary_2021['TweetVolume'] / summary_2021['TweetVolume'].sum()
summary_2021['Year'] = 2021
summary_2021['SourceType'] = 'local_x_raw_2021'
summary_2021['SourceURL'] = str(RAW_TWEETS_CSV)
summary_2021['Notes'] = 'Computed from local pre-election X raw tweets with simple lexicon sentiment.'
summary_2021['EstimatedXMentions'] = summary_2021['TweetVolume']
summary_2021['TotalMentions'] = np.nan
summary_2021['XChannelShareWithinParty'] = 1.0

summary_2025 = pd.DataFrame([
    {'Party': 'Liberal', 'TotalMentions': 82690, 'MentionShare': 0.29, 'PositivePct': 0.50, 'NegativePct': 0.41, 'XChannelShareWithinParty': 0.37},
    {'Party': 'Conservative', 'TotalMentions': 69254, 'MentionShare': 0.24, 'PositivePct': 0.47, 'NegativePct': 0.43, 'XChannelShareWithinParty': 0.31},
    {'Party': 'NDP', 'TotalMentions': 33317, 'MentionShare': 0.12, 'PositivePct': 0.57, 'NegativePct': 0.30, 'XChannelShareWithinParty': 0.26},
    {'Party': 'Bloc Québécois', 'TotalMentions': 21793, 'MentionShare': 0.08, 'PositivePct': 0.65, 'NegativePct': 0.21, 'XChannelShareWithinParty': 0.21},
    {'Party': 'Green', 'TotalMentions': 13642, 'MentionShare': 0.05, 'PositivePct': 0.58, 'NegativePct': 0.26, 'XChannelShareWithinParty': 0.26},
])
summary_2025['EstimatedXMentions'] = summary_2025['TotalMentions'] * summary_2025['XChannelShareWithinParty']
summary_2025['TweetVolume'] = summary_2025['EstimatedXMentions']
summary_2025['SupportIndex'] = summary_2025['PositivePct'] - summary_2025['NegativePct']
summary_2025['XShare'] = summary_2025['EstimatedXMentions'] / summary_2025['EstimatedXMentions'].sum()
summary_2025['Year'] = 2025
summary_2025['SourceType'] = 'public_2025_x_proxy'
summary_2025['SourceURL'] = ARTICLE_URL
summary_2025['Notes'] = (
    '2025 values transcribed from Mozaik Charts 5/6/7. '
    'Total mentions from Chart-5, sentiment from Chart-6, X channel share from Chart-7.'
)

summary = pd.concat([
    summary_2021[[
        'Year', 'Party', 'TweetVolume', 'TotalMentions', 'EstimatedXMentions',
        'PositivePct', 'NegativePct', 'SupportIndex', 'XShare', 'XChannelShareWithinParty',
        'SourceType', 'SourceURL', 'Notes',
    ]],
    summary_2025[[
        'Year', 'Party', 'TweetVolume', 'TotalMentions', 'EstimatedXMentions',
        'PositivePct', 'NegativePct', 'SupportIndex', 'XShare', 'XChannelShareWithinParty',
        'SourceType', 'SourceURL', 'Notes',
    ]],
], ignore_index=True)

proj_support = project_linear(summary[summary['Year'].isin([2021, 2025])], 'SupportIndex', 2029)
proj_xshare = project_linear(summary[summary['Year'].isin([2021, 2025])], 'XShare', 2029)
summary_2029 = proj_support.merge(proj_xshare, on=['Party', 'Year'], how='inner')
summary_2029['XShare'] = summary_2029['XShare'].clip(lower=0.0)
summary_2029['XShare'] = summary_2029['XShare'] / summary_2029['XShare'].sum()
summary_2029['SupportIndex'] = summary_2029['SupportIndex'].clip(-1.0, 1.0)
summary_2029['TweetVolume'] = np.nan
summary_2029['TotalMentions'] = np.nan
summary_2029['EstimatedXMentions'] = np.nan
summary_2029['PositivePct'] = np.nan
summary_2029['NegativePct'] = np.nan
summary_2029['XChannelShareWithinParty'] = np.nan
summary_2029['SourceType'] = 'linear_projection_2029'
summary_2029['SourceURL'] = ARTICLE_URL
summary_2029['Notes'] = 'SupportIndex and XShare linearly projected from 2021 and 2025 values.'
summary = pd.concat([summary, summary_2029[summary.columns]], ignore_index=True)

summary.to_csv(OUT_CSV, index=False)

print('Source URLs:')
print(f'- 2025 retrospective article: {ARTICLE_URL}')
print(f'- Chart 5 (total mentions): {CHART5_URL}')
print(f'- Chart 6 (sentiment): {CHART6_URL}')
print(f'- Chart 7 (X channel share): {CHART7_URL}')
print('')
print(f'Saved: {OUT_CSV}')
print('')
print(summary.sort_values(['Year', 'Party']).round(4).to_string(index=False))
