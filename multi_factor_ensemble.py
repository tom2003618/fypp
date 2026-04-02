# Multi-Factor Ensemble: combine all 5 prediction models into one weighted prediction
# Uses inverse-MAE weighting: models with lower 2025 error get higher weight.
#
# Models:
#   1. Economy (AB/BC/ON/QC weighted)     — linear regression on GDP composite
#   2. NLP (X/RoBERTa)                    — ridge regression on SupportIndex + log(XShare)
#   3. Education (Short-cycle tertiary)   — linear regression on education attainment
#   4. Age/Generation (IPF)               — iterative proportional fitting with age effects
#   5. Historical (Bayesian random walk)  — Gibbs-sampled hierarchical drift model
#
# Output:
#   outputs_demographics/ensemble_2025_predictions.csv
#   outputs_demographics/ensemble_2025_vs_actual.png
#   outputs_demographics/ensemble_2029_projections.csv
#   outputs_demographics/ensemble_2029_projections.png

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root ──────────────────────────────────────────────────────────
def find_project_root():
    markers = ['.git', 'canada_federal_vote_share_2000_2025.csv', 'econ_2021_2025.py']
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if any((candidate / m).exists() for m in markers):
            return candidate
    return Path.cwd()

ROOT = find_project_root()
OUT_DIR = ROOT / 'outputs_demographics'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARTIES = ['Liberal', 'Conservative', 'NDP', 'Bloc Québécois', 'Green', 'Others']
MAIN_PARTIES = PARTIES[:-1]
ACTUAL_2025 = {'Liberal': 43.76, 'Conservative': 41.31, 'NDP': 6.29,
               'Bloc Québécois': 6.29, 'Green': 1.22, 'Others': 1.13}

# ── Helper: load actual 2025 from CSV if available ────────────────────────
actual_csv = ROOT / 'outputs' / 'national_vote_share_clean.csv'
if actual_csv.exists():
    _adf = pd.read_csv(actual_csv)
    _a25 = _adf[_adf['Year'] == 2025]
    if not _a25.empty:
        ACTUAL_2025 = dict(zip(_a25['Party'], _a25['VoteShare']))

# ── Collect 2025 predictions from each model ─────────────────────────────

def load_economy_nlp_2025():
    """Factor-based: Economy + X/NLP predictions for 2025."""
    path = OUT_DIR / 'factor_based_2025_predictions.csv'
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df['Year'] == 2025]
    results = {}
    for factor in df['Factor'].unique():
        sub = df[df['Factor'] == factor][['Party', 'PredictedVoteShare']]
        results[factor] = dict(zip(sub['Party'], sub['PredictedVoteShare']))
    return results

def load_education_2025():
    """Education attainment: best-fit education level prediction."""
    fit_path = OUT_DIR / 'education_attainment_fit_summary_2025.csv'
    pred_path = OUT_DIR / 'education_attainment_predictions_2025.csv'
    if not fit_path.exists() or not pred_path.exists():
        return None, None
    fit = pd.read_csv(fit_path)
    best_factor = fit.sort_values('FitMAE_2025').iloc[0]['Factor']
    pred = pd.read_csv(pred_path)
    sub = pred[(pred['Year'] == 2025) & (pred['Factor'] == best_factor)][['Party', 'PredictedVoteShare']]
    return best_factor, dict(zip(sub['Party'], sub['PredictedVoteShare']))

def load_voter_generation_2025():
    """Voter generation preference model: aggregate age bands to national."""
    model_path = OUT_DIR / 'voter_generation_party_preference_model_2011_2025.csv'
    weight_path = OUT_DIR / 'voter_generation_age_weight_history_2011_2025.csv'
    if not model_path.exists() or not weight_path.exists():
        return None
    model = pd.read_csv(model_path)
    weights = pd.read_csv(weight_path)
    d25 = model[model['Year'] == 2025]
    w25 = weights[weights['Year'] == 2025][['AgeBand', 'Weight']]
    if d25.empty or w25.empty:
        return None
    merged = d25.merge(w25, on='AgeBand')
    national = merged.groupby('Party').apply(
        lambda g: (g['SupportPct'] * g['Weight']).sum(), include_groups=False
    ).reset_index()
    national.columns = ['Party', 'PredictedVoteShare']
    return dict(zip(national['Party'], national['PredictedVoteShare']))

def load_historical_bayesian_2025():
    """Bayesian hierarchical random walk prediction."""
    path = ROOT / 'outputs' / 'prediction_2025_vs_actual.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return dict(zip(df['Party'], df['PredictedVoteShare']))

# ── Collect 2029 projections ───────────────────────────────────────────────

def load_economy_nlp_2029():
    """Factor-based: Economy + X/NLP projections for 2029."""
    path = OUT_DIR / 'factor_next_election_predictions.csv'
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    results = {}
    for scenario in df['Scenario'].unique():
        if 'Average' in scenario:
            continue
        sub = df[df['Scenario'] == scenario][['Party', 'PredictedVoteShare']]
        results[scenario] = dict(zip(sub['Party'], sub['PredictedVoteShare']))
    return results

def load_x_nlp_2029():
    """X/NLP standalone 2029 projection (replicate ridge regression)."""
    x_path = OUT_DIR / 'x_nlp_party_summary_2021_2025.csv'
    votes_path = ROOT / 'outputs' / 'national_vote_share_clean.csv'
    if not x_path.exists() or not votes_path.exists():
        return None
    x_df = pd.read_csv(x_path)
    votes = pd.read_csv(votes_path)
    x_df = x_df[x_df['Party'].isin(MAIN_PARTIES)].copy()
    for c in ['SupportIndex', 'XShare']:
        x_df[c] = pd.to_numeric(x_df[c], errors='coerce')
    actual_2021 = votes[(votes['Year'] == 2021) & (votes['Party'].isin(MAIN_PARTIES))]
    train = x_df[x_df['Year'] == 2021][['Party', 'SupportIndex', 'XShare']].merge(
        actual_2021[['Party', 'VoteShare']], on='Party')
    if len(train) < len(MAIN_PARTIES):
        return None
    train['VoteShareMain'] = train['VoteShare'] / train['VoteShare'].sum()
    X = np.column_stack([np.ones(len(train)), train['SupportIndex'].values,
                         np.log(train['XShare'].clip(lower=1e-6).values)])
    y = np.log(train['VoteShareMain'].clip(lower=1e-6).values)
    penalty = np.eye(3); penalty[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + 0.2 * penalty, X.T @ y)
    others_base = float(votes[(votes['Year'] == 2021) & (votes['Party'] == 'Others')]['VoteShare'].iloc[0])

    frame_2029 = x_df[x_df['Year'] == 2029][['Party', 'SupportIndex', 'XShare']]
    frame_2029 = frame_2029.set_index('Party').reindex(MAIN_PARTIES).reset_index()
    if frame_2029[['SupportIndex', 'XShare']].isna().any().any():
        return None
    Z = np.column_stack([np.ones(len(frame_2029)), frame_2029['SupportIndex'].values,
                         np.log(frame_2029['XShare'].clip(lower=1e-6).values)])
    logits = Z @ beta
    ms = np.exp(logits - logits.max())
    ms /= ms.sum()
    pred = ms * (100 - others_base)
    result = dict(zip(MAIN_PARTIES, pred))
    result['Others'] = others_base
    return result

def load_education_2029():
    """Education attainment: best-fit education level 2029 projection."""
    fit_path = OUT_DIR / 'education_attainment_fit_summary_2025.csv'
    pred_path = OUT_DIR / 'education_attainment_predictions_2029.csv'
    if not fit_path.exists() or not pred_path.exists():
        return None, None
    fit = pd.read_csv(fit_path)
    best_factor = fit.sort_values('FitMAE_2025').iloc[0]['Factor']
    pred = pd.read_csv(pred_path)
    sub = pred[(pred['Year'] == 2029) & (pred['Factor'] == best_factor)][['Party', 'PredictedVoteShare']]
    if sub.empty:
        return None, None
    return best_factor, dict(zip(sub['Party'], sub['PredictedVoteShare']))

def load_voter_generation_2029():
    """Voter generation preference model: aggregate 2029 age bands to national."""
    model_path = OUT_DIR / 'voter_generation_party_preference_model_2011_2025.csv'
    weight_path = OUT_DIR / 'voter_generation_age_weight_history_2011_2025.csv'
    if not model_path.exists() or not weight_path.exists():
        return None
    model = pd.read_csv(model_path)
    weights = pd.read_csv(weight_path)
    d29 = model[model['Year'] == 2029]
    w29 = weights[weights['Year'] == 2029][['AgeBand', 'Weight']]
    if d29.empty or w29.empty:
        return None
    merged = d29.merge(w29, on='AgeBand')
    national = merged.groupby('Party').apply(
        lambda g: (g['SupportPct'] * g['Weight']).sum(), include_groups=False
    ).reset_index()
    national.columns = ['Party', 'PredictedVoteShare']
    return dict(zip(national['Party'], national['PredictedVoteShare']))

# ── Compute MAE for each model ────────────────────────────────────────────

def compute_mae(pred_dict):
    errors = []
    for party in PARTIES:
        if party in pred_dict and party in ACTUAL_2025:
            errors.append(abs(pred_dict[party] - ACTUAL_2025[party]))
    return np.mean(errors) if errors else 999.0

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Collect all 2025 predictions
    models_2025 = {}

    econ_nlp = load_economy_nlp_2025()
    for name, pred in econ_nlp.items():
        models_2025[name] = pred

    edu_name, edu_pred = load_education_2025()
    if edu_pred:
        models_2025[f'Education ({edu_name.replace("Education: ", "")})'] = edu_pred

    gen_pred = load_voter_generation_2025()
    if gen_pred:
        models_2025['Age/Generation (IPF)'] = gen_pred

    hist_pred = load_historical_bayesian_2025()
    if hist_pred:
        # Normalize People's into Others if present
        if "People's" in hist_pred:
            hist_pred['Others'] = hist_pred.get('Others', 0) + hist_pred.pop("People's")
        models_2025['Historical (Bayesian RW)'] = hist_pred

    if not models_2025:
        print("No model predictions found. Run prerequisite scripts first.")
        return

    # Compute MAE and inverse-MAE weights
    mae_dict = {name: compute_mae(pred) for name, pred in models_2025.items()}
    inv_mae = {name: 1.0 / mae for name, mae in mae_dict.items()}
    total_inv = sum(inv_mae.values())
    weights = {name: v / total_inv for name, v in inv_mae.items()}

    # ── Print formulas ────────────────────────────────────────────────────
    print("=" * 70)
    print("MULTI-FACTOR ENSEMBLE: 5-Model Weighted Prediction")
    print("=" * 70)
    print()
    print("Individual model formulas:")
    print("-" * 70)
    print("1. Economy:    VoteShare = Intercept(p) + Slope(p) × z(GDP_composite)")
    print("               → population-weighted across AB/BC/ON/QC (86.5%)")
    print("2. NLP:        log(share) = β₀ + β₁·SupportIndex + β₂·log(XShare)")
    print("               → Ridge λ=0.2, fitted on 2021 actuals")
    print("3. Education:  VoteShare = Intercept(p) + Slope(p) × z(ShortCycleTertiary%)")
    print("               → population-weighted across AB/BC/ON/QC")
    print("4. Generation: IPF(age_effect × national_target), 3000 iterations")
    print("               → population-weighted across age bands")
    print("5. Historical: Bayesian hierarchical random walk (Gibbs, 2000 draws)")
    print("               → drift + innovation from 1867-2021 series")
    print("-" * 70)
    print()
    print("Ensemble weighting formula:")
    print("  weight(i) = (1 / MAE_i) / Σ(1 / MAE_j)")
    print("  Ensemble(party) = Σ weight(i) × prediction_i(party)")
    print()

    # Display weights table
    print(f"{'Model':<40} {'MAE':>8} {'Weight':>8}")
    print("-" * 58)
    for name in sorted(weights, key=lambda n: -weights[n]):
        print(f"{name:<40} {mae_dict[name]:>8.2f} {weights[name]:>8.1%}")
    print()

    # Compute ensemble 2025
    ensemble_2025 = {}
    for party in PARTIES:
        val = 0.0
        for name, w in weights.items():
            pred = models_2025[name].get(party, 0.0)
            val += w * pred
        ensemble_2025[party] = val
    # Normalize
    total = sum(ensemble_2025.values())
    if total > 0:
        ensemble_2025 = {p: v / total * 100.0 for p, v in ensemble_2025.items()}

    # Build results DataFrame
    rows = []
    for party in PARTIES:
        row = {'Party': party, 'Actual_2025': ACTUAL_2025.get(party, 0.0),
               'Ensemble_Predicted': ensemble_2025.get(party, 0.0)}
        for name in models_2025:
            row[name] = models_2025[name].get(party, 0.0)
        rows.append(row)
    results_df = pd.DataFrame(rows)
    results_df['Error'] = results_df['Ensemble_Predicted'] - results_df['Actual_2025']
    results_df['AbsError'] = results_df['Error'].abs()

    ensemble_mae = results_df['AbsError'].mean()
    ensemble_rmse = np.sqrt((results_df['Error'] ** 2).mean())

    print("2025 Ensemble Prediction vs Actual:")
    print(results_df[['Party', 'Ensemble_Predicted', 'Actual_2025', 'Error']].round(2).to_string(index=False))
    print(f"\nEnsemble MAE:  {ensemble_mae:.2f}")
    print(f"Ensemble RMSE: {ensemble_rmse:.2f}")

    # Compare vs individual MAEs
    print(f"\n{'Model':<40} {'MAE':>8}")
    print("-" * 50)
    print(f"{'>>> ENSEMBLE <<<':<40} {ensemble_mae:>8.2f}")
    for name in sorted(mae_dict, key=lambda n: mae_dict[n]):
        print(f"{name:<40} {mae_dict[name]:>8.2f}")

    results_df.to_csv(OUT_DIR / 'ensemble_2025_predictions.csv', index=False)
    print(f"\nSaved: {OUT_DIR / 'ensemble_2025_predictions.csv'}")

    # ── Plot 2025: Ensemble vs Actual ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(PARTIES))
    w_bar = 0.12
    actual_vals = [ACTUAL_2025.get(p, 0) for p in PARTIES]

    # Plot each model as a thin bar
    model_names = list(models_2025.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    for i, name in enumerate(model_names):
        vals = [models_2025[name].get(p, 0) for p in PARTIES]
        bars = ax.bar(x + i * w_bar, vals, w_bar * 0.9, label=f'{name} (w={weights[name]:.1%})',
               color=colors[i], alpha=0.6)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=5, rotation=90)

    # Ensemble as thick dark bar
    ens_vals = [ensemble_2025.get(p, 0) for p in PARTIES]
    bars_ens = ax.bar(x + len(model_names) * w_bar, ens_vals, w_bar * 0.9,
           label=f'ENSEMBLE (MAE={ensemble_mae:.2f})', color='black', alpha=0.85)
    for bar, v in zip(bars_ens, ens_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=5.5, fontweight='bold', rotation=90)

    # Actual as markers
    ax.scatter(x + (len(model_names) / 2) * w_bar, actual_vals, color='red',
               s=120, zorder=5, marker='D', label='Actual 2025')

    ax.set_xticks(x + (len(model_names) / 2) * w_bar)
    ax.set_xticklabels(PARTIES, rotation=30, ha='right')
    ax.set_ylabel('Vote Share (%)')
    ax.set_title('Multi-Factor Ensemble: 2025 Predictions vs Actual\n'
                 'weight(i) = (1/MAE_i) / Σ(1/MAE_j)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    png_2025 = OUT_DIR / 'ensemble_2025_vs_actual.png'
    fig.savefig(png_2025, dpi=200)
    plt.close(fig)
    print(f"Saved: {png_2025}")

    # ── 2029 Projections ──────────────────────────────────────────────────
    models_2029 = {}

    econ_nlp_2029 = load_economy_nlp_2029()
    for name, pred in econ_nlp_2029.items():
        models_2029[name] = pred

    # Education 2029
    edu_factor_2029, edu_pred_2029 = load_education_2029()
    if edu_pred_2029:
        # Strip "Education: " prefix from factor name to match 2025 model name
        short_name = edu_factor_2029.replace('Education: ', '')
        models_2029[f'Education ({short_name})'] = edu_pred_2029

    # Voter generation 2029
    gen_pred_2029 = load_voter_generation_2029()
    if gen_pred_2029:
        models_2029['Age/Generation (IPF)'] = gen_pred_2029

    if models_2029:
        # Use 2025 MAE-based weights (only for models that have 2029 projections)
        available_2029 = {n: weights.get(n, 0) for n in models_2029}
        # Re-weight to models that also had 2025 predictions
        for name in list(available_2029.keys()):
            if name not in weights:
                # Use similar model's weight
                if 'NLP' in name or 'X/NLP' in name:
                    available_2029[name] = weights.get('Political leaning (X/NLP)', 0.1)
        total_w = sum(available_2029.values())
        if total_w > 0:
            weights_2029 = {n: v / total_w for n, v in available_2029.items()}
        else:
            weights_2029 = {n: 1.0 / len(available_2029) for n in available_2029}

        ensemble_2029 = {}
        for party in PARTIES:
            val = sum(weights_2029[n] * models_2029[n].get(party, 0) for n in models_2029)
            ensemble_2029[party] = val
        total = sum(ensemble_2029.values())
        if total > 0:
            ensemble_2029 = {p: v / total * 100.0 for p, v in ensemble_2029.items()}

        rows_2029 = []
        for party in PARTIES:
            row = {'Party': party, 'Ensemble_2029': ensemble_2029.get(party, 0)}
            for name in models_2029:
                row[name] = models_2029[name].get(party, 0)
            rows_2029.append(row)
        df_2029 = pd.DataFrame(rows_2029)
        df_2029.to_csv(OUT_DIR / 'ensemble_2029_projections.csv', index=False)
        print(f"\nSaved: {OUT_DIR / 'ensemble_2029_projections.csv'}")

        print("\n2029 Ensemble Projection:")
        print(f"{'Model':<40} {'Weight':>8}")
        print("-" * 50)
        for n in sorted(weights_2029, key=lambda k: -weights_2029[k]):
            print(f"{n:<40} {weights_2029[n]:>8.1%}")
        print()
        print(df_2029[['Party', 'Ensemble_2029']].round(2).to_string(index=False))

        # Plot 2029
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        x2 = np.arange(len(PARTIES))
        w2 = 0.15
        colors_2029 = plt.cm.Set2(np.linspace(0, 1, len(models_2029)))
        for i, name in enumerate(models_2029):
            vals = [models_2029[name].get(p, 0) for p in PARTIES]
            bars = ax2.bar(x2 + i * w2, vals, w2 * 0.9, label=f'{name} (w={weights_2029[name]:.1%})',
                    color=colors_2029[i], alpha=0.7)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                             f'{v:.1f}', ha='center', va='bottom', fontsize=5.5, rotation=90)
        ens_2029_vals = [ensemble_2029.get(p, 0) for p in PARTIES]
        bars_ens = ax2.bar(x2 + len(models_2029) * w2, ens_2029_vals, w2 * 0.9,
                label='ENSEMBLE', color='black', alpha=0.85)
        for bar, v in zip(bars_ens, ens_2029_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=6, fontweight='bold', rotation=90)
        # Add 2025 actual as reference
        ax2.scatter(x2 + (len(models_2029) / 2) * w2, actual_vals, color='red',
                    s=100, zorder=5, marker='D', label='Actual 2025 (reference)', alpha=0.5)

        ax2.set_xticks(x2 + (len(models_2029) / 2) * w2)
        ax2.set_xticklabels(PARTIES, rotation=30, ha='right')
        ax2.set_ylabel('Vote Share (%)')
        ax2.set_title('Multi-Factor Ensemble: 2029 Projection\n'
                       'Weighted by 2025 inverse-MAE (models with 2029 data only)')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        fig2.tight_layout()
        png_2029 = OUT_DIR / 'ensemble_2029_projections.png'
        fig2.savefig(png_2029, dpi=200)
        plt.close(fig2)
        print(f"Saved: {png_2029}")
    else:
        print("\nNo 2029 projections available.")

if __name__ == '__main__':
    main()
