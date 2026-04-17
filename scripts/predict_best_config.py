"""
Predict Best Config vs SA Baseline
====================================
Tests whether the OLS prediction model can identify configurations that
actually outperform the best single-agent model on each benchmark.

Search: brute-force over all 5 architectures × 9 models = 45 combinations.
        Each combo maps to one feature vector → one OLS prediction → argmax.

Workflow per benchmark:
  1. Fit OLS model on all 4 benchmarks
  2. Build feature vectors for all 45 (architecture × model) combos
  3. Predict performance for each via fitted OLS coefficients
  4. Select predicted-best config; look up its ACTUAL performance
  5. Compare against best actual SA model performance
  6. Report gap — if small, prediction model adds no value

Run:
  python scripts/predict_best_config.py
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Import shared constants and helpers from the existing analysis script
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from extended_mixed_effects_6benchmarks import (  # noqa: E402
    ORIGINAL_DATA,
    COORDINATION_METRICS,
    INTELLIGENCE_SCORES,
    TOOL_COUNTS,
    AGENT_COUNTS,
    FORMULA,
    FEATURE_COLS,
    build_dataframe,
    standardize,
    normalize_model_name,
)

ORIGINAL_MODELS = [
    'claude-3-7-sonnet-20250219',
    'claude-sonnet-4-20250514',
    'claude-sonnet-4-5',
    'gemini-2.0-flash',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gpt-5',
    'gpt-5-mini',
    'gpt-5-nano',
]
ARCHITECTURES = list(COORDINATION_METRICS.keys())
BENCHMARKS_4  = ['browsecomp_plus_sampled_100', 'finance-agent', 'plancraft-test', 'workbench']


# ===========================================================================
# Feature builder  (brute-force: one row per combo)
# ===========================================================================

def build_combo_features(benchmark: str, df_raw: pd.DataFrame,
                         intel_mean: float) -> pd.DataFrame:
    """
    Build one raw (un-standardized) feature row for every (arch × model) combo
    on the given benchmark — 45 rows total.

    single_agent_baseline = actual SA performance of that model on this
    benchmark (same methodology as the original regression).
    """
    n_tools   = TOOL_COUNTS[benchmark]
    log_tools = np.log1p(n_tools)

    sa_rows = (
        df_raw[(df_raw['dataset'] == benchmark) & (df_raw['architecture'] == 'single-agent')]
        .set_index('model_short')['performance']
    )
    actual_perf = (
        df_raw[df_raw['dataset'] == benchmark]
        .set_index(['architecture', 'model_short'])['performance']
    )

    rows = []
    for arch in ARCHITECTURES:
        coord    = COORDINATION_METRICS[arch]
        n_agents = AGENT_COUNTS[arch]
        log_agents   = np.log1p(n_agents)
        log_overhead = np.log1p(coord['overhead_pct'])
        log_error_amp = np.log1p(coord['error_amplification'])

        for model_name in ORIGINAL_MODELS:
            model_short  = normalize_model_name(model_name)
            intelligence = INTELLIGENCE_SCORES.get(model_short)
            if intelligence is None:
                continue

            sa_baseline = sa_rows.get(model_short, np.nan)
            if np.isnan(sa_baseline):
                continue

            actual = actual_perf.get((arch, model_short), np.nan)
            intel_c  = intelligence - intel_mean
            intel_sq = intel_c ** 2

            rows.append({
                # identifiers (not used in regression)
                'architecture': arch,
                'model':        model_name,
                'model_short':  model_short,
                'performance':  actual,          # actual result (for eval only)
                # raw FEATURE_COLS
                'intelligence_centered':    intel_c,
                'intelligence_sq_centered': intel_sq,
                'n_tools':                  n_tools,
                'log_tools':                log_tools,
                'n_agents':                 n_agents,
                'log_agents':               log_agents,
                'overhead_pct':             coord['overhead_pct'],
                'log_overhead':             log_overhead,
                'message_density':          coord['message_density'],
                'redundancy':               coord['redundancy'],
                'efficiency':               coord['efficiency'],
                'error_amplification':      coord['error_amplification'],
                'log_error_amp':            log_error_amp,
                'success_per_1k':           coord['success_per_1k_tokens'],
                'single_agent_baseline':    sa_baseline,
                # interaction terms
                'intel_x_efficiency':   intel_c * coord['efficiency'],
                'error_x_baseline':     coord['error_amplification'] * sa_baseline,
                'overhead_x_tools':     coord['overhead_pct'] * n_tools,
                'redundancy_x_agents':  coord['redundancy'] * n_agents,
                'msg_density_x_intel':  coord['message_density'] * intel_c,
                'efficiency_x_tools':   coord['efficiency'] * n_tools,
                'baseline_x_agents':    sa_baseline * log_agents,
                'intel_x_tools':        intel_c * log_tools,
                'error_x_tools':        coord['error_amplification'] * n_tools,
            })

    return pd.DataFrame(rows)


def predict_combos(combo_raw: pd.DataFrame, scaler: StandardScaler,
                   fitted_model) -> pd.DataFrame:
    """Standardize features with the given scaler, then predict."""
    combo_scaled = combo_raw.copy()
    combo_scaled[FEATURE_COLS] = scaler.transform(combo_raw[FEATURE_COLS])
    out = combo_raw.copy()
    out['predicted'] = fitted_model.predict(combo_scaled)
    return out


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_benchmark(benchmark: str, combo_df: pd.DataFrame) -> dict:
    """
    Given all 45 combos with 'predicted' and 'performance' (actual), compute:
      - predicted-best config  → actual performance
      - oracle best            → actual performance (ceiling)
      - best SA model          → actual performance (baseline)
      - gap: predicted-best actual − best SA actual
    """
    valid = combo_df.dropna(subset=['performance', 'predicted'])

    best_pred   = valid.loc[valid['predicted'].idxmax()]
    best_oracle = valid.loc[valid['performance'].idxmax()]
    sa_valid    = valid[valid['architecture'] == 'single-agent']
    best_sa          = sa_valid.loc[sa_valid['performance'].idxmax()]
    best_sa_predicted = sa_valid.loc[sa_valid['predicted'].idxmax()]

    return {
        'benchmark':       benchmark,
        # prediction model's pick (best overall)
        'pred_arch':       best_pred['architecture'],
        'pred_model':      best_pred['model_short'],
        'pred_predicted':  best_pred['predicted'],
        'pred_actual':     best_pred['performance'],
        # prediction model's pick (best SA only)
        'pred_sa_model':      best_sa_predicted['model_short'],
        'pred_sa_predicted':  best_sa_predicted['predicted'],
        'pred_sa_actual':     best_sa_predicted['performance'],
        # oracle best overall
        'oracle_arch':     best_oracle['architecture'],
        'oracle_model':    best_oracle['model_short'],
        'oracle_actual':   best_oracle['performance'],
        # oracle best SA
        'sa_model':        best_sa['model_short'],
        'sa_actual':       best_sa['performance'],
        # key metrics
        'gap_actual':      best_pred['performance']   - best_sa['performance'],
        'gap_oracle':      best_oracle['performance'] - best_sa['performance'],
        'pred_error':      best_pred['predicted']     - best_pred['performance'],
        # keep for top-K printout
        '_combo_df':       valid,
    }


# ===========================================================================
# Printing
# ===========================================================================

def sep(char='=', width=92):
    print(char * width)


def verdict(gap: float) -> str:
    if gap > 0.02:
        return 'BEATS SA'
    if gap >= -0.02:
        return 'marginal'
    return 'WORSE than SA'


def print_summary_table(results: list):
    hdr = f"  {'type':<18}  {'model':<25}  {'architecture':<28}  {'performance':>11}"
    for r in results:
        print()
        sep('-')
        print(f"  {r['benchmark']}")
        sep('-')
        print(hdr)
        sep('-')

        # Predicted section
        print(f"  {'pred (fitted)':<18}  {r['pred_model']:<25}  {r['pred_arch']:<28}  {r['pred_predicted']:>11.3f}")
        print(f"  {'pred (actual)':<18}  {r['pred_model']:<25}  {r['pred_arch']:<28}  {r['pred_actual']:>11.3f}")
        g1 = r['pred_actual'] - r['pred_predicted']
        print(f"    overestimation error (pred fitted − pred actual): {g1:>+.3f}")
        print()
        print(f"  {'pred SA (fitted)':<18}  {r['pred_sa_model']:<25}  {'single-agent':<28}  {r['pred_sa_predicted']:>11.3f}")
        print(f"  {'pred SA (actual)':<18}  {r['pred_sa_model']:<25}  {'single-agent':<28}  {r['pred_sa_actual']:>11.3f}")
        g2 = r['pred_sa_actual'] - r['pred_sa_predicted']
        print(f"    overestimation error (pred SA fitted − pred SA actual): {g2:>+.3f}")

        sep('-')

        # Actual section
        print(f"  {'actual best':<18}  {r['oracle_model']:<25}  {r['oracle_arch']:<28}  {r['oracle_actual']:>11.3f}")
        print(f"  {'actual SA':<18}  {r['sa_model']:<25}  {'single-agent':<28}  {r['sa_actual']:>11.3f}")
        g3 = r['oracle_actual'] - r['sa_actual']
        print(f"    ceiling (actual best − actual SA, max gain possible from config choice): {g3:>+.3f}")

        sep('-')


# ===========================================================================
# In-sample evaluation
# ===========================================================================

def run_insample(df_raw: pd.DataFrame, scaler: StandardScaler,
                 fitted_model, intel_mean: float) -> list:

    results = []
    for bm in BENCHMARKS_4:
        combo_raw = build_combo_features(bm, df_raw, intel_mean)
        combo_df  = predict_combos(combo_raw, scaler, fitted_model)
        r = evaluate_benchmark(bm, combo_df)
        results.append(r)

    print_summary_table(results)
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    df_raw, intel_mean = build_dataframe(ORIGINAL_DATA)
    df_scaled, scaler  = standardize(df_raw, FEATURE_COLS)
    assert len(df_raw) == 180, f"Expected 180 rows, got {len(df_raw)}"
    fitted_model = smf.ols(FORMULA, data=df_scaled).fit()
    run_insample(df_raw, scaler, fitted_model, intel_mean)


if __name__ == "__main__":
    main()
