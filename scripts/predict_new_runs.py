"""
Quick Prediction Test: Mixed-Effects Model on New Runs
======================================================
Trains the 6-benchmark mixed-effects model (Eq. 1 from the paper),
then predicts performance for user-supplied new configurations.

Usage:
  # Edit NEW_RUNS below, then:
  python scripts/predict_new_runs.py

  # Or pass a CSV with columns: dataset,architecture,provider,model,actual_performance
  python scripts/predict_new_runs.py --csv path/to/new_runs.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# EDIT HERE: your new runs to predict
# Format: [dataset, architecture, provider, model, actual_performance]
# Set actual_performance to None if you only want the prediction
# ==============================================================================
NEW_RUNS = [
    # Examples for plancraft-test with a new model:
    ['plancraft-test', 'single-agent',              'anthropic', 'minimax-m2.7', 0.3695652173913043],
    ['plancraft-test', 'multi-agent-centralized',   'anthropic', 'minimax-m2.7', 0.34],
    ['plancraft-test', 'multi-agent-decentralized', 'anthropic', 'minimax-m2.7', 0.43],
    ['plancraft-test', 'multi-agent-hybrid',        'anthropic', 'minimax-m2.7', 0.40],
    ['plancraft-test', 'multi-agent-independent',   'anthropic', 'minimax-m2.7', 0.14],
    # ['workbench', 'single-agent',              'anthropic', 'minimax-m2.7', 0.26],
    # ['workbench', 'multi-agent-centralized',   'anthropic', 'minimax-m2.7', 0.20],
    # ['workbench', 'multi-agent-decentralized', 'anthropic', 'minimax-m2.7', 0.30],
    # ['workbench', 'multi-agent-hybrid',        'anthropic', 'minimax-m2.7', 0.22],
    # ['workbench', 'multi-agent-independent',   'anthropic', 'minimax-m2.7', 0.15],
]

# ==============================================================================
# TRAINING DATA — 180 original + 90 new = 270 (6 benchmarks)
# Copied from extended_mixed_effects_6benchmarks.py
# ==============================================================================
ORIGINAL_DATA = [
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'anthropic', 'claude-3-7-sonnet-20250219', 0.3434343434343434],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-20250514',   0.32323232323232326],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-5',          0.42857142857142855],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'gemini',    'gemini-2.0-flash',           0.22],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'gemini',    'gemini-2.5-flash',           0.31],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'gemini',    'gemini-2.5-pro',             0.37],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'openai',    'gpt-5',                      0.34],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'openai',    'gpt-5-mini',                 0.26],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'openai',    'gpt-5-nano',                 0.27],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'anthropic', 'claude-3-7-sonnet-20250219', 0.29292929292929293],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-20250514',   0.37373737373737376],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-5',          0.43434343434343436],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'gemini',    'gemini-2.0-flash',           0.18],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-flash',           0.26],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-pro',             0.43],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'openai',    'gpt-5',                      0.5],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'openai',    'gpt-5-mini',                 0.33],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'openai',    'gpt-5-nano',                 0.32],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'anthropic', 'claude-3-7-sonnet-20250219', 0.3333333333333333],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-20250514',   0.41414141414141414],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-5',          0.40404040404040403],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'gemini',    'gemini-2.0-flash',           0.2],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-flash',           0.32],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-pro',             0.4],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'openai',    'gpt-5',                      0.38],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'openai',    'gpt-5-mini',                 0.24],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'openai',    'gpt-5-nano',                 0.33],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'anthropic', 'claude-3-7-sonnet-20250219', 0.18181818181818182],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-20250514',   0.1111111111111111],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-5',          0.1414141414141414],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'gemini',    'gemini-2.0-flash',           0.1],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'gemini',    'gemini-2.5-flash',           0.21],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'gemini',    'gemini-2.5-pro',             0.24],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'openai',    'gpt-5',                      0.44],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'openai',    'gpt-5-mini',                 0.25],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'openai',    'gpt-5-nano',                 0.18],
    ['browsecomp_plus_sampled_100', 'single-agent',              'anthropic', 'claude-3-7-sonnet-20250219', 0.26262626262626265],
    ['browsecomp_plus_sampled_100', 'single-agent',              'anthropic', 'claude-sonnet-4-20250514',   0.29292929292929293],
    ['browsecomp_plus_sampled_100', 'single-agent',              'anthropic', 'claude-sonnet-4-5',          0.3434343434343434],
    ['browsecomp_plus_sampled_100', 'single-agent',              'gemini',    'gemini-2.0-flash',           0.17],
    ['browsecomp_plus_sampled_100', 'single-agent',              'gemini',    'gemini-2.5-flash',           0.28],
    ['browsecomp_plus_sampled_100', 'single-agent',              'gemini',    'gemini-2.5-pro',             0.36],
    ['browsecomp_plus_sampled_100', 'single-agent',              'openai',    'gpt-5',                      0.37],
    ['browsecomp_plus_sampled_100', 'single-agent',              'openai',    'gpt-5-mini',                 0.41],
    ['browsecomp_plus_sampled_100', 'single-agent',              'openai',    'gpt-5-nano',                 0.37],
    ['finance-agent', 'multi-agent-centralized',   'anthropic', 'claude-3-7-sonnet-20250219', 0.3],
    ['finance-agent', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-20250514',   0.42],
    ['finance-agent', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-5',          0.46],
    ['finance-agent', 'multi-agent-centralized',   'gemini',    'gemini-2.0-flash',           0.7],
    ['finance-agent', 'multi-agent-centralized',   'gemini',    'gemini-2.5-flash',           0.74],
    ['finance-agent', 'multi-agent-centralized',   'gemini',    'gemini-2.5-pro',             0.78],
    ['finance-agent', 'multi-agent-centralized',   'openai',    'gpt-5',                      0.8],
    ['finance-agent', 'multi-agent-centralized',   'openai',    'gpt-5-mini',                 0.72],
    ['finance-agent', 'multi-agent-centralized',   'openai',    'gpt-5-nano',                 0.76],
    ['finance-agent', 'multi-agent-decentralized', 'anthropic', 'claude-3-7-sonnet-20250219', 0.2],
    ['finance-agent', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-20250514',   0.32],
    ['finance-agent', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-5',          0.44],
    ['finance-agent', 'multi-agent-decentralized', 'gemini',    'gemini-2.0-flash',           0.72],
    ['finance-agent', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-flash',           0.74],
    ['finance-agent', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-pro',             0.76],
    ['finance-agent', 'multi-agent-decentralized', 'openai',    'gpt-5',                      0.78],
    ['finance-agent', 'multi-agent-decentralized', 'openai',    'gpt-5-mini',                 0.76],
    ['finance-agent', 'multi-agent-decentralized', 'openai',    'gpt-5-nano',                 0.76],
    ['finance-agent', 'multi-agent-hybrid',        'anthropic', 'claude-3-7-sonnet-20250219', 0.24],
    ['finance-agent', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-20250514',   0.38],
    ['finance-agent', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-5',          0.46],
    ['finance-agent', 'multi-agent-hybrid',        'gemini',    'gemini-2.0-flash',           0.68],
    ['finance-agent', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-flash',           0.74],
    ['finance-agent', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-pro',             0.76],
    ['finance-agent', 'multi-agent-hybrid',        'openai',    'gpt-5',                      0.78],
    ['finance-agent', 'multi-agent-hybrid',        'openai',    'gpt-5-mini',                 0.66],
    ['finance-agent', 'multi-agent-hybrid',        'openai',    'gpt-5-nano',                 0.74],
    ['finance-agent', 'multi-agent-independent',   'anthropic', 'claude-3-7-sonnet-20250219', 0.12],
    ['finance-agent', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-20250514',   0.16],
    ['finance-agent', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-5',          0.28],
    ['finance-agent', 'multi-agent-independent',   'gemini',    'gemini-2.0-flash',           0.62],
    ['finance-agent', 'multi-agent-independent',   'gemini',    'gemini-2.5-flash',           0.68],
    ['finance-agent', 'multi-agent-independent',   'gemini',    'gemini-2.5-pro',             0.76],
    ['finance-agent', 'multi-agent-independent',   'openai',    'gpt-5',                      0.76],
    ['finance-agent', 'multi-agent-independent',   'openai',    'gpt-5-mini',                 0.78],
    ['finance-agent', 'multi-agent-independent',   'openai',    'gpt-5-nano',                 0.76],
    ['finance-agent', 'single-agent',              'anthropic', 'claude-3-7-sonnet-20250219', 0.3],
    ['finance-agent', 'single-agent',              'anthropic', 'claude-sonnet-4-20250514',   0.32],
    ['finance-agent', 'single-agent',              'anthropic', 'claude-sonnet-4-5',          0.28],
    ['finance-agent', 'single-agent',              'gemini',    'gemini-2.0-flash',           0.1],
    ['finance-agent', 'single-agent',              'gemini',    'gemini-2.5-flash',           0.16],
    ['finance-agent', 'single-agent',              'gemini',    'gemini-2.5-pro',             0.58],
    ['finance-agent', 'single-agent',              'openai',    'gpt-5',                      0.62],
    ['finance-agent', 'single-agent',              'openai',    'gpt-5-mini',                 0.54],
    ['finance-agent', 'single-agent',              'openai',    'gpt-5-nano',                 0.24],
    ['plancraft-test', 'multi-agent-centralized',   'anthropic', 'claude-3-7-sonnet-20250219', 0.1919191919191919],
    ['plancraft-test', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-20250514',   0.1717171717171717],
    ['plancraft-test', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-5',          0.1919191919191919],
    ['plancraft-test', 'multi-agent-centralized',   'gemini',    'gemini-2.0-flash',           0.3],
    ['plancraft-test', 'multi-agent-centralized',   'gemini',    'gemini-2.5-flash',           0.38],
    ['plancraft-test', 'multi-agent-centralized',   'gemini',    'gemini-2.5-pro',             0.34],
    ['plancraft-test', 'multi-agent-centralized',   'openai',    'gpt-5',                      0.32],
    ['plancraft-test', 'multi-agent-centralized',   'openai',    'gpt-5-mini',                 0.35],
    ['plancraft-test', 'multi-agent-centralized',   'openai',    'gpt-5-nano',                 0.29],
    ['plancraft-test', 'multi-agent-decentralized', 'anthropic', 'claude-3-7-sonnet-20250219', 0.1111111111111111],
    ['plancraft-test', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-20250514',   0.20202020202020202],
    ['plancraft-test', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-5',          0.16161616161616163],
    ['plancraft-test', 'multi-agent-decentralized', 'gemini',    'gemini-2.0-flash',           0.44],
    ['plancraft-test', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-flash',           0.41],
    ['plancraft-test', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-pro',             0.38],
    ['plancraft-test', 'multi-agent-decentralized', 'openai',    'gpt-5',                      0.46],
    ['plancraft-test', 'multi-agent-decentralized', 'openai',    'gpt-5-mini',                 0.45],
    ['plancraft-test', 'multi-agent-decentralized', 'openai',    'gpt-5-nano',                 0.38],
    ['plancraft-test', 'multi-agent-hybrid',        'anthropic', 'claude-3-7-sonnet-20250219', 0.30303030303030304],
    ['plancraft-test', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-20250514',   0.2828282828282828],
    ['plancraft-test', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-5',          0.3434343434343434],
    ['plancraft-test', 'multi-agent-hybrid',        'gemini',    'gemini-2.0-flash',           0.32],
    ['plancraft-test', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-flash',           0.41],
    ['plancraft-test', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-pro',             0.42],
    ['plancraft-test', 'multi-agent-hybrid',        'openai',    'gpt-5',                      0.336],
    ['plancraft-test', 'multi-agent-hybrid',        'openai',    'gpt-5-mini',                 0.35],
    ['plancraft-test', 'multi-agent-hybrid',        'openai',    'gpt-5-nano',                 0.35],
    ['plancraft-test', 'multi-agent-independent',   'anthropic', 'claude-3-7-sonnet-20250219', 0.09090909090909091],
    ['plancraft-test', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-20250514',   0.09090909090909091],
    ['plancraft-test', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-5',          0.0707070707070707],
    ['plancraft-test', 'multi-agent-independent',   'gemini',    'gemini-2.0-flash',           0.15],
    ['plancraft-test', 'multi-agent-independent',   'gemini',    'gemini-2.5-flash',           0.19],
    ['plancraft-test', 'multi-agent-independent',   'gemini',    'gemini-2.5-pro',             0.14],
    ['plancraft-test', 'multi-agent-independent',   'openai',    'gpt-5',                      0.28],
    ['plancraft-test', 'multi-agent-independent',   'openai',    'gpt-5-mini',                 0.28],
    ['plancraft-test', 'multi-agent-independent',   'openai',    'gpt-5-nano',                 0.24],
    ['plancraft-test', 'single-agent',              'anthropic', 'claude-3-7-sonnet-20250219', 0.5959595959595959],
    ['plancraft-test', 'single-agent',              'anthropic', 'claude-sonnet-4-20250514',   0.6767676767676768],
    ['plancraft-test', 'single-agent',              'anthropic', 'claude-sonnet-4-5',          0.7676767676767676],
    ['plancraft-test', 'single-agent',              'gemini',    'gemini-2.0-flash',           0.52],
    ['plancraft-test', 'single-agent',              'gemini',    'gemini-2.5-flash',           0.51],
    ['plancraft-test', 'single-agent',              'gemini',    'gemini-2.5-pro',             0.51],
    ['plancraft-test', 'single-agent',              'openai',    'gpt-5',                      0.61],
    ['plancraft-test', 'single-agent',              'openai',    'gpt-5-mini',                 0.54],
    ['plancraft-test', 'single-agent',              'openai',    'gpt-5-nano',                 0.38],
    ['workbench', 'multi-agent-centralized',   'anthropic', 'claude-3-7-sonnet-20250219', 0.63],
    ['workbench', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-20250514',   0.68],
    ['workbench', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-5',          0.72],
    ['workbench', 'multi-agent-centralized',   'gemini',    'gemini-2.0-flash',           0.52],
    ['workbench', 'multi-agent-centralized',   'gemini',    'gemini-2.5-flash',           0.58],
    ['workbench', 'multi-agent-centralized',   'gemini',    'gemini-2.5-pro',             0.66],
    ['workbench', 'multi-agent-centralized',   'openai',    'gpt-5',                      0.64],
    ['workbench', 'multi-agent-centralized',   'openai',    'gpt-5-mini',                 0.6],
    ['workbench', 'multi-agent-centralized',   'openai',    'gpt-5-nano',                 0.56],
    ['workbench', 'multi-agent-decentralized', 'anthropic', 'claude-3-7-sonnet-20250219', 0.67],
    ['workbench', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-20250514',   0.72],
    ['workbench', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-5',          0.81],
    ['workbench', 'multi-agent-decentralized', 'gemini',    'gemini-2.0-flash',           0.52],
    ['workbench', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-flash',           0.58],
    ['workbench', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-pro',             0.69],
    ['workbench', 'multi-agent-decentralized', 'openai',    'gpt-5',                      0.76],
    ['workbench', 'multi-agent-decentralized', 'openai',    'gpt-5-mini',                 0.62],
    ['workbench', 'multi-agent-decentralized', 'openai',    'gpt-5-nano',                 0.61],
    ['workbench', 'multi-agent-hybrid',        'anthropic', 'claude-3-7-sonnet-20250219', 0.66],
    ['workbench', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-20250514',   0.71],
    ['workbench', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-5',          0.74],
    ['workbench', 'multi-agent-hybrid',        'gemini',    'gemini-2.0-flash',           0.55],
    ['workbench', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-flash',           0.63],
    ['workbench', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-pro',             0.66],
    ['workbench', 'multi-agent-hybrid',        'openai',    'gpt-5',                      0.6],
    ['workbench', 'multi-agent-hybrid',        'openai',    'gpt-5-mini',                 0.56],
    ['workbench', 'multi-agent-hybrid',        'openai',    'gpt-5-nano',                 0.48],
    ['workbench', 'multi-agent-independent',   'anthropic', 'claude-3-7-sonnet-20250219', 0.55],
    ['workbench', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-20250514',   0.65],
    ['workbench', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-5',          0.7],
    ['workbench', 'multi-agent-independent',   'gemini',    'gemini-2.0-flash',           0.53],
    ['workbench', 'multi-agent-independent',   'gemini',    'gemini-2.5-flash',           0.54],
    ['workbench', 'multi-agent-independent',   'gemini',    'gemini-2.5-pro',             0.56],
    ['workbench', 'multi-agent-independent',   'openai',    'gpt-5',                      0.59],
    ['workbench', 'multi-agent-independent',   'openai',    'gpt-5-mini',                 0.45],
    ['workbench', 'multi-agent-independent',   'openai',    'gpt-5-nano',                 0.44],
    ['workbench', 'single-agent',              'anthropic', 'claude-3-7-sonnet-20250219', 0.53],
    ['workbench', 'single-agent',              'anthropic', 'claude-sonnet-4-20250514',   0.64],
    ['workbench', 'single-agent',              'anthropic', 'claude-sonnet-4-5',          0.65],
    ['workbench', 'single-agent',              'gemini',    'gemini-2.0-flash',           0.55],
    ['workbench', 'single-agent',              'gemini',    'gemini-2.5-flash',           0.63],
    ['workbench', 'single-agent',              'gemini',    'gemini-2.5-pro',             0.64],
    ['workbench', 'single-agent',              'openai',    'gpt-5',                      0.7],
    ['workbench', 'single-agent',              'openai',    'gpt-5-mini',                 0.7],
    ['workbench', 'single-agent',              'openai',    'gpt-5-nano',                 0.62],
]

COORDINATION_METRICS = {
    'single-agent': {
        'overhead_pct': 0.0, 'message_density': 0.00, 'redundancy': 0.00,
        'efficiency': 0.466, 'error_amplification': 1.0, 'success_per_1k_tokens': 67.7,
    },
    'multi-agent-independent': {
        'overhead_pct': 58.0, 'message_density': 0.00, 'redundancy': 0.48,
        'efficiency': 0.234, 'error_amplification': 17.2, 'success_per_1k_tokens': 42.4,
    },
    'multi-agent-decentralized': {
        'overhead_pct': 263.0, 'message_density': 0.41, 'redundancy': 0.50,
        'efficiency': 0.132, 'error_amplification': 7.8, 'success_per_1k_tokens': 23.9,
    },
    'multi-agent-centralized': {
        'overhead_pct': 285.0, 'message_density': 0.39, 'redundancy': 0.41,
        'efficiency': 0.120, 'error_amplification': 4.4, 'success_per_1k_tokens': 21.5,
    },
    'multi-agent-hybrid': {
        'overhead_pct': 515.0, 'message_density': 0.24, 'redundancy': 0.46,
        'efficiency': 0.074, 'error_amplification': 5.1, 'success_per_1k_tokens': 13.6,
    },
}

INTELLIGENCE_SCORES = {
    "claude-3-7-sonnet": 35, "claude-sonnet-4": 39, "claude-sonnet-4-5": 52,
    "gemini-2.0-flash": 19, "gemini-2.5-flash": 27, "gemini-2.5-pro": 35,
    "gpt-5-nano": 27, "gpt-5-mini": 41, "gpt-5": 45,
    "gemini-3-flash-preview": 46,
    "minimax-m2.7": 50,
    # Add new models here with their Intelligence Index score:
    # "your-new-model": XX,
}

TOOL_COUNTS = {
    'browsecomp_plus_sampled_100': 3, 'workbench': 16,
    'plancraft-test': 4, 'finance-agent': 5,
    'swebench-verified': 7, 'terminalbench': 2,
}

AGENT_COUNTS = {
    'single-agent': 1, 'multi-agent-independent': 3,
    'multi-agent-centralized': 4, 'multi-agent-decentralized': 3,
    'multi-agent-hybrid': 4,
}

FORMULA = """performance ~ intelligence_centered + intelligence_sq_centered + log_tools + log_agents +
             log_overhead + message_density + redundancy + efficiency + log_error_amp +
             single_agent_baseline +
             intel_x_efficiency + error_x_baseline + overhead_x_tools +
             redundancy_x_agents + msg_density_x_intel + efficiency_x_tools +
             baseline_x_agents + intel_x_tools + error_x_tools"""

FEATURE_COLS = [
    'intelligence_centered', 'intelligence_sq_centered',
    'n_tools', 'log_tools', 'n_agents', 'log_agents',
    'overhead_pct', 'log_overhead',
    'message_density', 'redundancy', 'efficiency',
    'error_amplification', 'log_error_amp', 'success_per_1k',
    'single_agent_baseline',
    'intel_x_efficiency', 'error_x_baseline', 'overhead_x_tools',
    'redundancy_x_agents', 'msg_density_x_intel', 'efficiency_x_tools',
    'baseline_x_agents', 'intel_x_tools', 'error_x_tools',
]


def normalize_model_name(model_name: str) -> str:
    import re
    short = re.sub(r'-\d{8}$', '', model_name)
    mapping = {
        'claude-3-7-sonnet': 'claude-3-7-sonnet',
        'claude-sonnet-4': 'claude-sonnet-4',
        'claude-sonnet-4-5': 'claude-sonnet-4-5',
        'gemini-2.0-flash': 'gemini-2.0-flash',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-3-flash-preview': 'gemini-3-flash-preview',
        'gpt-5-nano': 'gpt-5-nano',
        'gpt-5-mini': 'gpt-5-mini',
        'gpt-5': 'gpt-5',
    }
    if short in mapping:
        return mapping[short]
    short2 = re.sub(r'-\d{8}$', '', short)
    if short2 in mapping:
        return mapping[short2]
    return short


def load_new_csv_data(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    required = {'dataset', 'agent_type', 'provider', 'model', 'resolved'}
    if required.issubset(set(df.columns)):
        grp = df.groupby(['dataset', 'agent_type', 'provider', 'model']).agg(
            resolution_rate=('resolved', 'mean'),
        ).reset_index()
        return [
            [r['dataset'], r['agent_type'], r['provider'], r['model'], r['resolution_rate']]
            for _, r in grp.iterrows()
        ]
    # Alternative: direct format with actual_performance column
    required_alt = {'dataset', 'architecture', 'provider', 'model'}
    if not required_alt.issubset(set(df.columns)):
        sys.exit(f"CSV must have columns {required} or {required_alt}")
    perf_col = 'actual_performance' if 'actual_performance' in df.columns else 'performance'
    if perf_col not in df.columns:
        sys.exit(f"CSV must have '{perf_col}' column")
    return [
        [r['dataset'], r['architecture'], r['provider'], r['model'],
         r[perf_col] if pd.notna(r[perf_col]) else None]
        for _, r in df.iterrows()
    ]


def build_training_dataframe(data_rows: list):
    """Build feature-engineered DataFrame from training data."""
    df = pd.DataFrame(data_rows, columns=['dataset', 'architecture', 'vendor', 'model', 'performance'])
    df['model_short'] = df['model'].apply(normalize_model_name)
    df['intelligence'] = df['model_short'].map(INTELLIGENCE_SCORES)
    unmapped = df[df['intelligence'].isna()]['model_short'].unique()
    if len(unmapped) > 0:
        raise ValueError(f"No intelligence score for: {unmapped.tolist()}\nAdd to INTELLIGENCE_SCORES.")

    df['n_tools'] = df['dataset'].map(TOOL_COUNTS)
    df['n_agents'] = df['architecture'].map(AGENT_COUNTS)

    for key in ['overhead_pct', 'message_density', 'redundancy', 'efficiency',
                 'error_amplification', 'success_per_1k_tokens']:
        col = key.replace('_tokens', '')
        df[col] = df['architecture'].map({k: v[key] for k, v in COORDINATION_METRICS.items()})

    sa_perf = df[df['architecture'] == 'single-agent'][['dataset', 'model_short', 'performance']].copy()
    sa_perf.columns = ['dataset', 'model_short', 'single_agent_baseline']
    df = df.merge(sa_perf, on=['dataset', 'model_short'], how='left')

    dataset_sa_mean = df[df['architecture'] == 'single-agent'].groupby('dataset')['performance'].mean()
    mask_sa = df['architecture'] == 'single-agent'
    df.loc[mask_sa, 'single_agent_baseline'] = df.loc[mask_sa, 'dataset'].map(dataset_sa_mean)

    intel_mean = df['intelligence'].mean()
    df['intelligence_centered'] = df['intelligence'] - intel_mean
    df['intelligence_sq_centered'] = df['intelligence_centered'] ** 2

    df['log_tools'] = np.log1p(df['n_tools'])
    df['log_agents'] = np.log1p(df['n_agents'])
    df['log_overhead'] = np.log1p(df['overhead_pct'])
    df['log_error_amp'] = np.log1p(df['error_amplification'])

    df['intel_x_efficiency']    = df['intelligence_centered'] * df['efficiency']
    df['error_x_baseline']      = df['error_amplification']  * df['single_agent_baseline']
    df['overhead_x_tools']      = df['overhead_pct']         * df['n_tools']
    df['redundancy_x_agents']   = df['redundancy']           * df['n_agents']
    df['msg_density_x_intel']   = df['message_density']      * df['intelligence_centered']
    df['efficiency_x_tools']    = df['efficiency']           * df['n_tools']
    df['baseline_x_agents']     = df['single_agent_baseline'] * np.log1p(df['n_agents'])
    df['intel_x_tools']         = df['intelligence_centered'] * np.log1p(df['n_tools'])
    df['error_x_tools']         = df['error_amplification']  * df['n_tools']

    return df, intel_mean


def prepare_new_rows_for_prediction(new_runs, training_df, intel_mean, scaler):
    """
    Build feature vectors for new runs using training-set parameters.
    For single_agent_baseline: uses the new run's own SA performance if provided,
    otherwise falls back to training data for that (dataset, model).
    """
    rows = []
    for run in new_runs:
        dataset, architecture, provider, model = run[0], run[1], run[2], run[3]
        actual = run[4] if len(run) > 4 else None

        model_short = normalize_model_name(model)
        intelligence = INTELLIGENCE_SCORES.get(model_short)
        if intelligence is None:
            print(f"  WARNING: No intelligence score for '{model_short}'. Skipping.")
            continue

        n_tools = TOOL_COUNTS.get(dataset)
        if n_tools is None:
            print(f"  WARNING: No tool count for dataset '{dataset}'. Add to TOOL_COUNTS. Skipping.")
            continue

        n_agents = AGENT_COUNTS.get(architecture)
        if n_agents is None:
            print(f"  WARNING: Unknown architecture '{architecture}'. Skipping.")
            continue

        coord = COORDINATION_METRICS.get(architecture)
        if coord is None:
            print(f"  WARNING: No coordination metrics for '{architecture}'. Skipping.")
            continue

        # Single-agent baseline lookup
        # First check if the user provided SA performance in new_runs
        sa_baseline = None
        for r in new_runs:
            if r[0] == dataset and r[1] == 'single-agent' and normalize_model_name(r[3]) == model_short:
                if r[4] is not None:
                    sa_baseline = r[4]
                break

        # Fallback: look up from training data
        if sa_baseline is None:
            mask = (training_df['dataset'] == dataset) & (training_df['model_short'] == model_short) & \
                   (training_df['architecture'] == 'single-agent')
            if mask.any():
                sa_baseline = training_df.loc[mask, 'performance'].values[0]

        # For single-agent architecture: use dataset-level SA mean
        if architecture == 'single-agent':
            ds_sa = training_df[(training_df['dataset'] == dataset) &
                                (training_df['architecture'] == 'single-agent')]
            if len(ds_sa) > 0:
                sa_baseline = ds_sa['performance'].mean()
            elif actual is not None:
                sa_baseline = actual  # use own performance as rough estimate

        if sa_baseline is None:
            print(f"  WARNING: Cannot determine single_agent_baseline for "
                  f"({dataset}, {model_short}). Using dataset SA mean from training data.")
            ds_sa = training_df[(training_df['dataset'] == dataset) &
                                (training_df['architecture'] == 'single-agent')]
            sa_baseline = ds_sa['performance'].mean() if len(ds_sa) > 0 else 0.5

        intel_centered = intelligence - intel_mean
        row = {
            'dataset': dataset, 'architecture': architecture,
            'vendor': provider, 'model': model, 'model_short': model_short,
            'actual_performance': actual,
            'intelligence': intelligence,
            'intelligence_centered': intel_centered,
            'intelligence_sq_centered': intel_centered ** 2,
            'n_tools': n_tools, 'log_tools': np.log1p(n_tools),
            'n_agents': n_agents, 'log_agents': np.log1p(n_agents),
            'overhead_pct': coord['overhead_pct'],
            'log_overhead': np.log1p(coord['overhead_pct']),
            'message_density': coord['message_density'],
            'redundancy': coord['redundancy'],
            'efficiency': coord['efficiency'],
            'error_amplification': coord['error_amplification'],
            'log_error_amp': np.log1p(coord['error_amplification']),
            'success_per_1k': coord['success_per_1k_tokens'],
            'single_agent_baseline': sa_baseline,
            'intel_x_efficiency': intel_centered * coord['efficiency'],
            'error_x_baseline': coord['error_amplification'] * sa_baseline,
            'overhead_x_tools': coord['overhead_pct'] * n_tools,
            'redundancy_x_agents': coord['redundancy'] * n_agents,
            'msg_density_x_intel': coord['message_density'] * intel_centered,
            'efficiency_x_tools': coord['efficiency'] * n_tools,
            'baseline_x_agents': sa_baseline * np.log1p(n_agents),
            'intel_x_tools': intel_centered * np.log1p(n_tools),
            'error_x_tools': coord['error_amplification'] * n_tools,
        }
        rows.append(row)

    df_new = pd.DataFrame(rows)
    if len(df_new) == 0:
        return df_new

    # Standardize using training scaler
    df_new_scaled = df_new.copy()
    df_new_scaled[FEATURE_COLS] = scaler.transform(df_new[FEATURE_COLS])
    return df_new_scaled


def main():
    parser = argparse.ArgumentParser(description="Predict new runs with the mixed-effects model")
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV file with new runs (columns: dataset,architecture,provider,model,actual_performance)')
    parser.add_argument('--per-instance-csv', type=str, default=None,
                        help='Per-instance CSV (columns: dataset,agent_type,provider,model,resolved)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Load training data (6 benchmarks)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("MIXED-EFFECTS MODEL — PREDICTION ON NEW RUNS")
    print("=" * 80)

    # Try to load swebench/terminalbench data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'per_instance_results_swe_tb.csv')
    all_training_rows = list(ORIGINAL_DATA)

    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        grp = df_csv.groupby(['dataset', 'agent_type', 'provider', 'model']).agg(
            resolution_rate=('resolved', 'mean'),
        ).reset_index()
        for _, row in grp.iterrows():
            all_training_rows.append([
                row['dataset'], row['agent_type'], row['provider'],
                row['model'], row['resolution_rate'],
            ])
        n_benchmarks = 6
    else:
        print(f"  NOTE: {csv_path} not found. Training on 4 benchmarks (180 rows) only.")
        n_benchmarks = 4

    # ------------------------------------------------------------------
    # Step 2: Build and fit model
    # ------------------------------------------------------------------
    print(f"\n  Training on {n_benchmarks} benchmarks ({len(all_training_rows)} rows)...")
    df_train, intel_mean = build_training_dataframe(all_training_rows)

    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_train_scaled[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])

    model = smf.ols(FORMULA, data=df_train_scaled).fit()

    print(f"  Model fitted: R²_train = {model.rsquared:.4f}, "
          f"Adj R² = {model.rsquared_adj:.4f}, "
          f"N = {len(df_train_scaled)}")
    print(f"  Intelligence centering offset = {intel_mean:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Collect new runs
    # ------------------------------------------------------------------
    new_runs = list(NEW_RUNS)

    if args.per_instance_csv:
        new_runs.extend(load_new_csv_data(args.per_instance_csv))

    if args.csv:
        df_input = pd.read_csv(args.csv)
        for _, r in df_input.iterrows():
            actual = r.get('actual_performance', r.get('performance', None))
            if pd.isna(actual):
                actual = None
            new_runs.append([r['dataset'], r['architecture'], r['provider'], r['model'], actual])

    if not new_runs:
        print("\n  No new runs to predict!")
        print("  Edit NEW_RUNS in this script, or pass --csv / --per-instance-csv")
        print("\n  Example NEW_RUNS entry:")
        print("    ['plancraft-test', 'single-agent', 'gemini', 'gemini-3-flash-preview', 0.55]")
        print("\n  Known models:", sorted(INTELLIGENCE_SCORES.keys()))
        print("  Known datasets:", sorted(TOOL_COUNTS.keys()))
        print("  Known architectures:", sorted(AGENT_COUNTS.keys()))
        return

    # ------------------------------------------------------------------
    # Step 4: Prepare and predict
    # ------------------------------------------------------------------
    print(f"\n  Predicting {len(new_runs)} new configuration(s)...")

    df_pred = prepare_new_rows_for_prediction(new_runs, df_train, intel_mean, scaler)
    if len(df_pred) == 0:
        print("  No valid configurations to predict. Check warnings above.")
        return

    df_pred['predicted'] = model.predict(df_pred)

    # ------------------------------------------------------------------
    # Step 5: Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)

    hdr = "{:<20s} {:<28s} {:<22s} {:>6s} {:>10s} {:>10s} {:>10s} {:>8s}"
    row_fmt = "{:<20s} {:<28s} {:<22s} {:>6d} {:>10.4f} {:>10s} {:>10s} {:>8s}"
    print(hdr.format("Dataset", "Architecture", "Model", "Intel",
                      "Predicted", "Actual", "Residual", "MAPE"))
    print("-" * 120)

    residuals = []
    for _, r in df_pred.iterrows():
        actual = r['actual_performance']
        pred = r['predicted']
        if actual is not None and not np.isnan(actual):
            resid = actual - pred
            mape = abs(resid / actual) * 100 if actual != 0 else float('inf')
            residuals.append(resid)
            actual_s = f"{actual:.4f}"
            resid_s = f"{resid:+.4f}"
            mape_s = f"{mape:.1f}%"
        else:
            actual_s = "  N/A"
            resid_s = "  N/A"
            mape_s = "N/A"

        print(row_fmt.format(
            r['dataset'][:20], r['architecture'][:28], r['model_short'][:22],
            int(r['intelligence']), pred, actual_s, resid_s, mape_s
        ))

    # ------------------------------------------------------------------
    # Step 6: Summary statistics
    # ------------------------------------------------------------------
    if residuals:
        residuals = np.array(residuals)
        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"  Configurations with actuals : {len(residuals)}")
        print(f"  Mean Absolute Error (MAE)   : {np.mean(np.abs(residuals)):.4f}")
        print(f"  Root Mean Sq Error (RMSE)   : {np.sqrt(np.mean(residuals**2)):.4f}")
        print(f"  Mean Residual (bias)        : {np.mean(residuals):+.4f}")
        print(f"  Max |residual|              : {np.max(np.abs(residuals)):.4f}")
        actuals_arr = df_pred[df_pred['actual_performance'].notna()]['actual_performance'].values
        preds_arr = df_pred[df_pred['actual_performance'].notna()]['predicted'].values
        ss_res = np.sum((actuals_arr - preds_arr) ** 2)
        ss_tot = np.sum((actuals_arr - actuals_arr.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        print(f"  R² (pred vs actual)         : {r2:.4f}")

        # Architecture-level breakdown
        print("\n  Per-architecture breakdown:")
        for arch in df_pred['architecture'].unique():
            mask = (df_pred['architecture'] == arch) & df_pred['actual_performance'].notna()
            sub = df_pred[mask]
            if len(sub) == 0:
                continue
            resids = sub['actual_performance'].values - sub['predicted'].values
            print(f"    {arch:<28s}  n={len(sub):>2d}  "
                  f"MAE={np.mean(np.abs(resids)):.4f}  "
                  f"bias={np.mean(resids):+.4f}")
    else:
        print("\n  No actual values provided — showing predictions only.")
        print("  Add actual_performance to compare with predictions.")

    # ------------------------------------------------------------------
    # Step 7: Show plancraft-specific context if relevant
    # ------------------------------------------------------------------
    plancraft_preds = df_pred[df_pred['dataset'] == 'plancraft-test']
    if len(plancraft_preds) > 0:
        print("\n" + "-" * 80)
        print("PLANCRAFT CONTEXT (from training data)")
        print("-" * 80)
        pc_train = df_train[df_train['dataset'] == 'plancraft-test']
        sa_mean = pc_train[pc_train['architecture'] == 'single-agent']['performance'].mean()
        print(f"  SA mean performance (training): {sa_mean:.3f}")
        print(f"  SA performance > 0.45 threshold: YES — paper predicts MAS degradation")
        print()
        print("  Training data performance by architecture:")
        arch_means = pc_train.groupby('architecture')['performance'].agg(['mean', 'std'])
        for arch, row in arch_means.iterrows():
            print(f"    {arch:<28s}  mean={row['mean']:.3f}  std={row['std']:.3f}")

    print()


if __name__ == "__main__":
    main()
