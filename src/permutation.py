#!/usr/bin/env python3
"""
Permutation Test for Statistical Significance - FAST VERSION.
Only tests baselines (instant), skips ML in permutations.
"""

import csv
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

from src.baselines import BASELINES, evaluate_selection


# Feature columns
FEATURE_COLS = ["gap", "freq_7", "freq_14", "freq_30", "freq_60", "freq_90", "prev_hit"]
TARGET_COL = "target"
NUMBER_COL = "number"
GAME_COL = "game_name"
SEQ_ID_COL = "seq_id"
DRAW_IDX_COL = "target_draw_idx"


def load_features(csv_path: str = "data/features.csv") -> Dict[str, Dict]:
    """Load features and pre-group by game and draw_idx for efficiency."""
    game_raw = defaultdict(list)
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in FEATURE_COLS + [TARGET_COL, NUMBER_COL, SEQ_ID_COL, DRAW_IDX_COL]:
                row[col] = int(row[col])
            game_raw[row[GAME_COL]].append(row)
    
    game_data = {}
    for game, feats in game_raw.items():
        feats.sort(key=lambda x: (x[DRAW_IDX_COL], x[NUMBER_COL]))
        
        by_draw = defaultdict(list)
        for f in feats:
            by_draw[f[DRAW_IDX_COL]].append(f)
        
        draw_indices = sorted(by_draw.keys())
        
        draw_arrays = {}
        for d_idx in draw_indices:
            draw_feats = by_draw[d_idx]
            draw_arrays[d_idx] = {
                'features': np.array([[f[c] for c in FEATURE_COLS] for f in draw_feats], dtype=float),
                'targets': np.array([f[TARGET_COL] for f in draw_feats], dtype=int),
                'numbers': np.array([f[NUMBER_COL] for f in draw_feats], dtype=int),
                'actual': sorted([f[NUMBER_COL] for f in draw_feats if f[TARGET_COL] == 1])
            }
        
        game_data[game] = {
            'draw_indices': draw_indices,
            'draw_arrays': draw_arrays,
            'first_eval_idx': next(i for i, d in enumerate(draw_indices) if d >= 100)
        }
    
    return game_data


def permute_targets(game_data: Dict) -> Dict:
    """Create a permuted version by shuffling target labels across draws."""
    draw_arrays = game_data['draw_arrays']
    draw_indices = game_data['draw_indices']
    
    shuffled_indices = np.random.permutation(draw_indices)
    draw_mapping = dict(zip(draw_indices, shuffled_indices))
    
    permuted_arrays = {}
    for d_idx in draw_indices:
        orig = draw_arrays[d_idx]
        shuffle_idx = draw_mapping[d_idx]
        shuffle_targets = draw_arrays[shuffle_idx]['targets']
        
        permuted_arrays[d_idx] = {
            'features': orig['features'].copy(),
            'targets': shuffle_targets.copy(),
            'numbers': orig['numbers'].copy(),
            'actual': draw_arrays[shuffle_idx]['actual'].copy()
        }
    
    return {
        'draw_indices': draw_indices.copy(),
        'draw_arrays': permuted_arrays,
        'first_eval_idx': game_data['first_eval_idx']
    }


def run_baseline_evaluation_fast(game_data: Dict, baseline_name: str, baseline_func) -> List[Dict]:
    """Run a single baseline strategy across all evaluation draws."""
    results = []
    draw_arrays = game_data['draw_arrays']
    draw_indices = game_data['draw_indices']
    start_pos = game_data['first_eval_idx']
    
    for draw_idx in draw_indices[start_pos:]:
        draw_data = draw_arrays[draw_idx]
        
        draw_feats = []
        for i in range(90):
            draw_feats.append({
                c: draw_data['features'][i, j] for j, c in enumerate(FEATURE_COLS)
            })
            draw_feats[-1][NUMBER_COL] = draw_data['numbers'][i]
            draw_feats[-1][TARGET_COL] = draw_data['targets'][i]
        
        selected = baseline_func(draw_feats)
        actual = draw_data['actual']
        
        eval_result = evaluate_selection(selected, actual)
        eval_result["draw_idx"] = draw_idx
        eval_result["model"] = baseline_name
        
        results.append(eval_result)
    
    return results


def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics from per-draw results."""
    if not results:
        return {}
    
    total = len(results)
    hits_dist = Counter(r["hit_count"] for r in results)
    wins = sum(1 for r in results if r["success"])
    avg_hits = float(np.mean([r["hit_count"] for r in results]))
    
    return {
        "total_draws": total,
        "win_rate": wins / total if total > 0 else 0,
        "avg_hits": avg_hits,
        "hits_distribution": dict(sorted(hits_dist.items())),
        "wins": wins,
    }


def run_permutation_test(n_iterations: int = 200):
    """Run permutation test - FAST (baselines only)."""
    print(f"Loading features...")
    game_data = load_features()
    
    games = list(game_data.keys())
    
    # Only test baselines (instant)
    models_to_test = {
        "Random": BASELINES["Random"],
        "Frequency (freq_30)": BASELINES["Frequency (freq_30)"],
        "Gap (Cold)": BASELINES["Gap (Cold)"],
        "Gap (Hot)": BASELINES["Gap (Hot)"],
        "Combined Heuristic": BASELINES["Combined Heuristic"],
    }
    
    # Actual win rates
    actual_win_rates = {}
    
    print("\nComputing actual win rates...")
    for game in games:
        actual_win_rates[game] = {}
        for model_name, model_func in models_to_test.items():
            results = run_baseline_evaluation_fast(game_data[game], model_name, model_func)
            metrics = aggregate_results(results)
            actual_win_rates[game][model_name] = metrics["win_rate"]
            print(f"  {game} - {model_name}: {metrics['win_rate']*100:.2f}%")
    
    # Permutation iterations
    print(f"\nRunning {n_iterations} permutation iterations...")
    permuted_win_rates = {game: {m: [] for m in models_to_test} for game in games}
    
    for iteration in range(n_iterations):
        if iteration % 20 == 0:
            print(f"  Iteration {iteration}/{n_iterations}")
        
        for game in games:
            permuted = permute_targets(game_data[game])
            
            for model_name, model_func in models_to_test.items():
                results = run_baseline_evaluation_fast(permuted, model_name, model_func)
                metrics = aggregate_results(results)
                permuted_win_rates[game][model_name].append(metrics["win_rate"])
    
    # Compute p-values
    print("\n" + "=" * 80)
    print("PERMUTATION TEST RESULTS")
    print("=" * 80)
    
    significant_results = []
    
    for game in games:
        print(f"\n{game}:")
        for model_name in models_to_test:
            actual = actual_win_rates[game][model_name]
            permuted = permuted_win_rates[game][model_name]
            
            p_value = np.mean(np.array(permuted) >= actual)
            perm_mean = np.mean(permuted)
            perm_std = np.std(permuted)
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"  {model_name:<25} Actual: {actual*100:.2f}%  "
                  f"Null: {perm_mean*100:.2f}% ± {perm_std*100:.2f}%  "
                  f"p = {p_value:.4f} {significance}")
            
            if p_value < 0.05:
                significant_results.append({
                    "game": game,
                    "model": model_name,
                    "actual_win_rate": actual,
                    "null_mean": perm_mean,
                    "p_value": p_value
                })
    
    # Overall
    print("\n" + "=" * 80)
    print("OVERALL SIGNIFICANCE:")
    print("=" * 80)
    
    for model_name in models_to_test:
        actuals = [actual_win_rates[g][model_name] for g in games]
        all_permuted = []
        for g in games:
            all_permuted.extend(permuted_win_rates[g][model_name])
        
        overall_actual = np.mean(actuals)
        overall_p = np.mean(np.array(all_permuted) >= overall_actual)
        
        significance = "***" if overall_p < 0.001 else "**" if overall_p < 0.01 else "*" if overall_p < 0.05 else "ns"
        
        print(f"  {model_name:<25} Actual: {overall_actual*100:.2f}%  "
              f"Null: {np.mean(all_permuted)*100:.2f}% ± {np.std(all_permuted)*100:.2f}%  "
              f"p = {overall_p:.4f} {significance}")
    
    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)
    
    if significant_results:
        print("Statistically significant edges detected (p < 0.05):")
        for r in significant_results:
            print(f"  ✓ {r['game']} - {r['model']}: p = {r['p_value']:.4f}")
    else:
        print("No statistically significant edges detected at p < 0.05 level.")
        print("All models perform within random chance expectations.")
    
    return {
        "actual_win_rates": actual_win_rates,
        "permuted_win_rates": permuted_win_rates,
        "significant": significant_results
    }


if __name__ == "__main__":
    run_permutation_test(n_iterations=200)