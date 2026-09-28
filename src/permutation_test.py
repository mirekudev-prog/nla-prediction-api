#!/usr/bin/env python3
"""
Full Permutation Test for Statistical Significance.
Runs label-shuffling tests (100+ iterations) across evaluation draws.
Tests all models: LR, RF, GB, Ensemble, baselines.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime

from src.baselines import BASELINES, evaluate_selection
from src.ml_models import (SimpleLogisticRegression, SimpleRandomForest, 
                            SimpleGradientBoosting)
from src.ensemble import create_default_ensemble, SoftVotingEnsemble


# Feature columns (must match features.py)
FEATURE_COLS = [
    "gap", "freq_7", "freq_14", "freq_30", "freq_60", "freq_90", "prev_hit",
    "pair_30_avg", "pair_90_avg",
    "sum_digits", "is_odd", "is_high",
    "prev_sum", "prev_odd_ratio", "prev_high_ratio"
]
TARGET_COL = "target"
NUMBER_COL = "number"
GAME_COL = "game_name"
SEQ_ID_COL = "seq_id"
DRAW_IDX_COL = "target_draw_idx"
WARMUP_WINDOW = 90
NUMBERS = 90


def load_features(csv_path: str = "data/features.csv") -> Dict[str, Dict]:
    """Load features and pre-group by game and draw_idx."""
    from collections import defaultdict
    import csv
    
    game_raw = defaultdict(list)
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in FEATURE_COLS + [TARGET_COL, NUMBER_COL, SEQ_ID_COL, DRAW_IDX_COL]:
                if col in row:
                    row[col] = float(row[col]) if col in ["pair_30_avg", "pair_90_avg", "prev_odd_ratio", "prev_high_ratio"] else int(row[col])
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


def prepare_ml_data_fast(game_data: Dict, draw_idx: int, train_window: int = 30) -> Tuple:
    """Fast ML data preparation using pre-grouped arrays."""
    draw_arrays = game_data['draw_arrays']
    draw_indices = game_data['draw_indices']
    
    if draw_idx not in draw_arrays:
        return None, None, None, None, None
    
    pos = draw_indices.index(draw_idx)
    train_start = max(0, pos - train_window)
    train_draw_indices = draw_indices[train_start:pos]
    
    if len(train_draw_indices) < 5:
        return None, None, None, None, None
    
    train_feats = []
    train_targets = []
    for d in train_draw_indices:
        train_feats.append(draw_arrays[d]['features'])
        train_targets.append(draw_arrays[d]['targets'])
    
    X_train = np.vstack(train_feats)
    y_train = np.concatenate(train_targets)
    
    test = draw_arrays[draw_idx]
    X_test = test['features']
    test_numbers = test['numbers']
    actual = test['actual']
    
    if len(np.unique(y_train)) < 2:
        return None, None, None, None, None
    
    return X_train, y_train, X_test, test_numbers, actual


def run_model_evaluation(game_data: Dict, model, model_name: str, 
                          retrain_freq: int = 50, is_ensemble: bool = False) -> List[Dict]:
    """Run model evaluation with walk-forward retraining."""
    results = []
    draw_indices = game_data['draw_indices']
    start_pos = game_data['first_eval_idx']
    
    for draw_idx in draw_indices[start_pos:]:
        actual = game_data['draw_arrays'][draw_idx]['actual']
        
        should_retrain = (not hasattr(model, 'is_fitted') or not model.is_fitted) or \
                         (draw_idx - getattr(model, 'last_train_idx', -1) >= retrain_freq)
        
        if should_retrain:
            X_train, y_train, X_test, test_numbers, _ = prepare_ml_data_fast(game_data, draw_idx)
            if X_train is None:
                selected = sorted(np.random.choice(range(1, 91), 7, replace=False))
            else:
                model.fit(X_train, y_train)
                model.last_train_idx = draw_idx
                
                proba = model.predict_proba(X_test)
                prob_1 = proba[:, 1]
                top7_indices = np.argsort(prob_1)[-7:][::-1]
                selected = sorted([test_numbers[i] for i in top7_indices])
        else:
            _, _, X_test, test_numbers, _ = prepare_ml_data_fast(game_data, draw_idx)
            if X_test is None:
                selected = sorted(np.random.choice(range(1, 91), 7, replace=False))
            else:
                proba = model.predict_proba(X_test)
                prob_1 = proba[:, 1]
                top7_indices = np.argsort(prob_1)[-7:][::-1]
                selected = sorted([test_numbers[i] for i in top7_indices])
        
        eval_result = evaluate_selection(selected, actual)
        eval_result["draw_idx"] = draw_idx
        eval_result["model"] = model_name
        
        results.append(eval_result)
    
    return results


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


def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics from per-draw results."""
    if not results:
        return {}
    
    total = len(results)
    from collections import Counter
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


def run_permutation_test(n_iterations: int = 100, 
                          games: List[str] = None,
                          output_path: str = "data/permutation_results.json") -> Dict:
    """
    Run full permutation test.
    
    Args:
        n_iterations: Number of permutation iterations
        games: List of games to test (None = all)
        output_path: Path to save results
    """
    print(f"Loading features for permutation test...")
    game_data = load_features()
    
    all_games = list(game_data.keys())
    if games:
        all_games = [g for g in games if g in all_games]
    
    # Models to test
    models_to_test = {
        "Random": ("baseline", BASELINES["Random"]),
        "Frequency (freq_30)": ("baseline", BASELINES["Frequency (freq_30)"]),
        "Gap (Cold)": ("baseline", BASELINES["Gap (Cold)"]),
        "Combined Heuristic": ("baseline", BASELINES["Combined Heuristic"]),
        "Logistic Regression": ("ml", SimpleLogisticRegression(learning_rate=0.1, max_iter=200)),
        "Random Forest": ("ml", SimpleRandomForest(n_estimators=15, max_depth=3, random_state=42)),
        "Gradient Boosting": ("ml", SimpleGradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=2, random_state=42)),
        "Soft Voting Ensemble": ("ensemble", create_default_ensemble()),
    }
    
    # Actual win rates
    actual_win_rates = {}
    
    print("\nComputing actual win rates...")
    for game in all_games:
        print(f"  {game}...")
        actual_win_rates[game] = {}
        
        for model_name, (model_type, model_func) in models_to_test.items():
            if model_type == "baseline":
                results = run_baseline_evaluation_fast(game_data[game], model_name, model_func)
            elif model_type == "ensemble":
                results = run_model_evaluation(game_data[game], model_func, model_name, is_ensemble=True)
            else:
                results = run_model_evaluation(game_data[game], model_func, model_name)
            
            metrics = aggregate_results(results)
            actual_win_rates[game][model_name] = metrics["win_rate"]
            print(f"    {model_name}: {metrics['win_rate']*100:.2f}% (n={metrics['total_draws']})")
    
    # Permutation iterations
    print(f"\nRunning {n_iterations} permutation iterations...")
    permuted_win_rates = {game: {m: [] for m in models_to_test} for game in all_games}
    
    for iteration in range(n_iterations):
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}/{n_iterations}")
        
        for game in all_games:
            permuted = permute_targets(game_data[game])
            
            for model_name, (model_type, model_func) in models_to_test.items():
                if model_type == "baseline":
                    results = run_baseline_evaluation_fast(permuted, model_name, model_func)
                elif model_type == "ensemble":
                    # For ensemble, need fresh instance
                    fresh_ensemble = create_default_ensemble()
                    results = run_model_evaluation(permuted, fresh_ensemble, model_name, is_ensemble=True)
                else:
                    # For ML models, need fresh instance
                    if model_name == "Logistic Regression":
                        fresh_model = SimpleLogisticRegression(learning_rate=0.1, max_iter=200)
                    elif model_name == "Random Forest":
                        fresh_model = SimpleRandomForest(n_estimators=15, max_depth=3, random_state=42)
                    elif model_name == "Gradient Boosting":
                        fresh_model = SimpleGradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=2, random_state=42)
                    else:
                        fresh_model = model_func
                    results = run_model_evaluation(permuted, fresh_model, model_name)
                
                metrics = aggregate_results(results)
                permuted_win_rates[game][model_name].append(metrics["win_rate"])
    
    # Compute p-values
    print("\n" + "=" * 90)
    print("PERMUTATION TEST RESULTS")
    print("=" * 90)
    
    significant_results = []
    
    for game in all_games:
        print(f"\n{game}:")
        for model_name in models_to_test:
            actual = actual_win_rates[game][model_name]
            permuted = permuted_win_rates[game][model_name]
            
            p_value = np.mean(np.array(permuted) >= actual)
            perm_mean = np.mean(permuted)
            perm_std = np.std(permuted)
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"  {model_name:<28} Actual: {actual*100:.2f}%  "
                  f"Null: {perm_mean*100:.2f}% ± {perm_std*100:.2f}%  "
                  f"p = {p_value:.4f} {significance}")
            
            if p_value < 0.05:
                significant_results.append({
                    "game": game,
                    "model": model_name,
                    "actual_win_rate": actual,
                    "null_mean": perm_mean,
                    "null_std": perm_std,
                    "p_value": p_value
                })
    
    # Overall summary
    print("\n" + "=" * 90)
    print("OVERALL SIGNIFICANCE (averaged across games):")
    print("=" * 90)
    
    overall_results = {}
    
    for model_name in models_to_test:
        actuals = [actual_win_rates[g][model_name] for g in all_games]
        all_permuted = []
        for g in all_games:
            all_permuted.extend(permuted_win_rates[g][model_name])
        
        overall_actual = np.mean(actuals)
        overall_p = np.mean(np.array(all_permuted) >= overall_actual)
        overall_null_mean = np.mean(all_permuted)
        overall_null_std = np.std(all_permuted)
        
        significance = "***" if overall_p < 0.001 else "**" if overall_p < 0.01 else "*" if overall_p < 0.05 else "ns"
        
        overall_results[model_name] = {
            "actual": overall_actual,
            "null_mean": overall_null_mean,
            "null_std": overall_null_std,
            "p_value": overall_p,
            "significance": significance
        }
        
        print(f"  {model_name:<28} Actual: {overall_actual*100:.2f}%  "
              f"Null: {overall_null_mean*100:.2f}% ± {overall_null_std*100:.2f}%  "
              f"p = {overall_p:.4f} {significance}")
    
    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT:")
    print("=" * 90)
    
    if significant_results:
        print("Statistically significant edges detected (p < 0.05):")
        for r in significant_results:
            print(f"  ✓ {r['game']} - {r['model']}: p = {r['p_value']:.4f}")
    else:
        print("No statistically significant edges detected at p < 0.05 level.")
        print("All models perform within random chance expectations.")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_iterations": n_iterations,
        "games_tested": all_games,
        "actual_win_rates": {g: {m: float(v) for m, v in rates.items()} 
                            for g, rates in actual_win_rates.items()},
        "permuted_win_rates": {g: {m: [float(v) for v in vals] 
                                  for m, vals in rates.items()} 
                              for g, rates in permuted_win_rates.items()},
        "overall_results": {m: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                               for k, v in res.items()} 
                           for m, res in overall_results.items()},
        "significant_results": significant_results,
        "verdict": "significant" if significant_results else "not_significant"
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run permutation test")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--games", type=str, nargs="+", help="Games to test")
    parser.add_argument("--output", type=str, default="data/permutation_results.json")
    
    args = parser.parse_args()
    
    run_permutation_test(n_iterations=args.iterations, games=args.games, output_path=args.output)