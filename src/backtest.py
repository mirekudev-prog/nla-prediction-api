#!/usr/bin/env python3
"""
Walk-Forward Backtesting Engine for Ghana NLA 5/90.
Evaluates baselines and ML models under strict temporal constraints.
Optimized version with pre-grouped data.
"""

import csv
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

from src.baselines import BASELINES, evaluate_selection
from src.ml_models import SimpleLogisticRegression


# Feature columns for ML models
FEATURE_COLS = ["gap", "freq_7", "freq_14", "freq_30", "freq_60", "freq_90", "prev_hit"]
TARGET_COL = "target"
NUMBER_COL = "number"
GAME_COL = "game_name"
SEQ_ID_COL = "seq_id"
DRAW_IDX_COL = "target_draw_idx"


def load_features(csv_path: str = "data/features.csv") -> Dict[str, Dict]:
    """Load features and pre-group by game and draw_idx for efficiency."""
    # First load all
    game_raw = defaultdict(list)
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in FEATURE_COLS + [TARGET_COL, NUMBER_COL, SEQ_ID_COL, DRAW_IDX_COL]:
                row[col] = int(row[col])
            game_raw[row[GAME_COL]].append(row)
    
    # Pre-process each game
    game_data = {}
    for game, feats in game_raw.items():
        # Sort by draw_idx then number
        feats.sort(key=lambda x: (x[DRAW_IDX_COL], x[NUMBER_COL]))
        
        # Group by draw_idx
        by_draw = defaultdict(list)
        for f in feats:
            by_draw[f[DRAW_IDX_COL]].append(f)
        
        draw_indices = sorted(by_draw.keys())
        
        # Pre-extract arrays for each draw
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


def prepare_ml_data_fast(game_data: Dict, draw_idx: int, train_window: int = 50) -> Tuple:
    """
    Fast ML data preparation using pre-grouped arrays.
    train_window: number of past draws to use for training
    """
    draw_arrays = game_data['draw_arrays']
    draw_indices = game_data['draw_indices']
    
    if draw_idx not in draw_arrays:
        return None, None, None, None
    
    # Find position of draw_idx
    pos = draw_indices.index(draw_idx)
    
    # Training draws: previous train_window draws
    train_start = max(0, pos - train_window)
    train_draw_indices = draw_indices[train_start:pos]
    
    if len(train_draw_indices) < 5:  # Need at least 5 draws * 90 = 450 samples
        return None, None, None, None
    
    # Concatenate training data
    train_feats = []
    train_targets = []
    for d in train_draw_indices:
        train_feats.append(draw_arrays[d]['features'])
        train_targets.append(draw_arrays[d]['targets'])
    
    X_train = np.vstack(train_feats)
    y_train = np.concatenate(train_targets)
    
    # Test data
    test = draw_arrays[draw_idx]
    X_test = test['features']
    test_numbers = test['numbers']
    actual = test['actual']
    
    # Check we have both classes
    if len(np.unique(y_train)) < 2:
        return None, None, None, None
    
    return X_train, y_train, X_test, test_numbers, actual


def run_baseline_evaluation_fast(game_data: Dict, baseline_name: str, baseline_func) -> List[Dict]:
    """Run a single baseline strategy across all evaluation draws."""
    results = []
    draw_arrays = game_data['draw_arrays']
    draw_indices = game_data['draw_indices']
    start_pos = game_data['first_eval_idx']
    
    for draw_idx in draw_indices[start_pos:]:
        draw_data = draw_arrays[draw_idx]
        
        # Get features for this draw (90 numbers)
        # Convert to list of dicts for baseline function
        draw_feats = []
        for i in range(90):
            draw_feats.append({
                c: draw_data['features'][i, j] for j, c in enumerate(FEATURE_COLS)
            })
            draw_feats[-1][NUMBER_COL] = draw_data['numbers'][i]
            draw_feats[-1][TARGET_COL] = draw_data['targets'][i]
        
        # Get prediction
        selected = baseline_func(draw_feats)
        
        # Evaluate
        actual = draw_data['actual']
        eval_result = evaluate_selection(selected, actual)
        eval_result["draw_idx"] = draw_idx
        eval_result["model"] = baseline_name
        
        results.append(eval_result)
    
    return results


def run_ml_evaluation_fast(game_data: Dict, model_class, model_name: str, retrain_freq: int = 20) -> List[Dict]:
    """Run ML model evaluation with walk-forward retraining (optimized)."""
    results = []
    model = None
    last_train_idx = -1
    draw_indices = game_data['draw_indices']
    start_pos = game_data['first_eval_idx']
    
    for draw_idx in draw_indices[start_pos:]:
        actual = game_data['draw_arrays'][draw_idx]['actual']
        
        # Check if we need to retrain
        should_retrain = (model is None) or (draw_idx - last_train_idx >= retrain_freq)
        
        if should_retrain:
            X_train, y_train, X_test, test_numbers, _ = prepare_ml_data_fast(game_data, draw_idx)
            if X_train is None:
                selected = sorted(np.random.choice(range(1, 91), 7, replace=False))
            else:
                model = model_class()
                model.fit(X_train, y_train)
                last_train_idx = draw_idx
                
                proba = model.predict_proba(X_test)
                prob_1 = proba[:, 1]
                top7_indices = np.argsort(prob_1)[-7:][::-1]
                selected = sorted([test_numbers[i] for i in top7_indices])
        else:
            # Use existing model
            _, _, X_test, test_numbers, _ = prepare_ml_data_fast(game_data, draw_idx)
            if X_test is None:
                selected = sorted(np.random.choice(range(1, 91), 7, replace=False))
            else:
                proba = model.predict_proba(X_test)
                prob_1 = proba[:, 1]
                top7_indices = np.argsort(prob_1)[-7:][::-1]
                selected = sorted([test_numbers[i] for i in top7_indices])
        
        # Evaluate
        eval_result = evaluate_selection(selected, actual)
        eval_result["draw_idx"] = draw_idx
        eval_result["model"] = model_name
        
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


def print_comparison_table(all_results: Dict[str, Dict[str, Any]]):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("WALK-FORWARD BACKTESTING RESULTS")
    print("=" * 100)
    
    header = f"{'Game':<20} {'Model':<25} {'Draws':>6} {'Win Rate':>10} {'Avg Hits':>10} {'Dist (0-5)':<30}"
    print(header)
    print("-" * 100)
    
    for game, models in all_results.items():
        first = True
        for model, metrics in models.items():
            if first:
                game_str = game
                first = False
            else:
                game_str = ""
            
            dist_str = " ".join(f"{metrics['hits_distribution'].get(i, 0):>3}" for i in range(6))
            print(f"{game_str:<20} {model:<25} {metrics['total_draws']:>6} "
                  f"{metrics['win_rate']*100:>9.1f}% {metrics['avg_hits']:>9.3f}  [{dist_str}]")
    
    print("=" * 100)


def run_full_backtest():
    """Run complete backtesting across all games and models."""
    print("Loading features...")
    game_data = load_features()
    
    print(f"Games loaded: {list(game_data.keys())}")
    for game, data in game_data.items():
        draws = len(data['draw_indices'])
        eval_draws = len(data['draw_indices']) - data['first_eval_idx']
        print(f"  {game}: {draws} draws, {eval_draws} evaluation draws")
    
    all_results = {}
    
    for game, data in game_data.items():
        print(f"\nEvaluating {game}...")
        game_results = {}
        
        # Baselines (only key ones for speed)
        key_baselines = {
            "Random": BASELINES["Random"],
            "Frequency (freq_30)": BASELINES["Frequency (freq_30)"],
            "Gap (Cold)": BASELINES["Gap (Cold)"],
            "Gap (Hot)": BASELINES["Gap (Hot)"],
            "Combined Heuristic": BASELINES["Combined Heuristic"],
        }
        
        for name, func in key_baselines.items():
            print(f"  {name}...", end=" ", flush=True)
            results = run_baseline_evaluation_fast(data, name, func)
            game_results[name] = aggregate_results(results)
            print(f"Win Rate: {game_results[name]['win_rate']*100:.1f}%")
        
        # ML Model
        print(f"  Logistic Regression...", end=" ", flush=True)
        lr_results = run_ml_evaluation_fast(data, SimpleLogisticRegression, "Logistic Regression", retrain_freq=50)
        game_results["Logistic Regression"] = aggregate_results(lr_results)
        print(f"Win Rate: {game_results['Logistic Regression']['win_rate']*100:.1f}%")
        
        all_results[game] = game_results
    
    print_comparison_table(all_results)
    
    # Overall summary
    print("\nOVERALL SUMMARY (averaged across games):")
    print("-" * 80)
    model_names = list(next(iter(all_results.values())).keys())
    for model in model_names:
        win_rates = [all_results[g][model]["win_rate"] for g in all_results]
        avg_hits = [all_results[g][model]["avg_hits"] for g in all_results]
        print(f"{model:<25} Win Rate: {np.mean(win_rates)*100:.2f}% ± {np.std(win_rates)*100:.2f}%  "
              f"Avg Hits: {np.mean(avg_hits):.3f}")
    
    return all_results


if __name__ == "__main__":
    run_full_backtest()