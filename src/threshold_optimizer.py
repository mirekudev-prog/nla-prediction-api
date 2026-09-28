#!/usr/bin/env python3
"""
Walk-Forward Threshold & Rank Optimization.
Per-game probability cutoff tuning and rank selection optimization.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from src.baselines import BASELINES, evaluate_selection
from src.ml_models import (SimpleLogisticRegression, SimpleRandomForest, 
                            SimpleGradientBoosting)
from src.ensemble import create_default_ensemble


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


class ThresholdOptimizer:
    """
    Walk-forward threshold optimization for probability cutoffs and rank selection.
    """
    
    def __init__(self, 
                 prob_thresholds: List[float] = None,
                 rank_cutoffs: List[int] = None):
        """
        Args:
            prob_thresholds: List of probability thresholds to test
            rank_cutoffs: List of top-k ranks to test
        """
        self.prob_thresholds = prob_thresholds or np.arange(0.01, 0.21, 0.01).tolist()
        self.rank_cutoffs = rank_cutoffs or [5, 6, 7, 8, 9, 10]
        
        self.best_thresholds = {}  # game -> {model -> best_config}
    
    def optimize_game(self, game_data: Dict, model, model_name: str,
                      retrain_freq: int = 50) -> Dict[str, Any]:
        """
        Optimize threshold/rank for a single game using walk-forward validation.
        Uses first 50% of evaluation draws for tuning, last 50% for testing.
        """
        draw_indices = game_data['draw_indices']
        start_pos = game_data['first_eval_idx']
        eval_indices = draw_indices[start_pos:]
        
        if len(eval_indices) < 20:
            return {"error": "Not enough evaluation draws"}
        
        # Split: first 50% for tuning, last 50% for validation
        split_idx = len(eval_indices) // 2
        tune_indices = eval_indices[:split_idx]
        val_indices = eval_indices[split_idx:]
        
        print(f"  Tuning on {len(tune_indices)} draws, validating on {len(val_indices)} draws")
        
        # Collect predictions on tuning set
        tune_predictions = []  # List of (draw_idx, prob_1, numbers, actual)
        
        model_copy = self._clone_model(model)
        
        for draw_idx in tune_indices:
            actual = game_data['draw_arrays'][draw_idx]['actual']
            
            should_retrain = (not hasattr(model_copy, 'is_fitted') or not model_copy.is_fitted) or \
                             (draw_idx - getattr(model_copy, 'last_train_idx', -1) >= retrain_freq)
            
            if should_retrain:
                X_train, y_train, X_test, test_numbers, _ = prepare_ml_data_fast(game_data, draw_idx)
                if X_train is None:
                    continue
                model_copy.fit(X_train, y_train)
                model_copy.last_train_idx = draw_idx
                
                proba = model_copy.predict_proba(X_test)
                prob_1 = proba[:, 1]
            else:
                _, _, X_test, test_numbers, _ = prepare_ml_data_fast(game_data, draw_idx)
                if X_test is None:
                    continue
                proba = model_copy.predict_proba(X_test)
                prob_1 = proba[:, 1]
            
            tune_predictions.append({
                'draw_idx': draw_idx,
                'prob_1': prob_1,
                'numbers': test_numbers,
                'actual': actual
            })
        
        # Evaluate each threshold/rank on tuning set
        best_config = self._find_best_config(tune_predictions)
        
        # Validate on validation set
        val_metrics = self._evaluate_config(val_indices, game_data, model, model_name,
                                            best_config, retrain_freq)
        
        return {
            "best_config": best_config,
            "validation_metrics": val_metrics,
            "tune_draws": len(tune_indices),
            "val_draws": len(val_indices)
        }
    
    def _find_best_config(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Find best threshold/rank configuration on tuning predictions."""
        best_score = -1
        best_config = {}
        
        # Test probability thresholds
        for thresh in self.prob_thresholds:
            wins = 0
            total = 0
            for pred in predictions:
                prob_1 = pred['prob_1']
                numbers = pred['numbers']
                actual = pred['actual']
                
                selected = sorted([numbers[i] for i, p in enumerate(prob_1) if p >= thresh])
                # If fewer than 5 selected, take top 5; if more than 10, take top 10
                if len(selected) < 5:
                    top_indices = np.argsort(prob_1)[-5:][::-1]
                    selected = sorted([numbers[i] for i in top_indices])
                elif len(selected) > 10:
                    selected = selected[:10]
                
                eval_result = evaluate_selection(selected, actual)
                if eval_result["success"]:
                    wins += 1
                total += 1
            
            win_rate = wins / total if total > 0 else 0
            if win_rate > best_score:
                best_score = win_rate
                best_config = {
                    "type": "probability",
                    "threshold": thresh,
                    "tune_win_rate": win_rate
                }
        
        # Test rank cutoffs
        for k in self.rank_cutoffs:
            wins = 0
            total = 0
            for pred in predictions:
                prob_1 = pred['prob_1']
                numbers = pred['numbers']
                actual = pred['actual']
                
                top_indices = np.argsort(prob_1)[-k:][::-1]
                selected = sorted([numbers[i] for i in top_indices])
                
                eval_result = evaluate_selection(selected, actual)
                if eval_result["success"]:
                    wins += 1
                total += 1
            
            win_rate = wins / total if total > 0 else 0
            if win_rate > best_score:
                best_score = win_rate
                best_config = {
                    "type": "rank",
                    "top_k": k,
                    "tune_win_rate": win_rate
                }
        
        return best_config
    
    def _evaluate_config(self, val_indices: List[int], game_data: Dict, 
                         model, model_name: str, config: Dict, retrain_freq: int) -> Dict:
        """Evaluate best config on validation set."""
        model_copy = self._clone_model(model)
        
        wins = 0
        total = 0
        hits_dist = defaultdict(int)
        
        for draw_idx in val_indices:
            actual = game_data['draw_arrays'][draw_idx]['actual']
            
            should_retrain = (not hasattr(model_copy, 'is_fitted') or not model_copy.is_fitted) or \
                             (draw_idx - getattr(model_copy, 'last_train_idx', -1) >= retrain_freq)
            
            if should_retrain:
                X_train, y_train, X_test, test_numbers, _ = prepare_ml_data_fast(game_data, draw_idx)
                if X_train is None:
                    continue
                model_copy.fit(X_train, y_train)
                model_copy.last_train_idx = draw_idx
                
                proba = model_copy.predict_proba(X_test)
                prob_1 = proba[:, 1]
            else:
                _, _, X_test, test_numbers, _ = prepare_ml_data_fast(game_data, draw_idx)
                if X_test is None:
                    continue
                proba = model_copy.predict_proba(X_test)
                prob_1 = proba[:, 1]
            
            # Apply config
            if config["type"] == "probability":
                thresh = config["threshold"]
                selected = sorted([test_numbers[i] for i, p in enumerate(prob_1) if p >= thresh])
                if len(selected) < 5:
                    top_indices = np.argsort(prob_1)[-5:][::-1]
                    selected = sorted([test_numbers[i] for i in top_indices])
                elif len(selected) > 10:
                    selected = selected[:10]
            else:
                k = config["top_k"]
                top_indices = np.argsort(prob_1)[-k:][::-1]
                selected = sorted([test_numbers[i] for i in top_indices])
            
            eval_result = evaluate_selection(selected, actual)
            if eval_result["success"]:
                wins += 1
            total += 1
            hits_dist[eval_result["hit_count"]] += 1
        
        return {
            "win_rate": wins / total if total > 0 else 0,
            "total_draws": total,
            "wins": wins,
            "hits_distribution": dict(hits_dist)
        }
    
    def _clone_model(self, model):
        """Create a fresh copy of the model."""
        if hasattr(model, 'models'):  # Ensemble
            return create_default_ensemble()
        elif isinstance(model, SimpleLogisticRegression):
            return SimpleLogisticRegression(learning_rate=0.1, max_iter=200)
        elif isinstance(model, SimpleRandomForest):
            return SimpleRandomForest(n_estimators=15, max_depth=3, random_state=42)
        elif isinstance(model, SimpleGradientBoosting):
            return SimpleGradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=2, random_state=42)
        else:
            return model


def run_threshold_optimization(
    games: List[str] = None,
    models_to_test: Dict = None,
    output_path: str = "data/optimal_thresholds.json"
) -> Dict:
    """
    Run threshold optimization across all games and models.
    """
    print("Loading features for threshold optimization...")
    game_data = load_features()
    
    all_games = list(game_data.keys())
    if games:
        all_games = [g for g in games if g in all_games]
    
    if models_to_test is None:
        models_to_test = {
            "Logistic Regression": SimpleLogisticRegression(learning_rate=0.1, max_iter=200),
            "Random Forest": SimpleRandomForest(n_estimators=15, max_depth=3, random_state=42),
            "Gradient Boosting": SimpleGradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=2, random_state=42),
            "Soft Voting Ensemble": create_default_ensemble(),
        }
    
    optimizer = ThresholdOptimizer()
    results = {}
    
    for game in all_games:
        print(f"\n{'='*60}")
        print(f"Optimizing thresholds for {game}")
        print(f"{'='*60}")
        results[game] = {}
        
        for model_name, model in models_to_test.items():
            print(f"\n  {model_name}...")
            try:
                result = optimizer.optimize_game(game_data[game], model, model_name)
                results[game][model_name] = result
                
                if "error" not in result:
                    config = result["best_config"]
                    val = result["validation_metrics"]
                    print(f"    Best: {config['type']} = {config.get('threshold', config.get('top_k'))}")
                    print(f"    Tune WR: {config.get('tune_win_rate', 0)*100:.2f}%")
                    print(f"    Val WR: {val['win_rate']*100:.2f}%")
                else:
                    print(f"    Error: {result['error']}")
            except Exception as e:
                print(f"    Error: {e}")
                results[game][model_name] = {"error": str(e)}
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "games": all_games,
        "models_tested": list(models_to_test.keys()),
        "prob_thresholds_tested": optimizer.prob_thresholds,
        "rank_cutoffs_tested": optimizer.rank_cutoffs,
        "results": results
    }
    
    with open(output_path, "w") as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(output), f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    for game in all_games:
        print(f"\n{game}:")
        for model_name in models_to_test:
            if model_name in results[game] and "error" not in results[game][model_name]:
                config = results[game][model_name]["best_config"]
                val = results[game][model_name]["validation_metrics"]
                param = config.get('threshold', config.get('top_k'))
                print(f"  {model_name:<28} {config['type']}={param}  "
                      f"Tune: {config.get('tune_win_rate',0)*100:.1f}%  "
                      f"Val: {val['win_rate']*100:.1f}%")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run threshold optimization")
    parser.add_argument("--games", type=str, nargs="+", help="Games to optimize")
    parser.add_argument("--output", type=str, default="data/optimal_thresholds.json")
    
    args = parser.parse_args()
    
    run_threshold_optimization(games=args.games, output_path=args.output)