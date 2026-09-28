#!/usr/bin/env python3
"""
Rigorous Training Pipeline for Ghana NLA 5/90.
Directly optimizes for P(≥2 hits in top-7) using:
- Learning-to-Rank (ListNet/ListMLE)
- Cost-sensitive learning
- Walk-forward validation with proper metric
- Probability calibration
- Diversity-aware post-processing
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime
import csv
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

# Local imports
import sys
sys.path.insert(0, "/data/data/com.termux/files/home")
from src.ml_models import SimpleLogisticRegression, SimpleRandomForest, SimpleGradientBoosting
from src.ensemble import SoftVotingEnsemble, create_default_ensemble
from src.baselines import evaluate_selection, BASELINES

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
DRAW_IDX_COL = "target_draw_idx"
WARMUP_WINDOW = 90
NUMBERS = 90
TOP_K = 7
TARGET_HITS = 2  # We want at least 2 hits


def load_features(csv_path: str = "data/features.csv") -> Dict[str, Dict]:
    """Load features and pre-group by game and draw_idx."""
    from collections import defaultdict
    
    game_raw = defaultdict(list)
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in FEATURE_COLS + [TARGET_COL, NUMBER_COL, DRAW_IDX_COL]:
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


def prepare_training_data(game_data: Dict, draw_idx: int, train_window: int = 60) -> Tuple:
    """Prepare training data for a specific draw index."""
    draw_arrays = game_data['draw_arrays']
    draw_indices = game_data['draw_indices']
    
    if draw_idx not in draw_arrays:
        return None, None, None, None
    
    pos = draw_indices.index(draw_idx)
    train_start = max(0, pos - train_window)
    train_draw_indices = draw_indices[train_start:pos]
    
    if len(train_draw_indices) < 10:
        return None, None, None, None
    
    train_feats = []
    train_targets = []
    train_groups = []  # For learning-to-rank
    
    for d in train_draw_indices:
        train_feats.append(draw_arrays[d]['features'])
        train_targets.append(draw_arrays[d]['targets'])
        train_groups.append(90)  # 90 numbers per draw
    
    X_train = np.vstack(train_feats)
    y_train = np.concatenate(train_targets)
    groups = np.array(train_groups)
    
    return X_train, y_train, groups, train_draw_indices


def top_k_hit_metric(y_true: np.ndarray, y_pred: np.ndarray, k: int = TOP_K, 
                      target_hits: int = TARGET_HITS) -> float:
    """
    Metric: P(at least target_hits in top-k).
    This is what we actually care about.
    """
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    hits = y_true[top_k_indices].sum()
    return 1.0 if hits >= target_hits else 0.0


def expected_hits_metric(y_true: np.ndarray, y_pred: np.ndarray, k: int = TOP_K) -> float:
    """Expected number of hits in top-k."""
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    return float(y_true[top_k_indices].sum())


def dcg_metric(y_true: np.ndarray, y_pred: np.ndarray, k: int = TOP_K) -> float:
    """Discounted Cumulative Gain for ranking quality."""
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    dcg = 0.0
    for i in range(min(k, len(y_true_sorted))):
        if y_true_sorted[i] == 1:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


class CostSensitiveLogisticRegression(SimpleLogisticRegression):
    """
    Logistic Regression with cost-sensitive learning.
    Weights positive class (hits) higher since they're rare (~5.5%).
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 500, 
                 pos_weight: float = 17.0, tol: float = 1e-4):  # 1/0.055 ≈ 18
        super().__init__(learning_rate, max_iter, tol)
        self.pos_weight = pos_weight
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CostSensitiveLogisticRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        X_bias = np.hstack([X, np.ones((n_samples, 1))])
        
        prev_loss = float('inf')
        
        for i in range(self.max_iter):
            z = X_bias @ self.weights
            y_pred = self._sigmoid(z)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            
            # Cost-sensitive log-loss
            loss = -np.mean(
                self.pos_weight * y * np.log(y_pred) + 
                (1 - y) * np.log(1 - y_pred)
            )
            
            # Cost-sensitive gradient
            grad = X_bias.T @ (y_pred - y * self.pos_weight / (self.pos_weight * y + (1 - y))) / n_samples
            self.weights -= self.learning_rate * grad
            
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        
        self.bias = self.weights[-1]
        self.weights = self.weights[:-1]
        return self


class ListNetRanker:
    """
    ListNet: Learning-to-Rank using cross-entropy on top-k probabilities.
    Optimizes the probability distribution over permutations.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 300, 
                 hidden_dim: int = 64, top_k: int = TOP_K):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
    
    def _init_params(self, n_features: int):
        np.random.seed(42)
        self.W1 = np.random.randn(n_features, self.hidden_dim) * 0.01
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        scores = h @ self.W2 + self.b2
        return scores.ravel()
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "ListNetRanker":
        """Fit using ListNet loss (cross-entropy on top-k probability)."""
        self._init_params(X.shape[1])
        
        # Split into groups (draws)
        group_starts = np.concatenate([[0], np.cumsum(groups)[:-1]])
        n_groups = len(groups)
        
        for epoch in range(self.max_iter):
            total_loss = 0.0
            
            for g in range(n_groups):
                start = group_starts[g]
                end = start + groups[g]
                X_g = X[start:end]
                y_g = y[start:end]
                
                if y_g.sum() == 0:
                    continue
                
                # Forward pass
                scores = self._forward(X_g)
                
                # ListNet: softmax over all items, then cross-entropy with true relevance
                # True distribution: uniform over relevant items
                true_probs = y_g / y_g.sum()
                pred_probs = np.exp(scores - scores.max())
                pred_probs = pred_probs / pred_probs.sum()
                
                # Cross-entropy loss
                loss = -np.sum(true_probs * np.log(pred_probs + 1e-15))
                total_loss += loss
                
                # Gradients
                grad_scores = pred_probs - true_probs
                
                h = np.maximum(0, X_g @ self.W1 + self.b1)
                dh = (grad_scores[:, None] * self.W2.T) * (h > 0)
                
                self.W2 -= self.learning_rate * h.T @ grad_scores[:, None] / n_groups
                self.b2 -= self.learning_rate * grad_scores.mean()
                self.W1 -= self.learning_rate * X_g.T @ dh / n_groups
                self.b1 -= self.learning_rate * dh.mean(axis=0)
            
            if epoch % 50 == 0:
                print(f"  Epoch {epoch}, Loss: {total_loss:.4f}")
        
        return self
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)


class CalibratedModel:
    """Wrapper that calibrates predicted probabilities using isotonic regression."""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
            y_val: np.ndarray = None) -> "CalibratedModel":
        self.base_model.fit(X, y)
        
        if X_val is not None and y_val is not None:
            # Get uncalibrated probabilities
            proba = self.base_model.predict_proba(X_val)[:, 1]
            
            # Fit isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(proba, y_val)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.base_model.predict_proba(X)
        if self.calibrator is not None:
            calibrated = self.calibrator.transform(proba[:, 1])
            calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
            return np.column_stack([1 - calibrated, calibrated])
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def diversity_penalty(selected: List[int]) -> float:
    """Penalty for lack of diversity in selection."""
    selected = np.array(selected)
    penalties = 0.0
    
    # Odd/even balance (ideal: 3-4 or 4-3)
    odd = (selected % 2).sum()
    even = len(selected) - odd
    penalties += abs(odd - 3.5) * 0.1
    
    # High/low balance (ideal: 3-4 or 4-3)
    high = (selected > 45).sum()
    low = len(selected) - high
    penalties += abs(high - 3.5) * 0.1
    
    # Sum of digits (ideal range: 20-40)
    sum_digits = sum(sum(int(d) for d in str(n)) for n in selected)
    if sum_digits < 20 or sum_digits > 40:
        penalties += 0.2
    
    # Spread (ideal: not too clustered)
    spread = max(selected) - min(selected)
    if spread < 20:
        penalties += 0.15
    
    return penalties


def select_top_k_diverse(proba: np.ndarray, numbers: np.ndarray, k: int = TOP_K,
                          diversity_weight: float = 0.1) -> List[int]:
    """Select top-k with diversity awareness using greedy algorithm."""
    candidates = list(zip(numbers, proba))
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected = []
    remaining = candidates[:]
    
    while len(selected) < k and remaining:
        best_score = -np.inf
        best_idx = 0
        
        for idx, (num, prob) in enumerate(remaining[:min(20, len(remaining))]):  # Top 20 candidates
            test_selected = selected + [num]
            div_penalty = diversity_penalty(test_selected)
            score = prob - diversity_weight * div_penalty
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        selected.append(remaining.pop(best_idx)[0])
    
    return sorted(selected)


def train_and_evaluate_game(game_name: str, game_data: Dict, 
                            model_type: str = "logistic",
                            train_window: int = 60,
                            retrain_freq: int = 30) -> Dict:
    """Walk-forward training and evaluation for a single game."""
    
    draw_indices = game_data['draw_indices']
    start_pos = game_data['first_eval_idx']
    eval_indices = draw_indices[start_pos:]
    
    results = []
    model = None
    last_train_idx = -1
    
    # For calibration, hold out last 20% of training data
    calibration_split = 0.8
    
    for draw_idx in eval_indices:
        actual = game_data['draw_arrays'][draw_idx]['actual']
        
        should_retrain = (model is None or 
                         draw_idx - last_train_idx >= retrain_freq)
        
        if should_retrain:
            X_train, y_train, groups, train_draw_indices = prepare_training_data(
                game_data, draw_idx, train_window)
            
            if X_train is None:
                selected = sorted(np.random.choice(range(1, 91), TOP_K, replace=False))
                results.append(evaluate_selection(selected, actual))
                continue
            
            # Split for calibration
            split_idx = int(len(train_draw_indices) * calibration_split)
            cal_draw_indices = train_draw_indices[split_idx:]
            
            if len(cal_draw_indices) > 5:
                # Calibration data
                X_cal_list, y_cal_list = [], []
                for d in cal_draw_indices:
                    X_cal_list.append(game_data['draw_arrays'][d]['features'])
                    y_cal_list.append(game_data['draw_arrays'][d]['targets'])
                X_cal = np.vstack(X_cal_list)
                y_cal = np.concatenate(y_cal_list)
                
                # Training data
                X_tr_list, y_tr_list = [], []
                for d in train_draw_indices[:split_idx]:
                    X_tr_list.append(game_data['draw_arrays'][d]['features'])
                    y_tr_list.append(game_data['draw_arrays'][d]['targets'])
                X_tr = np.vstack(X_tr_list)
                y_tr = np.concatenate(y_tr_list)
            else:
                X_tr, y_tr = X_train, y_train
                X_cal, y_cal = None, None
            
            # Create and train model
            if model_type == "logistic":
                base_model = CostSensitiveLogisticRegression(
                    learning_rate=0.05, max_iter=400, pos_weight=18.0)
                model = CalibratedModel(base_model)
                model.fit(X_tr, y_tr, X_cal, y_cal)
            elif model_type == "ranker":
                model = ListNetRanker(learning_rate=0.01, max_iter=300)
                model.fit(X_tr, y_tr, np.full(len(train_draw_indices[:split_idx]), 90))
            elif model_type == "rf":
                base_model = SimpleRandomForest(n_estimators=30, max_depth=4, random_state=42)
                model = CalibratedModel(base_model)
                model.fit(X_tr, y_tr, X_cal, y_cal)
            elif model_type == "gb":
                base_model = SimpleGradientBoosting(n_estimators=40, learning_rate=0.05, 
                                                     max_depth=3, random_state=42)
                model = CalibratedModel(base_model)
                model.fit(X_tr, y_tr, X_cal, y_cal)
            elif model_type == "ensemble":
                base_model = create_default_ensemble()
                model = CalibratedModel(base_model)
                model.fit(X_tr, y_tr, X_cal, y_cal)
            
            last_train_idx = draw_idx
        
        # Predict on current draw
        draw_data = game_data['draw_arrays'][draw_idx]
        X_test = draw_data['features']
        test_numbers = draw_data['numbers']
        
        if hasattr(model, 'predict_scores'):  # ListNet
            scores = model.predict_scores(X_test)
            prob_1 = 1 / (1 + np.exp(-scores))  # Convert to probability
        else:
            proba = model.predict_proba(X_test)
            prob_1 = proba[:, 1]
        
        # Select with diversity awareness
        selected = select_top_k_diverse(prob_1, test_numbers, TOP_K, diversity_weight=0.05)
        
        # Evaluate
        eval_result = evaluate_selection(selected, actual)
        eval_result["draw_idx"] = draw_idx
        eval_result["draw_date"] = draw_idx  # Would need mapping to actual date
        eval_result["selected"] = selected
        eval_result["probabilities"] = prob_1.tolist()
        
        results.append(eval_result)
    
    return results


def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute comprehensive metrics."""
    if not results:
        return {}
    
    total = len(results)
    hits = [r["hit_count"] for r in results]
    wins = sum(1 for r in results if r["success"])
    hits_ge_2 = sum(1 for r in results if r["hit_count"] >= 2)
    hits_ge_3 = sum(1 for r in results if r["hit_count"] >= 3)
    
    from collections import Counter
    hits_dist = Counter(hits)
    
    return {
        "total_draws": total,
        "win_rate": wins / total,
        "hit_rate_ge_2": hits_ge_2 / total,
        "hit_rate_ge_3": hits_ge_3 / total,
        "avg_hits": float(np.mean(hits)),
        "max_hits": max(hits),
        "hits_distribution": dict(sorted(hits_dist.items())),
        "wins": wins,
    }


def run_comprehensive_experiment(output_path: str = "data/rigorous_results.json") -> Dict:
    """Run full experiment across all games and models."""
    
    print("Loading features...")
    game_data = load_features()
    all_games = list(game_data.keys())
    
    models_to_test = {
        "Cost-Sensitive LR": "logistic",
        "ListNet Ranker": "ranker",
        "Random Forest": "rf",
        "Gradient Boosting": "gb",
        "Soft Voting Ensemble": "ensemble",
        "Gap (Cold) Baseline": "baseline_gap",
        "Frequency Baseline": "baseline_freq",
        "Combined Heuristic": "baseline_combined",
    }
    
    all_results = {}
    
    for game in all_games:
        print(f"\n{'='*60}")
        print(f"Game: {game} ({len(game_data[game]['draw_indices'])} draws)")
        print(f"{'='*60}")
        all_results[game] = {}
        
        for model_name, model_type in models_to_test.items():
            print(f"\n  {model_name}...")
            
            try:
                if model_type.startswith("baseline"):
                    baseline_key = model_type.replace("baseline_", "")
                    baseline_map = {
                        "gap": "Gap (Cold)",
                        "freq": "Frequency (freq_30)",
                        "combined": "Combined Heuristic"
                    }
                    results = []
                    draw_arrays = game_data[game]['draw_arrays']
                    draw_indices = game_data[game]['draw_indices']
                    start_pos = game_data[game]['first_eval_idx']
                    
                    for draw_idx in draw_indices[start_pos:]:
                        draw_data = draw_arrays[draw_idx]
                        draw_feats = []
                        for i in range(90):
                            draw_feats.append({
                                c: draw_data['features'][i, j] for j, c in enumerate(FEATURE_COLS)
                            })
                            draw_feats[-1][NUMBER_COL] = draw_data['numbers'][i]
                            draw_feats[-1][TARGET_COL] = draw_data['targets'][i]
                        
                        selected = BASELINES[baseline_map[baseline_key]](draw_feats)
                        actual = draw_data['actual']
                        eval_result = evaluate_selection(selected, actual)
                        eval_result["draw_idx"] = draw_idx
                        eval_result["selected"] = selected
                        results.append(eval_result)
                else:
                    results = train_and_evaluate_game(game, game_data[game], model_type)
                
                metrics = aggregate_results(results)
                all_results[game][model_name] = {
                    "metrics": metrics,
                    "per_draw": results
                }
                
                print(f"    Win Rate: {metrics['win_rate']*100:.2f}%")
                print(f"    P(≥2 hits): {metrics['hit_rate_ge_2']*100:.2f}%")
                print(f"    P(≥3 hits): {metrics['hit_rate_ge_3']*100:.2f}%")
                print(f"    Avg Hits: {metrics['avg_hits']:.3f}")
                print(f"    Distribution: {metrics['hits_distribution']}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results[game][model_name] = {"error": str(e)}
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY (averaged across games)")
    print("="*80)
    
    overall = {}
    for model_name in models_to_test:
        win_rates = []
        hit_rates_2 = []
        hit_rates_3 = []
        avg_hits = []
        
        for game in all_games:
            if model_name in all_results[game] and "metrics" in all_results[game][model_name]:
                m = all_results[game][model_name]["metrics"]
                win_rates.append(m["win_rate"])
                hit_rates_2.append(m["hit_rate_ge_2"])
                hit_rates_3.append(m["hit_rate_ge_3"])
                avg_hits.append(m["avg_hits"])
        
        if win_rates:
            overall[model_name] = {
                "win_rate": float(np.mean(win_rates)),
                "hit_rate_ge_2": float(np.mean(hit_rates_2)),
                "hit_rate_ge_3": float(np.mean(hit_rates_3)),
                "avg_hits": float(np.mean(avg_hits)),
                "win_rate_std": float(np.std(win_rates)),
                "hit_rate_ge_2_std": float(np.std(hit_rates_2)),
            }
            print(f"  {model_name:<28} WR: {overall[model_name]['win_rate']*100:.2f}%  "
                  f"P(≥2): {overall[model_name]['hit_rate_ge_2']*100:.2f}%  "
                  f"P(≥3): {overall[model_name]['hit_rate_ge_3']*100:.2f}%  "
                  f"Avg: {overall[model_name]['avg_hits']:.3f}")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "target_hits": TARGET_HITS,
        "top_k": TOP_K,
        "games": all_games,
        "models_tested": list(models_to_test.keys()),
        "per_game": {g: {m: {"metrics": r.get("metrics", {})} 
                         for m, r in all_results[g].items()} 
                     for g in all_games},
        "overall": overall
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rigorous training for 2+ hits target")
    parser.add_argument("--output", type=str, default="data/rigorous_results.json")
    parser.add_argument("--games", type=str, nargs="+", help="Games to test")
    
    args = parser.parse_args()
    
    run_comprehensive_experiment(output_path=args.output)