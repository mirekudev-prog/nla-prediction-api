#!/usr/bin/env python3
"""
Production Top-7 Number Generator for Ghana NLA 5/90.
Generates predictions for the next upcoming draw using the best model.
"""

import argparse
import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.db import init_db, get_draws
from src.features import compute_features_for_game
from src.ml_models import SimpleLogisticRegression, SimpleRandomForest, SimpleGradientBoosting
from src.baselines import evaluate_selection


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


# Pre-computed static number properties
NUMBER_PROPS = {}
for k in range(1, NUMBERS + 1):
    digits = [int(d) for d in str(k)]
    NUMBER_PROPS[k] = {
        "sum_digits": sum(digits),
        "is_odd": k % 2,
        "is_high": 1 if k > 45 else 0,
    }


def get_latest_draw_state(game_name: str) -> Tuple[List[Dict], int, List[int]]:
    """
    Get the latest draw state for a game.
    Returns: (all_draws, next_draw_idx, actual_numbers_of_latest_draw)
    """
    draws = get_draws(game_name)
    if not draws:
        raise ValueError(f"No draws found for {game_name}")
    
    # Latest draw is the last one (highest seq_id)
    latest = draws[-1]
    actual_numbers = [latest["n1"], latest["n2"], latest["n3"], latest["n4"], latest["n5"]]
    
    # Next draw index
    next_draw_idx = len([d for d in draws if d["game_name"] == game_name])
    
    return draws, next_draw_idx, actual_numbers


def compute_live_features(game_name: str, draws: list, target_draw_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feature matrix for the NEXT draw (target_draw_idx).
    Uses ALL historical draws including the latest.
    """
    # Extract winning numbers for each draw
    draw_numbers = []
    for d in draws:
        draw_numbers.append([d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]])
    
    # Initialize tracking
    last_seen = np.full(NUMBERS + 1, -1, dtype=int)
    freq_windows = {
        7: np.zeros(NUMBERS + 1, dtype=int),
        14: np.zeros(NUMBERS + 1, dtype=int),
        30: np.zeros(NUMBERS + 1, dtype=int),
        60: np.zeros(NUMBERS + 1, dtype=int),
        90: np.zeros(NUMBERS + 1, dtype=int),
    }
    pair_windows = {
        30: np.zeros((NUMBERS + 1, NUMBERS + 1), dtype=int),
        90: np.zeros((NUMBERS + 1, NUMBERS + 1), dtype=int),
    }
    past_draws = []
    
    # Process ALL historical draws to build up state
    for t, current_draw in enumerate(draw_numbers):
        # Update history (including the latest draw)
        for k in current_draw:
            last_seen[k] = t
        
        # Update pair co-occurrence
        for window_size in pair_windows:
            if len(past_draws) >= window_size:
                oldest = past_draws[-window_size]
                for i, a in enumerate(oldest):
                    for b in oldest[i+1:]:
                        pair_windows[window_size][a, b] -= 1
                        pair_windows[window_size][b, a] -= 1
            
            for i, a in enumerate(current_draw):
                for b in current_draw[i+1:]:
                    pair_windows[window_size][a, b] += 1
                    pair_windows[window_size][b, a] += 1
        
        past_draws.append(current_draw)
        
        # Update frequency windows
        for window_size in freq_windows:
            if len(past_draws) <= window_size:
                for k in current_draw:
                    freq_windows[window_size][k] += 1
            else:
                oldest = past_draws[-window_size - 1]
                for k in oldest:
                    freq_windows[window_size][k] -= 1
                for k in current_draw:
                    freq_windows[window_size][k] += 1
    
    # Now compute features for the NEXT draw (target_draw_idx = len(draw_numbers))
    # Using all history up to now
    t = target_draw_idx
    prev_draw = draw_numbers[-1] if draw_numbers else []
    prev_sum = sum(prev_draw)
    prev_odd = sum(1 for n in prev_draw if n % 2 == 1)
    prev_high = sum(1 for n in prev_draw if n > 45)
    prev_odd_ratio = prev_odd / 5.0 if prev_draw else 0.5
    prev_high_ratio = prev_high / 5.0 if prev_draw else 0.5
    
    features = []
    numbers = []
    
    for k in range(1, NUMBERS + 1):
        gap = t - last_seen[k] if last_seen[k] >= 0 else 999
        
        freq_7 = freq_windows[7][k]
        freq_14 = freq_windows[14][k]
        freq_30 = freq_windows[30][k]
        freq_60 = freq_windows[60][k]
        freq_90 = freq_windows[90][k]
        
        prev_hit = 1 if k in prev_draw else 0
        
        if len(past_draws) >= 30:
            pair_30_avg = float(np.mean([pair_windows[30][k, n] for n in prev_draw]))
            pair_90_avg = float(np.mean([pair_windows[90][k, n] for n in prev_draw]))
        else:
            pair_30_avg = 0.0
            pair_90_avg = 0.0
        
        props = NUMBER_PROPS[k]
        sum_digits = props["sum_digits"]
        is_odd = props["is_odd"]
        is_high = props["is_high"]
        
        features.append([
            gap, freq_7, freq_14, freq_30, freq_60, freq_90, prev_hit,
            pair_30_avg, pair_90_avg,
            sum_digits, is_odd, is_high,
            prev_sum, prev_odd_ratio, prev_high_ratio
        ])
        numbers.append(k)
    
    return np.array(features, dtype=float), np.array(numbers, dtype=int)


def train_best_model(draws: list) -> Any:
    """
    Train the best performing model on all available data.
    Based on backtest results, Logistic Regression performs best overall.
    """
    # Build features for all historical draws (excluding the very latest for training)
    game_name = draws[0]["game_name"]
    draw_numbers = []
    for d in draws:
        draw_numbers.append([d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]])
    
    # Use the same feature computation but collect training data
    # We need features and targets for draws >= WARMUP_WINDOW
    last_seen = np.full(NUMBERS + 1, -1, dtype=int)
    freq_windows = {w: np.zeros(NUMBERS + 1, dtype=int) for w in [7, 14, 30, 60, 90]}
    pair_windows = {w: np.zeros((NUMBERS + 1, NUMBERS + 1), dtype=int) for w in [30, 90]}
    past_draws = []
    
    X_all = []
    y_all = []
    
    for t, current_draw in enumerate(draw_numbers):
        if t >= WARMUP_WINDOW:
            prev_draw = draw_numbers[t - 1] if t > 0 else []
            prev_sum = sum(prev_draw)
            prev_odd = sum(1 for n in prev_draw if n % 2 == 1)
            prev_high = sum(1 for n in prev_draw if n > 45)
            prev_odd_ratio = prev_odd / 5.0 if prev_draw else 0.5
            prev_high_ratio = prev_high / 5.0 if prev_draw else 0.5
            
            for k in range(1, NUMBERS + 1):
                gap = t - last_seen[k] if last_seen[k] >= 0 else 999
                
                freq_7 = freq_windows[7][k]
                freq_14 = freq_windows[14][k]
                freq_30 = freq_windows[30][k]
                freq_60 = freq_windows[60][k]
                freq_90 = freq_windows[90][k]
                
                prev_hit = 1 if (t > 0 and k in draw_numbers[t - 1]) else 0
                
                if len(past_draws) >= 30:
                    pair_30_avg = float(np.mean([pair_windows[30][k, n] for n in prev_draw]))
                    pair_90_avg = float(np.mean([pair_windows[90][k, n] for n in prev_draw]))
                else:
                    pair_30_avg = 0.0
                    pair_90_avg = 0.0
                
                props = NUMBER_PROPS[k]
                features.append([
                    gap, freq_7, freq_14, freq_30, freq_60, freq_90, prev_hit,
                    pair_30_avg, pair_90_avg,
                    props["sum_digits"], props["is_odd"], props["is_high"],
                    prev_sum, prev_odd_ratio, prev_high_ratio
                ])
                y_all.append(1 if k in current_draw else 0)
        
        # Update history
        for k in current_draw:
            last_seen[k] = t
        
        for window_size in pair_windows:
            if len(past_draws) >= window_size:
                oldest = past_draws[-window_size]
                for i, a in enumerate(oldest):
                    for b in oldest[i+1:]:
                        pair_windows[window_size][a, b] -= 1
                        pair_windows[window_size][b, a] -= 1
            for i, a in enumerate(current_draw):
                for b in current_draw[i+1:]:
                    pair_windows[window_size][a, b] += 1
                    pair_windows[window_size][b, a] += 1
        
        past_draws.append(current_draw)
        
        for window_size in freq_windows:
            if len(past_draws) <= window_size:
                for k in current_draw:
                    freq_windows[window_size][k] += 1
            else:
                oldest = past_draws[-window_size - 1]
                for k in oldest:
                    freq_windows[window_size][k] -= 1
                for k in current_draw:
                    freq_windows[window_size][k] += 1
    
    X = np.array(X_all, dtype=float)
    y = np.array(y_all, dtype=int)
    
    # Train Logistic Regression (best overall performer)
    model = SimpleLogisticRegression(learning_rate=0.01, max_iter=200)
    model.fit(X, y)
    
    return model


def calculate_selection_metrics(selected: List[int]) -> Dict[str, Any]:
    """Calculate descriptive metrics for a selection of 7 numbers."""
    selected = sorted(selected)
    odd_count = sum(1 for n in selected if n % 2 == 1)
    even_count = 7 - odd_count
    high_count = sum(1 for n in selected if n > 45)
    low_count = 7 - high_count
    sum_digits = sum(sum(int(d) for d in str(n)) for n in selected)
    avg_gap = None  # Would need historical data
    
    return {
        "numbers": selected,
        "odd": odd_count,
        "even": even_count,
        "high": high_count,
        "low": low_count,
        "sum_digits": sum_digits,
        "avg": sum(selected) / 7,
        "spread": max(selected) - min(selected)
    }


def format_output(metrics: Dict[str, Any], game_name: str, model_name: str) -> str:
    """Format the output for display."""
    nums = metrics["numbers"]
    formatted = f"[{', '.join(str(n).zfill(2) for n in nums)}]"
    
    output = f"""
╔════════════════════════════════════════════════════════════════════╗
║  GHANA NLA 5/90 - NEXT DRAW PREDICTION                           ║
╠════════════════════════════════════════════════════════════════════╣
║  Game: {game_name:<50} ║
║  Model: {model_name:<49} ║
║  Top 7 Numbers: {formatted:<43} ║
╠════════════════════════════════════════════════════════════════════╣
║  Selection Metrics:                                              ║
║    Odd/Even: {metrics['odd']} odd / {metrics['even']} even                                    ║
║    High/Low: {metrics['high']} high (46-90) / {metrics['low']} low (1-45)                            ║
║    Sum of Digits: {metrics['sum_digits']:<3}                                    ║
║    Average: {metrics['avg']:.1f}                                           ║
║    Spread (max-min): {metrics['spread']:<3}                                     ║
╚════════════════════════════════════════════════════════════════════╝
"""
    return output


class FeatureScaler:
    """Simple feature standardization."""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def main():
    parser = argparse.ArgumentParser(description="Generate Top-7 predictions for NLA 5/90")
    parser.add_argument("--game", type=str, required=True, 
                       choices=["Monday Special", "Lucky Tuesday", "MidWeek", 
                                "Fortune Thursday", "Friday Bonanza", 
                                "National Weekly", "Sunday Aseda"],
                       help="Game name to predict")
    parser.add_argument("--model", type=str, default="logistic",
                       choices=["logistic", "rf", "gb", "gap_cold"],
                       help="Model to use for prediction")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run without saving to database")
    
    args = parser.parse_args()
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    print(f"Initializing database connection...")
    init_db()
    
    print(f"Fetching latest draws for {args.game}...")
    draws, next_draw_idx, latest_actual = get_latest_draw_state(args.game)
    print(f"  Total historical draws: {len(draws)}")
    print(f"  Latest draw actual: {sorted(latest_actual)}")
    print(f"  Next draw index: {next_draw_idx}")
    
    print(f"Computing live features for next draw...")
    X_live, numbers = compute_live_features(args.game, draws, next_draw_idx)
    
    print(f"Training model...")
    if args.model == "logistic":
        model = SimpleLogisticRegression(learning_rate=0.1, max_iter=300)
    elif args.model == "rf":
        model = SimpleRandomForest(n_estimators=15, max_depth=3, random_state=42)
    elif args.model == "gb":
        model = SimpleGradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=2, random_state=42)
    elif args.model == "gap_cold":
        print("Using Gap (Cold) baseline heuristic...")
        from src.baselines import BASELINES
        
        draw_feats = []
        for i in range(90):
            draw_feats.append({
                c: X_live[i, j] for j, c in enumerate(FEATURE_COLS)
            })
            draw_feats[-1][NUMBER_COL] = numbers[i]
            draw_feats[-1][TARGET_COL] = 0
        
        selected = BASELINES["Gap (Cold)"](draw_feats)
        metrics = calculate_selection_metrics(selected)
        print(format_output(metrics, args.game, "Gap (Cold) Baseline"))
        return
    
    # Build training data with feature standardization
    draw_numbers = []
    for d in draws:
        draw_numbers.append([d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]])
    
    last_seen = np.full(NUMBERS + 1, -1, dtype=int)
    freq_windows = {w: np.zeros(NUMBERS + 1, dtype=int) for w in [7, 14, 30, 60, 90]}
    pair_windows = {w: np.zeros((NUMBERS + 1, NUMBERS + 1), dtype=int) for w in [30, 90]}
    past_draws = []
    
    X_train = []
    y_train = []
    
    for t, current_draw in enumerate(draw_numbers):
        if t >= WARMUP_WINDOW:
            prev_draw = draw_numbers[t - 1] if t > 0 else []
            prev_sum = sum(prev_draw)
            prev_odd = sum(1 for n in prev_draw if n % 2 == 1)
            prev_high = sum(1 for n in prev_draw if n > 45)
            prev_odd_ratio = prev_odd / 5.0 if prev_draw else 0.5
            prev_high_ratio = prev_high / 5.0 if prev_draw else 0.5
            
            for k in range(1, NUMBERS + 1):
                gap = t - last_seen[k] if last_seen[k] >= 0 else 999
                
                freq_7 = freq_windows[7][k]
                freq_14 = freq_windows[14][k]
                freq_30 = freq_windows[30][k]
                freq_60 = freq_windows[60][k]
                freq_90 = freq_windows[90][k]
                
                prev_hit = 1 if (t > 0 and k in draw_numbers[t - 1]) else 0
                
                if len(past_draws) >= 30:
                    pair_30_avg = float(np.mean([pair_windows[30][k, n] for n in prev_draw]))
                    pair_90_avg = float(np.mean([pair_windows[90][k, n] for n in prev_draw]))
                else:
                    pair_30_avg = 0.0
                    pair_90_avg = 0.0
                
                props = NUMBER_PROPS[k]
                X_train.append([
                    gap, freq_7, freq_14, freq_30, freq_60, freq_90, prev_hit,
                    pair_30_avg, pair_90_avg,
                    props["sum_digits"], props["is_odd"], props["is_high"],
                    prev_sum, prev_odd_ratio, prev_high_ratio
                ])
                y_train.append(1 if k in current_draw else 0)
        
        for k in current_draw:
            last_seen[k] = t
        
        for window_size in pair_windows:
            if len(past_draws) >= window_size:
                oldest = past_draws[-window_size]
                for i, a in enumerate(oldest):
                    for b in oldest[i+1:]:
                        pair_windows[window_size][a, b] -= 1
                        pair_windows[window_size][b, a] -= 1
            for i, a in enumerate(current_draw):
                for b in current_draw[i+1:]:
                    pair_windows[window_size][a, b] += 1
                    pair_windows[window_size][b, a] += 1
        
        past_draws.append(current_draw)
        
        for window_size in freq_windows:
            if len(past_draws) <= window_size:
                for k in current_draw:
                    freq_windows[window_size][k] += 1
            else:
                oldest = past_draws[-window_size - 1]
                for k in oldest:
                    freq_windows[window_size][k] -= 1
                for k in current_draw:
                    freq_windows[window_size][k] += 1
    
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)
    
    # Standardize features
    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_live_scaled = scaler.transform(X_live)
    
    model.fit(X_train_scaled, y_train)
    
    # Predict on live features
    proba = model.predict_proba(X_live_scaled)
    prob_1 = proba[:, 1]
    
    # Select top 7
    top7_indices = np.argsort(prob_1)[-7:][::-1]
    selected = sorted([numbers[i] for i in top7_indices])
    
    metrics = calculate_selection_metrics(selected)
    
    model_names = {
        "logistic": "Logistic Regression",
        "rf": "Random Forest",
        "gb": "Gradient Boosting"
    }
    
    print(format_output(metrics, args.game, model_names.get(args.model, args.model)))
    
    print("\nTop 20 numbers by predicted probability:")
    sorted_indices = np.argsort(prob_1)[::-1]
    for rank, idx in enumerate(sorted_indices[:20], 1):
        marker = " ★" if numbers[idx] in selected else ""
        print(f"  {rank:2d}. {numbers[idx]:2d}  (p={prob_1[idx]:.4f}){marker}")


if __name__ == "__main__":
    main()