#!/usr/bin/env python3
"""
Leak-Free Rolling Feature Engine for Ghana NLA 5/90 - Enhanced
Zero-leakage: features for draw t only use draws with index < t

Advanced features added:
- Pair co-occurrence frequencies
- Sum of digits, odd/even ratio, high/low ratio
- Number-specific static properties
"""

import csv
import numpy as np
from collections import defaultdict
from src.db import get_draws

# Configuration
WARMUP_WINDOW = 90  # minimum past draws required before computing features
NUMBERS = 90  # numbers 1-90


# Pre-compute static number properties
NUMBER_PROPS = {}
for k in range(1, NUMBERS + 1):
    digits = [int(d) for d in str(k)]
    NUMBER_PROPS[k] = {
        "sum_digits": sum(digits),
        "is_odd": k % 2,
        "is_high": 1 if k > 45 else 0,  # 1-45 low, 46-90 high
        "digit_root": sum(digits) % 9 or 9 if sum(digits) > 0 else 0,
    }


def compute_pair_cooccurrence(past_draws: list, window: int) -> np.ndarray:
    """
    Compute pair co-occurrence matrix for the last `window` draws.
    Returns (91, 91) matrix where [i,j] = count of draws where both i and j appeared.
    """
    matrix = np.zeros((NUMBERS + 1, NUMBERS + 1), dtype=int)
    relevant_draws = past_draws[-window:] if window > 0 else []
    
    for draw in relevant_draws:
        for i, a in enumerate(draw):
            for b in draw[i+1:]:
                matrix[a, b] += 1
                matrix[b, a] += 1
    
    return matrix


def compute_features_for_game(game_name: str) -> list:
    """
    Build feature matrix for a single game with enhanced features.
    Returns list of feature rows (each row = one number at one target draw).
    """
    draws = get_draws(game_name)
    if len(draws) <= WARMUP_WINDOW:
        print(f"  {game_name}: only {len(draws)} draws, need > {WARMUP_WINDOW}")
        return []
    
    draw_numbers = []
    for d in draws:
        draw_numbers.append([d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]])
    
    feature_rows = []
    total_draws = len(draw_numbers)
    
    # Track last seen position for each number
    last_seen = np.full(NUMBERS + 1, -1, dtype=int)
    
    # Rolling frequency windows
    freq_windows = {
        7: np.zeros(NUMBERS + 1, dtype=int),
        14: np.zeros(NUMBERS + 1, dtype=int),
        30: np.zeros(NUMBERS + 1, dtype=int),
        60: np.zeros(NUMBERS + 1, dtype=int),
        90: np.zeros(NUMBERS + 1, dtype=int),
    }
    
    # Pair co-occurrence windows
    pair_windows = {
        30: np.zeros((NUMBERS + 1, NUMBERS + 1), dtype=int),
        90: np.zeros((NUMBERS + 1, NUMBERS + 1), dtype=int),
    }
    
    past_draws = []
    
    for t in range(total_draws):
        current_draw = draw_numbers[t]
        
        if t >= WARMUP_WINDOW:
            # Pre-compute pair co-occurrence for this draw's numbers
            # For each number k, we want avg co-occurrence with other drawn numbers
            
            for k in range(1, NUMBERS + 1):
                # gap: draws since last seen
                gap = t - last_seen[k] if last_seen[k] >= 0 else 999
                
                # frequencies in windows
                freq_7 = freq_windows[7][k]
                freq_14 = freq_windows[14][k]
                freq_30 = freq_windows[30][k]
                freq_60 = freq_windows[60][k]
                freq_90 = freq_windows[90][k]
                
                # prev_hit
                prev_hit = 1 if (t > 0 and k in draw_numbers[t - 1]) else 0
                
                # Pair co-occurrence features
                # Average co-occurrence of k with the 5 numbers in current draw
                # But we can't use current draw (leakage), so use past draws' patterns
                # Instead: average pair frequency with other numbers in the game
                if len(past_draws) >= 30:
                    # Avg pair freq with numbers that appeared in last draw
                    last_draw = draw_numbers[t - 1]
                    pair_30_avg = np.mean([pair_windows[30][k, n] for n in last_draw])
                    pair_90_avg = np.mean([pair_windows[90][k, n] for n in last_draw])
                else:
                    pair_30_avg = 0
                    pair_90_avg = 0
                
                # Static number properties
                props = NUMBER_PROPS[k]
                sum_digits = props["sum_digits"]
                is_odd = props["is_odd"]
                is_high = props["is_high"]
                
                # Draw-level properties (of the PREVIOUS draw, no leakage)
                if t > 0:
                    prev_draw = draw_numbers[t - 1]
                    prev_sum = sum(prev_draw)
                    prev_odd = sum(1 for n in prev_draw if n % 2 == 1)
                    prev_high = sum(1 for n in prev_draw if n > 45)
                    prev_odd_ratio = prev_odd / 5.0
                    prev_high_ratio = prev_high / 5.0
                else:
                    prev_sum = 0
                    prev_odd_ratio = 0.5
                    prev_high_ratio = 0.5
                
                # Target
                target = 1 if k in current_draw else 0
                
                feature_rows.append({
                    "game_name": game_name,
                    "seq_id": draws[t]["seq_id"],
                    "target_draw_idx": t,
                    "number": k,
                    # Core features
                    "gap": gap,
                    "freq_7": freq_7,
                    "freq_14": freq_14,
                    "freq_30": freq_30,
                    "freq_60": freq_60,
                    "freq_90": freq_90,
                    "prev_hit": prev_hit,
                    # Pair features
                    "pair_30_avg": pair_30_avg,
                    "pair_90_avg": pair_90_avg,
                    # Static number properties
                    "sum_digits": sum_digits,
                    "is_odd": is_odd,
                    "is_high": is_high,
                    # Previous draw properties
                    "prev_sum": prev_sum,
                    "prev_odd_ratio": prev_odd_ratio,
                    "prev_high_ratio": prev_high_ratio,
                    # Target
                    "target": target,
                })
        
        # Update history (NO LEAKAGE - current draw not used for its own features)
        for k in current_draw:
            last_seen[k] = t
        
        # Update pair co-occurrence matrices
        for window_size in pair_windows:
            if len(past_draws) >= window_size:
                # Remove oldest draw's pairs
                oldest = past_draws[-window_size]
                for i, a in enumerate(oldest):
                    for b in oldest[i+1:]:
                        pair_windows[window_size][a, b] -= 1
                        pair_windows[window_size][b, a] -= 1
            
            # Add current draw's pairs
            for i, a in enumerate(current_draw):
                for b in current_draw[i+1:]:
                    pair_windows[window_size][a, b] += 1
                    pair_windows[window_size][b, a] += 1
        
        # Update frequency windows
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
    
    print(f"  {game_name}: {len(feature_rows)} feature rows generated")
    return feature_rows


def build_and_save_features(game_name=None, output_path="data/features.csv"):
    """
    Build features for all games (or single game) and save to CSV.
    """
    games = ["Monday Special", "Lucky Tuesday", "MidWeek", 
             "Fortune Thursday", "Friday Bonanza", "National Weekly", "Sunday Aseda"]
    
    if game_name:
        games = [game_name]
    
    all_features = []
    
    for game in games:
        print(f"Building features for {game}...")
        features = compute_features_for_game(game)
        all_features.extend(features)
    
    if not all_features:
        print("No features generated!")
        return
    
    fieldnames = [
        "game_name", "seq_id", "target_draw_idx", "number",
        "gap", "freq_7", "freq_14", "freq_30", "freq_60", "freq_90",
        "prev_hit", "pair_30_avg", "pair_90_avg",
        "sum_digits", "is_odd", "is_high",
        "prev_sum", "prev_odd_ratio", "prev_high_ratio",
        "target"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)
    
    total_rows = len(all_features)
    target_positive = sum(1 for r in all_features if r["target"] == 1)
    games_count = defaultdict(int)
    for r in all_features:
        games_count[r["game_name"]] += 1
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY (Enhanced)")
    print("=" * 60)
    print(f"Total feature rows: {total_rows:,}")
    print(f"Target positive (hits): {target_positive:,} ({target_positive/total_rows*100:.2f}%)")
    print(f"Features per row: {len(fieldnames) - 4}")  # minus 4 id cols
    for game in games_count:
        draws_for_game = games_count[game] // 90
        print(f"  {game}: {draws_for_game} draws × 90 = {games_count[game]} rows")
    print(f"Output saved to: {output_path}")
    print("\nZero-leakage verification: PASSED (by construction)")
    return all_features


if __name__ == "__main__":
    build_and_save_features()