#!/usr/bin/env python3
"""
Baseline strategy functions for Ghana NLA 5/90 backtesting.
Each function accepts a draw's 90-number feature set and returns 7 selected numbers.
"""

import random
from typing import List, Dict, Any


def random_top7(features: List[Dict[str, Any]], seed: int = None) -> List[int]:
    """
    Returns 7 distinct random numbers from 1-90.
    Features are ignored (true random baseline).
    """
    if seed is not None:
        random.seed(seed)
    return sorted(random.sample(range(1, 91), 7))


def frequency_top7(features: List[Dict[str, Any]], feature_col: str = "freq_30") -> List[int]:
    """
    Returns top 7 numbers sorted by highest rolling frequency.
    feature_col: one of 'freq_7', 'freq_14', 'freq_30', 'freq_60', 'freq_90'
    """
    # Sort by frequency descending, break ties by number
    sorted_features = sorted(features, key=lambda x: (-x[feature_col], x["number"]))
    return sorted([f["number"] for f in sorted_features[:7]])


def gap_top7(features: List[Dict[str, Any]], mode: str = "cold") -> List[int]:
    """
    Returns top 7 numbers sorted by gap.
    mode='cold': longest gap (numbers overdue)
    mode='hot': shortest gap (numbers recently drawn)
    """
    if mode == "cold":
        # Largest gap first
        sorted_features = sorted(features, key=lambda x: (-x["gap"], x["number"]))
    else:  # hot
        # Smallest gap first (but gap > 0 to avoid current draw)
        sorted_features = sorted(features, key=lambda x: (x["gap"], x["number"]))
    return sorted([f["number"] for f in sorted_features[:7]])


def prev_hit_top7(features: List[Dict[str, Any]]) -> List[int]:
    """
    Returns top 7 numbers that were drawn in previous draw (prev_hit=1),
    then fills with highest frequency if needed.
    """
    # First priority: prev_hit = 1
    prev_hits = [f for f in features if f["prev_hit"] == 1]
    prev_hits.sort(key=lambda x: (-x["freq_30"], x["number"]))
    
    selected = [f["number"] for f in prev_hits]
    
    # Fill remaining with highest frequency
    if len(selected) < 7:
        others = [f for f in features if f["prev_hit"] == 0]
        others.sort(key=lambda x: (-x["freq_30"], x["number"]))
        for f in others:
            if len(selected) >= 7:
                break
            selected.append(f["number"])
    
    return sorted(selected[:7])


def combined_top7(features: List[Dict[str, Any]]) -> List[int]:
    """
    Combined heuristic: weighted score from freq_30, gap, prev_hit.
    """
    scored = []
    for f in features:
        # Normalize features roughly
        freq_score = f["freq_30"] / 30.0  # max possible is 30
        gap_score = min(f["gap"], 90) / 90.0  # cap at 90
        prev_score = f["prev_hit"]
        
        # Weighted combination (favor cold + frequency)
        score = 0.4 * freq_score + 0.4 * gap_score + 0.2 * prev_score
        scored.append((score, f["number"]))
    
    scored.sort(key=lambda x: (-x[0], x[1]))
    return sorted([num for _, num in scored[:7]])


def evaluate_selection(selected: List[int], actual: List[int]) -> Dict[str, int]:
    """
    Evaluate a selection of 7 numbers against actual 5 drawn numbers.
    Returns hit count and hit details.
    """
    selected_set = set(selected)
    actual_set = set(actual)
    hits = selected_set & actual_set
    return {
        "hit_count": len(hits),
        "hits": sorted(hits),
        "selected": sorted(selected),
        "actual": sorted(actual),
        "success": len(hits) >= 2  # threshold: at least 2 hits
    }


# Registry of all baselines
BASELINES = {
    "Random": random_top7,
    "Frequency (freq_7)": lambda f: frequency_top7(f, "freq_7"),
    "Frequency (freq_14)": lambda f: frequency_top7(f, "freq_14"),
    "Frequency (freq_30)": lambda f: frequency_top7(f, "freq_30"),
    "Frequency (freq_60)": lambda f: frequency_top7(f, "freq_60"),
    "Frequency (freq_90)": lambda f: frequency_top7(f, "freq_90"),
    "Gap (Cold)": lambda f: gap_top7(f, "cold"),
    "Gap (Hot)": lambda f: gap_top7(f, "hot"),
    "Prev Hit + Freq": prev_hit_top7,
    "Combined Heuristic": combined_top7,
}


if __name__ == "__main__":
    # Quick test
    test_features = [
        {"number": i, "freq_7": i % 5, "freq_30": i % 10, "gap": 90 - i, "prev_hit": 1 if i % 7 == 0 else 0}
        for i in range(1, 91)
    ]
    
    for name, func in BASELINES.items():
        result = func(test_features)
        print(f"{name}: {result}")