#!/usr/bin/env python3
"""
Ensemble Models for Ghana NLA 5/90.
Soft-voting and rank-averaging ensembles combining LR, RF, GB.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


class BaseEnsemble:
    """Base class for ensemble methods."""
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Args:
            models: Dict of {model_name: model_instance}
            weights: Dict of {model_name: weight}, defaults to equal weights
        """
        self.models = models
        self.model_names = list(models.keys())
        
        if weights is None:
            self.weights = {name: 1.0 / len(models) for name in self.model_names}
        else:
            # Normalize weights
            total = sum(weights.get(name, 0) for name in self.model_names)
            self.weights = {name: weights.get(name, 0) / total for name in self.model_names}
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEnsemble":
        """Fit all base models."""
        for name, model in self.models.items():
            print(f"  Fitting {name}...")
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def get_base_probas(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predicted probabilities from all base models."""
        probas = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probas[name] = model.predict_proba(X)[:, 1]  # Class 1 probability
            else:
                # Fallback for models without predict_proba
                preds = model.predict(X)
                probas[name] = preds.astype(float)
        return probas


class SoftVotingEnsemble(BaseEnsemble):
    """
    Soft Voting Ensemble: Average predicted probabilities.
    Best for calibrated probability estimates.
    """
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns: (n_samples, 2) array with [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        base_probas = self.get_base_probas(X)
        n_samples = X.shape[0]
        
        # Weighted average of class 1 probabilities
        ensemble_prob_1 = np.zeros(n_samples)
        for name, prob_1 in base_probas.items():
            ensemble_prob_1 += self.weights[name] * prob_1
        
        prob_0 = 1 - ensemble_prob_1
        return np.column_stack([prob_0, ensemble_prob_1])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


class RankAveragingEnsemble(BaseEnsemble):
    """
    Rank Averaging Ensemble: Average rank positions across models.
    More robust to miscalibrated probabilities.
    """
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Convert average ranks to pseudo-probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        base_probas = self.get_base_probas(X)
        n_samples = X.shape[0]
        
        # Compute ranks for each model (1 = highest probability)
        ranks = {}
        for name, prob_1 in base_probas.items():
            # Rank in descending order (highest prob = rank 1)
            ranks[name] = n_samples - np.argsort(np.argsort(prob_1))
        
        # Weighted average of ranks
        avg_rank = np.zeros(n_samples)
        for name, rank in ranks.items():
            avg_rank += self.weights[name] * rank
        
        # Convert average rank to probability-like score
        # Higher avg_rank (better rank) → higher probability
        max_rank = len(self.model_names) * n_samples
        min_rank = len(self.model_names)
        
        # Normalize to [0, 1]
        prob_1 = (max_rank - avg_rank) / (max_rank - min_rank)
        prob_1 = np.clip(prob_1, 1e-6, 1 - 1e-6)
        
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray, top_k: int = 7) -> np.ndarray:
        """Predict top-k for each sample (for our use case: 90 numbers)."""
        proba = self.predict_proba(X)
        prob_1 = proba[:, 1]
        
        # For each sample, we need to select top_k numbers
        # This is handled at the application level
        return (prob_1 >= np.partition(prob_1, -top_k)[-top_k]).astype(int)


class WeightedSoftVotingEnsemble(SoftVotingEnsemble):
    """
    Soft Voting with learnable weights via validation.
    """
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray, 
                         metric: str = "log_loss") -> Dict[str, float]:
        """
        Optimize ensemble weights on validation set.
        """
        from scipy.optimize import minimize
        
        base_probas = self.get_base_probas(X_val)
        model_probas = np.column_stack([base_probas[name] for name in self.model_names])
        
        def objective(w):
            w = np.clip(w, 1e-6, 1.0)
            w = w / w.sum()
            ensemble_p = model_probas @ w
            ensemble_p = np.clip(ensemble_p, 1e-15, 1 - 1e-15)
            if metric == "log_loss":
                return -np.mean(y_val * np.log(ensemble_p) + (1 - y_val) * np.log(1 - ensemble_p))
            elif metric == "brier":
                return np.mean((ensemble_p - y_val) ** 2)
            return 0
        
        # Start with equal weights
        x0 = np.ones(len(self.model_names)) / len(self.model_names)
        bounds = [(1e-6, 1.0) for _ in self.model_names]
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
        
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, 
                         method='SLSQP', options={'maxiter': 100})
        
        if result.success:
            optimal_weights = {name: float(w) for name, w in zip(self.model_names, result.x)}
            self.weights = optimal_weights
            print(f"  Optimized weights: {optimal_weights}")
            return optimal_weights
        
        return self.weights


def create_default_ensemble() -> SoftVotingEnsemble:
    """Create default ensemble with LR, RF, GB."""
    from src.ml_models import SimpleLogisticRegression, SimpleRandomForest, SimpleGradientBoosting
    
    models = {
        "logistic": SimpleLogisticRegression(learning_rate=0.1, max_iter=300),
        "random_forest": SimpleRandomForest(n_estimators=15, max_depth=3, random_state=42),
        "gradient_boosting": SimpleGradientBoosting(n_estimators=20, learning_rate=0.1, 
                                                     max_depth=2, random_state=42)
    }
    
    return SoftVotingEnsemble(models)


def create_weighted_ensemble(weights: Dict[str, float] = None) -> SoftVotingEnsemble:
    """Create ensemble with custom weights."""
    ensemble = create_default_ensemble()
    if weights:
        ensemble.weights = weights
    return ensemble


# Evaluation helper
def evaluate_ensemble(ensemble: BaseEnsemble, X: np.ndarray, y: np.ndarray, 
                      top_k: int = 7) -> Dict[str, float]:
    """Evaluate ensemble on test set."""
    proba = ensemble.predict_proba(X)
    prob_1 = proba[:, 1]
    
    # Rank-based metrics
    sorted_indices = np.argsort(prob_1)[::-1]
    top_k_indices = sorted_indices[:top_k]
    
    hits = y[top_k_indices].sum()
    precision_at_k = hits / top_k
    
    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y, prob_1)
    except:
        auc = 0.5
    
    return {
        "hits_at_k": int(hits),
        "precision_at_k": float(precision_at_k),
        "auc": float(auc),
        "top_k_indices": top_k_indices.tolist()
    }


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    X = np.random.randn(200, 15)
    y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5 > 0).astype(int)
    
    print("Testing SoftVotingEnsemble...")
    ensemble = create_default_ensemble()
    ensemble.fit(X, y)
    
    proba = ensemble.predict_proba(X[:10])
    print(f"  Probas shape: {proba.shape}")
    print(f"  Class 1 probs: {proba[:, 1]}")
    
    print("\nTesting RankAveragingEnsemble...")
    rank_ensemble = RankAveragingEnsemble(ensemble.models)
    rank_ensemble.fit(X, y)
    
    proba = rank_ensemble.predict_proba(X[:10])
    print(f"  Probas shape: {proba.shape}")
    print(f"  Class 1 probs: {proba[:, 1]}")
    
    print("\nEnsemble module ready!")