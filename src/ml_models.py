#!/usr/bin/env python3
"""
Simple ML models using only numpy (no scikit-learn dependency).
Includes: Logistic Regression, Random Forest, Gradient Boosting
"""

import numpy as np
from typing import List, Tuple, Optional


class SimpleLogisticRegression:
    """
    Logistic Regression with gradient descent.
    Binary classification for each number (hit or not).
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.classes_ = np.array([0, 1])
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLogisticRegression":
        n_samples, n_features = X.shape
        
        # Initialize weights (with bias term)
        self.weights = np.zeros(n_features + 1)
        
        prev_loss = float('inf')
        X_bias = np.hstack([X, np.ones((n_samples, 1))])
        
        for i in range(self.max_iter):
            z = X_bias @ self.weights
            y_pred = self._sigmoid(z)
            
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            
            grad = X_bias.T @ (y_pred - y) / n_samples
            self.weights -= self.learning_rate * grad
            
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        
        self.bias = self.weights[-1]
        self.weights = self.weights[:-1]
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        prob_1 = self._sigmoid(z)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class SimpleDecisionTree:
    """
    Simple decision tree for classification/regression.
    Used as base for Random Forest and Gradient Boosting.
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 10, 
                 min_samples_leaf: int = 5, regression: bool = False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.regression = regression
        self.tree = None
    
    def _impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0
        if self.regression:
            return np.var(y)
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)  # Gini
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        n_samples, n_features = X.shape
        best_impurity = float('inf')
        best_feature = None
        best_threshold = None
        
        # Random subset of features
        n_feat_subset = max(1, int(np.sqrt(n_features)))
        feature_indices = np.random.choice(n_features, n_feat_subset, replace=False)
        
        for feature_idx in feature_indices:
            values = X[:, feature_idx]
            percentiles = np.percentile(values, [25, 50, 75])
            for threshold in percentiles:
                left_mask = values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                imp_left = self._impurity(y[left_mask])
                imp_right = self._impurity(y[right_mask])
                impurity = (np.sum(left_mask) * imp_left + np.sum(right_mask) * imp_right) / n_samples
                
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        n_samples = len(y)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            (not self.regression and len(np.unique(y)) == 1)):
            if self.regression:
                return {"leaf": True, "value": float(np.mean(y))}
            return {"leaf": True, "value": np.bincount(y).argmax()}
        
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            if self.regression:
                return {"leaf": True, "value": float(np.mean(y))}
            return {"leaf": True, "value": np.bincount(y).argmax()}
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDecisionTree":
        self.tree = self._build_tree(X, y, 0)
        return self
    
    def _predict_one(self, x: np.ndarray, node: dict) -> float:
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        if self.regression:
            # For regression, return as probability-like
            proba = np.zeros((len(preds), 2))
            proba[:, 1] = preds
            proba[:, 0] = 1 - preds
            return proba
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds.astype(int)] = 1.0
        return proba


class SimpleRandomForest:
    """
    Random Forest with simple decision trees.
    """
    
    def __init__(self, n_estimators: int = 15, max_depth: int = 3, 
                 min_samples_split: int = 30, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRandomForest":
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        self.trees = []
        
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_proba = np.zeros((X.shape[0], 2))
        for tree in self.trees:
            all_proba += tree.predict_proba(X)
        return all_proba / len(self.trees)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class SimpleGradientBoosting:
    """
    Gradient Boosting for binary classification.
    Uses decision trees as weak learners.
    Optimized for speed in Termux environment.
    """
    
    def __init__(self, n_estimators: int = 20, learning_rate: float = 0.1,
                 max_depth: int = 2, min_samples_split: int = 30,
                 subsample: float = 0.8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.init_pred = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _logodds(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleGradientBoosting":
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        pos_rate = np.mean(y)
        self.init_pred = self._logodds(pos_rate)
        
        y_pred_logodds = np.full(n_samples, self.init_pred)
        
        for i in range(self.n_estimators):
            # Subsample
            if self.subsample < 1.0:
                indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                X_sub = X[indices]
                y_sub = y[indices]
                pred_sub = y_pred_logodds[indices]
            else:
                X_sub = X
                y_sub = y
                pred_sub = y_pred_logodds
            
            # Pseudo-residuals for log-loss
            probs = self._sigmoid(pred_sub)
            residuals = y_sub - probs
            
            # Fit regression tree to residuals
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                regression=True
            )
            tree.fit(X_sub, residuals)
            
            update = tree.predict(X)
            y_pred_logodds += self.learning_rate * update
            
            self.trees.append(tree)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_pred_logodds = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            y_pred_logodds += self.learning_rate * tree.predict(X)
        prob_1 = self._sigmoid(y_pred_logodds)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    print("Testing Logistic Regression...")
    lr = SimpleLogisticRegression(learning_rate=0.1, max_iter=100)
    lr.fit(X, y)
    proba = lr.predict_proba(X[:5])
    print(f"Proba shape: {proba.shape}")
    print(f"First 5 probs: {proba[:, 1]}")
    
    print("\nTesting Random Forest...")
    rf = SimpleRandomForest(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)
    proba = rf.predict_proba(X[:5])
    print(f"Proba shape: {proba.shape}")
    print(f"First 5 probs: {proba[:, 1]}")
    
    print("\nTesting Gradient Boosting...")
    gb = SimpleGradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=2, random_state=42)
    gb.fit(X, y)
    proba = gb.predict_proba(X[:5])
    print(f"Proba shape: {proba.shape}")
    print(f"First 5 probs: {proba[:, 1]}")