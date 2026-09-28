# Phase 5 Implementation Plan

## Directory Structure
```
src/
├── __init__.py
├── backtest.py              # Basic walk-forward (Phase 3)
├── backtest_advanced.py     # Advanced ML evaluation (Phase 4)
├── baselines.py             # Baseline strategies
├── db.py                    # Neon PostgreSQL abstraction
├── features.py              # Enhanced feature engineering (16 features)
├── fetch_online.py          # Web scraper for historical data
├── generate.py              # Production CLI for next-draw predictions
├── ml_models.py             # LR, RF, GB, Ensemble (Phase 4+5)
├── permutation.py           # Fast permutation test (Phase 3)
├── permutation_test.py      # Full permutation test with ML (Phase 5)
├── ensemble.py              # NEW: Soft-voting/rank-averaging ensemble
├── threshold_optimizer.py   # NEW: Walk-forward threshold tuning
└── app.py                   # NEW: FastAPI REST API endpoint

Dockerfile                   # NEW: Containerization
requirements.txt             # NEW: Python dependencies
```

## Component Specifications

### 1. `src/ensemble.py` - Ensemble Predictions
- **SoftVotingEnsemble**: Average predicted probabilities across LR, RF, GB
- **RankAveragingEnsemble**: Average rank positions, break ties by probability
- **WeightedEnsemble**: Configurable weights per model (default: equal)
- **API**: `fit(models_dict, X, y)`, `predict_proba(X)` → ensemble probabilities

### 2. `src/permutation_test.py` - Full Permutation Testing
- Label-shuffling across evaluation draws (100+ iterations)
- Test all models: LR, RF, GB, Ensemble, baselines
- Calculate p-values vs empirical null
- Output: significance table, save results to `data/permutation_results.json`

### 3. `src/threshold_optimizer.py` - Threshold/Rank Optimization
- Per-game probability cutoff tuning (grid search: 0.01-0.20)
- Per-game rank cutoff (top 5, 6, 7, 8, 9, 10)
- Walk-forward validation of threshold selection
- Output: optimal thresholds per game, save to `data/optimal_thresholds.json`

### 4. `app.py` - FastAPI REST API
- **Endpoints**:
  - `POST /predict` - Generate top-7 for a game
  - `GET /games` - List available games
  - `GET /health` - Health check
  - `GET /model/info` - Model metadata
- **Schemas**: Pydantic models for request/response
- **Config**: Environment-based (DATABASE_URL, MODEL_PATH)
- **Dockerfile**: Multi-stage build, python:3.11-slim

### 5. `requirements.txt` - Dependencies
```
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0
pg8000==1.31.5
python-dotenv==1.0.0
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.3.2
joblib==1.3.2
```

---

Let me start implementing each component.