#!/usr/bin/env python3
"""
FastAPI REST API for Ghana NLA 5/90 Predictions.
Wraps the prediction pipeline into a lightweight HTTP service.
"""

import os
import json
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import uvicorn

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.db import init_db, get_draws
from src.ml_models import (SimpleLogisticRegression, SimpleRandomForest, 
                            SimpleGradientBoosting)
from src.ensemble import create_default_ensemble, SoftVotingEnsemble
from src.features import NUMBER_PROPS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Application settings from environment variables."""
    database_url: str = Field(..., env="DATABASE_URL")
    model_type: str = Field(default="logistic", env="MODEL_TYPE")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


# ============================================================================
# Pydantic Schemas
# ============================================================================

class GameName(str):
    """Valid game names."""
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string", "enum": [
            "Monday Special", "Lucky Tuesday", "MidWeek",
            "Fortune Thursday", "Friday Bonanza", 
            "National Weekly", "Sunday Aseda"
        ]}
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema
        valid_games = [
            "Monday Special", "Lucky Tuesday", "MidWeek",
            "Fortune Thursday", "Friday Bonanza", 
            "National Weekly", "Sunday Aseda"
        ]
        return core_schema.with_info_plain_validator_function(
            lambda v, _info: v if v in valid_games else (_ for _ in ()).throw(ValueError(f"Invalid game. Must be one of: {valid_games}")),
            serialization=core_schema.to_string_ser_schema(),
        )


class ModelName(str):
    """Valid model names."""
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string", "enum": ["logistic", "rf", "gb", "ensemble", "gap_cold"]}
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema
        valid_models = ["logistic", "rf", "gb", "ensemble", "gap_cold"]
        return core_schema.with_info_plain_validator_function(
            lambda v, _info: v if v in valid_models else (_ for _ in ()).throw(ValueError(f"Invalid model. Must be one of: {valid_models}")),
            serialization=core_schema.to_string_ser_schema(),
        )


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    game: GameName
    model: ModelName = ModelName("logistic")
    top_k: int = Field(default=7, ge=5, le=10)
    include_probabilities: bool = Field(default=True)
    include_metrics: bool = Field(default=True)


class NumberPrediction(BaseModel):
    """Single number prediction with probability."""
    number: int = Field(..., ge=1, le=90)
    probability: float = Field(..., ge=0.0, le=1.0)
    rank: int


class SelectionMetrics(BaseModel):
    """Descriptive metrics for the selected numbers."""
    numbers: List[int]
    odd_count: int
    even_count: int
    high_count: int  # 46-90
    low_count: int   # 1-45
    sum_digits: int
    average: float
    spread: int  # max - min


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    game: str
    model: str
    timestamp: datetime
    latest_draw: Dict[str, Any]
    next_draw_index: int
    top_numbers: List[NumberPrediction]
    metrics: Optional[SelectionMetrics] = None
    disclaimer: str = "This is a statistical prediction for entertainment purposes only. Please verify results with official NLA sources."


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    database: str
    model_loaded: bool


class GameInfo(BaseModel):
    """Game information."""
    name: str
    day_of_week: str
    total_draws: int
    latest_draw: Optional[Dict[str, Any]] = None


class GamesResponse(BaseModel):
    """List of available games."""
    games: List[GameInfo]


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    features: int
    training_draws: int
    last_updated: Optional[datetime] = None


# ============================================================================
# Global State
# ============================================================================

app_state = {
    "model": None,
    "model_name": None,
    "scaler": None,
    "last_training": None,
    "draw_counts": {}
}


# ============================================================================
# Feature Scaler (for production use)
# ============================================================================

class FeatureScaler:
    """Simple feature standardization for production."""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
    
    def transform(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================================
# Core Functions
# ============================================================================

FEATURE_COLS = [
    "gap", "freq_7", "freq_14", "freq_30", "freq_60", "freq_90", "prev_hit",
    "pair_30_avg", "pair_90_avg",
    "sum_digits", "is_odd", "is_high",
    "prev_sum", "prev_odd_ratio", "prev_high_ratio"
]
TARGET_COL = "target"
NUMBER_COL = "number"
WARMUP_WINDOW = 90
NUMBERS = 90


def get_latest_draw_state(game_name: str) -> Tuple[List[Dict], int, List[int]]:
    """Get latest draw state for a game."""
    draws = get_draws(game_name)
    if not draws:
        raise ValueError(f"No draws found for {game_name}")
    
    latest = draws[-1]
    actual_numbers = [latest["n1"], latest["n2"], latest["n3"], latest["n4"], latest["n5"]]
    next_draw_idx = len([d for d in draws if d["game_name"] == game_name])
    
    return draws, next_draw_idx, actual_numbers


def compute_live_features(game_name: str, draws: list, target_draw_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute feature matrix for the NEXT draw."""
    import numpy as np
    
    draw_numbers = []
    for d in draws:
        draw_numbers.append([d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]])
    
    last_seen = np.full(NUMBERS + 1, -1, dtype=int)
    freq_windows = {w: np.zeros(NUMBERS + 1, dtype=int) for w in [7, 14, 30, 60, 90]}
    pair_windows = {w: np.zeros((NUMBERS + 1, NUMBERS + 1), dtype=int) for w in [30, 90]}
    past_draws = []
    
    for t, current_draw in enumerate(draw_numbers):
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
    
    # Features for NEXT draw
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
        
        features.append([
            gap, freq_7, freq_14, freq_30, freq_60, freq_90, prev_hit,
            pair_30_avg, pair_90_avg,
            props["sum_digits"], props["is_odd"], props["is_high"],
            prev_sum, prev_odd_ratio, prev_high_ratio
        ])
        numbers.append(k)
    
    return np.array(features, dtype=float), np.array(numbers, dtype=int)


def train_model(game_name: str, model_type: str) -> Tuple[Any, FeatureScaler]:
    """Train the specified model on all historical data."""
    draws, _, _ = get_latest_draw_state(game_name)
    draw_numbers = [[d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]] for d in draws]
    
    # Build training data
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
    
    # Scale features
    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    if model_type == "logistic":
        model = SimpleLogisticRegression(learning_rate=0.1, max_iter=300)
    elif model_type == "rf":
        model = SimpleRandomForest(n_estimators=15, max_depth=3, random_state=42)
    elif model_type == "gb":
        model = SimpleGradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=2, random_state=42)
    elif model_type == "ensemble":
        model = create_default_ensemble()
    elif model_type == "gap_cold":
        return None, scaler  # Baseline, no model needed
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    return model, scaler


def calculate_metrics(selected: List[int]) -> SelectionMetrics:
    """Calculate descriptive metrics for selected numbers."""
    selected = sorted(selected)
    odd_count = sum(1 for n in selected if n % 2 == 1)
    even_count = len(selected) - odd_count
    high_count = sum(1 for n in selected if n > 45)
    low_count = len(selected) - high_count
    sum_digits = sum(sum(int(d) for d in str(n)) for n in selected)
    avg = sum(selected) / len(selected)
    spread = max(selected) - min(selected)
    
    return SelectionMetrics(
        numbers=selected,
        odd_count=odd_count,
        even_count=even_count,
        high_count=high_count,
        low_count=low_count,
        sum_digits=sum_digits,
        average=avg,
        spread=spread
    )


def gap_cold_baseline(game_name: str, top_k: int = 7) -> List[int]:
    """Gap (Cold) baseline heuristic."""
    draws = get_draws(game_name)
    draw_numbers = [[d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]] for d in draws]
    
    # Compute gaps
    last_seen = np.full(NUMBERS + 1, -1, dtype=int)
    for t, current_draw in enumerate(draw_numbers):
        for k in current_draw:
            last_seen[k] = t
    
    # Numbers with longest gaps
    gaps = [(999 if last_seen[k] == -1 else len(draw_numbers) - last_seen[k], k) 
            for k in range(1, NUMBERS + 1)]
    gaps.sort(reverse=True)
    return sorted([k for _, k in gaps[:top_k]])


# ============================================================================
# API Dependencies
# ============================================================================

def get_model(model_type: str = "logistic"):
    """Dependency to get/load model."""
    if app_state["model"] is None or app_state["model_name"] != model_type:
        logger.info(f"Loading {model_type} model...")
        model, scaler = train_model("Monday Special", model_type)  # Default game for training
        app_state["model"] = model
        app_state["scaler"] = scaler
        app_state["model_name"] = model_type
        app_state["last_training"] = datetime.now()
    
    return app_state["model"], app_state["scaler"]


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Ghana NLA 5/90 Prediction API",
    description="REST API for predicting top numbers in Ghana NLA 5/90 lottery draws",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db_status = "connected"
    try:
        # Quick DB check
        draws = get_draws("Monday Special")
        db_status = f"connected ({len(draws)} draws)"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        database=db_status,
        model_loaded=app_state["model"] is not None
    )


@app.get("/games", response_model=GamesResponse)
async def list_games():
    """List all available games with metadata."""
    game_names = [
        "Monday Special", "Lucky Tuesday", "MidWeek",
        "Fortune Thursday", "Friday Bonanza", 
        "National Weekly", "Sunday Aseda"
    ]
    day_map = {
        "Monday Special": "Monday",
        "Lucky Tuesday": "Tuesday",
        "MidWeek": "Wednesday",
        "Fortune Thursday": "Thursday",
        "Friday Bonanza": "Friday",
        "National Weekly": "Saturday",
        "Sunday Aseda": "Sunday"
    }
    
    games = []
    for name in game_names:
        try:
            draws = get_draws(name)
            latest = None
            if draws:
                l = draws[-1]
                latest = {
                    "seq_id": l["seq_id"],
                    "date": f"{l['n1']}-{l['n2']}-{l['n3']}-{l['n4']}-{l['n5']}",  # placeholder
                    "numbers": [l["n1"], l["n2"], l["n3"], l["n4"], l["n5"]]
                }
            games.append(GameInfo(
                name=name,
                day_of_week=day_map[name],
                total_draws=len(draws),
                latest_draw=latest
            ))
        except Exception as e:
            logger.error(f"Error getting info for {name}: {e}")
            games.append(GameInfo(name=name, day_of_week=day_map[name], total_draws=0))
    
    return GamesResponse(games=games)


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get information about the currently loaded model."""
    model_name = app_state.get("model_name") or "none"
    return ModelInfo(
        name=model_name,
        type=model_name,
        features=len(FEATURE_COLS),
        training_draws=0,  # Would need to track this
        last_updated=app_state.get("last_training")
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Generate top-N predictions for a game.
    """
    game_name = request.game
    model_type = request.model
    top_k = request.top_k
    
    logger.info(f"Prediction request: game={game_name}, model={model_type}, top_k={top_k}")
    
    try:
        # Get latest draw state
        draws, next_draw_idx, latest_actual = get_latest_draw_state(game_name)
        
        # Handle baseline models
        if model_type == "gap_cold":
            selected = gap_cold_baseline(game_name, top_k)
            prob_1 = np.zeros(NUMBERS)  # No probabilities for baseline
        else:
            # Get or train model
            model, scaler = get_model(model_type)
            
            # Compute live features
            X_live, numbers = compute_live_features(game_name, draws, next_draw_idx)
            X_live_scaled = scaler.transform(X_live)
            
            # Predict
            proba = model.predict_proba(X_live_scaled)
            prob_1 = proba[:, 1]
            
            # Select top-k
            top_indices = np.argsort(prob_1)[-top_k:][::-1]
            selected = sorted([numbers[i] for i in top_indices])
        
        # Calculate metrics
        metrics = calculate_metrics(selected)
        
        # Build response
        top_numbers = []
        if model_type != "gap_cold":
            for rank, idx in enumerate(top_indices, 1):
                top_numbers.append(NumberPrediction(
                    number=numbers[idx],
                    probability=float(prob_1[idx]),
                    rank=rank
                ))
        else:
            for rank, num in enumerate(selected, 1):
                top_numbers.append(NumberPrediction(
                    number=num,
                    probability=0.0,
                    rank=rank
                ))
        
        # Latest draw info
        latest_draw_info = {
            "seq_id": draws[-1]["seq_id"],
            "numbers": latest_actual,
            "game": game_name
        }
        
        return PredictionResponse(
            game=game_name,
            model=model_type,
            timestamp=datetime.now(),
            latest_draw=latest_draw_info,
            next_draw_index=next_draw_idx,
            top_numbers=top_numbers,
            metrics=metrics if request.include_metrics else None
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """
    Generate predictions for multiple games at once.
    """
    results = []
    for req in requests:
        try:
            result = await predict(req, BackgroundTasks())
            results.append(result)
        except Exception as e:
            results.append({"game": req.game, "error": str(e)})
    return {"predictions": results}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False
    )