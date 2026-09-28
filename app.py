#!/usr/bin/env python3
"""
FastAPI REST API for Ghana NLA 5/90 Predictions.
Minimal version for Vercel deployment (no numpy/ML dependencies).
"""

import os
import json
import logging
import random
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Application settings from environment variables."""
    database_url: str = Field(..., env="DATABASE_URL")
    model_type: str = Field(default="gap_cold", env="MODEL_TYPE")
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
        return {"type": "string", "enum": ["gap_cold", "frequency", "random"]}
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema
        valid_models = ["gap_cold", "frequency", "random"]
        return core_schema.with_info_plain_validator_function(
            lambda v, _info: v if v in valid_models else (_ for _ in ()).throw(ValueError(f"Invalid model. Must be one of: {valid_models}")),
            serialization=core_schema.to_string_ser_schema(),
        )


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    game: GameName
    model: ModelName = ModelName("gap_cold")
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
    "model_loaded": False,
    "last_training": None,
    "draw_counts": {}
}


# ============================================================================
# Core Functions (Pure Python - No NumPy)
# ============================================================================

NUMBERS = 90
TOP_K = 7


def get_latest_draw_state(game_name: str) -> Tuple[List[Dict], int, List[int]]:
    """Get latest draw state for a game."""
    draws = get_draws(game_name)
    if not draws:
        raise ValueError(f"No draws found for {game_name}")
    
    latest = draws[-1]
    actual_numbers = [latest["n1"], latest["n2"], latest["n3"], latest["n4"], latest["n5"]]
    next_draw_idx = len([d for d in draws if d["game_name"] == game_name])
    
    return draws, next_draw_idx, actual_numbers


def gap_cold_baseline(game_name: str, top_k: int = TOP_K) -> List[int]:
    """Gap (Cold) baseline heuristic - pure Python."""
    draws = get_draws(game_name)
    draw_numbers = [[d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]] for d in draws]
    
    # Compute gaps
    last_seen = {k: -1 for k in range(1, NUMBERS + 1)}
    for t, current_draw in enumerate(draw_numbers):
        for k in current_draw:
            last_seen[k] = t
    
    # Numbers with longest gaps
    gaps = []
    for k in range(1, NUMBERS + 1):
        gap = 999 if last_seen[k] == -1 else len(draw_numbers) - last_seen[k]
        gaps.append((gap, k))
    
    gaps.sort(reverse=True)
    return sorted([k for _, k in gaps[:top_k]])


def frequency_baseline(game_name: str, top_k: int = TOP_K, window: int = 30) -> List[int]:
    """Frequency baseline - most frequent numbers in recent window."""
    draws = get_draws(game_name)
    draw_numbers = [[d["n1"], d["n2"], d["n3"], d["n4"], d["n5"]] for d in draws]
    
    # Count frequencies in recent window
    recent = draw_numbers[-window:] if len(draw_numbers) >= window else draw_numbers
    freq = {k: 0 for k in range(1, NUMBERS + 1)}
    for draw in recent:
        for k in draw:
            freq[k] += 1
    
    # Sort by frequency descending
    sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted([k for k, _ in sorted_nums[:top_k]])


def random_baseline(top_k: int = TOP_K) -> List[int]:
    """Random selection baseline."""
    return sorted(random.sample(range(1, NUMBERS + 1), top_k))


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


# ============================================================================
# API Dependencies
# ============================================================================


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
        draws = get_draws("Monday Special")
        db_status = f"connected ({len(draws)} draws)"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        database=db_status,
        model_loaded=True  # Always true for baseline models
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
                    "draw_date": l.get("draw_date"),
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
    return ModelInfo(
        name="gap_cold",
        type="baseline",
        features=0,
        training_draws=0,
        last_updated=None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Generate top-N predictions for a game using baseline models.
    """
    game_name = request.game
    model_type = request.model
    top_k = request.top_k
    
    logger.info(f"Prediction request: game={game_name}, model={model_type}, top_k={top_k}")
    
    try:
        # Get latest draw state
        draws, next_draw_idx, latest_actual = get_latest_draw_state(game_name)
        
        # Select numbers based on model
        if model_type == "gap_cold":
            selected = gap_cold_baseline(game_name, top_k)
        elif model_type == "frequency":
            selected = frequency_baseline(game_name, top_k)
        elif model_type == "random":
            selected = random_baseline(top_k)
        else:
            selected = gap_cold_baseline(game_name, top_k)
        
        # Calculate metrics
        metrics = calculate_metrics(selected)
        
        # Build response
        top_numbers = []
        for rank, num in enumerate(selected, 1):
            top_numbers.append(NumberPrediction(
                number=num,
                probability=1.0 / top_k,  # Equal probability for baselines
                rank=rank
            ))
        
        # Latest draw info
        latest_draw_info = {
            "seq_id": draws[-1]["seq_id"],
            "draw_date": draws[-1].get("draw_date"),
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