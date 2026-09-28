#!/usr/bin/env python3
"""
Continuous Learning System for Ghana NLA 5/90
- Learns machine number influence
- Learns cross-day dependencies (Wed→Thu, etc.)
- Incremental retraining with concept drift detection
- Feature engineering for 5/90 mechanics
"""

import os
import sys
import time
import json
import logging
import schedule
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import numpy as np
import requests
import pg8000
import joblib
from html.parser import HTMLParser
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── CONFIG ───
GAMES = [
    "Monday Special", "Lucky Tuesday", "MidWeek", "Fortune Thursday",
    "Friday Bonanza", "National Weekly", "Sunday Aseda"
]

GAME_ORDER = ["Sunday Aseda", "Monday Special", "Lucky Tuesday", "MidWeek", 
              "Fortune Thursday", "Friday Bonanza", "National Weekly"]

GAME_DAY = {
    "Monday Special": 0, "Lucky Tuesday": 1, "MidWeek": 2,
    "Fortune Thursday": 3, "Friday Bonanza": 4,
    "National Weekly": 5, "Sunday Aseda": 6
}

DAY_TO_GAME = {v: k for k, v in GAME_DAY.items()}

API_URL = "https://vercel-deploy-phi-umber.vercel.app"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://www.ghanayello.com/lottery/results/history",
}

DB_URL = os.getenv("DATABASE_URL", "")

# ─── DATABASE ───
def get_db():
    if not DB_URL: raise RuntimeError("DATABASE_URL not set")
    url = DB_URL.replace("postgresql://", "")
    creds, rest = url.split("@", 1)
    user, pwd = creds.split(":", 1)
    host_port, db = rest.split("/", 1)
    host_port = host_port.split("?")[0]
    host, port = (host_port.split(":") + ["5432"])[:2]
    return pg8000.connect(user=user, password=pwd, host=host, port=int(port), database=db, ssl_context=True)

# ─── FEATURE ENGINEERING ───
class FeatureEngine:
    """Builds features that capture machine influence & cross-day dependencies"""
    
    def __init__(self):
        self.game_idx = {g: i for i, g in enumerate(GAMES)}
        
    def build_features(self, draws: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix X and target y from historical draws.
        Each row = one number (1-90) at one draw time.
        """
        if len(draws) < 100:
            return np.array([]), np.array([])
        
        draws = sorted(draws, key=lambda d: d['draw_date'])
        n_draws = len(draws)
        n_numbers = 90
        
        # Precompute game sequences
        game_seq = [self.game_idx.get(d['game_name'], 0) for d in draws]
        
        # Rolling caches for efficiency
        X_rows = []
        y_rows = []
        
        # Caches for rolling stats
        last_seen = {g: {n: -999 for n in range(1, 91)} for g in GAMES}
        freq_cache = {g: {w: {n: 0 for n in range(1, 91)} for w in [7, 14, 30, 60, 90]} for g in GAMES}
        pair_cache = {g: {w: np.zeros((91, 91), dtype=int) for w in [30, 90]} for g in GAMES}
        machine_freq = {g: {n: 0 for n in range(1, 91)} for g in GAMES}
        machine_pair = {g: np.zeros((91, 91), dtype=int) for g in GAMES}
        
        past_draws_by_game = {g: [] for g in GAMES}
        
        for t, draw in enumerate(draws):
            game = draw['game_name']
            g_idx = self.game_idx[game]
            win_nums = set(draw['win_numbers'])
            mach_nums = set(draw.get('machine_numbers', []))
            date = draw['draw_date']
            
            # Update caches with this draw
            for n in win_nums:
                last_seen[game][n] = t
            
            for w in [7, 14, 30, 60, 90]:
                for n in win_nums:
                    freq_cache[game][w][n] += 1
            
            # Machine number influence
            for n in mach_nums:
                machine_freq[game][n] += 1
                for m in mach_nums:
                    if n != m:
                        machine_pair[game][n, m] += 1
            
            # Pair co-occurrence
            for w in [30, 90]:
                for i, a in enumerate(win_nums):
                    for b in list(win_nums)[i+1:]:
                        pair_cache[game][w][a, b] += 1
                        pair_cache[game][w][b, a] += 1
            
            # Maintain rolling windows
            past_draws_by_game[game].append(draw)
            for w in [7, 14, 30, 60, 90]:
                if len(past_draws_by_game[game]) > w:
                    old = past_draws_by_game[game][-w-1]
                    for n in old['win_numbers']:
                        freq_cache[game][w][n] -= 1
            
            # Only create training rows after warmup
            if t < 50:
                continue
                
            prev_game = draws[t-1]['game_name'] if t > 0 else game
            prev_win = set(draws[t-1]['win_numbers'])
            prev_mach = set(draws[t-1].get('machine_numbers', []))
            prev_date = draws[t-1]['draw_date']
            
            # Cross-day features
            days_since = (date - prev_date).days if isinstance(date, datetime) else 1
            same_week = date.isocalendar().week == prev_date.isocalendar().week if isinstance(date, datetime) else False
            
            for n in range(1, 91):
                # Target
                y = 1 if n in win_nums else 0
                
                # ── Features ──
                feats = []
                
                # 1. Gap (draws since last hit in THIS game)
                gap = t - last_seen[game][n]
                feats.append(min(gap, 999) / 100.0)
                
                # 2. Frequency windows (this game)
                for w in [7, 14, 30, 60, 90]:
                    feats.append(freq_cache[game][w][n] / max(w, 1))
                
                # 3. Machine number frequency (this game)
                feats.append(machine_freq[game][n] / max(len(past_draws_by_game[game]), 1))
                
                # 4. Machine pair affinity (this game)
                if mach_nums:
                    feats.append(np.mean([machine_pair[game][n, m] for m in mach_nums]) / 10.0)
                else:
                    feats.append(0.0)
                
                # 5. Prev day hit (same game or different)
                feats.append(1.0 if n in prev_win else 0.0)
                
                # 6. Prev day machine hit
                feats.append(1.0 if n in prev_mach else 0.0)
                
                # 7. Pair co-occurrence with prev day numbers
                if prev_win:
                    feats.append(np.mean([pair_cache[game][30][n, p] for p in prev_win]) / 10.0)
                    feats.append(np.mean([pair_cache[game][90][n, p] for p in prev_win]) / 10.0)
                else:
                    feats.extend([0.0, 0.0])
                
                # 8. Static number properties
                feats.append(n / 90.0)  # normalized value
                feats.append(n % 2)     # odd/even
                feats.append(1.0 if n > 45 else 0.0)  # high/low
                feats.append(sum(int(d) for d in str(n)) / 18.0)  # digit sum normalized
                
                # 9. Cross-game features (prev game's stats)
                prev_g_idx = self.game_idx.get(prev_game, 0)
                feats.append(prev_g_idx / 6.0)
                
                # 10. Temporal features
                feats.append(days_since / 7.0)
                feats.append(1.0 if same_week else 0.0)
                feats.append(date.weekday() / 6.0 if isinstance(date, datetime) else 0.0)
                
                X_rows.append(feats)
                y_rows.append(y)
        
        if not X_rows:
            return np.array([]), np.array([])
            
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)

feature_engine = FeatureEngine()

# ─── INCREMENTAL MODEL ───
class IncrementalModel:
    """Online learning with concept drift detection"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.performance_window = deque(maxlen=100)
        self.drift_threshold = 0.15
        self.version = 0
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Incremental update"""
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        
        if self.model is None:
            self.model = SGDClassifier(
                loss='log_loss', penalty='l2', alpha=0.001,
                learning_rate='adaptive', eta0=0.01,
                random_state=42, n_jobs=-1
            )
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-8
        
        # Scale
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Update
        classes = np.array([0, 1])
        self.model.partial_fit(X_scaled, y, classes=classes)
        self.version += 1
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full((X.shape[0], 2), 0.5)
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Track performance for drift detection"""
        if self.model is None:
            return 0.5
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        proba = self.model.predict_proba(X_scaled)[:, 1]
        pred = (proba > 0.5).astype(int)
        acc = (pred == y).mean()
        self.performance_window.append(acc)
        
        # Drift detection
        if len(self.performance_window) >= 50:
            recent = np.mean(list(self.performance_window)[-20:])
            older = np.mean(list(self.performance_window)[-50:-20])
            if older - recent > self.drift_threshold:
                logger.warning(f"🚨 Drift detected in {self.name}: {older:.3f} → {recent:.3f}")
        
        return acc


# ─── CONTINUOUS LEARNER ───
class ContinuousLearner:
    def __init__(self):
        self.models = {
            'sgd': IncrementalModel('sgd'),
            'sgd_balanced': IncrementalModel('sgd_balanced'),
        }
        self.fe = FeatureEngine()
        self.fetcher = DataFetcher()
        self.running = False
        self.last_retrain = {}
        self.performance_log = []
        
    def start(self):
        self.running = True
        logger.info("🚀 Continuous Learner STARTED")
        
        # Schedules
        schedule.every().day.at("02:00").do(self.daily_cycle)
        schedule.every().hour.do(self.hourly_check)
        schedule.every(6).hours.do(self.retrain)
        schedule.every().day.at("03:00").do(self.evaluate)
        
        # Initial
        self.daily_cycle()
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)
    
    def stop(self):
        self.running = False
    
    def daily_cycle(self) -> dict:
        """2 AM UTC: Fetch yesterday's results → retrain → predict today → return results"""
        logger.info("📅 Starting daily cycle...")
        today_game = self._today_game()
        yesterday_game = self._yesterday_game()
        yesterday_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        today_date = datetime.utcnow().strftime("%Y-%m-%d")
        today_day = datetime.utcnow().strftime("%A")
        
        # 1. Fetch yesterday's results
        logger.info(f"📥 Fetching {yesterday_game} for {yesterday_date}...")
        results = self._fetch_results(yesterday_game, yesterday_date)
        
        yesterday_win = []
        yesterday_machine = []
        
        if results:
            row = results[0]
            win = self._extract_numbers(row.get('cell2_title', ''))
            mach = self._extract_numbers(row.get('cell3_title', ''))
            date_str = row.get('cell0_data', '')
            
            if win:
                self._save_to_db(yesterday_game, win, mach, date_str)
                logger.info(f"✅ {yesterday_game}: {win} | Machine: {mach}")
                yesterday_win = win
                yesterday_machine = mach
            else:
                logger.warning(f"No results for {yesterday_game}")
        
        # 2. Retrain all models
        self.retrain()
        
        # 3. Predict today
        prediction = self.predict_today()
        
        # 4. Evaluate performance
        self.evaluate()
        
        # Return structured results for GitHub Actions
        return {
            "game": self._today_game(),
            "day": datetime.utcnow().strftime("%A"),
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "numbers": prediction if prediction else [],
            "metrics": "Generated",
            "yesterday_game": yesterday_game,
            "yesterday_win": yesterday_win,
            "yesterday_machine": yesterday_machine
        }
    
    def retrain(self):
        """Full retrain with all historical data"""
        logger.info("🧠 Retraining models...")
        
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT game_name, draw_date, win_n1,win_n2,win_n3,win_n4,win_n5,win_n6,
                   mach_n1,mach_n2,mach_n3,mach_n4,mach_n5,mach_n6
            FROM draws ORDER BY draw_date
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()
        
        if len(rows) < 200:
            logger.warning("Not enough data for retrain")
            return
        
        # Convert to draws format
        draws = []
        for r in rows:
            win = [r[i] for i in range(2, 7) if r[i]]
            mach = [r[i] for i in range(8, 13) if r[i]]
            draws.append({
                'game_name': r[0], 'draw_date': r[1],
                'win_numbers': win, 'machine_numbers': mach
            })
        
        # Build features
        X, y = self.fe.build_features(draws)
        if len(X) == 0:
            return
        
        # Train each model
        for name, model in self.models.items():
            if name == 'sgd_balanced':
                # Oversample positive class
                pos = y == 1
                neg = y == 0
                pos_idx = np.where(pos)[0]
                neg_idx = np.where(neg)[0]
                # Balance 1:4 ratio
                n_pos = pos.sum()
                n_neg_needed = min(len(neg_idx), n_pos * 4)
                neg_sample = np.random.choice(neg_idx, n_neg_needed, replace=False)
                idx = np.concatenate([pos_idx, neg_sample])
                np.random.shuffle(idx)
                X_bal, y_bal = X[idx], y[idx]
                model.partial_fit(X_bal, y_bal)
            else:
                model.partial_fit(X, y)
        
        logger.info("✅ Retrain complete")
    
    def evaluate(self):
        """Evaluate on recent data"""
        logger.info("📊 Evaluating...")
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT game_name, draw_date, win_n1,win_n2,win_n3,win_n4,win_n5
            FROM draws ORDER BY draw_date DESC LIMIT 100
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()
        
        if len(rows) < 20:
            return
        
        draws = [{'game_name': r[0], 'draw_date': r[1], 'win_numbers': [r[i] for i in range(2,7) if r[i]]} 
                 for r in rows]
        draws.reverse()
        
        X, y = self.fe.build_features(draws)
        if len(X) == 0:
            return
        
        for name, model in self.models.items():
            acc = model.evaluate(X, y)
            logger.info(f"  {name}: accuracy = {acc:.3f}")
    
    def predict_today(self):
        today_game = self._today_game()
        logger.info(f"🎯 Predicting {today_game}...")
        
        # Get latest draws for features
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT game_name, draw_date, win_n1,win_n2,win_n3,win_n4,win_n5,
                   mach_n1,mach_n2,mach_n3,mach_n4,mach_n5
            FROM draws WHERE game_name = %s ORDER BY draw_date DESC LIMIT 200
        """, (today_game,))
        rows = cur.fetchall()
        cur.close(); conn.close()
        
        draws = []
        for r in rows:
            win = [r[i] for i in range(2, 7) if r[i]]
            mach = [r[i] for i in range(7, 12) if r[i]]
            draws.append({'game_name': r[0], 'draw_date': r[1], 'win_numbers': win, 'machine_numbers': mach})
        draws.reverse()
        
        X, _ = self.fe.build_features(draws)
        if len(X) == 0:
            return
        
        # Use last draw's features (90 numbers)
        last_X = X[-90:]
        
        # Ensemble prediction
        probs = np.mean([m.predict_proba(last_X)[:, 1] for m in self.models.values()], axis=0)
        top7 = np.argsort(probs)[-7:][::-1] + 1
        
        logger.info(f"🎯 {today_game} prediction: {list(top7)}")
        return [int(n) for n in top7]
    
    def _today_game(self) -> str:
        return DAY_TO_GAME[datetime.utcnow().weekday()]
    
    def _yesterday_game(self) -> str:
        return DAY_TO_GAME[(datetime.utcnow() - timedelta(days=1)).weekday()]
    
    def _fetch_results(self, game: str, date: str) -> List[Dict]:
        data = {"data[Lottery][name]": game, "data[Lottery][date]": date, "_method": "POST"}
        try:
            resp = requests.post("https://www.ghanayello.com/lottery/results/history",
                               data=data, headers=HEADERS, timeout=60)
            parser = HistoryParser()
            parser.feed(resp.text)
            return parser.draws
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return []
    
    def _extract_numbers(self, title: str) -> List[int]:
        m = re.search(r"(?:Winning|Machine)\s*Numbers?\s+([\d\-]+)", title)
        return [int(n) for n in m.group(1).split("-")] if m else []
    
    def _save_to_db(self, game: str, win: List[int], mach: List[int], date: str):
        if not DB_URL: return
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM draws WHERE game_name=%s AND draw_date=%s", (game, date))
        if not cur.fetchone():
            cur.execute("SELECT COALESCE(MAX(seq_id),0)+1 FROM draws")
            seq = cur.fetchone()[0]
            mach = mach + [None]*(5-len(mach))
            cur.execute("""INSERT INTO draws (seq_id, game_name, day_of_week, draw_date,
                        win_n1,win_n2,win_n3,win_n4,win_n5,win_n6,
                        mach_n1,mach_n2,mach_n3,mach_n4,mach_n5,mach_n6,event_id)
                    VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s)""",
                (seq, game, GAME_DAY[game], date,
                 *win, 0, *mach, None, "Lotto Draw Event"))
            conn.commit()
        cur.close(); conn.close()


class DataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def fetch(self, game: str, date: str = "") -> List[Dict]:
        data = {"data[Lottery][name]": game, "data[Lottery][date]": date, "_method": "POST"}
        try:
            resp = self.session.post("https://www.ghanayello.com/lottery/results/history", 
                                   data=data, headers=HEADERS, timeout=60)
            parser = HistoryParser()
            parser.feed(resp.text)
            return parser.draws
        except Exception as e:
            logger.error(f"Fetch {game}: {e}")
            return []


class HistoryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_idx = 0
        self.current = {}
        self.draws = []
        self.row_count = 0
    
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table": self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True; self.cell_idx = 0; self.current = {}
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.current[f"cell{self.cell_idx}_title"] = attrs.get("title", "")
            self.current[f"cell{self.cell_idx}_data"] = ""
        elif tag == "div" and self.in_cell:
            cls = attrs.get("class", "")
            if "lotto_no_r" in cls or "lotto_no_w" in cls:
                self.current.setdefault("numbers", []).append("")
    
    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False; self.cell_idx += 1
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if self.row_count > 0 and self.current: self.draws.append(self.current)
            self.row_count += 1
        elif tag == "table": self.in_table = False
    
    def handle_data(self, data):
        if self.in_cell:
            self.current[f"cell{self.cell_idx}_data"] = (self.current.get(f"cell{self.cell_idx}_data", "") + data).strip()


# ─── MAIN ───
if __name__ == "__main__":
    learner = ContinuousLearner()
    try:
        learner.start()
    except KeyboardInterrupt:
        learner.stop()
        logger.info("👋 Stopped")