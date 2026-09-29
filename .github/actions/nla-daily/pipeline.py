#!/usr/bin/env python3
"""
GitHub Actions Daily NLA Pipeline
- Fetches completed draws from ghanayello.com
- Saves to Neon DB
- Generates prediction LOCALLY (no Vercel API dependency)
- Commits prediction to repo
- Time-aware: uses Ghana time (UTC+0)
"""

import os
import sys
import json
import random
from datetime import datetime, timezone, timedelta

# ─── TIME CONFIGURATION ───
GHANA_TZ = timezone.utc
DRAW_HOUR = 20
DRAW_MINUTE = 0

GAME_SCHEDULE = {
    0: "Monday Special", 1: "Lucky Tuesday", 2: "MidWeek",
    3: "Fortune Thursday", 4: "Friday Bonanza",
    5: "National Weekly", 6: "Sunday Aseda"
}

def now_ghana():
    return datetime.now(GHANA_TZ)

def get_game_for_date(dt=None):
    dt = dt or now_ghana()
    return GAME_SCHEDULE[dt.weekday()]

def get_previous_game(dt=None):
    dt = dt or now_ghana()
    prev_day = (dt - timedelta(days=1)).weekday()
    return GAME_SCHEDULE[prev_day]

def is_draw_completed(dt=None):
    dt = dt or now_ghana()
    return dt.hour >= DRAW_HOUR

def get_draw_date_for_game(game, reference=None):
    ref = reference or now_ghana()
    target_weekday = {v: k for k, v in GAME_SCHEDULE.items()}[game]
    days_diff = (target_weekday - ref.weekday()) % 7
    draw_date = ref + timedelta(days=days_diff)
    return draw_date.strftime('%Y-%m-%d')

# ─── PREDICTION LOGIC (LOCAL - NO EXTERNAL API) ───
NUMBERS = 90
TOP_K = 7

def get_draws_from_db(game_name: str = None):
    """Fetch draws from Neon DB"""
    import pg8000
    DATABASE_URL = os.getenv("DATABASE_URL")
    url = DATABASE_URL.replace("postgresql://", "")
    creds, rest = url.split("@", 1)
    user, pwd = creds.split(":", 1)
    host_port, db = rest.split("/", 1)[0], rest.split("/", 1)[1].split("?")[0]
    host, port = (host_port.split(":") + ["5432"])[:2]

    conn = pg8000.connect(user=user, password=pwd, host=host, port=int(port), database=db, ssl_context=True)
    cur = conn.cursor()

    if game_name:
        cur.execute("""SELECT seq_id, game_name, day_of_week, draw_date,
                      win_n1,win_n2,win_n3,win_n4,win_n5,win_n6,
                      mach_n1,mach_n2,mach_n3,mach_n4,mach_n5,mach_n6,event_id
                      FROM draws WHERE game_name=%s ORDER BY seq_id""", (game_name,))
    else:
        cur.execute("""SELECT seq_id, game_name, day_of_week, draw_date,
                      win_n1,win_n2,win_n3,win_n4,win_n5,win_n6,
                      mach_n1,mach_n2,mach_n3,mach_n4,mach_n5,mach_n6,event_id
                      FROM draws ORDER BY seq_id""")
    
    cols = [desc[0] for desc in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def gap_cold_baseline(game_name: str, top_k: int = TOP_K) -> list:
    """Gap (Cold) baseline heuristic - pure Python."""
    draws = get_draws_from_db(game_name)
    if not draws:
        return list(range(1, top_k + 1))
    
    draw_numbers = [[d["win_n1"], d["win_n2"], d["win_n3"], d["win_n4"], d["win_n5"]] for d in draws]
    
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


def gap_cold_with_probs(game_name: str, top_k: int = TOP_K):
    """Gap cold with equal probabilities"""
    nums = gap_cold_baseline(game_name, top_k)
    probs = [1.0 / top_k] * top_k
    return nums, probs


def calculate_metrics(numbers: list) -> dict:
    """Calculate selection metrics"""
    odd_count = sum(1 for n in numbers if n % 2 == 1)
    even_count = len(numbers) - odd_count
    high_count = sum(1 for n in numbers if n > 45)
    low_count = len(numbers) - high_count
    sum_digits = sum(numbers)
    average = sum_digits / len(numbers)
    spread = max(numbers) - min(numbers)
    return {
        "odd_count": odd_count, "even_count": even_count,
        "high_count": high_count, "low_count": low_count,
        "sum_digits": sum_digits, "average": average, "spread": spread
    }

# ─── MAIN PIPELINE ───
def main():
    now = now_ghana()
    print(f"🕐 Pipeline started: {now.isoformat()} (Ghana time)")
    print(f"🎯 Current game: {get_game_for_date(now)}")
    print(f"✅ Draw completed: {is_draw_completed(now)}")
    print(f"⏰ Draw time: {DRAW_HOUR}:{DRAW_MINUTE:02d}")

    # Determine which games to fetch
    yesterday_game = get_previous_game(now)
    yesterday_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
    games_to_fetch = [(yesterday_game, yesterday_date)]
    
    if is_draw_completed(now):
        today_game = get_game_for_date(now)
        today_date = now.strftime('%Y-%m-%d')
        games_to_fetch.append((today_game, today_date))
        print(f"📥 Fetching: {yesterday_game} ({yesterday_date}) + {today_game} ({today_date}) (draw completed)")
    else:
        print(f"📥 Fetching: {yesterday_game} ({yesterday_date}) only (draw not yet completed)")

    # Fetch and save each game
    saved_any = False
    for game, draw_date in games_to_fetch:
        print(f"\n🔍 Fetching {game} for {draw_date}...")
        result = fetch_and_save(game, draw_date)
        if result:
            saved_any = True

    # Generate prediction for TODAY'S game (the upcoming draw)
    today_game = get_game_for_date(now)
    prediction_date = get_draw_date_for_game(today_game, now)
    print(f"\n🔮 Generating LOCAL prediction for {today_game} ({prediction_date})...")
    
    prediction = generate_local_prediction(today_game, prediction_date)
    if not prediction:
        print("❌ Prediction failed")
        # Don't exit - continue with commit if we have predictions
        print("⚠️ Continuing without new prediction...")
    else:
        # Save prediction to repo
        save_prediction(prediction, today_game, prediction_date)
    print("✅ Pipeline complete")

    # Save prediction to repo
    save_prediction(prediction, today_game, prediction_date)
    print("✅ Pipeline complete")


def fetch_and_save(game, draw_date):
    """Fetch draw from ghanayello.com and save to Neon"""
    import requests
    import re
    from html.parser import HTMLParser

    # Fetch HTML with headers matching fetch_online.py (worked locally)
    data = {"data[Lottery][name]": game, "data[Lottery][date]": draw_date, "_method": "POST"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://www.ghanayello.com/lottery/results/history",
    }
    
    try:
        resp = requests.post("https://www.ghanayello.com/lottery/results/history", 
                            data=data, headers=headers, timeout=60)
        if resp.status_code != 200:
            print(f"  ❌ HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Fetch error: {e}")
        return False

    # Parse using same logic as fetch_online.py
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_table = False
            self.in_row = False
            self.in_cell = False
            self.cell_index = 0
            self.current_cell_data = ""
            self.current_row = {}
            self.draws = []
            self.header_passed = False
            self.capture_number = False
        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if tag == "table":
                self.in_table = True
                self.header_passed = False
            elif tag == "tr" and self.in_table:
                self.in_row = True
                self.cell_index = 0
                self.current_row = {}
            elif tag == "td" and self.in_row:
                self.in_cell = True
                self.current_cell_data = ""
                self.current_row[f"cell{self.cell_index}_title"] = attrs_dict.get("title", "")
            elif tag == "div" and self.in_cell:
                classes = attrs_dict.get("class", "")
                if "lotto_no_r" in classes:
                    self.current_row.setdefault("numbers", []).append("")
                    self.capture_number = True
        def handle_endtag(self, tag):
            if tag == "td" and self.in_cell:
                self.in_cell = False
                self.current_row[f"cell{self.cell_index}_data"] = self.current_cell_data.strip()
                self.cell_index += 1
                self.capture_number = False
            elif tag == "tr" and self.in_row:
                self.in_row = False
                if self.header_passed and self.current_row:
                    self.draws.append(self.current_row)
                self.header_passed = True
            elif tag == "table":
                self.in_table = False
        def handle_data(self, data):
            if self.in_cell:
                self.current_cell_data += data
            if self.capture_number and self.in_cell and "numbers" in self.current_row:
                if self.current_row["numbers"]:
                    self.current_row["numbers"][-1] += data.strip()

    parser = Parser()
    parser.feed(resp.text)

    if not parser.draws:
        print(f"  ⚠️ No draw data found for {game} {draw_date}")
        return False

    # Use cell DATA (content), not title attributes - matches fetch_online.py
    row = parser.draws[0]
    date_str = row.get("cell0_data", "")
    numbers_title = row.get("cell2_title", "")
    mach_title = row.get("cell3_title", "")
    
    win_m = re.search(r"Winning Numbers\s+([\d\-]+)", numbers_title)
    mach_m = re.search(r"Machine Numbers\s+([\d\-]+)", mach_title)
    
    if not win_m:
        print(f"  ⚠️ No winning numbers found")
        return False

    win = "-".join(win_m.group(1).split("-")[:5])
    mach = mach_m.group(1) if mach_m else ""

    # Save to Neon
    save_to_neon(game, draw_date, win, mach)
    return True


def save_to_neon(game, draw_date, win, mach):
    import pg8000

    DATABASE_URL = os.getenv("DATABASE_URL")
    url = DATABASE_URL.replace("postgresql://", "")
    creds, rest = url.split("@", 1)
    user, pwd = creds.split(":", 1)
    host_port, db = rest.split("/", 1)[0], rest.split("/", 1)[1].split("?")[0]
    host, port = (host_port.split(":") + ["5432"])[:2]

    win_nums = [int(x) for x in win.split("-")]
    mach_nums = [int(x) for x in mach.split("-")] if mach else [None]*5
    win_nums += [0]*(5-len(win_nums))
    mach_nums += [None]*(5-len(mach_nums))

    conn = pg8000.connect(user=user, password=pwd, host=host, port=int(port), database=db, ssl_context=True)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM draws WHERE game_name=%s AND draw_date=%s", (game, draw_date))
    if not cur.fetchone():
        cur.execute("SELECT COALESCE(MAX(seq_id),0)+1 FROM draws")
        seq = cur.fetchone()[0]
        cur.execute("""INSERT INTO draws (seq_id, game_name, day_of_week, draw_date,
                    win_n1,win_n2,win_n3,win_n4,win_n5,win_n6,
                    mach_n1,mach_n2,mach_n3,mach_n4,mach_n5,mach_n6,event_id)
                    VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s)""",
            (seq, game, {"Monday Special":"Monday","Lucky Tuesday":"Tuesday","MidWeek":"Wednesday",
                          "Fortune Thursday":"Thursday","Friday Bonanza":"Friday",
                          "National Weekly":"Saturday","Sunday Aseda":"Sunday"}[game],
             draw_date,
             win_nums[0],win_nums[1],win_nums[2],win_nums[3],win_nums[4],0,
             mach_nums[0],mach_nums[1],mach_nums[2],mach_nums[3],mach_nums[4],None,
             "Lotto Draw Event"))
        conn.commit()
        print(f"  ✅ Saved: {game} {draw_date} Win={win} Mach={mach}")
    else:
        print(f"  ⏭️ Already exists: {game} {draw_date}")
    cur.close()
    conn.close()


def generate_local_prediction(game, draw_date):
    """Generate prediction locally using gap_cold algorithm"""
    try:
        # Get numbers and probabilities
        numbers, probs = gap_cold_with_probs(game, TOP_K)
        
        # Calculate metrics
        metrics = calculate_metrics(numbers)
        metrics_str = f"Odd/Even: {metrics['odd_count']}/{metrics['even_count']} | High/Low: {metrics['high_count']}/{metrics['low_count']} | Sum: {metrics['sum_digits']} | Avg: {metrics['average']:.1f} | Spread: {metrics['spread']}"
        
        # Get latest draw for reference
        draws = get_draws_from_db(game)
        latest_draw = None
        if draws:
            latest = draws[-1]
            latest_draw = {
                "seq_id": latest["seq_id"],
                "draw_date": str(latest["draw_date"]),
                "numbers": [latest["win_n1"], latest["win_n2"], latest["win_n3"], latest["win_n4"], latest["win_n5"]],
                "game": game
            }
        
        prediction = {
            "date": draw_date,
            "day": datetime.now(timezone.utc).strftime("%A"),
            "game": game,
            "model": "gap_cold",
            "numbers": [str(n) for n in numbers],
            "metrics": metrics_str,
            "raw": {
                "game": game,
                "model": "gap_cold",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latest_draw": latest_draw,
                "next_draw_index": len(draws) if draws else 0,
                "top_numbers": [{"number": n, "probability": p, "rank": i+1} for i, (n, p) in enumerate(zip(numbers, probs))],
                "metrics": metrics,
                "disclaimer": "This is a statistical prediction for entertainment purposes only. Please verify results with official NLA sources."
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"  ✅ Prediction: {numbers}")
        print(f"  📊 Metrics: {metrics_str}")
        return prediction
        
    except Exception as e:
        print(f"  ❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_prediction(prediction, game, draw_date):
    import os
    os.makedirs("predictions", exist_ok=True)
    
    filepath = f"predictions/{draw_date}.json"
    with open(filepath, "w") as f:
        json.dump(prediction, f, indent=2)
    
    raw_path = f"predictions/{draw_date}_raw.json"
    with open(raw_path, "w") as f:
        json.dump(prediction["raw"], f, indent=2)
    
    print(f"  💾 Saved prediction: {filepath}")


if __name__ == "__main__":
    main()