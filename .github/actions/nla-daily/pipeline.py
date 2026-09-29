#!/usr/bin/env python3
"""
GitHub Actions Daily NLA Pipeline
- Fetches completed draws from ghanayello.com
- Saves to Neon DB
- Generates prediction via Vercel API
- Commits prediction to repo
- Time-aware: uses Ghana time (UTC+0)
"""

import os
import sys
import json
import subprocess
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

# ─── MAIN ───
def main():
    now = now_ghana()
    print(f"🕐 Pipeline started: {now.isoformat()} (Ghana time)")
    print(f"🎯 Current game: {get_game_for_date(now)}")
    print(f"✅ Draw completed: {is_draw_completed(now)}")
    print(f"⏰ Draw time: {DRAW_HOUR}:{DRAW_MINUTE:02d}")

    # Determine which games to fetch
    # 1. Yesterday's game (always - fetch with yesterday's date)
    # 2. Today's game IF draw already completed (fetch with today's date)
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
    print(f"\n🔮 Generating prediction for {today_game} ({prediction_date})...")
    
    prediction = generate_prediction(today_game, prediction_date)
    if not prediction:
        print("❌ Prediction failed")
        sys.exit(1)

    # Save prediction to repo
    save_prediction(prediction, today_game, prediction_date)
    print("✅ Pipeline complete")

def fetch_and_save(game, draw_date):
    """Fetch draw from ghanayello.com and save to Neon"""
    import requests
    import re
    from html.parser import HTMLParser

    # Fetch HTML
    data = {"data[Lottery][name]": game, "data[Lottery][date]": draw_date, "_method": "POST"}
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://www.ghanayello.com/lottery/results/history",
        "User-Agent": "Mozilla/5.0",
    }
    resp = requests.post("https://www.ghanayello.com/lottery/results/history", 
                        data=data, headers=headers, timeout=60)
    if resp.status_code != 200:
        print(f"  ❌ HTTP {resp.status_code}")
        return False

    # Parse
    class Parser(HTMLParser):
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

    parser = Parser()
    parser.feed(resp.text)

    if not parser.draws:
        print(f"  ⚠️ No draw data found for {game} {draw_date}")
        return False

    row = parser.draws[0]
    win_title = row.get("cell2_title", "")
    mach_title = row.get("cell3_title", "")
    
    win_m = re.search(r"Winning Numbers\s+([\d\-]+)", win_title)
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
    import os

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

def generate_prediction(game, draw_date):
    import requests
    API_URL = "https://vercel-deploy-phi-umber.vercel.app"
    
    resp = requests.post(f"{API_URL}/predict", 
        headers={"Content-Type": "application/json"},
        json={"game": game, "model": "gap_cold", "top_k": 7},
        timeout=30)
    
    if resp.status_code != 200:
        print(f"  ❌ Prediction failed: {resp.text}")
        return None
    
    data = resp.json()
    numbers = [str(n["number"]) for n in data["top_numbers"]]
    metrics = data["metrics"]
    metrics_str = f"Odd/Even: {metrics['odd_count']}/{metrics['even_count']} | High/Low: {metrics['high_count']}/{metrics['low_count']} | Sum: {metrics['sum_digits']} | Avg: {metrics['average']:.1f} | Spread: {metrics['spread']}"
    
    return {
        "date": draw_date,
        "day": datetime.now(timezone.utc).strftime("%A"),
        "game": game,
        "model": "gap_cold",
        "numbers": numbers,
        "metrics": metrics_str,
        "raw": data,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }

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