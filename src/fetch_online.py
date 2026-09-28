#!/usr/bin/env python3
"""
Ghana NLA 5/90 Historical Data Fetcher
Scrapes historical draw data from ghanayello.com - fetches all history per game
"""

import requests
import csv
import re
import time
from html.parser import HTMLParser
from datetime import datetime
from typing import List, Dict, Optional


# Target games with their standard names and day of week
TARGET_GAMES = {
    "Monday Special": "Monday",
    "Lucky Tuesday": "Tuesday",
    "MidWeek": "Wednesday",
    "Fortune Thursday": "Thursday",
    "Friday Bonanza": "Friday",
    "National Weekly": "Saturday",
    "Sunday Aseda": "Sunday",
}

BASE_URL = "https://www.ghanayello.com/lottery/results/history"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://www.ghanayello.com/lottery/results/history",
}


class DrawTableParser(HTMLParser):
    """Parse the draw results table from the HTML response."""
    
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


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string like '14 September, 2026 - Monday' to datetime."""
    try:
        date_part = date_str.split(" - ")[0].strip()
        return datetime.strptime(date_part, "%d %B, %Y")
    except Exception:
        return None


def extract_numbers_from_title(title: str) -> List[int]:
    """Extract 5 winning numbers from title like 'Winning Numbers 63-60-57-55-16'."""
    match = re.search(r"Winning Numbers\s+([\d\-]+)", title)
    if match:
        nums_str = match.group(1)
        nums = [int(n) for n in nums_str.split("-")]
        return nums
    return []


def validate_draw(numbers: List[int]) -> bool:
    """Validate draw: exactly 5 numbers, each 1-90, no duplicates."""
    if len(numbers) != 5:
        return False
    if any(n < 1 or n > 90 for n in numbers):
        return False
    if len(set(numbers)) != 5:
        return False
    return True


def fetch_draws_for_game(game: str, session: requests.Session) -> List[Dict]:
    """Fetch all historical draws for a specific game."""
    data = {
        "data[Lottery][name]": game,
        "data[Lottery][date]": "",  # Empty = all time
        "_method": "POST",
    }
    
    try:
        resp = session.post(BASE_URL, data=data, headers=HEADERS, timeout=60)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}")
            return []
        
        parser = DrawTableParser()
        parser.feed(resp.text)
        
        results = []
        for row in parser.draws:
            # Use cell DATA (content), not title attributes
            date_str = row.get("cell0_data", "")
            game_str = row.get("cell1_data", "")
            numbers_title = row.get("cell2_title", "")
            
            dt = parse_date(date_str)
            if not dt:
                continue
            
            numbers = extract_numbers_from_title(numbers_title)
            if not validate_draw(numbers):
                continue
            
            # Extract game name from cell data (contains link text)
            # e.g. "Monday Special Results"
            normalized_game = game_str.replace(" Results", "")
            if normalized_game not in TARGET_GAMES:
                continue
            
            results.append({
                "date": dt,
                "game_name": normalized_game,
                "day_of_week": TARGET_GAMES[normalized_game],
                "numbers": numbers,
            })
        
        return results
    
    except Exception as e:
        print(f"  Error: {e}")
        return []


def main():
    print("=" * 60)
    print("Ghana NLA 5/90 Historical Data Fetcher")
    print("=" * 60)
    print(f"Fetching all history for {len(TARGET_GAMES)} games...")
    
    all_draws = []
    seen = set()
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    for game in TARGET_GAMES.keys():
        print(f"\nFetching {game}...", end=" ", flush=True)
        draws = fetch_draws_for_game(game, session)
        
        new_count = 0
        for d in draws:
            key = (d["game_name"], d["date"].strftime("%Y-%m-%d"))
            if key not in seen:
                seen.add(key)
                all_draws.append(d)
                new_count += 1
        
        print(f"got {len(draws)} draws ({new_count} new unique)")
        time.sleep(0.5)  # Be polite
    
    # Sort by date (oldest first)
    all_draws.sort(key=lambda x: x["date"])
    
    # Assign sequential seq_id
    for i, draw in enumerate(all_draws, 1):
        draw["seq_id"] = i
    
    # Save to CSV
    output_path = "data/draws.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "game_name", "day_of_week", "draw_date", "n1", "n2", "n3", "n4", "n5"])
        for draw in all_draws:
            writer.writerow([
                draw["seq_id"],
                draw["game_name"],
                draw["day_of_week"],
                draw["date"].strftime("%Y-%m-%d"),
                draw["numbers"][0],
                draw["numbers"][1],
                draw["numbers"][2],
                draw["numbers"][3],
                draw["numbers"][4],
            ])
    
    # Print execution report
    print("\n" + "=" * 60)
    print("EXECUTION REPORT")
    print("=" * 60)
    print(f"Total rows saved: {len(all_draws)}")
    print("\nCounts per game:")
    for game in TARGET_GAMES.keys():
        count = sum(1 for d in all_draws if d["game_name"] == game)
        print(f"  {game}: {count}")
    
    if all_draws:
        print(f"\nDate range: {all_draws[0]['date'].strftime('%Y-%m-%d')} to {all_draws[-1]['date'].strftime('%Y-%m-%d')}")
    
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()