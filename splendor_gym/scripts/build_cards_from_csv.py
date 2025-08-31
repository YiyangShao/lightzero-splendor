import csv
import json
import argparse
import os
from typing import List, Dict, Any

BONUS_MAP = {
	"black": "onyx",
	"blue": "sapphire",
	"white": "diamond",
	"green": "emerald",
	"red": "ruby",
}

EXPECTED_HEADERS = [
	"Level",
	"Gem color",
	"PV",
	"Price",
	"Illustration",
	"(w)hite",
	"bl(u)e",
	"(g)reen",
	"(r)ed",
	"blac(k)",
]


def _find_header(reader: csv.reader) -> (List[str], List[List[str]]):
	rows = list(reader)
	header_idx = None
	for i, row in enumerate(rows):
		if len(row) >= 10 and row[0].strip() == "Level" and row[1].strip().startswith("Gem"):
			header_idx = i
			break
	if header_idx is None:
		raise ValueError("Could not find header row with 'Level' and 'Gem color'")
	header = rows[header_idx]
	data_rows = rows[header_idx + 1 :]
	return header, data_rows


def parse_cards(csv_path: str) -> List[Dict[str, Any]]:
	with open(csv_path, newline="", encoding="utf-8") as f:
		reader = csv.reader(f)
		header, data_rows = _find_header(reader)
		# Build column index map
		name_to_idx = {name: header.index(name) for name in EXPECTED_HEADERS if name in header}
		cards: List[Dict[str, Any]] = []
		current_tier = None
		current_bonus = None
		for row in data_rows:
			if not row or all((cell or "").strip() == "" for cell in row):
				continue
			level = (row[name_to_idx["Level"]] if "Level" in name_to_idx else "").strip()
			if level in ("1", "2", "3"):
				current_tier = int(level)
			gem_color = (row[name_to_idx["Gem color"]] if "Gem color" in name_to_idx else "").strip().lower()
			if gem_color in BONUS_MAP:
				current_bonus = BONUS_MAP[gem_color]
			# Parse cost columns
			cost: Dict[str, int] = {}
			has_cost = False
			for human, col in zip(["diamond", "sapphire", "emerald", "ruby", "onyx"], ["(w)hite", "bl(u)e", "(g)reen", "(r)ed", "blac(k)"]):
				val = row[name_to_idx[col]] if col in name_to_idx and name_to_idx[col] < len(row) else ""
				val = (val or "").strip()
				if val.isdigit():
					iv = int(val)
					if iv > 0:
						cost[human] = iv
						has_cost = True
			pv_str = (row[name_to_idx["PV"]] if "PV" in name_to_idx else "").strip()
			points = int(pv_str) if pv_str.isdigit() else 0
			if has_cost and current_tier is not None and current_bonus is not None:
				cards.append({
					"tier": current_tier,
					"points": points,
					"bonus": current_bonus,
					"cost": cost,
				})
		return cards


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--csv", required=True, help="Path to Splendor cards CSV (with headers)")
	parser.add_argument("--out", default=None, help="Output cards.json path (default engine/data/cards.json)")
	args = parser.parse_args()
	out = args.out or os.path.join(os.path.dirname(__file__), "..", "engine", "data", "cards.json")
	cards = parse_cards(args.csv)
	os.makedirs(os.path.dirname(out), exist_ok=True)
	with open(out, "w", encoding="utf-8") as f:
		json.dump(cards, f, ensure_ascii=False)
	print(f"Wrote {len(cards)} cards to {out}")


if __name__ == "__main__":
	main() 