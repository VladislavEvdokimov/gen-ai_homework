"""Convert reviews.csv → input/reviews.txt"""
import csv
from pathlib import Path

csv_path = Path("starter/input/reviews.csv")
txt_path = Path("input/reviews.txt")
txt_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = [row[0].strip() for row in reader if row and row[0].strip()]

with open(txt_path, "w", encoding="utf-8") as f:
    for i, text in enumerate(rows, 1):
        f.write(f"Review {i}: {text}\n\n")

print(f"Done: {len(rows)} reviews → {txt_path}")
