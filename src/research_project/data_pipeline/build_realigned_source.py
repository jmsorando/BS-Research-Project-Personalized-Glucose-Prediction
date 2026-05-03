#!/usr/bin/env python3
"""
Generate an updated patient_extract CSV with CGM-realigned meal times.

Reads the original patient extract and corrected_meal_times output, maps each
food item row to its meal bundle, and applies the time shift from realignment.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from research_project import config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = cfg.SOURCE_DIR
OUT = cfg.OUTPUT_DATA_DIR


def _resolve_source_extract() -> Path:
    if cfg.PIPELINE_INPUTS_META.exists():
        try:
            with open(cfg.PIPELINE_INPUTS_META, "r", encoding="utf-8") as f:
                meta = json.load(f)
            diary_file = str(meta.get("diary_file", "")).strip()
            if diary_file:
                candidate = SRC / diary_file
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    return cfg.discover_latest_extract(SRC)


def main():
    source_path = _resolve_source_extract()
    print("Loading source CSV...")
    source = pd.read_csv(source_path, low_memory=False)
    source = source.copy()
    print(f"  Source file: {source_path.name}")
    print(f"  {len(source)} rows, {len(source.columns)} columns")

    print("Loading corrected_meal_times_ALL.csv...")
    corrected = pd.read_csv(OUT / "corrected_meal_times_ALL.csv", low_memory=False)
    print(f"  {len(corrected)} meal bundles")

    source["Patient Id"] = source["Patient Id"].astype(str).str.strip()
    source["_date"] = pd.to_datetime(source["Date"], format="mixed", dayfirst=False).dt.date

    def parse_time_to_dt(row):
        try:
            pts = str(row["Time consumed at"]).split(":")
            h = int(pts[0])
            m = int(pts[1]) if len(pts) > 1 else 0
            d = row["_date"]
            return datetime(d.year, d.month, d.day, h, m)
        except Exception:
            return pd.NaT

    source["_parsed_time"] = source.apply(parse_time_to_dt, axis=1)
    corrected["myfood24_id"] = corrected["myfood24_id"].astype(str).str.strip()
    corrected["date"] = pd.to_datetime(corrected["date"]).dt.date
    corrected["reported_time"] = pd.to_datetime(corrected["reported_time"], errors="coerce")
    corrected["corrected_time"] = pd.to_datetime(corrected["corrected_time"], errors="coerce")
    corrected["time_shift_min"] = pd.to_numeric(corrected["time_shift_min"], errors="coerce").fillna(0.0)

    bundle_lookup = defaultdict(list)
    for _, row in corrected.iterrows():
        key = (str(row["myfood24_id"]), row["date"])
        foods = set(f.strip() for f in str(row["food_items"]).split(";") if f.strip())
        bundle_lookup[key].append(
            {
                "meal_label": str(row["meal_label"]),
                "food_items_set": foods,
                "food_items_str": str(row["food_items"]),
                "reported_time": row["reported_time"],
                "time_shift_min": row["time_shift_min"],
                "n_items": int(row["n_items"]) if pd.notna(row["n_items"]) else 1,
                "_matched_count": 0,
            }
        )

    print("Matching source rows to meal bundles...")
    shifts = np.zeros(len(source))
    matched_count = 0
    for i, row in source.iterrows():
        key = (str(row["Patient Id"]), row["_date"])
        bundles = bundle_lookup.get(key, [])
        if not bundles:
            continue
        food_name = str(row["Food name"]).strip()
        meal = str(row["Meal"]).strip()
        parsed_time = row["_parsed_time"]
        best_bundle = None
        best_score = -1.0
        for b in bundles:
            score = 0.0
            if food_name in b["food_items_set"]:
                score += 100
            elif food_name in b["food_items_str"]:
                score += 50
            if meal == b["meal_label"]:
                score += 20
            if pd.notna(parsed_time) and pd.notna(b["reported_time"]):
                diff_min = abs((parsed_time - b["reported_time"]).total_seconds() / 60)
                score += max(0, 30 - diff_min / 5)
            if b["_matched_count"] >= b["n_items"]:
                score -= 50
            if score > best_score:
                best_score = score
                best_bundle = b
        if best_bundle is not None and best_score > 0:
            shifts[i] = best_bundle["time_shift_min"]
            best_bundle["_matched_count"] += 1
            matched_count += 1
    print(f"  Matched {matched_count}/{len(source)} rows")

    print("Applying time shifts...")
    new_times = []
    for i, row in source.iterrows():
        shift = shifts[i]
        parsed = row["_parsed_time"]
        if pd.notna(parsed) and shift != 0:
            corrected_dt = parsed + timedelta(minutes=shift)
            new_times.append(f"{corrected_dt.hour}:{corrected_dt.minute:02d}")
        else:
            new_times.append(row["Time consumed at"])
    source["Time consumed at"] = new_times
    source["time_shift_min"] = shifts
    source.drop(columns=["_date", "_parsed_time"], inplace=True)
    out_path = OUT / "patient_extract1602_realigned.csv"
    source.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.name}")
    print(f"  {len(source)} rows, {len(source.columns)} columns")
    shifted = (shifts != 0).sum()
    print(f"  {shifted} rows had time shifts applied")
    if shifted > 0:
        nonzero = shifts[shifts != 0]
        print(f"  Mean shift:   {nonzero.mean():.1f} min")
        print(f"  Median shift: {np.median(nonzero):.1f} min")
        print(f"  Range: [{nonzero.min():.1f}, {nonzero.max():.1f}] min")


if __name__ == "__main__":
    main()

