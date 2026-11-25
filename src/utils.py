"""
Shared Utilities and Enterprise Attendance Logging & Analytics.
Fulfills:
  - AGENT.md §8: Individual Innovative Feature:
      * Automated Enterprise Attendance Logging to CSV.
      * Anti-Spam Check-in cooldown (prevents multi-logging within N seconds).
      * Daily Employee Well-being / Mood Analytics (HR Sentiment Index).
  - AGENT.md §14: Visualization and evaluation reporting helpers.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class AttendanceLogger:
    """
    Automated Enterprise Attendance & Mood Logger.
    Persists check-ins to outputs/attendance_logs/attendance_YYYY-MM-DD.csv.
    Includes check-in cooldown to prevent spamming logs when standing before the camera.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/attendance_logs",
        cooldown_seconds: float = 30.0
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = cooldown_seconds
        self._recent_checkins: Dict[str, float] = {}

    def get_log_file(self, date_str: Optional[str] = None) -> Path:
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        return self.output_dir / f"attendance_{date_str}.csv"

    def log_checkin(
        self,
        name: str,
        liveness_score: float,
        is_live: bool,
        similarity_score: float,
        emotion: str,
        emotion_confidence: float
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        if name != "Unknown" and name != "Spoof Attack":
            last_time = self._recent_checkins.get(name, 0.0)
            if now - last_time < self.cooldown_seconds:
                return None
            self._recent_checkins[name] = now

        now_dt = datetime.now()
        date_str = now_dt.strftime("%Y-%m-%d")
        time_str = now_dt.strftime("%H:%M:%S")

        status = "VERIFIED_PRESENT"
        if not is_live:
            status = "SECURITY_ALERT_SPOOF"
        elif name == "Unknown":
            status = "UNENROLLED_VISITOR"

        record = {
            "date": date_str,
            "time": time_str,
            "employee_name": name,
            "checkin_status": status,
            "liveness_status": "PASS" if is_live else "FAIL",
            "liveness_score": round(float(liveness_score), 4),
            "similarity_score": round(float(similarity_score), 4),
            "detected_emotion": emotion,
            "emotion_confidence": round(float(emotion_confidence), 4)
        }

        log_path = self.get_log_file(date_str)
        file_exists = log_path.is_file()

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        return record

    def get_daily_analytics(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        log_path = self.get_log_file(date_str)
        if not log_path.is_file():
            return {
                "date": date_str or datetime.now().strftime("%Y-%m-%d"),
                "total_events": 0,
                "unique_employees": 0,
                "verified_present": [],
                "spoof_attempts": 0,
                "mood_distribution": {},
                "workplace_sentiment_index": "N/A"
            }

        records = []
        with log_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        total_events = len(records)
        verified_records = [r for r in records if r["checkin_status"] == "VERIFIED_PRESENT"]
        spoof_records = [r for r in records if r["checkin_status"] == "SECURITY_ALERT_SPOOF"]

        unique_employees = sorted(list({r["employee_name"] for r in verified_records}))
        emotions = [r["detected_emotion"] for r in verified_records]
        emotion_counts = dict(Counter(emotions))

        positive_count = emotion_counts.get("Happy", 0)
        neutral_count = emotion_counts.get("Neutral", 0)
        total_verified = len(verified_records)
        if total_verified > 0:
            sentiment_score = ((positive_count * 1.0) + (neutral_count * 0.5)) / total_verified * 100.0
            sentiment_index = f"{sentiment_score:.1f}% Positive/Neutral"
        else:
            sentiment_index = "N/A"

        return {
            "date": date_str or datetime.now().strftime("%Y-%m-%d"),
            "total_events": total_events,
            "unique_employees_count": len(unique_employees),
            "verified_employees": unique_employees,
            "spoof_attacks_blocked": len(spoof_records),
            "mood_distribution": emotion_counts,
            "workplace_sentiment_index": sentiment_index
        }
