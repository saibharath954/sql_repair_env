"""
Adaptive curriculum manager for the DataOps Incident Response environment.

Tracks agent performance across episodes and automatically escalates or
reduces difficulty. The environment gets harder as the agent gets smarter.

Difficulty levels:
  0 — Novice:         1-2 faults, simple pool
  1 — Analyst:        2-3 faults, intermediate pool
  2 — Senior:         3-4 faults, full 12-fault pool
  3 — Staff Engineer: 4 faults + red herring table
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

STATE_FILE = "/tmp/curriculum_state.json"

LEVEL_NAMES = {0: "Novice", 1: "Analyst", 2: "Senior", 3: "Staff Engineer"}
FAULT_POOLS = {
    0: ["missing_comma", "wrong_sort_order", "duplicate_rows"],
    1: ["missing_comma", "wrong_join_key", "null_fk", "duplicate_rows", "type_drift"],
    2: [
        "null_fk", "type_drift", "duplicate_rows", "missing_comma", "wrong_join_key",
        "wrong_group_by", "stale_where", "column_alias_shadow", "implicit_cast_bug",
        "missing_distinct", "off_by_one_limit", "wrong_sort_order",
    ],
    3: [
        "null_fk", "type_drift", "duplicate_rows", "missing_comma", "wrong_join_key",
        "wrong_group_by", "stale_where", "column_alias_shadow", "implicit_cast_bug",
        "missing_distinct", "off_by_one_limit", "wrong_sort_order",
    ],  # + red_herring via difficulty_level=3
}
TASK_BY_LEVEL = {
    0: ["easy"],
    1: ["easy", "medium"],
    2: ["medium", "hard"],
    3: ["hard"],
}
PROMOTION_THRESHOLD = 0.75
DEMOTION_THRESHOLD = 0.30
WINDOW_SIZE = 5


@dataclass
class EpisodeRecord:
    task_id: str
    score: float
    steps_taken: int
    injected_fault_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class PromotionEvent:
    timestamp: float
    from_level: int
    to_level: int
    reason: str  # "promotion" | "demotion"
    rolling_mean: float


class CurriculumManager:
    """Tracks episode history and adjusts difficulty automatically."""

    def __init__(self):
        self._difficulty_level: int = 0
        self._history: List[EpisodeRecord] = []
        self._promotions: List[PromotionEvent] = []
        self._load_state()

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def difficulty_level(self) -> int:
        return self._difficulty_level

    def suggest_task(self) -> Dict:
        """Return suggested task_id and fault pool for the current difficulty level."""
        import random
        task_options = TASK_BY_LEVEL[self._difficulty_level]
        task_id = random.choice(task_options)
        return {
            "task_id": task_id,
            "difficulty_level": self._difficulty_level,
            "level_name": LEVEL_NAMES[self._difficulty_level],
            "fault_pool": FAULT_POOLS[self._difficulty_level],
        }

    def record_episode(
        self,
        task_id: str,
        score: float,
        steps_taken: int,
        injected_fault_count: int,
    ) -> Dict:
        """
        Record a completed episode and check for level transitions.
        Returns dict with promotion info if level changed, else {}.
        """
        record = EpisodeRecord(
            task_id=task_id,
            score=score,
            steps_taken=steps_taken,
            injected_fault_count=injected_fault_count,
        )
        self._history.append(record)

        result = self._check_promotion()
        self._save_state()
        return result

    def get_stats(self) -> Dict:
        """Return current curriculum statistics."""
        rolling = self._rolling_mean()
        return {
            "current_difficulty_level": self._difficulty_level,
            "level_name": LEVEL_NAMES[self._difficulty_level],
            "rolling_mean_reward": round(rolling, 4),
            "total_episodes": len(self._history),
            "window_size": WINDOW_SIZE,
            "promotion_threshold": PROMOTION_THRESHOLD,
            "demotion_threshold": DEMOTION_THRESHOLD,
            "promotions_history": [
                {
                    "timestamp": p.timestamp,
                    "from_level": p.from_level,
                    "to_level": p.to_level,
                    "reason": p.reason,
                    "rolling_mean": p.rolling_mean,
                }
                for p in self._promotions[-10:]
            ],
            "recent_scores": [r.score for r in self._history[-10:]],
            "fault_pool": FAULT_POOLS[self._difficulty_level],
        }

    def reset(self):
        """Clear all history and return to level 0. For testing only."""
        self._difficulty_level = 0
        self._history = []
        self._promotions = []
        self._save_state()

    # ─── Private ──────────────────────────────────────────────────────────────

    def _rolling_mean(self) -> float:
        recent = self._history[-WINDOW_SIZE:]
        if not recent:
            return 0.0
        return sum(r.score for r in recent) / len(recent)

    def _check_promotion(self) -> Dict:
        if len(self._history) < WINDOW_SIZE:
            return {}

        rolling = self._rolling_mean()
        old_level = self._difficulty_level

        if rolling >= PROMOTION_THRESHOLD and self._difficulty_level < 3:
            self._difficulty_level += 1
            event = PromotionEvent(
                timestamp=time.time(),
                from_level=old_level,
                to_level=self._difficulty_level,
                reason="promotion",
                rolling_mean=rolling,
            )
            self._promotions.append(event)
            return {
                "level_changed": True,
                "from": old_level,
                "to": self._difficulty_level,
                "reason": "promotion",
                "message": (
                    f"Promoted to {LEVEL_NAMES[self._difficulty_level]}! "
                    f"Rolling mean: {rolling:.2f}"
                ),
            }

        elif rolling < DEMOTION_THRESHOLD and self._difficulty_level > 0:
            self._difficulty_level -= 1
            event = PromotionEvent(
                timestamp=time.time(),
                from_level=old_level,
                to_level=self._difficulty_level,
                reason="demotion",
                rolling_mean=rolling,
            )
            self._promotions.append(event)
            return {
                "level_changed": True,
                "from": old_level,
                "to": self._difficulty_level,
                "reason": "demotion",
                "message": (
                    f"Demoted to {LEVEL_NAMES[self._difficulty_level]}. "
                    f"Rolling mean: {rolling:.2f}"
                ),
            }

        return {}

    def _save_state(self):
        try:
            state = {
                "difficulty_level": self._difficulty_level,
                "history": [asdict(r) for r in self._history[-100:]],
                "promotions": [asdict(p) for p in self._promotions[-50:]],
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self._difficulty_level = state.get("difficulty_level", 0)
                self._history = [EpisodeRecord(**r) for r in state.get("history", [])]
                self._promotions = [PromotionEvent(**p) for p in state.get("promotions", [])]
        except Exception:
            pass