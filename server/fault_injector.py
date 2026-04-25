"""
Stochastic fault injector for the DataOps Incident Response environment.

At reset() time, randomly selects 2-4 faults from a pool of 12 and applies them
to the episode's in-memory SQLite database and/or the broken_query string.

This is the core innovation: no two episodes are the same, so the agent cannot
memorize fixes — it must genuinely reason about what broke.
"""

import random
import re
import sqlite3
from enum import Enum
from typing import Dict, List, Optional, Tuple


class FaultType(Enum):
    # Data-level faults (corrupt the database)
    NULL_FK = "null_fk"
    TYPE_DRIFT = "type_drift"
    DUPLICATE_ROWS = "duplicate_rows"

    # Query-level faults (break the SQL query)
    MISSING_COMMA = "missing_comma"
    WRONG_JOIN_KEY = "wrong_join_key"
    WRONG_GROUP_BY = "wrong_group_by"
    STALE_WHERE = "stale_where"
    COLUMN_ALIAS_SHADOW = "column_alias_shadow"
    IMPLICIT_CAST_BUG = "implicit_cast_bug"
    MISSING_DISTINCT = "missing_distinct"
    OFF_BY_ONE_LIMIT = "off_by_one_limit"
    WRONG_SORT_ORDER = "wrong_sort_order"


# Which faults are data-level (vs query-level)
_DATA_FAULTS = frozenset({FaultType.NULL_FK, FaultType.TYPE_DRIFT, FaultType.DUPLICATE_ROWS})


class FaultInjector:
    """
    Randomly selects and applies faults to an episode's SQLite database and broken query.

    Design principles:
    - Always inject at least 1 data fault + 1 query fault so agent must diagnose both
    - Faults are deterministic given the same seed (for reproducibility)
    - Hints are specific to the actual faults injected (not generic)
    - Easy to extend: add a new FaultType and a handler method
    """

    LEVEL_POOLS = {
        0: [FaultType.MISSING_COMMA, FaultType.WRONG_SORT_ORDER, FaultType.DUPLICATE_ROWS],
        1: [
            FaultType.MISSING_COMMA, FaultType.WRONG_JOIN_KEY, FaultType.NULL_FK,
            FaultType.DUPLICATE_ROWS, FaultType.TYPE_DRIFT,
        ],
        2: list(FaultType),
        3: list(FaultType),  # all 12 + red herring table
    }

    def __init__(self, difficulty_level: int = 2):
        self.difficulty_level = min(max(difficulty_level, 0), 3)

    def inject(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        broken_query: str,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Inject 2-4 random faults into the episode.

        Returns:
            {
                "injected_faults": [list of FaultType.value strings],
                "broken_query":    str (modified query with query-level faults applied),
                "fault_count":     int,
            }
        """
        rng = random.Random(seed)
        pool = self.LEVEL_POOLS[self.difficulty_level]

        num_faults = rng.randint(2, min(4, len(pool)))
        selected = rng.sample(pool, num_faults)

        # Ensure at least 1 data fault and 1 query fault
        data_faults = [f for f in selected if f in _DATA_FAULTS]
        query_faults = [f for f in selected if f not in _DATA_FAULTS]

        if not data_faults:
            data_candidates = [f for f in pool if f in _DATA_FAULTS]
            if data_candidates:
                pick = rng.choice(data_candidates)
                if len(selected) < 4:
                    selected.append(pick)
                else:
                    selected[-1] = pick

        if not query_faults:
            query_candidates = [f for f in pool if f not in _DATA_FAULTS]
            if query_candidates:
                pick = rng.choice(query_candidates)
                if len(selected) < 4:
                    selected.append(pick)
                else:
                    selected[0] = pick

        # Apply data-level faults
        for fault in selected:
            if fault == FaultType.NULL_FK:
                self._apply_null_fk(conn, task_id, rng)
            elif fault == FaultType.TYPE_DRIFT:
                self._apply_type_drift(conn, task_id)
            elif fault == FaultType.DUPLICATE_ROWS:
                self._apply_duplicate_rows(conn, task_id, rng)

        # Apply query-level faults (chained)
        modified_query = broken_query
        for fault in selected:
            if fault == FaultType.MISSING_COMMA:
                modified_query = self._apply_missing_comma(modified_query, rng)
            elif fault == FaultType.WRONG_JOIN_KEY:
                modified_query = self._apply_wrong_join_key(modified_query, task_id)
            elif fault == FaultType.WRONG_GROUP_BY:
                modified_query = self._apply_wrong_group_by(modified_query, task_id)
            elif fault == FaultType.STALE_WHERE:
                modified_query = self._apply_stale_where(modified_query, rng)
            elif fault == FaultType.COLUMN_ALIAS_SHADOW:
                modified_query = self._apply_column_alias_shadow(modified_query, rng)
            elif fault == FaultType.IMPLICIT_CAST_BUG:
                modified_query = self._apply_implicit_cast_bug(modified_query)
            elif fault == FaultType.MISSING_DISTINCT:
                modified_query = self._apply_missing_distinct(modified_query)
            elif fault == FaultType.OFF_BY_ONE_LIMIT:
                modified_query = self._apply_off_by_one_limit(modified_query)
            elif fault == FaultType.WRONG_SORT_ORDER:
                modified_query = self._apply_wrong_sort_order(modified_query)

        # Level 3: add red herring table
        if self.difficulty_level == 3:
            self._inject_red_herring_table(conn, task_id, rng)

        return {
            "injected_faults": [f.value for f in selected],
            "broken_query": modified_query,
            "fault_count": len(selected),
        }

    # ─── Data-level fault appliers ────────────────────────────────────────────

    def _apply_null_fk(self, conn: sqlite3.Connection, task_id: str, rng: random.Random):
        """Insert rows referencing non-existent parent IDs."""
        phantom_id = rng.choice([99, 999, 9999])
        try:
            if task_id == "easy":
                conn.execute(
                    "INSERT INTO employees VALUES (?, ?, ?, ?)",
                    (100 + phantom_id, f"Ghost_{phantom_id}", 50000.0, "Engineering"),
                )
            elif task_id == "medium":
                conn.execute(
                    "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
                    (100 + phantom_id, phantom_id, 10, 5, "2024-06-01"),
                )
            elif task_id == "hard":
                conn.execute(
                    "INSERT INTO transactions VALUES (?, ?, ?, ?)",
                    (9000 + phantom_id, phantom_id, "999.99", "2024-06-01"),
                )
            conn.commit()
        except Exception:
            pass

    def _apply_type_drift(self, conn: sqlite3.Connection, task_id: str):
        """Simulate a column that was loaded as TEXT instead of REAL."""
        try:
            if task_id == "easy":
                conn.executescript("""
                    ALTER TABLE employees RENAME TO employees_backup;
                    CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                        salary TEXT NOT NULL, dept TEXT NOT NULL);
                    INSERT INTO employees SELECT id, name, CAST(salary AS TEXT), dept
                        FROM employees_backup;
                    DROP TABLE employees_backup;
                """)
            elif task_id == "medium":
                conn.executescript("""
                    ALTER TABLE products RENAME TO products_backup;
                    CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT NOT NULL,
                        category TEXT NOT NULL, unit_price TEXT NOT NULL);
                    INSERT INTO products SELECT product_id, product_name, category,
                        CAST(unit_price AS TEXT) FROM products_backup;
                    DROP TABLE products_backup;
                """)
            conn.commit()
        except Exception:
            pass

    def _apply_duplicate_rows(self, conn: sqlite3.Connection, task_id: str, rng: random.Random):
        """Insert exact duplicate rows to inflate aggregates."""
        try:
            if task_id == "easy":
                conn.execute("INSERT INTO employees SELECT * FROM employees WHERE id=1")
            elif task_id == "medium":
                conn.execute("INSERT INTO orders SELECT * FROM orders WHERE order_id=1")
            elif task_id == "hard":
                conn.execute("INSERT INTO transactions SELECT * FROM transactions WHERE txn_id=1001")
            conn.commit()
        except Exception:
            pass

    def _inject_red_herring_table(self, conn: sqlite3.Connection, task_id: str, rng: random.Random):
        """Level 3: inject a plausible-looking but incorrect table to distract the agent."""
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS legacy_data (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL,
                    note TEXT
                );
                INSERT INTO legacy_data VALUES (1, 'deprecated_entry', 99999.0, 'DO NOT USE - legacy system');
                INSERT INTO legacy_data VALUES (2, 'old_record', 12345.0, 'Migrated 2022-01-01');
            """)
            conn.commit()
        except Exception:
            pass

    # ─── Query-level fault appliers ───────────────────────────────────────────

    def _apply_missing_comma(self, query: str, rng: random.Random) -> str:
        """Remove a comma between SELECT columns."""
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.IGNORECASE | re.DOTALL)
        if select_match:
            cols = select_match.group(1)
            if "," in cols:
                new_cols = cols.replace(",", "", 1)
                query = query[: select_match.start(1)] + new_cols + query[select_match.end(1) :]
        return query

    def _apply_wrong_join_key(self, query: str, task_id: str) -> str:
        """Replace a valid JOIN column with a non-existent one."""
        replacements = {
            "medium": [
                ("o.product_id = p.product_id", "o.product_name = p.product_name"),
                ("c.customer_id = o.customer_id", "c.customer_name = o.customer_name"),
            ],
            "hard": [
                ("c.customer_id = t.customer_id", "c.customer_name = t.customer_name"),
            ],
        }
        for old, new in replacements.get(task_id, []):
            if old in query:
                return query.replace(old, new)
        return query

    def _apply_wrong_group_by(self, query: str, task_id: str) -> str:
        """Replace GROUP BY column with a wrong one."""
        replacements = {
            "medium": [
                ("GROUP BY p.category", "GROUP BY o.product_id"),
                ("GROUP BY p.category", "GROUP BY o.order_id"),
            ],
        }
        for old, new in replacements.get(task_id, []):
            if old in query:
                return query.replace(old, new)
        return query

    def _apply_stale_where(self, query: str, rng: random.Random) -> str:
        """Replace a WHERE literal with a wrong value."""
        stale_map = {
            "'Engineering'": rng.choice(["'Engineering_OLD'", "'Eng'", "'engineering'"]),
            "'US'": "'USA'",
            "2024": "2023",
        }
        for old, new in stale_map.items():
            if old in query:
                return query.replace(old, new, 1)
        return query

    def _apply_column_alias_shadow(self, query: str, rng: random.Random) -> str:
        """Add an alias that shadows a real column, causing confusion."""
        query = re.sub(
            r"\bSUM\(([^)]+)\)\s+AS\s+(\w+)",
            r"SUM(\1) AS total",
            query,
            count=1,
            flags=re.IGNORECASE,
        )
        return query

    def _apply_implicit_cast_bug(self, query: str) -> str:
        """Remove CAST from a query that needs it for numeric computation."""
        query = re.sub(r"CAST\(([^)]+)\s+AS\s+REAL\)", r"\1", query, flags=re.IGNORECASE)
        return query

    def _apply_missing_distinct(self, query: str) -> str:
        """Remove DISTINCT from a subquery or main SELECT."""
        query = re.sub(r"\bSELECT\s+DISTINCT\b", "SELECT", query, flags=re.IGNORECASE)
        return query

    def _apply_off_by_one_limit(self, query: str) -> str:
        """Add a LIMIT that cuts off exactly one correct row."""
        if "LIMIT" not in query.upper():
            query = query.rstrip(";") + " LIMIT 2"
        return query

    def _apply_wrong_sort_order(self, query: str) -> str:
        """Flip ASC/DESC in ORDER BY."""
        if "DESC" in query.upper():
            query = re.sub(r"\bDESC\b", "ASC", query, flags=re.IGNORECASE)
        elif "ASC" in query.upper():
            query = re.sub(r"\bASC\b", "DESC", query, flags=re.IGNORECASE)
        else:
            if "ORDER BY" in query.upper():
                query = query.rstrip(";") + " ASC"
        return query

    # ─── Hint generation ─────────────────────────────────────────────────────

    HINT_TEMPLATES = {
        FaultType.NULL_FK: (
            "Check for orphaned rows — some rows may reference IDs that don't exist in the parent table.",
            "Use a LEFT JOIN or NOT IN to find rows with no matching parent record.",
            "Filter with INNER JOIN on the parent table, or add a WHERE customer_id IN (SELECT id FROM parent).",
        ),
        FaultType.TYPE_DRIFT: (
            "Inspect the schema carefully — a column may have been loaded with the wrong type.",
            "Try running: SELECT typeof(column_name) FROM table LIMIT 1 to check actual storage type.",
            "Wrap numeric columns with CAST(column AS REAL) before doing arithmetic.",
        ),
        FaultType.DUPLICATE_ROWS: (
            "Preview the data — there may be duplicate rows inflating aggregates.",
            "Use SELECT COUNT(*) vs SELECT COUNT(DISTINCT id) to spot duplicates.",
            "Wrap the source table in a subquery with SELECT DISTINCT to de-duplicate before aggregating.",
        ),
        FaultType.MISSING_COMMA: (
            "Look at the SELECT column list very carefully.",
            "Check whether all intended columns are separated by commas.",
            "A comma is missing between two column names in the SELECT clause.",
        ),
        FaultType.WRONG_JOIN_KEY: (
            "The JOIN condition may reference a column that doesn't exist.",
            "Use query_schema on both joined tables to verify the join columns exist.",
            "The ON clause uses a column name from the wrong table — check both sides of the =.",
        ),
        FaultType.WRONG_GROUP_BY: (
            "The GROUP BY clause may not match the intended aggregation.",
            "GROUP BY should use the column that defines the groups in SELECT, not an ID column.",
            "Change GROUP BY to use the category/name column, not the ID column.",
        ),
        FaultType.STALE_WHERE: (
            "The WHERE clause filter value may be outdated or wrong.",
            "Check the actual data values with inspect_data — the filter literal may not match.",
            "The WHERE clause uses a value that doesn't match any rows in the database.",
        ),
        FaultType.COLUMN_ALIAS_SHADOW: (
            "An alias in the query may be shadowing or renaming a column unexpectedly.",
            "Check the AS alias names in the SELECT list — they may conflict.",
            "Rename the AS alias to a unique, non-conflicting name.",
        ),
        FaultType.IMPLICIT_CAST_BUG: (
            "Arithmetic may be failing silently due to a type mismatch.",
            "Check if numeric columns are actually stored as TEXT — use CAST() for arithmetic.",
            "Wrap text-stored numbers with CAST(column AS REAL) before SUM() or multiplication.",
        ),
        FaultType.MISSING_DISTINCT: (
            "Aggregates may be inflated by duplicate source rows.",
            "Add DISTINCT inside the subquery that feeds the JOIN.",
            "Use SELECT DISTINCT txn_id, customer_id, amount inside a subquery before joining.",
        ),
        FaultType.OFF_BY_ONE_LIMIT: (
            "The result set may be truncated — check if all expected rows are returned.",
            "A LIMIT clause may be cutting off the last correct row.",
            "Remove the LIMIT clause or increase it to return all expected results.",
        ),
        FaultType.WRONG_SORT_ORDER: (
            "The sort order may be reversed.",
            "Check ORDER BY direction — the query may use ASC where DESC is required.",
            "Change ORDER BY ... ASC to ORDER BY ... DESC (or vice versa).",
        ),
    }

    def get_hints(self, injected_faults: List[str]) -> List[str]:
        """
        Return 3 progressive hints for the specific faults injected.
        Hint 0 = shown at step 5 (vague)
        Hint 1 = shown at step 10 (specific)
        Hint 2 = shown at step 15 (near-answer)
        """
        fault_enums = []
        for f in injected_faults:
            try:
                fault_enums.append(FaultType(f))
            except ValueError:
                pass

        if not fault_enums:
            return [
                "Hint: Use list_tables to see available tables, then query_schema to inspect their structure.",
                "Hint: Look carefully at the broken query — there may be issues in both the SQL and the data.",
                "Hint: Use inspect_data on each table and compare to what the query expects.",
            ]

        hints = ["", "", ""]
        for fe in fault_enums:
            templates = self.HINT_TEMPLATES.get(
                fe, ("Check the query and data.", "Inspect schema and data.", "Review the query logic.")
            )
            hints[0] += f" | {templates[0]}" if hints[0] else templates[0]
            hints[1] += f" | {templates[1]}" if hints[1] else templates[1]
            hints[2] += f" | {templates[2]}" if hints[2] else templates[2]

        return [
            f"Hint (Step 5): {hints[0]}",
            f"Hint (Step 10): {hints[1]}",
            f"Hint (Step 15): {hints[2]}",
        ]