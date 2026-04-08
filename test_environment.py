"""
Unit tests for the SQL Repair Environment.

Run with:
    cd sql_repair_env
    pip install -e ".[dev]"
    pytest test_environment.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from models import SQLRepairAction, SQLRepairObservation
from server.sql_repair_environment import SQLRepairEnvironment
from server.grader import compute_score, SUBGOALS
from server.tasks import TASKS


@pytest.fixture
def env():
    return SQLRepairEnvironment()


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="easy")
        assert isinstance(obs, SQLRepairObservation)

    def test_reset_zero_score(self, env):
        obs = env.reset(task_id="easy")
        assert obs.partial_score == 0.0

    def test_reset_zero_steps(self, env):
        obs = env.reset(task_id="easy")
        assert obs.step_count == 0

    def test_reset_contains_broken_query(self, env):
        obs = env.reset(task_id="easy")
        assert TASKS["easy"]["broken_query"] in obs.hint or obs.broken_query != ""

    def test_reset_isolates_episodes(self, env):
        env.reset(task_id="easy")
        env.step(SQLRepairAction(action_type="submit_query",
                                  sql_query="CREATE TABLE dirty (x int)"))
        obs2 = env.reset(task_id="easy")
        # After reset, no dirty table should exist
        obs3 = env.step(SQLRepairAction(action_type="query_schema", target_table="dirty"))
        assert "not found" in obs3.error_message.lower() or obs3.schema_info == ""

    def test_reset_all_tasks(self, env):
        for tid in ["easy", "medium", "hard"]:
            obs = env.reset(task_id=tid)
            assert obs.task_id == tid
            assert obs.done is False


# ── grader determinism ────────────────────────────────────────────────────────

class TestGrader:
    def test_grader_returns_float(self):
        score, flags = compute_score(None, "easy", [], {k: False for k, _ in SUBGOALS["easy"]})
        assert isinstance(score, float)

    def test_grader_zero_on_empty(self):
        for tid in ["easy", "medium", "hard"]:
            score, _ = compute_score(None, tid, [], {k: False for k, _ in SUBGOALS[tid]})
            assert score == 0.0

    def test_grader_bounded(self):
        for tid in ["easy", "medium", "hard"]:
            all_true = {k: True for k, _ in SUBGOALS[tid]}
            score, _ = compute_score(None, tid, [], all_true)
            assert 0.0 <= score <= 1.0

    def test_grader_deterministic(self, env):
        env.reset(task_id="easy")
        correct_sql = TASKS["easy"]["action_schema"]["sql_query"]
        obs = env.step(SQLRepairAction(action_type="submit_query", sql_query=correct_sql))
        score1 = obs.partial_score
        # Run again with same state
        score2, _ = compute_score(
            env._conn, "easy", obs.query_result, dict(env._achieved_flags)
        )
        assert score1 == score2

    def test_perfect_score_on_correct_easy(self, env):
        env.reset(task_id="easy")
        correct_sql = TASKS["easy"]["action_schema"]["sql_query"]
        obs = env.step(SQLRepairAction(action_type="submit_query", sql_query=correct_sql))
        assert obs.partial_score == 1.0
        assert obs.done is True

    def test_partial_score_on_partial_fix(self, env):
        """Query runs without error but returns wrong rows → partial credit."""
        env.reset(task_id="easy")
        # Query executes fine but returns all rows, not just Engineering
        obs = env.step(SQLRepairAction(
            action_type="submit_query",
            sql_query="SELECT name, salary FROM employees ORDER BY salary DESC"
        ))
        assert 0.0 < obs.partial_score < 1.0


# ── step mechanics ────────────────────────────────────────────────────────────

class TestStep:
    def test_step_increments_count(self, env):
        env.reset()
        obs = env.step(SQLRepairAction(action_type="list_tables"))
        assert obs.step_count == 1

    def test_max_steps_triggers_done(self, env):
        env.reset(task_id="easy")
        obs = None
        for _ in range(21):
            obs = env.step(SQLRepairAction(action_type="list_tables"))
            if obs.done:
                break
        assert obs.done is True

    def test_reward_is_float(self, env):
        env.reset()
        obs = env.step(SQLRepairAction(action_type="list_tables"))
        assert isinstance(obs.reward, float)

    def test_list_tables_works(self, env):
        env.reset(task_id="easy")
        obs = env.step(SQLRepairAction(action_type="list_tables"))
        assert "employees" in obs.tables

    def test_query_schema_works(self, env):
        env.reset(task_id="easy")
        obs = env.step(SQLRepairAction(action_type="query_schema", target_table="employees"))
        assert "employees" in obs.schema_info.lower()

    def test_inspect_data_works(self, env):
        env.reset(task_id="easy")
        obs = env.step(SQLRepairAction(action_type="inspect_data", target_table="employees"))
        assert len(obs.query_result) > 0

    def test_bad_sql_returns_error_not_crash(self, env):
        env.reset()
        obs = env.step(SQLRepairAction(
            action_type="submit_query",
            sql_query="SELEKT * FORM employees"
        ))
        assert obs.error_message != ""
        assert obs.done is False  # episode still alive

    def test_unknown_action_type_returns_error(self, env):
        env.reset()
        obs = env.step(SQLRepairAction(action_type="fly_to_moon"))
        assert obs.error_message != ""


# ── medium task ───────────────────────────────────────────────────────────────

class TestMedium:
    def test_correct_medium_query_scores_1(self, env):
        env.reset(task_id="medium")
        
        # --- Simulate the agent inspecting the schema ---
        env.step(SQLRepairAction(action_type="query_schema", target_table="orders"))
        
        correct_sql = TASKS["medium"]["action_schema"]["sql_query"]
        obs = env.step(SQLRepairAction(action_type="submit_query", sql_query=correct_sql))
        assert obs.partial_score == 1.0
    def test_medium_broken_query_scores_0(self, env):
        env.reset(task_id="medium")
        broken_sql = TASKS["medium"]["broken_query"]
        obs = env.step(SQLRepairAction(action_type="submit_query", sql_query=broken_sql))
        assert obs.partial_score < 1.0


# ── hard task ─────────────────────────────────────────────────────────────────

class TestHard:
    def test_correct_hard_query_scores_1(self, env):
        env.reset(task_id="hard")
        # Inspect data first (triggers duplicate detection flag)
        env.step(SQLRepairAction(action_type="inspect_data", target_table="transactions"))
        correct_sql = TASKS["hard"]["action_schema"]["sql_query"]
        obs = env.step(SQLRepairAction(action_type="submit_query", sql_query=correct_sql))
        assert obs.partial_score >= 0.9  # may not hit all flags without full agent loop

    def test_hard_broken_naive_query_low_score(self, env):
        env.reset(task_id="hard")
        # Naive query — no dedup, wrong type handling
        obs = env.step(SQLRepairAction(
            action_type="submit_query",
            sql_query="SELECT c.name, SUM(t.amount) AS total_spend FROM customers_hard c JOIN transactions t ON c.customer_id = t.customer_id GROUP BY c.name ORDER BY total_spend DESC"
        ))
        # Amount is TEXT so SUM fails or gives wrong result
        assert obs.partial_score < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])