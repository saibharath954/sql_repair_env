"""
Task definitions for the SQL Repair environment.

Each task provides:
- description      : Human-readable goal statement.
- difficulty       : easy | medium | hard
- setup_sql        : DDL + DML to initialise the in-memory SQLite database.
- broken_query     : The defective query the agent must repair.
- expected_rows    : Canonical correct result set (list of dicts, sorted deterministically).
- expected_columns : Ordered list of column names that must appear in the result.
- action_schema    : Example action payload shown via GET /tasks (auto-documents the API).
- hints            : Progressive hint strings indexed 0-2 (shown after 5/10/15 stuck steps).
"""

from typing import Any, Dict, List

TASKS: Dict[str, Dict[str, Any]] = {

    # ─────────────────────────────────────────────────────────────────────────
    # EASY — single table, plain syntax error (missing comma between columns)
    # ─────────────────────────────────────────────────────────────────────────
    "easy": {
        "description": (
            "The HR system's salary report query has a syntax error. "
            "Fix it so it returns the name and salary of all Engineering employees, "
            "ordered by salary descending."
        ),
        "difficulty": "easy",
        "setup_sql": """
            CREATE TABLE employees (
                id      INTEGER PRIMARY KEY,
                name    TEXT    NOT NULL,
                salary  REAL    NOT NULL,
                dept    TEXT    NOT NULL
            );
            INSERT INTO employees VALUES
                (1, 'Alice',   90000.0, 'Engineering'),
                (2, 'Bob',     75000.0, 'Marketing'),
                (3, 'Carol',   95000.0, 'Engineering'),
                (4, 'David',   82000.0, 'Engineering'),
                (5, 'Eve',     68000.0, 'Sales');
        """,
        # Bug: missing comma between 'name' and 'salary'
        "broken_query": (
            "SELECT name salary FROM employees "
            "WHERE dept = 'Engineering' ORDER BY salary DESC"
        ),
        "expected_rows": [
            {"name": "Carol",  "salary": 95000.0},
            {"name": "Alice",  "salary": 90000.0},
            {"name": "David",  "salary": 82000.0},
        ],
        "expected_columns": ["name", "salary"],
        "action_schema": {
            "action_type": "submit_query",
            "sql_query": "SELECT name, salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC",
            "target_table": None,
        },
        "hints": [
            "Hint 1: Use query_schema or inspect_data to examine the employees table.",
            "Hint 2: Look carefully at the SELECT column list — something is missing between columns.",
            "Hint 3: A comma is missing between 'name' and 'salary' in the SELECT clause.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MEDIUM — three tables, broken JOIN key + wrong GROUP BY column
    # ─────────────────────────────────────────────────────────────────────────
    "medium": {
        "description": (
            "The sales dashboard query is broken: it uses the wrong JOIN key and "
            "groups by the wrong column. Fix it to return each product category with "
            "total revenue (unit_price * quantity), ordered by revenue descending."
        ),
        "difficulty": "medium",
        "setup_sql": """
            CREATE TABLE customers (
                customer_id   INTEGER PRIMARY KEY,
                name          TEXT NOT NULL,
                country       TEXT NOT NULL
            );
            CREATE TABLE orders (
                order_id      INTEGER PRIMARY KEY,
                customer_id   INTEGER NOT NULL,
                product_id    INTEGER NOT NULL,
                quantity      INTEGER NOT NULL,
                order_date    TEXT NOT NULL
            );
            CREATE TABLE products (
                product_id    INTEGER PRIMARY KEY,
                product_name  TEXT NOT NULL,
                category      TEXT NOT NULL,
                unit_price    REAL NOT NULL
            );

            INSERT INTO customers VALUES
                (1, 'Acme Corp',    'US'),
                (2, 'Globex',       'UK'),
                (3, 'Initech',      'US');

            INSERT INTO products VALUES
                (10, 'Widget A',  'Widgets',   25.00),
                (11, 'Widget B',  'Widgets',   40.00),
                (20, 'Gadget X',  'Gadgets',   75.00),
                (21, 'Gadget Y',  'Gadgets',  120.00),
                (30, 'Doohickey', 'Misc',      15.00);

            INSERT INTO orders VALUES
                (1, 1, 10, 4,  '2024-01-10'),
                (2, 1, 20, 2,  '2024-01-12'),
                (3, 2, 11, 6,  '2024-01-15'),
                (4, 2, 21, 1,  '2024-01-18'),
                (5, 3, 10, 10, '2024-01-20'),
                (6, 3, 30, 5,  '2024-01-22'),
                (7, 1, 21, 3,  '2024-02-01');
        """,
        # Bugs: JOIN uses o.product_name (doesn't exist), GROUP BY uses o.product_id not p.category
        "broken_query": (
            "SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue "
            "FROM orders o "
            "JOIN products p ON o.product_name = p.product_name "
            "GROUP BY o.product_id "
            "ORDER BY revenue DESC"
        ),
        "expected_rows": [
            {"category": "Gadgets", "revenue": 630.0},
            {"category": "Widgets", "revenue": 590.0},
            {"category": "Misc",    "revenue":  75.0},
        ],
        "expected_columns": ["category", "revenue"],
        "action_schema": {
            "action_type": "submit_query",
            "sql_query": (
                "SELECT p.category, SUM(p.unit_price * o.quantity) AS revenue "
                "FROM orders o JOIN products p ON o.product_id = p.product_id "
                "GROUP BY p.category ORDER BY revenue DESC"
            ),
            "target_table": None,
        },
        "hints": [
            "Hint 1: Use list_tables then query_schema on each table to understand the schema.",
            "Hint 2: The JOIN condition references a column that does not exist on the orders table.",
            "Hint 3: The GROUP BY must use p.category (not o.product_id) to aggregate by category.",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # HARD — dirty ETL: duplicate rows, NULL FK violations, type mismatches,
    #         then a complex multi-table analytical query to write correctly
    # ─────────────────────────────────────────────────────────────────────────
    "hard": {
        "description": (
            "A nightly ETL job loaded dirty data into three tables: "
            "(1) duplicate transactions exist, "
            "(2) some transactions reference non-existent customer IDs (NULL FK violation), "
            "(3) the 'amount' column was loaded as TEXT instead of REAL. "
            "Clean the data, then write a query that returns each customer's name and their "
            "total spend (sum of valid, non-duplicate transaction amounts), "
            "ordered by total_spend descending. Exclude transactions with invalid customer IDs."
        ),
        "difficulty": "hard",
        "setup_sql": """
            CREATE TABLE customers_hard (
                customer_id INTEGER PRIMARY KEY,
                name        TEXT NOT NULL
            );
            CREATE TABLE transactions (
                txn_id      INTEGER,
                customer_id INTEGER,
                amount      TEXT,         -- BUG: stored as TEXT, should be REAL
                txn_date    TEXT
            );

            INSERT INTO customers_hard VALUES
                (1, 'Alice'),
                (2, 'Bob'),
                (3, 'Carol');

            -- Clean rows
            INSERT INTO transactions VALUES (1001, 1, '250.00', '2024-03-01');
            INSERT INTO transactions VALUES (1002, 2, '180.50', '2024-03-02');
            INSERT INTO transactions VALUES (1003, 1, '320.75', '2024-03-03');
            INSERT INTO transactions VALUES (1004, 3, '95.00',  '2024-03-04');

            -- DUPLICATE rows (exact copies of 1001 and 1003)
            INSERT INTO transactions VALUES (1001, 1, '250.00', '2024-03-01');
            INSERT INTO transactions VALUES (1003, 1, '320.75', '2024-03-03');

            -- INVALID customer_id (customer 99 does not exist)
            INSERT INTO transactions VALUES (1005, 99, '500.00', '2024-03-05');

            -- Another duplicate
            INSERT INTO transactions VALUES (1002, 2, '180.50', '2024-03-02');
        """,
        "broken_query": (
            "SELECT c.name, SUM(t.amount) AS total_spend "
            "FROM customers_hard c "
            "JOIN transactions t ON c.customer_id = t.customer_id "
            "GROUP BY c.name ORDER BY total_spend DESC"
        ),
        "expected_rows": [
            {"name": "Alice", "total_spend": 570.75},
            {"name": "Bob",   "total_spend": 180.50},
            {"name": "Carol", "total_spend":  95.00},
        ],
        "expected_columns": ["name", "total_spend"],
        "action_schema": {
            "action_type": "submit_query",
            "sql_query": (
                "SELECT c.name, SUM(CAST(t.amount AS REAL)) AS total_spend "
                "FROM customers_hard c "
                "JOIN (SELECT DISTINCT txn_id, customer_id, amount FROM transactions) t "
                "  ON c.customer_id = t.customer_id "
                "GROUP BY c.customer_id, c.name "
                "ORDER BY total_spend DESC"
            ),
            "target_table": None,
        },
        "hints": [
            "Hint 1: Use inspect_data on the transactions table — look for duplicate txn_id rows and odd-looking amount values.",
            "Hint 2: De-duplicate with SELECT DISTINCT or GROUP BY txn_id. Also CAST(amount AS REAL) to fix the type issue.",
            "Hint 3: Use an INNER JOIN with customers_hard to exclude the orphan customer_id=99 rows automatically.",
        ],
    },
}