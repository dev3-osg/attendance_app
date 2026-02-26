"""
CONSTABLE – SQLite database helpers.
Tables:
  employees  – id (TEXT PK), name (TEXT), user_type (TEXT), aadhaar (TEXT), mobile (TEXT), registered_at (TEXT)
  attendance – id (INTEGER PK), employee_id (TEXT FK), timestamp (TEXT), date (TEXT), punch_type (TEXT 'in'|'out')
"""

import sqlite3
import os
from datetime import datetime, date

DB_DIR = os.path.join(os.path.dirname(__file__))
DB_PATH = os.path.join(DB_DIR, "constable.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist; add new columns to existing tables."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            user_type     TEXT DEFAULT 'employee',
            aadhaar       TEXT DEFAULT '',
            mobile        TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            date        TEXT NOT NULL,
            punch_type  TEXT NOT NULL DEFAULT 'in',
            at_iso      TEXT,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        );
    """)
    # Migrate: add punch_type to attendance if missing
    try:
        cur.execute("SELECT punch_type FROM attendance LIMIT 1")
    except sqlite3.OperationalError:
        cur.execute("ALTER TABLE attendance ADD COLUMN punch_type TEXT DEFAULT 'in'")
        cur.execute("UPDATE attendance SET punch_type = 'in' WHERE punch_type IS NULL OR punch_type = ''")
    # Migrate: add at_iso for cooldown if missing
    cur.execute("PRAGMA table_info(attendance)")
    cols = [r[1] for r in cur.fetchall()]
    if "at_iso" not in cols:
        cur.execute("ALTER TABLE attendance ADD COLUMN at_iso TEXT")
    # Migrate employees: add user_type, aadhaar, mobile if missing
    for col, default in [("user_type", "employee"), ("aadhaar", ""), ("mobile", "")]:
        try:
            cur.execute("SELECT " + col + " FROM employees LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE employees ADD COLUMN " + col + " TEXT DEFAULT '" + default.replace("'", "''") + "'")
            cur.execute("UPDATE employees SET " + col + " = ? WHERE " + col + " IS NULL", (default,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Employee helpers
# ---------------------------------------------------------------------------

def add_employee(employee_id: str, name: str, user_type: str = "employee", aadhaar: str = "", mobile: str = "") -> bool:
    """Insert or replace an employee record. Returns True on success."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO employees (id, name, registered_at, user_type, aadhaar, mobile)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (employee_id, name, datetime.now().isoformat(timespec="seconds"), user_type, aadhaar or "", mobile or ""),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB] add_employee error: {e}")
        return False
    finally:
        conn.close()


def get_employee(employee_id: str):
    """Return employee row or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM employees WHERE id = ?", (employee_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_employees():
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM employees ORDER BY registered_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_employee(employee_id: str) -> bool:
    """Delete an employee and their attendance records. Returns True on success."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM attendance WHERE employee_id = ?", (employee_id,))
        conn.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB] delete_employee error: {e}")
        return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Attendance: one punch in per day, then only punch out; 1 min cooldown; last punch out only
# ---------------------------------------------------------------------------

COOLDOWN_SECONDS = 60


def mark_attendance(employee_id: str) -> dict:
    """
    First time today = punch in only. Every other time = punch out only.
    1 min cooldown for same user. Only last punch out time is used for display.
    Returns {'status': 'success'|'cooldown', 'punch_type': 'in'|'out', 'timestamp': ...}
    """
    today = date.today().isoformat()
    now = datetime.now()
    now_iso = now.isoformat()
    ts = now.strftime("%I:%M %p")
    conn = get_connection()
    try:
        last_row = conn.execute(
            "SELECT at_iso FROM attendance WHERE employee_id = ? AND date = ? ORDER BY id DESC LIMIT 1",
            (employee_id, today),
        ).fetchone()
        if last_row and last_row["at_iso"]:
            try:
                last_dt = datetime.fromisoformat(last_row["at_iso"])
                if (now - last_dt).total_seconds() < COOLDOWN_SECONDS:
                    return {"status": "cooldown", "punch_type": None, "timestamp": None}
            except (ValueError, TypeError):
                pass

        has_any_today = conn.execute(
            "SELECT 1 FROM attendance WHERE employee_id = ? AND date = ? LIMIT 1",
            (employee_id, today),
        ).fetchone()
        next_punch = "out" if has_any_today else "in"

        conn.execute(
            "INSERT INTO attendance (employee_id, timestamp, date, punch_type, at_iso) VALUES (?, ?, ?, ?, ?)",
            (employee_id, ts, today, next_punch, now_iso),
        )
        conn.commit()
        return {"status": "success", "punch_type": next_punch, "timestamp": ts}
    finally:
        conn.close()


def get_today_attendance():
    """Return today's attendance: one row per employee with first punch_in and last punch_out."""
    today = date.today().isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT e.id, e.name,
                      (SELECT MIN(a.timestamp) FROM attendance a WHERE a.employee_id = e.id AND a.date = ? AND a.punch_type = 'in') AS punch_in,
                      (SELECT MAX(a.timestamp) FROM attendance a WHERE a.employee_id = e.id AND a.date = ? AND a.punch_type = 'out') AS punch_out
               FROM attendance a
               JOIN employees e ON a.employee_id = e.id
               WHERE a.date = ?
               GROUP BY e.id, e.name""",
            (today, today, today),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
