"""The queue database (SQLite on the local disk; the daemon and every CLI on this machine share it).

Job states: ``pending`` -> ``running`` -> ``done`` (exit 0) | ``failed`` (other exit) | ``lost`` (its
process vanished without an exit code); ``pending`` or ``running`` -> ``cancelled`` (a running job is
first ``cancelling`` until the daemon has killed it).
"""

import json
import sqlite3
import time

ACTIVE = ('pending', 'running', 'cancelling')
FINISHED = ('done', 'failed', 'lost', 'cancelled')

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, owner TEXT, command TEXT NOT NULL, cwd TEXT NOT NULL, env TEXT NOT NULL,
    mem REAL NOT NULL, util REAL NOT NULL, window REAL NOT NULL, ngpus INTEGER NOT NULL,
    hosts TEXT, mask INTEGER NOT NULL, priority INTEGER NOT NULL,
    state TEXT NOT NULL, reason TEXT,
    submitted REAL NOT NULL, started REAL, ended REAL,
    host TEXT, gpus TEXT, launch_used TEXT, exit_code INTEGER
);
CREATE TABLE IF NOT EXISTS gpus (
    host TEXT, idx INTEGER, name TEXT, total REAL, used REAL, util REAL, util_max REAL, window REAL,
    apps TEXT, updated REAL, PRIMARY KEY (host, idx)
);
CREATE TABLE IF NOT EXISTS hosts (host TEXT PRIMARY KEY, ok INTEGER, error TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(config):
    conn = sqlite3.connect(config.db, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(SCHEMA)
    return conn


def submit(conn, **job):
    job.update(state='pending', submitted=time.time(), env=json.dumps(job['env']),
               hosts=json.dumps(job['hosts']) if job.get('hosts') else None)
    keys = ', '.join(job)
    cursor = conn.execute(f'INSERT INTO jobs ({keys}) VALUES ({", ".join("?" * len(job))})', list(job.values()))
    return cursor.lastrowid


def jobs(conn, where='1', args=()):
    return [dict(row) for row in conn.execute(f'SELECT * FROM jobs WHERE {where} ORDER BY id', args)]


def job(conn, job_id):
    row = conn.execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone()
    return dict(row) if row else None


def update(conn, job_id, **fields):
    sets = ', '.join(f'{key} = ?' for key in fields)
    conn.execute(f'UPDATE jobs SET {sets} WHERE id = ?', [*fields.values(), job_id])


def cancel(conn, job_id):
    """Cancel a job: a pending one at once, a running one by the daemon. Its state afterwards (None: no job)."""
    conn.execute("UPDATE jobs SET state = 'cancelled', ended = ?, reason = NULL WHERE id = ? AND state = 'pending'",
                 (time.time(), job_id))
    conn.execute("UPDATE jobs SET state = 'cancelling' WHERE id = ? AND state = 'running'", (job_id,))
    found = job(conn, job_id)
    return found and found['state']


def get_meta(conn, key, default=None):
    row = conn.execute('SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
    return row['value'] if row else default


def set_meta(conn, key, value):
    conn.execute('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)', (key, str(value)))
