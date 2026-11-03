import sqlite3

from gpu_queue import db
from gpu_queue.config import Config


def test_migration_preserves_running_and_pending_jobs_and_old_daemon_inserts(tmp_path):
    conf = Config(state_dir=tmp_path)
    conf.defaults.update(cpus=2.5, ram=8.)
    legacy = sqlite3.connect(conf.db, isolation_level=None)
    legacy.executescript(db.SCHEMA)
    fields = dict(name='test', owner='test', command='sleep 10', cwd='/tmp', env={},
                  mem=4., util=30., window=30., ngpus=1, hosts=None, mask=1, priority=0)
    pending = db.submit(legacy, **fields)
    running = db.submit(legacy, **fields)
    db.update(legacy, running, state='running', started=123., host='a', gpus='0', launch_used='{"0":1}')
    upgraded = db.connect(conf)
    assert db.job(upgraded, pending)['state'] == 'pending'
    old_running = db.job(upgraded, running)
    assert (old_running['state'], old_running['started'], old_running['launch_used']) == ('running', 123., '{"0":1}')
    assert (old_running['cpus'], old_running['ram'], old_running['launch_host']) == (2.5, 8., None)
    legacy.execute('INSERT OR REPLACE INTO hosts VALUES (?, 1, NULL, ?)', ('a', 124.))
    another = db.submit(legacy, **fields)
    assert db.job(upgraded, another)['cpus'] == 2.5
    upgraded.close()
    upgraded = db.connect(conf)  # Idempotent; declarations are not overwritten on reconnect.
    db.update(upgraded, pending, cpus=7., ram=9.)
    reopened = db.connect(conf)
    assert (db.job(reopened, pending)['cpus'], db.job(reopened, pending)['ram']) == (7., 9.)
    reopened.close()
    upgraded.close()
    legacy.close()


def test_replaces_legacy_resource_metric_without_losing_jobs(tmp_path):
    conf = Config(state_dir=tmp_path)
    legacy = sqlite3.connect(conf.db, isolation_level=None)
    legacy.executescript(db.SCHEMA.replace('cpus REAL, cpu_used REAL', 'cpus INTEGER, load1 REAL'))
    legacy.execute("INSERT INTO host_resources VALUES ('a', 56, 100, 448, 440, 100)")
    fields = dict(command='sleep 10', cwd='/tmp', env={}, mem=4., util=30., window=30., ngpus=1, mask=1, priority=0)
    job_id = db.submit(legacy, **fields)
    db.update(legacy, job_id, state='running', started=100., launch_used='{}')
    legacy.close()
    conn = db.connect(conf)
    assert 'load1' not in {row['name'] for row in conn.execute('PRAGMA table_info(host_resources)')}
    assert not list(conn.execute('SELECT * FROM host_resources'))
    assert db.job(conn, job_id)['state'] == 'running'
    conn.execute("INSERT INTO host_resources VALUES ('a', 55.5, NULL, 448, 440, 101)")
    reopened = db.connect(conf)
    assert reopened.execute('SELECT cpus FROM host_resources').fetchone()[0] == 55.5
    reopened.close()
    conn.close()
