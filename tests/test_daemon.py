from dataclasses import replace
import json

from gpu_queue import daemon, db, hosts
from gpu_queue.config import Config


def test_daemon_tracks_running_jobs_while_cpu_window_fills_and_saves_new_metric(tmp_path, monkeypatch):
    conf = Config(state_dir=tmp_path, shared_dir=tmp_path / 'shared', hosts=['a'])
    conf.shared_dir.mkdir()
    service = daemon.Daemon(conf)
    job_id = db.submit(service.conn, command='true', cwd=str(tmp_path), env={}, mem=0., util=30.,
                       window=30., ngpus=0, mask=1, priority=0, cpus=16., ram=4.)
    previous = db.submit(service.conn, command='true', cwd=str(tmp_path), env={}, mem=0., util=30.,
                         window=30., ngpus=0, mask=1, priority=0, cpus=1., ram=4.)
    db.update(service.conn, previous, state='running', host='a', gpus='', started=100.)
    launched, followed = [], []
    monkeypatch.setattr(service, 'launch', lambda *args: launched.append(args))
    monkeypatch.setattr(service, 'follow', lambda job, *args: followed.append(job['id']))
    sample = hosts.Host('a', 56., None, 448., 440., 0, 0., 'scope')
    monkeypatch.setattr(hosts, 'sample', lambda *args: ([], set(), replace(sample)))
    try:
        for stamp in (0., 30., 60.):
            sample.cpu_time, sample.cpu_usage = stamp, stamp * 100_000
            monkeypatch.setattr(daemon.time, 'time', lambda: 1000. + stamp)
            service.tick()
            current = db.job(service.conn, job_id)
            if stamp < 60:
                assert current['state'] == 'pending' and 'still sampling' in current['reason']
                assert service.conn.execute('SELECT cpu_used FROM host_resources').fetchone()[0] is None
            else:
                assert current['state'] == 'running'
                assert json.loads(current['launch_host']) == {'cpu_used': .1, 'ram_available': 440.}
                assert len(launched) == 1
        assert followed == [previous] * 3
        # Restart loses CPU history, but keeps running jobs and their snapshots.
        restarted = daemon.Daemon(conf)
        try:
            assert restarted.cpu_history.add(replace(sample)) is None
            assert db.job(restarted.conn, job_id)['state'] == 'running'
        finally:
            restarted.pool.shutdown()
            restarted.conn.close()
    finally:
        service.pool.shutdown()
        service.conn.close()
