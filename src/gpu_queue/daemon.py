"""The daemon: one per machine (a lock in the state directory), started by the CLI when needed.

Every ``poll`` seconds it samples each host's CPU/RAM and GPUs over SSH (with the liveness of its running
jobs), finishes jobs whose exit code appeared, kills cancelled ones, and starts pending jobs that
fit (:mod:`gpu_queue.schedule`). A job runs as ``setsid bash <shared>/jobs/<id>.sh`` on its host;
the script records its pid, runs the command in the submission's directory and environment, then
writes the exit code. Job state lives in files on the shared filesystem, so a restarted daemon
picks its running jobs up again.
"""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import logging
import os
import shlex
import subprocess
import sys
import time

from gpu_queue import config as config_module, db, hosts, schedule

KILL_AFTER = 20.     # s after SIGTERM, SIGKILL
LOST_AFTER = 3       # samples with the job's process gone and no exit code
START_TIMEOUT = 120. # s for a started job's pid file to appear

log = logging.getLogger('gq')


def running_daemon(conf):
    """Pid of the live daemon, or None."""
    lock = conf.state_dir / 'daemon.lock'
    with open(lock, 'a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            text = handle.read().strip()
            return int(text) if text.isdigit() else -1
        fcntl.flock(handle, fcntl.LOCK_UN)
        return None


def ensure(conf):
    """Start the daemon unless one runs; its pid."""
    pid = running_daemon(conf)
    if pid is not None:
        return pid
    with open(conf.state_dir / 'daemon.log', 'a') as out:
        process = subprocess.Popen([sys.executable, '-m', 'gpu_queue.daemon'], stdout=out, stderr=out,
                                   stdin=subprocess.DEVNULL, start_new_session=True, env=os.environ.copy())
    for _ in range(50):
        if running_daemon(conf) is not None:
            break
        time.sleep(.1)
    return process.pid


class Daemon:
    def __init__(self, conf):
        self.conf = conf
        self.conn = db.connect(conf)
        self.history = schedule.History()
        self.missing = {}       # job id -> samples with its process gone
        self.killed = {}        # job id -> time of SIGTERM
        self.pool = ThreadPoolExecutor(max(4, len(conf.hosts)))

    def serve(self):
        lock = open(self.conf.state_dir / 'daemon.lock', 'a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info('another daemon runs')
            return
        lock.seek(0), lock.truncate(), lock.write(str(os.getpid())), lock.flush()
        log.info('daemon %d serving %s', os.getpid(), self.conf.hosts)
        while True:
            started = time.time()
            try:
                self.tick()
            except Exception:
                log.exception('tick failed')
            time.sleep(max(.5, self.conf.poll - (time.time() - started)))

    # -- one round ---------------------------------------------------------------------
    def tick(self):
        now = time.time()
        db.set_meta(self.conn, 'heartbeat', now)
        active = db.jobs(self.conn, "state IN ('running', 'cancelling')")
        pids = {job['id']: self.pid(job['id']) for job in active}
        futures = {host: self.pool.submit(hosts.sample, self.conf, host,
                                          [pids[j['id']] for j in active if j['host'] == host and pids[j['id']]])
                   for host in self.conf.hosts}
        gpus, alive, up, resources = [], {}, set(), []
        for host, future in futures.items():
            try:
                host_gpus, alive[host], resource = future.result()
            except Exception as error:
                log.warning('%s: %s', host, error)
                self.history.forget(host)
                self.conn.execute('INSERT OR REPLACE INTO hosts VALUES (?, 0, ?, ?)', (host, str(error), now))
                continue
            up.add(host)
            gpus += host_gpus
            resources.append(resource)
            self.conn.execute('INSERT OR REPLACE INTO host_resources VALUES (?, ?, ?, ?, ?, ?)',
                              (host, resource.cpus, resource.load1, resource.ram_total, resource.ram_available, now))
            self.conn.execute('INSERT OR REPLACE INTO hosts VALUES (?, 1, NULL, ?)', (host, now))
        self.history.add(now, gpus)
        self.snapshot(gpus, now)
        for job in active:
            if job['host'] in up:
                self.follow(job, pids[job['id']], alive[job['host']], now)
        self.start_jobs(gpus, resources, now)

    def snapshot(self, gpus, now):
        window = self.conf.defaults['window']
        self.conn.execute('BEGIN')
        self.conn.execute('DELETE FROM gpus WHERE host NOT IN (%s)' % ','.join('?' * len(self.conf.hosts)),
                          self.conf.hosts)
        for gpu in gpus:
            util_max = self.history.util_max((gpu.host, gpu.idx), now, window)
            self.conn.execute('INSERT OR REPLACE INTO gpus VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                              (gpu.host, gpu.idx, gpu.name, gpu.total, gpu.used, gpu.util, util_max, window,
                               json.dumps(gpu.apps), now))
        self.conn.execute('COMMIT')

    def pid(self, job_id):
        try:
            return int(self.conf.pid_file(job_id).read_text().strip())
        except (OSError, ValueError):
            return None

    def exit_code(self, job_id):
        try:
            return int(self.conf.exit_file(job_id).read_text().strip())
        except (OSError, ValueError):
            return None

    def follow(self, job, pid, alive, now):
        code = self.exit_code(job['id'])
        if job['state'] == 'cancelling':
            if pid is None or pid not in alive:
                db.update(self.conn, job['id'], state='cancelled', ended=now, exit_code=code)
                log.info('job %d cancelled', job['id'])
                return
            signal = 'KILL' if now - self.killed.setdefault(job['id'], now) > KILL_AFTER else 'TERM'
            hosts.run(self.conf, job['host'], f'kill -{signal} -- -{pid} 2>/dev/null; kill -{signal} {pid} 2>/dev/null; true')
            return
        if code is not None:
            db.update(self.conn, job['id'], state='done' if code == 0 else 'failed', ended=now, exit_code=code)
            log.info('job %d exited %d', job['id'], code)
            return
        if pid is None:
            if now - job['started'] > START_TIMEOUT:
                db.update(self.conn, job['id'], state='lost', ended=now, reason='its script never started')
            return
        if pid in alive:
            self.missing.pop(job['id'], None)
            return
        self.missing[job['id']] = self.missing.get(job['id'], 0) + 1
        if self.missing[job['id']] >= LOST_AFTER:
            db.update(self.conn, job['id'], state='lost', ended=now, reason='process gone without an exit code')
            log.info('job %d lost', job['id'])

    # -- starting ----------------------------------------------------------------------
    def start_jobs(self, gpus, resources, now):
        pending = db.jobs(self.conn, "state = 'pending'")
        if not pending:
            return
        running = db.jobs(self.conn, "state IN ('running', 'cancelling')")
        placements, reasons = schedule.place(pending, gpus, self.history, running, now, self.conf.warmup,
                                            resources, self.conf.cpu_target, self.conf.ram_reserve)
        for job_id, reason in reasons.items():
            db.update(self.conn, job_id, reason=reason)
        used = {(gpu.host, gpu.idx): gpu.used for gpu in gpus}
        host_samples = {h.host: {'load1': h.load1, 'ram_available': h.ram_available} for h in resources}
        for job, host, indices in placements:
            # Claim it first: a `gq cancel` in between leaves it cancelled.
            claimed = self.conn.execute("UPDATE jobs SET state = 'running', started = ?, host = ?, gpus = ?, "
                                        "launch_used = ?, launch_host = ?, reason = NULL WHERE id = ? AND state = 'pending'",
                                        (now, host, ','.join(map(str, indices)),
                                         json.dumps({str(i): used[(host, i)] for i in indices}),
                                         json.dumps(host_samples[host]), job['id'])).rowcount
            if not claimed:
                continue
            try:
                self.launch(job, host, indices)
                log.info('job %d started on %s GPU %s', job['id'], host, indices)
            except Exception as error:
                db.update(self.conn, job['id'], state='failed', ended=time.time(), reason=f'launch: {error}')
                log.warning('job %d launch failed: %s', job['id'], error)

    def launch(self, job, host, indices):
        conf, job_id = self.conf, job['id']
        for path in (conf.pid_file(job_id), conf.exit_file(job_id)):
            path.unlink(missing_ok=True)
        gpu_list = ','.join(map(str, indices))
        command = job['command'].replace('{gpus}', ' '.join(map(str, indices))).replace('{gpu}', str(indices[0]) if indices else '')
        env = json.loads(job['env'])
        if host != config_module.LOCAL:     # a proxy on this machine's loopback is not on the others
            env = {key: value for key, value in env.items() if not key.lower().endswith('_proxy')}
        env.update(GQ_JOB_ID=str(job_id), GQ_HOST=host, GQ_GPUS=gpu_list, CUDA_DEVICE_ORDER='PCI_BUS_ID')
        if job['mask']:
            env['CUDA_VISIBLE_DEVICES'] = gpu_list
        exit_file = shlex.quote(str(conf.exit_file(job_id)))
        lines = ['#!/bin/bash', f'echo $$ > {shlex.quote(str(conf.pid_file(job_id)))}',
                 *(f'export {key}={shlex.quote(value)}' for key, value in env.items()),
                 f'cd {shlex.quote(job["cwd"])} || {{ echo 97 > {exit_file}; exit 97; }}',
                 f'echo "[gq] job {job_id} on {host} GPU {gpu_list}, $(date \'+%F %T\')"',
                 f'echo "[gq] $ "{shlex.quote(command)}',
                 f'bash -c {shlex.quote(command)}',
                 'code=$?',
                 'echo "[gq] exit $code, $(date \'+%F %T\')"',
                 f'echo $code > {exit_file}.tmp && mv {exit_file}.tmp {exit_file}']
        script = conf.script(job_id)
        script.write_text('\n'.join(lines) + '\n')
        result = hosts.run(conf, host, f'setsid bash {shlex.quote(str(script))} > {shlex.quote(str(conf.log(job_id)))} '
                                       '2>&1 < /dev/null & disown; true')
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f'exit {result.returncode}')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    Daemon(config_module.load()).serve()


if __name__ == '__main__':
    main()
