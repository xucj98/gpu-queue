"""``gq``: submit jobs with CPU/RAM/GPU budgets, follow them, see host resources.

    gq submit [--mem 8] [--util 30] [--window 30] [--gpus 1] [--host H] [--name N] -- COMMAND ...
    gq ls [-a]                 running and pending jobs (-a: and the recently finished)
    gq hosts                   container cores, 60 s CPU usage, available RAM, running declarations
    gq gpus                    host resources and GPUs: memory, utilisation, running jobs
    gq show ID / gq log [-f] ID / gq cancel ID ...
    gq wait ID ...             one line per state change; exits when all finished (0: all succeeded)
    gq daemon [status|stop|restart|run]

In COMMAND, ``{gpus}`` becomes the job's physical GPU indices (space separated) and ``{gpu}``
the first; the job also sees ``GQ_GPUS`` (comma separated) and, unless ``--no-mask``,
``CUDA_VISIBLE_DEVICES`` set to them.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from gpu_queue import config as config_module, daemon, db

# Not carried from the submitting shell to the job.
ENV_SKIP = {'CUDA_VISIBLE_DEVICES', 'SSH_AUTH_SOCK', 'SSH_CLIENT', 'SSH_CONNECTION', 'SSH_TTY', 'DISPLAY',
            'HOSTNAME', 'PWD', 'OLDPWD', 'SHLVL', '_', 'TMUX', 'TMUX_PANE', 'TERM_SESSION_ID', 'XDG_RUNTIME_DIR'}


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return number


def _gpu_count(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be >= 0 (0 for CPU-only jobs)')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(prog='gq', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    conf = config_module.load()
    defaults = conf.defaults

    p = sub.add_parser('submit', help='queue a command')
    p.add_argument('--name')
    p.add_argument('--owner', default=os.environ.get('GQ_OWNER'), help='default: $GQ_OWNER, else the directory name')
    p.add_argument('--mem', type=float, default=defaults['mem'], help=f'GB free on each GPU (default {defaults["mem"]:g})')
    p.add_argument('--cpus', type=_positive, default=defaults['cpus'],
                   help=f'logical cores per job, may be fractional (default {defaults["cpus"]:g})')
    p.add_argument('--ram', type=_positive, default=defaults['ram'],
                   help=f'GiB RAM per job (default {defaults["ram"]:g})')
    p.add_argument('--util', type=float, default=defaults['util'],
                   help=f'utilisation below this %% over the window (default {defaults["util"]:g})')
    p.add_argument('--window', type=float, default=defaults['window'], help=f'seconds (default {defaults["window"]:g})')
    p.add_argument('--gpus', type=_gpu_count, default=1, help='GPUs on one host; 0 for CPU-only (default 1)')
    p.add_argument('--host', action='append', help='allowed host (repeatable; default any)')
    p.add_argument('--priority', type=int, default=0, help='higher first (default 0)')
    p.add_argument('--no-mask', action='store_true', help='do not set CUDA_VISIBLE_DEVICES')
    p.add_argument('--cwd', type=Path, default=Path.cwd())
    p.add_argument('--wait', action='store_true', help='then wait for it (as gq wait)')
    p.add_argument('command', nargs=argparse.REMAINDER, help='after --; one argument is run as a shell line')

    p = sub.add_parser('ls', help='running and pending jobs')
    p.add_argument('-a', '--all', action='store_true', help='also jobs finished in the last --since hours')
    p.add_argument('--since', type=float, default=6.)
    p.add_argument('--owner')
    sub.add_parser('gpus', help="the hosts' GPUs and the jobs on them")
    sub.add_parser('hosts', help='host CPU/RAM samples and running job declarations')
    p = sub.add_parser('show', help="a job's details")
    p.add_argument('id', type=int)
    p = sub.add_parser('log', help="a job's output")
    p.add_argument('id', type=int)
    p.add_argument('-f', '--follow', action='store_true')
    p.add_argument('-n', '--lines', type=int, default=40)
    p = sub.add_parser('cancel', help='cancel jobs (running ones are killed)')
    p.add_argument('ids', type=int, nargs='+')
    p = sub.add_parser('wait', help='follow jobs until they finish')
    p.add_argument('ids', type=int, nargs='+')
    p.add_argument('--timeout', type=float, help='seconds (exit 124 when reached)')
    p.add_argument('--quiet', action='store_true', help='only the final line per job')
    p = sub.add_parser('daemon', help='the scheduling daemon')
    p.add_argument('action', nargs='?', default='status', choices=('status', 'stop', 'restart', 'run'))

    args = parser.parse_args(argv)
    if args.cmd == 'daemon' and args.action == 'run':
        return daemon.main()
    conn = db.connect(conf)
    if args.cmd in ('submit', 'ls', 'gpus', 'hosts', 'wait'):
        daemon.ensure(conf)
    return globals()['cmd_' + args.cmd](conf, conn, args) or 0


# -- commands --------------------------------------------------------------------------------
def cmd_submit(conf, conn, args):
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        sys.exit('gq submit: no command (put it after --)')
    line = command[0] if len(command) == 1 else shlex.join(command)
    env = {key: value for key, value in os.environ.items() if key not in ENV_SKIP and not key.startswith('GQ_')}
    if args.host:
        unknown = set(args.host) - set(conf.hosts)
        if unknown:
            sys.exit(f'gq submit: unknown host(s) {", ".join(sorted(unknown))}; configured: {", ".join(conf.hosts)}')
    job_id = db.submit(conn, name=args.name or _default_name(command),
                       owner=args.owner or args.cwd.resolve().name, command=line, cwd=str(args.cwd.resolve()),
                       env=env, mem=args.mem, util=args.util, window=args.window, ngpus=args.gpus,
                       cpus=args.cpus, ram=args.ram,
                       hosts=args.host, mask=0 if args.no_mask else 1, priority=args.priority)
    print(job_id, flush=True)
    if args.wait:
        # Like srun: stopping the waiting client (Ctrl-C, or a batch runner's SIGTERM) cancels the job.
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            return cmd_wait(conf, conn, argparse.Namespace(ids=[job_id], timeout=None, quiet=False))
        except KeyboardInterrupt:
            db.cancel(conn, job_id)
            print(f'gq: cancelled job {job_id}', file=sys.stderr, flush=True)
            return 130


def cmd_ls(conf, conn, args):
    owner = ('AND owner = ?', (args.owner,)) if args.owner else ('', ())
    running = db.jobs(conn, f"state IN ('running', 'cancelling') {owner[0]}", owner[1])
    pending = db.jobs(conn, f"state = 'pending' {owner[0]}", owner[1])
    pending.sort(key=lambda job: (-job['priority'], job['id']))
    now = time.time()
    print(_daemon_line(conf, conn))
    print(f'\nRUNNING ({len(running)})')
    _table([('ID', 'NAME', 'OWNER', 'HOST:GPU', 'GPU MEM', 'CPUS', 'RAM', 'ELAPSED', 'STATE')] + [
        (j['id'], _cut(j['name'], 24), _cut(j['owner'], 16), f'{j["host"]}:{j["gpus"] or "-"}', f'{j["mem"]:g}G',
         f'{j["cpus"]:g}', f'{j["ram"]:g}G',
         _duration(now - j['started']), j['state']) for j in running])
    print(f'\nPENDING ({len(pending)})')
    _table([('#', 'ID', 'NAME', 'OWNER', 'NEEDS', 'WAITING', 'REASON')] + [
        (k + 1, j['id'], _cut(j['name'], 24), _cut(j['owner'], 16), _needs(j), _duration(now - j['submitted']),
         j['reason'] or '') for k, j in enumerate(pending)])
    if args.all:
        done = db.jobs(conn, f"state IN {db.FINISHED} AND ended > ? {owner[0]}", (now - args.since * 3600, *owner[1]))
        print(f'\nFINISHED in the last {args.since:g} h ({len(done)})')
        _table([('ID', 'NAME', 'OWNER', 'HOST:GPU', 'STATE', 'EXIT', 'RAN', 'ENDED')] + [
            (j['id'], _cut(j['name'], 24), _cut(j['owner'], 16), f'{j["host"] or "-"}:{j["gpus"] or "-"}', j['state'],
             '' if j['exit_code'] is None else j['exit_code'],
             _duration(j['ended'] - j['started']) if j['started'] else '-', _ago(now - j['ended']))
            for j in done[-30:]])


def cmd_gpus(conf, conn, args):
    now = time.time()
    print(_daemon_line(conf, conn))
    _host_table(conf, conn, now)
    print()
    jobs_on = {}
    for job in db.jobs(conn, "state IN ('running', 'cancelling')"):
        for idx in (job['gpus'] or '').split(','):
            if not idx:
                continue
            jobs_on.setdefault((job['host'], int(idx)), []).append(job)
    host_rows = {row['host']: row for row in conn.execute('SELECT * FROM hosts')}
    rows = [('HOST', 'GPU', 'NAME', 'MEM USED/TOTAL', 'FREE', 'UTIL', 'MAX', 'PROCS', 'GQ JOBS')]
    for host in conf.hosts:
        state = host_rows.get(host)
        if state is not None and not state['ok']:
            rows.append((host, '-', f'DOWN: {_cut(state["error"], 60)}', '', '', '', '', '', ''))
            continue
        for gpu in conn.execute('SELECT * FROM gpus WHERE host = ? ORDER BY idx', (host,)):
            apps = json.loads(gpu['apps'] or '[]')
            jobs = jobs_on.get((host, gpu['idx']), [])
            rows.append((host, gpu['idx'], gpu['name'].replace('NVIDIA GeForce ', ''),
                         f'{gpu["used"]:5.1f}/{gpu["total"]:.0f}G', f'{gpu["total"] - gpu["used"]:.1f}G',
                         f'{gpu["util"]:.0f}%', '-' if gpu['util_max'] is None else f'{gpu["util_max"]:.0f}%',
                         len(apps), ' '.join(f'{j["id"]}({_cut(j["owner"], 10)})' for j in jobs)))
    _table(rows)
    print(f'\nMAX: highest utilisation over the last {conf.defaults["window"]:g} s. PROCS: GPU processes '
          '(any owner; nvidia-smi pids are host pids, not matched to jobs).')


def cmd_hosts(conf, conn, args):
    print(_daemon_line(conf, conn))
    _host_table(conf, conn, time.time())


def _host_table(conf, conn, now):
    states = {row['host']: row for row in conn.execute('SELECT * FROM hosts')}
    samples = {row['host']: row for row in conn.execute('SELECT * FROM host_resources')}
    running = db.jobs(conn, "state IN ('running', 'cancelling')")
    rows = [('HOST', 'CORES', 'CPU USED 1m', 'CPU TARGET', 'RAM AVAIL/TOTAL', 'GQ CPUS', 'GQ RAM', 'SAMPLE')]
    for host in conf.hosts:
        state, resource = states.get(host), samples.get(host)
        jobs = [j for j in running if j['host'] == host]
        cpus, ram = sum(j['cpus'] for j in jobs), sum(j['ram'] for j in jobs)
        if not state or not state['ok'] or not resource:
            status = f'DOWN: {_cut(state["error"], 60)}' if state and not state['ok'] else 'still sampling'
            rows.append((host, '-', '-', '-', '-', f'{cpus:g}', f'{ram:g}G', status))
        else:
            cpu_used = '-' if resource['cpu_used'] is None else f'{resource["cpu_used"]:.2f}'
            status = 'CPU still sampling (60 s)' if resource['cpu_used'] is None else _ago(now - resource['updated'])
            rows.append((host, f'{resource["cpus"]:g}', cpu_used,
                         f'{resource["cpus"] * conf.cpu_target:.2f}',
                         f'{resource["ram_available"]:.1f}/{resource["ram_total"]:.1f}G',
                         f'{cpus:g}', f'{ram:g}G', status))
    _table(rows)
    print('  CPU USED 1m: container CPU cores used, mean over the last 60 s. CPU TARGET: cores.')
    print('  GQ CPUS/RAM: declarations of running/cancelling jobs; not measured usage. RAM: GiB.')


def cmd_show(conf, conn, args):
    job = db.job(conn, args.id) or sys.exit(f'gq: no job {args.id}')
    job['env'] = f'{len(json.loads(job["env"]))} variables'
    for key in ('submitted', 'started', 'ended'):
        if job[key]:
            job[key] = time.strftime('%F %T', time.localtime(job[key]))
    job['log'] = str(conf.log(args.id))
    for key, value in job.items():
        print(f'{key:12} {value}')


def cmd_log(conf, conn, args):
    path = conf.log(args.id)
    if not path.exists():
        sys.exit(f'gq: no log yet for job {args.id} ({path})')
    if not args.follow:
        return subprocess.call(['tail', '-n', str(args.lines), str(path)])
    process = subprocess.Popen(['tail', '-n', str(args.lines), '-F', str(path)])
    try:
        while (db.job(conn, args.id) or {}).get('state') in db.ACTIVE:
            time.sleep(2)
        time.sleep(1)
    except KeyboardInterrupt:
        pass
    process.send_signal(signal.SIGTERM)


def cmd_cancel(conf, conn, args):
    for job_id in args.ids:
        print(job_id, db.cancel(conn, job_id) or 'no such job')


def cmd_wait(conf, conn, args):
    """One line per state change of the jobs; exit 0 when all ended done, 1 otherwise, 124 on timeout."""
    deadline = args.timeout and time.time() + args.timeout
    seen = {}
    while True:
        jobs = {job_id: db.job(conn, job_id) for job_id in args.ids}
        for job_id, job in jobs.items():
            if job is None:
                sys.exit(f'gq: no job {job_id}')
            key = (job['state'], job['host'])
            if seen.get(job_id) != key:
                seen[job_id] = key
                if not args.quiet or job['state'] in db.FINISHED:
                    print(_event(conf, job), flush=True)
        if all(job['state'] in db.FINISHED for job in jobs.values()):
            return 0 if all(job['state'] == 'done' for job in jobs.values()) else 1
        if deadline and time.time() > deadline:
            return 124
        time.sleep(min(conf.poll, 5.))


def cmd_daemon(conf, conn, args):
    pid = daemon.running_daemon(conf)
    if args.action == 'status':
        print(_daemon_line(conf, conn))
        return 0 if pid else 1
    if pid and pid > 0:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if daemon.running_daemon(conf) is None:
                break
            time.sleep(.1)
        print(f'stopped daemon {pid}')
    if args.action == 'restart':
        print(f'started daemon {daemon.ensure(conf)}')


def _default_name(command):
    """The program, or the script it runs (``uv run python x.py`` -> ``x.py``)."""
    words = shlex.split(command[0]) if len(command) == 1 else command
    scripts = [w for w in words if w.endswith(('.py', '.sh'))]
    return Path(scripts[0] if scripts else words[0]).name


# -- formatting ------------------------------------------------------------------------------
def _event(conf, job):
    where = f' on {job["host"]}:{job["gpus"]}' if job['host'] else ''
    text = f'[gq] job {job["id"]} ({job["name"]}) {job["state"]}{where}'
    if job['state'] in db.FINISHED:
        if job['exit_code'] is not None:
            text += f', exit {job["exit_code"]}'
        if job['started']:
            text += f', ran {_duration(job["ended"] - job["started"])}'
        if job['reason']:
            text += f', {job["reason"]}'
        text += f'; log {conf.log(job["id"])}'
    return text


def _daemon_line(conf, conn):
    pid = daemon.running_daemon(conf)
    beat = float(db.get_meta(conn, 'heartbeat', 0))
    age = time.time() - beat
    status = f'daemon {pid}' if pid else 'daemon NOT running'
    return f'{status}, last sample {_ago(age) if beat else "never"}; hosts: {", ".join(conf.hosts)}'


def _needs(job):
    hosts = json.loads(job['hosts']) if job['hosts'] else None
    gpu = f'{job["ngpus"]}x >={job["mem"]:g}G, <{job["util"]:g}%/{job["window"]:g}s' if job['ngpus'] else 'CPU-only'
    text = f'{job["cpus"]:g} CPU, {job["ram"]:g}G RAM; {gpu}'
    return text + (f' @{",".join(hosts)}' if hosts else '')


def _table(rows):
    if len(rows) == 1:
        print('  (none)')
        return
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print('  ' + '  '.join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def _cut(text, width):
    text = text or ''
    return text if len(text) <= width else text[:width - 1] + '…'


def _duration(seconds):
    seconds = int(max(0, seconds))
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'


def _ago(seconds):
    return f'{_duration(seconds)} ago'


if __name__ == '__main__':
    sys.exit(main())
