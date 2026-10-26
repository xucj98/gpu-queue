"""Commands on the hosts (local, or over SSH with a shared connection) and GPU samples.

Process ids: inside these containers ``nvidia-smi`` reports host-namespace pids, unrelated to
the pids ``ps`` and ``kill`` see. Jobs are therefore tracked by the pid their script records
itself (container namespace, used only for ``ps`` and ``kill``), and GPU processes only by the
pids ``nvidia-smi`` reports, never matched against the former. A job's process counts as alive
unless it is gone or a zombie (the containers' init does not reap orphans).
"""

from dataclasses import dataclass
import subprocess

from gpu_queue.config import LOCAL


def run(config, host, script, timeout=20):
    """Run the bash ``script`` on ``host``; ``CompletedProcess`` (text)."""
    if host == LOCAL:
        command = ['bash', '-c', script]
    else:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'ControlMaster=auto',
                   '-o', f'ControlPath={config.state_dir}/ssh-%C', '-o', 'ControlPersist=600',
                   host, 'bash', '-c', _quote(script)]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def _quote(text):
    return "'" + text.replace("'", "'\"'\"'") + "'"


@dataclass
class Gpu:
    host: str
    idx: int
    uuid: str
    name: str
    total: float    # GB
    used: float     # GB
    util: float     # %
    apps: list      # [(nvidia-smi pid, GB)]


# s; a heavily loaded host answers slowly, and a timeout marks it down until the next sample
SAMPLE_TIMEOUT = 60

SAMPLE = """
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || exit 3
echo ---
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
echo ---
pids={pids}
[ -n "$pids" ] && ps -o pid=,stat= -p "$pids" 2>/dev/null | awk '$2 !~ /^Z/ {{print $1}}'
true
"""


def sample(config, host, pids=()):
    """``(gpus, alive)``: the GPUs of ``host`` and which of ``pids`` (container pids) are alive.

    One ``ps`` for all pids: a ``ps`` per pid outlasted the timeout with hundreds of running jobs."""
    result = run(config, host, SAMPLE.format(pids=','.join(str(int(p)) for p in pids)), timeout=SAMPLE_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip().splitlines()[-1:] or f'exit {result.returncode}')
    gpu_part, app_part, alive_part = (result.stdout.split('---\n') + ['', ''])[:3]
    apps = {}
    for line in app_part.strip().splitlines():
        fields = [f.strip() for f in line.split(',')]
        if len(fields) == 3 and fields[1].isdigit():
            apps.setdefault(fields[0], []).append((int(fields[1]), _gb(fields[2])))
    gpus = []
    for line in gpu_part.strip().splitlines():
        index, uuid, name, total, used, util = [f.strip() for f in line.split(',')]
        gpus.append(Gpu(host, int(index), uuid, name, _gb(total), _gb(used), _number(util), apps.get(uuid, [])))
    alive = {int(p) for p in alive_part.split()}
    return gpus, alive


def _gb(mib):
    return _number(mib) / 1024.


def _number(text):
    try:
        return float(text)
    except ValueError:      # [N/A]
        return 0.
