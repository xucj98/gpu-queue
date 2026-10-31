"""Configuration: ``$GQ_CONFIG`` or ``~/.config/gpu-queue/config.toml``, every key optional::

    hosts = ["localhost", "wuwen-4090-2"]     # "localhost": this machine, no SSH
    shared_dir = "/mnt/public/.../gpu-queue"  # job scripts, logs, exit codes: seen by every host
    state_dir = "~/.local/state/gpu-queue"    # the queue database, on the local disk
    poll = 5            # s between host/GPU samples
    warmup = 120        # s for CPU/RAM/GPU startup reservations
    cpu_target = 0.85   # fraction of logical cores
    ram_reserve = 0.05  # fraction of total RAM kept free
    [defaults]          # of `gq submit`
    mem = 4.0           # GB free
    util = 30.0         # % utilisation, highest over the window
    window = 30.0       # s
    cpus = 1.5          # logical cores, per job
    ram = 4.0           # GiB, per job
"""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tomllib

LOCAL = 'localhost'


@dataclass
class Config:
    hosts: list = field(default_factory=lambda: [LOCAL])
    shared_dir: Path = Path('~/.local/state/gpu-queue/shared').expanduser()
    state_dir: Path = Path('~/.local/state/gpu-queue').expanduser()
    poll: float = 5.
    warmup: float = 120.
    cpu_target: float = .85
    ram_reserve: float = .05
    defaults: dict = field(default_factory=lambda: {'mem': 4., 'util': 30., 'window': 30., 'cpus': 1.5, 'ram': 4.})

    @property
    def db(self):
        return self.state_dir / 'queue.db'

    def script(self, job_id):
        return self.shared_dir / 'jobs' / f'{job_id}.sh'

    def log(self, job_id):
        return self.shared_dir / 'logs' / f'{job_id}.log'

    def pid_file(self, job_id):
        return self.shared_dir / 'jobs' / f'{job_id}.pid'

    def exit_file(self, job_id):
        return self.shared_dir / 'jobs' / f'{job_id}.exit'


def path():
    return Path(os.environ.get('GQ_CONFIG', '~/.config/gpu-queue/config.toml')).expanduser()


def load():
    config = Config()
    file = path()
    if file.exists():
        data = tomllib.loads(file.read_text())
        config.hosts = list(data.get('hosts', config.hosts))
        for key in ('shared_dir', 'state_dir'):
            if key in data:
                setattr(config, key, Path(data[key]).expanduser())
        config.poll = float(data.get('poll', config.poll))
        config.warmup = float(data.get('warmup', config.warmup))
        config.cpu_target = float(data.get('cpu_target', config.cpu_target))
        config.ram_reserve = float(data.get('ram_reserve', config.ram_reserve))
        config.defaults.update(data.get('defaults', {}))
    if not 0. < config.cpu_target <= 1. or not 0. <= config.ram_reserve < 1.:
        raise ValueError('cpu_target must be in (0, 1]; ram_reserve in [0, 1)')
    for key in ('cpus', 'ram'):
        value = float(config.defaults[key])
        if not math.isfinite(value) or value <= 0.:
            raise ValueError(f'defaults.{key} must be positive and finite')
        config.defaults[key] = value
    config.state_dir.mkdir(parents=True, exist_ok=True)
    for sub in ('jobs', 'logs'):
        (config.shared_dir / sub).mkdir(parents=True, exist_ok=True)
    return config
