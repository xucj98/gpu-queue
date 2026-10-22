"""Configuration: ``$GQ_CONFIG`` or ``~/.config/gpu-queue/config.toml``, every key optional::

    hosts = ["localhost", "wuwen-4090-2"]     # "localhost": this machine, no SSH
    shared_dir = "/mnt/public/.../gpu-queue"  # job scripts, logs, exit codes: seen by every host
    state_dir = "~/.local/state/gpu-queue"    # the queue database, on the local disk
    poll = 5            # s between GPU samples
    warmup = 120        # s a new job's requested memory stays reserved on its GPUs
    [defaults]          # of `gq submit`
    mem = 4.0           # GB free
    util = 30.0         # % utilisation, highest over the window
    window = 30.0       # s
"""

from dataclasses import dataclass, field
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
    defaults: dict = field(default_factory=lambda: {'mem': 4., 'util': 30., 'window': 30.})

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
        config.defaults.update(data.get('defaults', {}))
    config.state_dir.mkdir(parents=True, exist_ok=True)
    for sub in ('jobs', 'logs'):
        (config.shared_dir / sub).mkdir(parents=True, exist_ok=True)
    return config
