from dataclasses import replace

import pytest

from gpu_queue.cpu import History
from gpu_queue.cpu_sample import read_cpu
from gpu_queue.hosts import Host


def sample(stamp, cores):
    return Host('a', 56., None, 448., 440., cores * stamp * 1_000_000, stamp, 'scope')


def test_full_minute_of_cgroup_usage_not_host_load():
    history = History()
    assert history.add(sample(0., 2.)) is None
    assert history.add(sample(30., 2.)) is None
    assert history.add(sample(60., 2.)) == 2.
    # Next minute is idle; the old busy interval slides out of the mean.
    assert history.add(replace(sample(90., 0.), cpu_usage=120_000_000)) == 1.
    assert history.add(replace(sample(120., 0.), cpu_usage=120_000_000)) == 0.


def test_irregular_polls_interpolate_the_window_boundary():
    history = History()
    for stamp in (10., 35., 55.):
        assert history.add(sample(stamp, 1.5)) is None
    assert history.add(sample(75., 1.5)) == 1.5  # boundary is t=15, not oldest sample at t=10
    assert history.add(sample(83., 1.5)) == 1.5


@pytest.mark.parametrize('changed', ['counter', 'clock', 'gap', 'scope', 'capacity'])
def test_restart_or_replacement_requires_a_fresh_window(changed):
    history = History()
    history.add(sample(100., 1.))
    assert history.add(sample(160., 1.)) == 1.
    next_sample = sample(165., 1.)
    if changed == 'counter':
        next_sample.cpu_usage = 0
    elif changed == 'clock':
        next_sample.cpu_time = 1.
    elif changed == 'gap':
        next_sample.cpu_time = 221.
    elif changed == 'scope':
        next_sample.cpu_scope = 'other'
    else:
        next_sample.cpus = 28.
    assert history.add(next_sample) is None
    assert history.add(replace(next_sample, cpu_time=next_sample.cpu_time + 60,
                               cpu_usage=next_sample.cpu_usage + 60_000_000)) == 1.


def test_forget_requires_resampling_only_that_host():
    history = History()
    history.add(sample(0., 0.))
    history.add(replace(sample(0., 0.), host='b'))
    history.forget('a')
    assert history.add(sample(60., 0.)) is None
    assert history.add(replace(sample(60., 0.), host='b')) == 0.


def test_probe_reads_current_cgroup_and_respects_ancestor_quota(tmp_path, monkeypatch):
    root = tmp_path / 'cgroup'
    group = root / 'parent' / 'container'
    group.mkdir(parents=True)
    (group / 'cpu.stat').write_text('usage_usec 12000000\nuser_usec 10000000\nsystem_usec 2000000\n')
    (group / 'cpu.max').write_text('max 100000\n')
    (group.parent / 'cpu.max').write_text('550000 100000\n')
    (root / 'cpu.max').write_text('800000 100000\n')
    proc = tmp_path / 'proc_cgroup'
    proc.write_text('0::/parent/container\n')
    monkeypatch.setattr('os.sched_getaffinity', lambda pid: set(range(16)))
    capacity, usage, stamp, scope = read_cpu(root, proc)
    assert capacity == 5.5 and usage == 12000000 and stamp > 0
    assert scope == str(group.stat().st_ino)
    monkeypatch.setattr('os.sched_getaffinity', lambda pid: {0, 1, 2, 3})
    assert read_cpu(root, proc)[0] == 4.
    proc.write_text('0::/\n')
    (root / 'cpu.stat').write_text('usage_usec 123\n')
    assert read_cpu(root, proc)[1] == 123
