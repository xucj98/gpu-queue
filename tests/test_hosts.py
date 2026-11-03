from subprocess import CompletedProcess

import pytest

from gpu_queue import hosts
from gpu_queue.config import Config


def test_samples_resources_and_liveness_in_one_command(monkeypatch):
    calls = []
    output = ('0, GPU-0, RTX 4090, 24576, 1024, 7\n---\n'
              'GPU-0, 42, 512\n---\n123\n---\n[128, 12500000, 100, "scope"]\n896\n850\n')

    def run(conf, host, script, timeout):
        calls.append((host, script, timeout))
        return CompletedProcess([], 0, output, '')

    monkeypatch.setattr(hosts, 'run', run)
    gpus, alive, resource = hosts.sample(Config(), 'a', [123, 456])
    assert len(calls) == 1 and 'pids=123,456' in calls[0][1]
    assert '/proc/meminfo' in calls[0][1] and 'cpu.stat' in calls[0][1]
    assert '/proc/loadavg' not in calls[0][1]
    assert alive == {123} and gpus[0].used == 1. and gpus[0].apps == [(42, .5)]
    assert resource == hosts.Host('a', 128, None, 896., 850., 12500000, 100., 'scope')


def test_cpu_host_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(hosts, 'run', lambda *a, **kw: CompletedProcess([], 0, '---\n---\n---\n[8, 2000000, 100, "scope"]\n32\n28\n', ''))
    gpus, alive, resource = hosts.sample(Config(), 'cpu')
    assert gpus == [] and alive == set() and resource.cpus == 8


@pytest.mark.parametrize('resource', ['[8, NaN, 100, "scope"]\n32\n28', '[8, 2, 100, "scope"]\n32\n33',
                                    '[0, 2, 100, "scope"]\n32\n28', '[8, 2, 100, "scope"]\n32'])
def test_bad_host_sample_fails_closed(monkeypatch, resource):
    monkeypatch.setattr(hosts, 'run', lambda *a, **kw: CompletedProcess([], 0, '---\n---\n---\n'+resource+'\n', ''))
    with pytest.raises(ValueError):
        hosts.sample(Config(), 'cpu')
