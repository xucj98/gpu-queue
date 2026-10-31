import argparse

import pytest

from gpu_queue import cli, config


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_cpu_ram_requests_are_positive_and_finite(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cli._positive(value)


def test_config_loads_resource_limits_without_changing_existing_defaults(tmp_path, monkeypatch):
    path = tmp_path / 'config.toml'
    path.write_text(f'state_dir = "{tmp_path}/state"\nshared_dir = "{tmp_path}/shared"\n'
                    'cpu_target = 0.8\nram_reserve = 0.1\n[defaults]\ncpus = 2.25\nram = 6\n')
    monkeypatch.setenv('GQ_CONFIG', str(path))
    conf = config.load()
    assert (conf.cpu_target, conf.ram_reserve) == (.8, .1)
    assert (conf.defaults['cpus'], conf.defaults['ram'], conf.defaults['mem']) == (2.25, 6., 4.)


@pytest.mark.parametrize('setting', ['cpu_target = 0', 'cpu_target = 1.1', 'ram_reserve = 1',
                                    '[defaults]\ncpus = -1', '[defaults]\nram = inf'])
def test_invalid_resource_defaults(tmp_path, monkeypatch, setting):
    path = tmp_path / 'config.toml'
    path.write_text(setting)
    monkeypatch.setenv('GQ_CONFIG', str(path))
    with pytest.raises(ValueError):
        config.load()
