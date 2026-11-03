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


def test_host_table_shows_container_cpu_and_sampling_state(tmp_path, capsys):
    from gpu_queue import db
    conf = config.Config(state_dir=tmp_path, hosts=['a'])
    conn = db.connect(conf)
    conn.execute("INSERT INTO hosts VALUES ('a', 1, NULL, 100)")
    conn.execute("INSERT INTO host_resources VALUES ('a', 55.5, NULL, 100, 90, 100)")
    cli._host_table(conf, conn, 101.)
    text = capsys.readouterr().out
    assert 'CPU USED 1m' in text and 'LOAD1' not in text
    assert 'CPU still sampling (60 s)' in text
    conn.execute("UPDATE host_resources SET cpu_used = 1.25 WHERE host = 'a'")
    cli._host_table(conf, conn, 101.)
    text = capsys.readouterr().out
    assert '55.5' in text and '1.25' in text and f'{55.5 * .85:.2f}' in text
    conn.close()
