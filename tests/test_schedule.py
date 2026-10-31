import json

from gpu_queue.hosts import Gpu, Host
from gpu_queue.schedule import History, host_reservations, place


def gpu(host, idx, used, util=0., total=24.):
    return Gpu(host, idx, f'{host}-{idx}', 'RTX 4090', total, used, util, [])


def job(job_id, mem=4., util=30., window=30., ngpus=1, hosts=None, priority=0, cpus=1.5, ram=4.):
    return {'id': job_id, 'mem': mem, 'util': util, 'window': window, 'ngpus': ngpus,
            'hosts': json.dumps(hosts) if hosts else None, 'priority': priority, 'cpus': cpus, 'ram': ram}


def resources(gpus):
    return [Host(host, 128, 0., 512., 500.) for host in sorted({g.host for g in gpus})]


def history(gpus, utils=None, span=60.):
    h = History()
    for t in range(0, int(span) + 1, 5):
        h.add(float(t), [gpu(g.host, g.idx, g.used, (utils or {}).get((g.host, g.idx), g.util)) for g in gpus])
    return h


def test_memory_and_utilisation_thresholds():
    gpus = [gpu('a', 0, 20.), gpu('a', 1, 2.), gpu('a', 2, 2.)]
    h = history(gpus, utils={('a', 1): 80.})
    placements, reasons = place([job(1, mem=8.)], gpus, h, [], 60., 120., resources(gpus))
    assert [(p[1], p[2]) for p in placements] == [('a', [2])]
    placements, reasons = place([job(2, mem=8., util=90.)], gpus, h, [], 60., 120., resources(gpus))
    assert placements[0][2] == [1] or placements[0][2] == [2]


def test_busy_within_window_only():
    gpus = [gpu('a', 0, 0.)]
    h = History()
    h.add(0., [gpu('a', 0, 0., 90.)])
    for t in range(5, 61, 5):
        h.add(float(t), [gpu('a', 0, 0., 0.)])
    assert place([job(1, window=30.)], gpus, h, [], 60., 120., resources(gpus))[0]
    assert not place([job(2, window=60.)], gpus, h, [], 60., 120., resources(gpus))[0]


def test_needs_samples_spanning_the_window():
    gpus = [gpu('a', 0, 0.)]
    h = history(gpus, span=10.)
    placements, reasons = place([job(1, window=30.)], gpus, h, [], 10., 120., resources(gpus))
    assert not placements and 'still sampling' in reasons[1]


def test_packs_by_memory_and_reserves_for_warming_jobs():
    gpus = [gpu('a', 0, 2.)]
    h = history(gpus)
    pending = [job(i, mem=6., util=100.) for i in range(1, 6)]
    placements, reasons = place(pending, gpus, h, [], 60., 120., resources(gpus))
    assert [p[0]['id'] for p in placements] == [1, 2, 3]          # 22 GB free: three of 6 GB
    # A job started 10 s ago asked for 6 GB and has taken 1 GB so far: 5 GB stay reserved (17 GB free).
    running = [{'host': 'a', 'gpus': '0', 'mem': 6., 'started': 50., 'launch_used': json.dumps({'0': 1.}), 'cpus': 1.5, 'ram': 4.}]
    placements, _ = place([job(9, mem=18., util=100.)], [gpu('a', 0, 2.)], h, running, 60., 120., resources(gpus))
    assert not placements
    assert not place([job(9, mem=18., util=100.)], [gpu('a', 0, 2.)], h, running, 200., 120., resources(gpus))[0]   # no recent samples
    h.add(200., [gpu('a', 0, 2.)])
    assert place([job(9, mem=18., util=100.)], [gpu('a', 0, 2.)], h, running, 200., 120., resources(gpus))[0]      # warm-up over


def test_multi_gpu_on_one_host_priority_and_backfill():
    gpus = [gpu('a', 0, 0.), gpu('b', 0, 0.), gpu('b', 1, 0.)]
    h = history(gpus)
    placements, _ = place([job(1, ngpus=2)], gpus, h, [], 60., 120., resources(gpus))
    assert placements[0][1] == 'b' and sorted(placements[0][2]) == [0, 1]
    pending = [job(1, mem=30.), job(2, mem=20.), job(3, mem=20., priority=5)]
    placements, reasons = place(pending, gpus, h, [], 60., 120., resources(gpus))
    assert [p[0]['id'] for p in placements] == [3, 2] and 1 in reasons


def test_allowed_hosts():
    gpus = [gpu('a', 0, 0.), gpu('b', 0, 20.)]
    h = history(gpus)
    placements, reasons = place([job(1, mem=8., hosts=['b'])], gpus, h, [], 60., 120., resources(gpus))
    assert not placements and '1 < 8G free' in reasons[1]


def plan(pending, hosts, running=(), now=60., warmup=120.):
    gpus = [gpu(h.host, 0, 0.) for h in hosts]
    return place(pending, gpus, history(gpus), running, now, warmup, hosts)


def warming(job_id, started=50., cpus=3., ram=20., before_load=4., before_ram=100.):
    return dict(job(job_id, cpus=cpus, ram=ram), host='a', gpus='', started=started,
                launch_used='{}', launch_host=json.dumps({'load1': before_load, 'ram_available': before_ram}))


def test_cpu_capacity_and_fractional_requests():
    hosts = [Host('a', 4, 2., 100., 90.)]
    assert plan([job(1, cpus=1.4)], hosts)[0]
    placements, reasons = plan([job(1, cpus=1.5)], hosts)
    assert not placements and 'CPU a:' in reasons[1]
    assert 'RAM' not in reasons[1]


def test_ram_available_less_total_reserve():
    hosts = [Host('a', 16, 0., 100., 12.)]
    assert plan([job(1, ram=7.)], hosts)[0]
    placements, reasons = plan([job(1, ram=8.)], hosts)
    assert not placements and 'RAM a:' in reasons[1]
    assert 'CPU' not in reasons[1]


def test_same_poll_cannot_start_hundreds_before_load_rises():
    hosts = [Host('a', 8, 0., 100., 100.)]
    placements, reasons = plan([job(i, ngpus=0) for i in range(1, 457)], hosts)
    assert [p[0]['id'] for p in placements] == [1, 2, 3, 4]
    assert len(reasons) == 452 and all('CPU' in r for r in reasons.values())


def test_warmup_accounts_for_background_cpu_and_unconsumed_ram():
    hosts = [Host('a', 10, 5., 100., 95.)]
    running = [warming(1)]
    cpu, ram = host_reservations(hosts, running, 60., 120.)
    assert cpu == {'a': 7.} and ram == {'a': 15.}
    assert not plan([job(2, cpus=2.)], hosts, running)[0]  # 7 + 2 > 8.5
    assert plan([job(2, cpus=1.5, ram=75.)], hosts, running)[0]
    assert not plan([job(2, cpus=1.5, ram=76.)], hosts, running)[0]
    cpu, ram = host_reservations(hosts, running, 171., 120.)
    assert cpu == {'a': 5.} and ram == {'a': 0.}


def test_warmup_growth_is_credited_once_for_a_whole_batch():
    hosts = [Host('a', 32, 6., 100., 90.)]
    running = [warming(1, cpus=3., ram=10.), warming(2, cpus=3., ram=10.)]
    cpu, ram = host_reservations(hosts, running, 60., 120.)
    assert cpu == {'a': 10.} and ram == {'a': 10.}
    # Crediting the host's 10 GiB growth to BOTH jobs would incorrectly reserve zero.
    assert not plan([job(3, ram=76.)], hosts, running)[0]


def test_warmup_overlapping_batches_and_measured_usage_above_declarations():
    hosts = [Host('a', 32, 9., 100., 87.)]
    running = [warming(1, started=40., cpus=3., ram=10.),
               warming(2, started=50., cpus=3., ram=10., before_load=5., before_ram=97.)]
    cpu, ram = host_reservations(hosts, running, 60., 120.)
    assert cpu == {'a': 10.} and ram == {'a': 7.}
    hosts[0].load1 = 15.
    hosts[0].ram_available = 70.
    cpu, ram = host_reservations(hosts, running, 60., 120.)
    assert cpu == {'a': 15.} and ram == {'a': 0.}


def test_old_jobs_without_launch_samples_reserve_conservatively():
    hosts = [Host('a', 16, 5., 100., 90.)]
    running = [warming(1)]
    running[0]['launch_host'] = None
    cpu, ram = host_reservations(hosts, running, 60., 120.)
    assert cpu == {'a': 8.} and ram == {'a': 20.}


def test_cpu_and_ram_backfill_and_host_restriction():
    hosts = [Host('a', 8, 5., 100., 90.), Host('b', 16, 0., 100., 90.)]
    pending = [job(1, cpus=20.), job(2, ram=90.), job(3, cpus=3., hosts=['a']), job(4, cpus=3.)]
    placements, reasons = plan(pending, hosts)
    assert [(j['id'], h) for j, h, _ in placements] == [(4, 'b')]
    assert 'CPU' in reasons[1] and 'RAM' in reasons[2] and 'CPU a:' in reasons[3]
    assert 'CPU b:' not in reasons[3]


def test_cpu_only_host_without_gpus_and_ram_same_poll_reservation():
    hosts = [Host('a', 128, 0., 100., 25.)]
    placements, reasons = place([job(1, ngpus=0, ram=15.), job(2, ngpus=0, ram=15.),
                                 job(3, ngpus=0, ram=5.)], [], History(), [], 60., 120., hosts)
    assert [(j['id'], h, indices) for j, h, indices in placements] == [(1, 'a', []), (3, 'a', [])]
    assert 'RAM' in reasons[2]


def test_missing_host_sample_does_not_use_stale_gpus():
    gpus = [gpu('a', 0, 0.)]
    placements, reasons = place([job(1)], gpus, history(gpus), [], 60., 120., [])
    assert not placements and 'CPU/RAM sample required' in reasons[1]
