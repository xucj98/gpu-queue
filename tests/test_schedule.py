import json

from gpu_queue.hosts import Gpu
from gpu_queue.schedule import History, place


def gpu(host, idx, used, util=0., total=24.):
    return Gpu(host, idx, f'{host}-{idx}', 'RTX 4090', total, used, util, [])


def job(job_id, mem=4., util=30., window=30., ngpus=1, hosts=None, priority=0):
    return {'id': job_id, 'mem': mem, 'util': util, 'window': window, 'ngpus': ngpus,
            'hosts': json.dumps(hosts) if hosts else None, 'priority': priority}


def history(gpus, utils=None, span=60.):
    h = History()
    for t in range(0, int(span) + 1, 5):
        h.add(float(t), [gpu(g.host, g.idx, g.used, (utils or {}).get((g.host, g.idx), g.util)) for g in gpus])
    return h


def test_memory_and_utilisation_thresholds():
    gpus = [gpu('a', 0, 20.), gpu('a', 1, 2.), gpu('a', 2, 2.)]
    h = history(gpus, utils={('a', 1): 80.})
    placements, reasons = place([job(1, mem=8.)], gpus, h, [], 60., 120.)
    assert [(p[1], p[2]) for p in placements] == [('a', [2])]
    placements, reasons = place([job(2, mem=8., util=90.)], gpus, h, [], 60., 120.)
    assert placements[0][2] == [1] or placements[0][2] == [2]


def test_busy_within_window_only():
    gpus = [gpu('a', 0, 0.)]
    h = History()
    h.add(0., [gpu('a', 0, 0., 90.)])
    for t in range(5, 61, 5):
        h.add(float(t), [gpu('a', 0, 0., 0.)])
    assert place([job(1, window=30.)], gpus, h, [], 60., 120.)[0]
    assert not place([job(2, window=60.)], gpus, h, [], 60., 120.)[0]


def test_needs_samples_spanning_the_window():
    gpus = [gpu('a', 0, 0.)]
    h = history(gpus, span=10.)
    placements, reasons = place([job(1, window=30.)], gpus, h, [], 10., 120.)
    assert not placements and 'still sampling' in reasons[1]


def test_packs_by_memory_and_reserves_for_warming_jobs():
    gpus = [gpu('a', 0, 2.)]
    h = history(gpus)
    pending = [job(i, mem=6., util=100.) for i in range(1, 6)]
    placements, reasons = place(pending, gpus, h, [], 60., 120.)
    assert [p[0]['id'] for p in placements] == [1, 2, 3]          # 22 GB free: three of 6 GB
    # A job started 10 s ago asked for 6 GB and has taken 1 GB so far: 5 GB stay reserved (17 GB free).
    running = [{'host': 'a', 'gpus': '0', 'mem': 6., 'started': 50., 'launch_used': json.dumps({'0': 1.})}]
    placements, _ = place([job(9, mem=18., util=100.)], [gpu('a', 0, 2.)], h, running, 60., 120.)
    assert not placements
    assert not place([job(9, mem=18., util=100.)], [gpu('a', 0, 2.)], h, running, 200., 120.)[0]   # no recent samples
    h.add(200., [gpu('a', 0, 2.)])
    assert place([job(9, mem=18., util=100.)], [gpu('a', 0, 2.)], h, running, 200., 120.)[0]      # warm-up over


def test_multi_gpu_on_one_host_priority_and_backfill():
    gpus = [gpu('a', 0, 0.), gpu('b', 0, 0.), gpu('b', 1, 0.)]
    h = history(gpus)
    placements, _ = place([job(1, ngpus=2)], gpus, h, [], 60., 120.)
    assert placements[0][1] == 'b' and sorted(placements[0][2]) == [0, 1]
    pending = [job(1, mem=30.), job(2, mem=20.), job(3, mem=20., priority=5)]
    placements, reasons = place(pending, gpus, h, [], 60., 120.)
    assert [p[0]['id'] for p in placements] == [3, 2] and 1 in reasons


def test_allowed_hosts():
    gpus = [gpu('a', 0, 0.), gpu('b', 0, 20.)]
    h = history(gpus)
    placements, reasons = place([job(1, mem=8., hosts=['b'])], gpus, h, [], 60., 120.)
    assert not placements and '1 < 8G free' in reasons[1]
