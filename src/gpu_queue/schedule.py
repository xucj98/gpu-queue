"""Placing pending jobs within host CPU/RAM and GPU budgets (no I/O: the daemon feeds it samples).

A GPU suits a job when, on an allowed host, its free memory (less what jobs started within
``warmup`` asked for but have not taken yet) is at least the job's ``mem`` and its utilisation
stayed below the job's ``util`` over the job's ``window`` (which the samples must cover). A job
asking for several GPUs gets them on one host. Jobs go by priority, then submission; a job
that does not fit does not hold back later ones that do (backfill). Of the suitable GPUs the
emptiest go first.
"""

from collections import deque
from itertools import groupby
import json


class History:
    """GPU samples ``(time, used GB, util %)`` per ``(host, index)`` over the last ``keep`` seconds."""

    def __init__(self, keep=600.):
        self.keep, self.samples = keep, {}

    def add(self, now, gpus):
        for gpu in gpus:
            samples = self.samples.setdefault((gpu.host, gpu.idx), deque())
            samples.append((now, gpu.used, gpu.util))
            while samples and samples[0][0] < now - self.keep:
                samples.popleft()

    def util_max(self, key, now, window):
        """Highest utilisation over the last ``window`` s, or None while the samples span less (or stopped)."""
        samples = self.samples.get(key)
        if not samples or samples[0][0] > now - window:
            return None
        recent = [util for t, _, util in samples if t >= now - window]
        return max(recent) if recent else None

    def forget(self, host):
        for key in [key for key in self.samples if key[0] == host]:
            del self.samples[key]


def reserved(running, gpus, now, warmup):
    """GB per ``(host, index)`` that jobs started within ``warmup`` asked for beyond what their GPU
    has gained since (their memory may still be coming)."""
    used = {(gpu.host, gpu.idx): gpu.used for gpu in gpus}
    out = {}
    for job in running:
        if job['started'] is None or now - job['started'] > warmup:
            continue
        before = json.loads(job['launch_used'] or '{}')
        for idx in _indices(job['gpus']):
            key = (job['host'], idx)
            gained = used.get(key, 0.) - before.get(str(idx), used.get(key, 0.))
            out[key] = out.get(key, 0.) + max(0., job['mem'] - gained)
    return out


def host_reservations(resources, running, now, warmup):
    """Effective CPU use and unconsumed RAM declarations, per host.

    Each launch sample anchors a *host-wide* reservation. Sum the declarations launched
    since that anchor, then credit the host's measured growth once, not once per job.
    Taking the largest overlapping reservation also keeps pre-existing CPU usage accounted for.
    Old jobs with no launch sample conservatively reserve their full declarations.
    """
    cpu, ram = {}, {}
    for host in resources:
        warming = [j for j in running if j['host'] == host.host and j['started'] is not None
                   and now - j['started'] <= warmup]
        # Snapshots from older versions use a different CPU metric and cannot anchor this one.
        known_jobs, unknown = [], []
        for job in warming:
            before = json.loads(job.get('launch_host') or '{}')
            (known_jobs if before.get('cpu_used') is not None else unknown).append(job)
        known = sorted(known_jobs, key=lambda j: j['started'], reverse=True)
        cpu_floor, ram_held, tail_cpu, tail_ram = host.cpu_used, 0., 0., 0.
        for _, batch in groupby(known, key=lambda j: j['started']):
            batch = list(batch)
            tail_cpu += sum(j['cpus'] for j in batch)
            tail_ram += sum(j['ram'] for j in batch)
            for job in batch:
                before = json.loads(job['launch_host'])
                cpu_floor = max(cpu_floor, before['cpu_used'] + tail_cpu)
                gained = max(0., before['ram_available'] - host.ram_available)
                ram_held = max(ram_held, tail_ram - gained)
        cpu[host.host] = cpu_floor + sum(j['cpus'] for j in unknown)
        ram[host.host] = ram_held + sum(j['ram'] for j in unknown)
    return cpu, ram


def place(pending, gpus, history, running, now, warmup, resources, cpu_target=.85, ram_reserve=.05):
    """``(placements, reasons)``; CPU/RAM are per host, GPU thresholds per card.

    Backfill is preserved: a job that does not fit never blocks later smaller jobs.
    The budgets below are decremented immediately, including within the same poll.
    """
    sampling = {h.host for h in resources if h.cpu_used is None}
    resources = [h for h in resources if h.cpu_used is not None]
    reserve = reserved(running, gpus, now, warmup)
    free = {(g.host, g.idx): g.total - g.used - reserve.get((g.host, g.idx), 0.) for g in gpus}
    cpu_used, ram_held = host_reservations(resources, running, now, warmup)
    ram_free = {h.host: h.ram_available - h.ram_total * ram_reserve - ram_held[h.host] for h in resources}
    by_host = {h.host: [] for h in resources}
    for gpu in gpus:
        if gpu.host in by_host:
            by_host[gpu.host].append(gpu)
    placements, reasons = [], {}
    for job in sorted(pending, key=lambda job: (-job['priority'], job['id'])):
        allowed = json.loads(job['hosts']) if job['hosts'] else None
        host_reasons = [f'CPU {host}: still sampling (60 s window)' for host in sorted(sampling)
                        if not allowed or host in allowed]
        candidates, short_mem, busy, warming = [], 0, 0, 0
        eligible = [h for h in resources if not allowed or h.host in allowed]
        for resource in eligible:
            host = resource.host
            cpu_ok = cpu_used[host] + job['cpus'] <= resource.cpus * cpu_target
            ram_ok = job['ram'] <= ram_free[host]
            if not cpu_ok:
                host_reasons.append(f'CPU {host}: {cpu_used[host]:.2f} + {job["cpus"]:g} > '
                                    f'{resource.cpus * cpu_target:.2f} cores')
            if not ram_ok:
                host_reasons.append(f'RAM {host}: {ram_free[host]:.2f}G available after reserves < {job["ram"]:g}G')
            suitable = []
            if job['ngpus']:
                for gpu in by_host[host]:
                    key = (host, gpu.idx)
                    util = history.util_max(key, now, job['window'])
                    if free[key] < job['mem']:
                        short_mem += 1
                    elif util is None:
                        warming += 1
                    elif util >= job['util']:
                        busy += 1
                    else:
                        suitable.append(gpu)
            if cpu_ok and ram_ok and len(suitable) >= job['ngpus']:
                suitable.sort(key=lambda gpu: (-free[(host, gpu.idx)], gpu.util))
                candidates.append((host, suitable[:job['ngpus']]))
        if candidates:
            host, chosen = max(candidates, key=lambda c: (sum(free[(c[0], g.idx)] for g in c[1]),
                               -sum(g.util for g in c[1]), -cpu_used[c[0]]))
            for gpu in chosen:
                free[(host, gpu.idx)] -= job['mem']
            cpu_used[host] += job['cpus']
            ram_free[host] -= job['ram']
            placements.append((job, host, [gpu.idx for gpu in chosen]))
        else:
            parts = list(host_reasons)
            if job['ngpus'] and eligible:
                total = sum(len(by_host[h.host]) for h in eligible)
                gpu_parts = [f'{short_mem} < {job["mem"]:g}G free'] if short_mem else []
                gpu_parts += [f'{busy} busy (util >= {job["util"]:g}% in {job["window"]:g}s)'] if busy else []
                gpu_parts += [f'{warming} still sampling'] if warming else []
                if not total or gpu_parts:
                    parts.append(f'need {job["ngpus"]} GPU(s) on one host; of {total}: ' + ', '.join(gpu_parts))
            reasons[job['id']] = '; '.join(parts) if eligible or host_reasons else 'no allowed host is up (CPU/RAM sample required)'
            if not reasons[job['id']]:
                reasons[job['id']] = f'need {job["ngpus"]} GPU(s) on one host'
    return placements, reasons


def _indices(text):
    return [int(i) for i in text.split(',')] if text else []
