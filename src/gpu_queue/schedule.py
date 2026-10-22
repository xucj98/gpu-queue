"""Placing pending jobs on idle GPUs (no I/O: the daemon feeds it samples).

A GPU suits a job when, on an allowed host, its free memory (less what jobs started within
``warmup`` asked for but have not taken yet) is at least the job's ``mem`` and its utilisation
stayed below the job's ``util`` over the job's ``window`` (which the samples must cover). A job
asking for several GPUs gets them on one host. Jobs go by priority, then submission; a job
that does not fit does not hold back later ones that do (backfill). Of the suitable GPUs the
emptiest go first.
"""

from collections import deque
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


def place(pending, gpus, history, running, now, warmup):
    """``(placements, reasons)``: ``[(job, host, [indices])]`` to start now, and why each other
    pending job waits (``{id: text}``)."""
    reserve = reserved(running, gpus, now, warmup)
    free = {(g.host, g.idx): g.total - g.used - reserve.get((g.host, g.idx), 0.) for g in gpus}
    by_host = {}
    for gpu in gpus:
        by_host.setdefault(gpu.host, []).append(gpu)
    placements, reasons = [], {}
    for job in sorted(pending, key=lambda job: (-job['priority'], job['id'])):
        allowed = json.loads(job['hosts']) if job['hosts'] else None
        candidates, short_mem, busy, warming = [], 0, 0, 0
        for host, host_gpus in by_host.items():
            if allowed and host not in allowed:
                continue
            suitable = []
            for gpu in host_gpus:
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
            if len(suitable) >= job['ngpus']:
                suitable.sort(key=lambda gpu: (-free[(host, gpu.idx)], gpu.util))
                candidates.append((host, suitable[:job['ngpus']]))
        if candidates:
            host, chosen = max(candidates, key=lambda c: (sum(free[(c[0], g.idx)] for g in c[1]), -sum(g.util for g in c[1])))
            for gpu in chosen:
                free[(host, gpu.idx)] -= job['mem']
            placements.append((job, host, [gpu.idx for gpu in chosen]))
        else:
            total = sum(len(g) for h, g in by_host.items() if not allowed or h in allowed)
            parts = [f'{short_mem} < {job["mem"]:g}G free'] if short_mem else []
            parts += [f'{busy} busy (util >= {job["util"]:g}% in {job["window"]:g}s)'] if busy else []
            parts += [f'{warming} still sampling'] if warming else []
            reasons[job['id']] = (f'need {job["ngpus"]} GPU(s) on one host; of {total}: ' + ', '.join(parts)
                                  if total else 'no allowed host is up')
    return placements, reasons


def _indices(text):
    return [int(i) for i in text.split(',')] if text else []
