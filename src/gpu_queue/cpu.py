"""A time-weighted, trailing 60-second mean of cgroup CPU usage, in cores."""

from collections import deque

WINDOW = 60.


class History:
    def __init__(self):
        self.samples = {}

    def forget(self, host):
        self.samples.pop(host, None)

    def add(self, resource):
        samples = self.samples.setdefault(resource.host, deque())
        stamp, usage = resource.cpu_time, resource.cpu_usage
        identity = (resource.cpu_scope, resource.cpus)
        if samples:
            last_time, last_usage, last_identity = samples[-1]
            if (stamp <= last_time or stamp - last_time > WINDOW or usage < last_usage
                    or identity != last_identity):
                samples.clear()
        samples.append((stamp, usage, identity))
        cutoff = stamp - WINDOW
        while len(samples) > 1 and samples[1][0] <= cutoff:
            samples.popleft()
        if samples[0][0] > cutoff:
            return None
        # Interpolate the cumulative counter at the window boundary for irregular polls.
        start_time, start_usage, _ = samples[0]
        if start_time < cutoff:
            end_time, end_usage, _ = samples[1]
            start_usage += (end_usage - start_usage) * (cutoff - start_time) / (end_time - start_time)
        return (usage - start_usage) / (WINDOW * 1_000_000)
