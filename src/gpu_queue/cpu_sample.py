"""Standalone CPU probe, sent to each host's python3 over the existing SSH call.

Read the current cgroup v2 and its descendants, never the host-wide load average.
The visible cgroup hierarchy, affinity and quotas define this container's capacity.
"""

import json
import os
from pathlib import Path
import time


def read_cpu(cgroup_root=Path("/sys/fs/cgroup"), proc_cgroup=Path("/proc/self/cgroup")):
    relative = next(line.split(":", 2)[2] for line in proc_cgroup.read_text().splitlines()
                    if line.startswith("0::"))
    root = cgroup_root
    group = root / relative.lstrip("/")
    # A cgroup namespace normally exposes the container itself at /.
    usage = dict(line.split() for line in (group / "cpu.stat").read_text().splitlines())
    stamp = time.monotonic()
    capacity = float(len(os.sched_getaffinity(0)))
    for parent in (group, *group.parents):
        quota_file = parent / "cpu.max"
        if quota_file.exists():
            quota, period = quota_file.read_text().split()
            if quota != "max":
                capacity = min(capacity, int(quota) / int(period))
        if parent == root:
            break
    return capacity, int(usage["usage_usec"]), stamp, str(group.stat().st_ino)


if __name__ == "__main__":
    print(json.dumps(read_cpu()))
