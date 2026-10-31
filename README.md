# gpu-queue（`gq`）

`gq` 是一个 GPU 任务队列，管理几台能通过 SSH 连到、共享文件系统的机器。

任务先排队，等到某台机器的 CPU、内存和 GPU 都放得下再启动。同一台机器上的所有 agent 和 shell 看到的是同一个队列。

## 安装

```bash
pipx install -e /mnt/public/xcj/tools/gpu-queue
```

配置文件在 `~/.config/gpu-queue/config.toml`，各项都可以省略（见 `src/gpu_queue/config.py`）：

```toml
hosts = ["localhost", "wuwen-4090-2", "wuwen-4090-3"]   # localhost：本机，不走 SSH
shared_dir = "/mnt/public/xcj/cache/gpu-queue"           # 任务脚本、日志、退出码
cpu_target = 0.85                                     # 最多使用逻辑核数的 85%
ram_reserve = 0.05                                    # 留出总内存的 5%

[defaults]                                           # 每个任务的默认申报量
cpus = 1.5                                           # 核，可以是小数
ram = 4.0                                            # GiB，整件任务合计
```

对每台机器的要求：
- 本机能免密 SSH 到它。
- 它看到的 `shared_dir` 和任务工作目录与本机是同一路径。

## 使用

```bash
gq submit --mem 8 --util 30 --window 30 -- uv run python train.py --gpu {gpu}
gq submit --gpus 2 --host wuwen-4090-2 --name sweep -- 'python a.py && python b.py'
gq submit --cpus 1.5 --ram 4 --mem 2 --util 85 -- python isaac_cpu.py
gq submit --gpus 0 --cpus 8 --ram 16 -- python cpu_only.py
gq ls            # 运行中（机器:GPU、已运行时长）与排队中（顺序、等待原因）；-a 加上最近结束的
gq hosts         # 各机器核数、load1、可用内存、运行任务的 CPU/内存申报合计
gq gpus          # 同上，加每张卡的显存、利用率（当前 / 30 s 内最高）和卡上的 gq 任务
gq log -f 12     # 任务输出，也在 <shared_dir>/logs/12.log
gq cancel 12 13  # 排队中的直接取消；运行中的给进程组发 SIGTERM，20 s 后 SIGKILL
gq wait 12 13    # 每次状态变化打印一行；全部成功退出码为 0，否则为 1
gq submit --wait -- CMD   # 提交并等待（退出码同 gq wait）；中断或终止这个客户端会取消任务，可作为批量脚本的 --launcher 前缀
gq show 12
```

### CPU 和内存

`--cpus` 是任务需要的逻辑核数，`--ram` 是任务合计需要的 GiB 内存，支持小数。默认 1.5 核、4 GiB，来自三台机器上 31 个 Isaac CPU 任务的进程组采样：平均约 1.06 核，RSS 最大 2.63 GiB。其他负载按实际需要申报，例如多线程训练应调高 `--cpus`。

每台机器同时满足：
- **CPU**：有效已用核数加本任务 `--cpus`，不超过逻辑核数 × `cpu_target`（默认 0.85）。
- **内存**：本任务 `--ram` 不超过 `MemAvailable` 减去启动预留，再减去总内存 × `ram_reserve`（默认 0.05）。

**启动预留**：`warmup` 秒内（默认 120 s），CPU 取实测 1 分钟 loadavg 与“启动时负载 + 此后启动任务的核数申报合计”中的较大值；内存预留尚未兑现的申报量，主机整体的内存增长只抵扣一次。同一轮启动的任务也立即扣除预算，避免启动初期负载还没上升就放入过多任务。排在前面但放不下的任务仍不阻塞后面的较小任务。

CPU 和内存是调度申报，不是进程的硬限制。loadavg 包含等待 I/O 的任务，可能比实际忙碌核数更保守。`gq ls` 的等待原因会注明 `CPU` 或 `RAM`；`gq hosts` 的 `GQ CPUS/RAM` 是正在运行（含取消中）任务的申报总量，不是实测用量。

**纯 CPU 任务**：`--gpus 0` 不分配 GPU、不检查显存和利用率窗口，只检查 CPU 和内存；默认把 `CUDA_VISIBLE_DEVICES` 设为空。支持未安装 `nvidia-smi` 的 CPU 主机。

### GPU 空闲阈值

同时满足下面两条，这张卡才算空闲：
- **显存**：空余不少于 `--mem` GB。
- **利用率**：最近 `--window` 秒内一直低于 `--util`%。

这三个参数和 CPU、内存申报量的默认值都在配置的 `[defaults]` 里，每个任务可以单独设置。

**显存预留**：任务启动后 `warmup` 秒内（默认 120 s），它申报但还没实际占上的显存，仍按申报量记在它的卡上。这样同一张卡按申报显存装任务，不会因为任务启动初期显存还没涨上来而被重复分配。

**多个任务挤在同一张卡上**：把 `--util` 设高一些。

### 任务的运行环境

**环境变量**
- 继承提交时 shell 的环境变量。
- 不继承 `CUDA_VISIBLE_DEVICES`，以及 SSH 和终端相关的变量。
- 在其他机器上运行时，也不继承代理变量（`*_proxy`）。本机的代理在 127.0.0.1 上，其他机器连不到。

**工作目录**：提交时的目录，或用 `--cwd` 指定。

**GPU 相关变量**
- `GQ_GPUS`：分到的物理 GPU 编号，逗号分隔。
- `CUDA_VISIBLE_DEVICES`：与 `GQ_GPUS` 相同。加 `--no-mask` 则不设置。
- `CUDA_DEVICE_ORDER`：设为 `PCI_BUS_ID`。
- `GQ_JOB_ID`、`GQ_HOST`：任务编号和所在机器。

**命令写法**
- 占位符：命令中的 `{gpus}` 替换为分到的全部编号（空格分隔），`{gpu}` 替换为其中第一个。
- `--` 后面只有一个参数时，按 shell 命令行执行；有多个参数时，按命令加参数执行。

**任务归属**：`--owner` 默认取 `$GQ_OWNER`，没有设置时取目录名，用来区分不同 agent 的任务。

## 在 Claude Code 中使用

- **任务结束时收到通知**：用 `run_in_background` 运行 `gq wait 12 13`。它退出时 harness 会通知 agent，退出码表示是否全部成功。
- **逐条跟踪进展**：用 Monitor 运行 `gq wait 12 13`。每次状态变化（在某台机器上启动、结束、失败）都会产生一条事件。
- **区分自己的任务**：把 `GQ_OWNER` 设为会话名或 worktree 名。

## 工作方式

**调度进程**
- 每台机器只运行一个调度进程。没有在跑时，`gq` 会自动启动它。
- 管理命令：`gq daemon status|stop|restart`。
- 日志在状态目录里。

**每 `poll` 秒（默认 5 s）的工作**
- 每台机器只发一次命令，通过复用的 SSH 连接采样 `nvidia-smi`、逻辑核数、`/proc/loadavg`、`/proc/meminfo`，同时检查各任务的进程是否还在。
- 已写出退出码的任务记为结束。
- 已取消的任务把进程杀掉。
- 排队中放得下的任务按优先级、再按提交顺序启动（见 `src/gpu_queue/schedule.py`）。排在前面但放不下的任务，不会挡住后面放得下的任务。

**任务的执行**
- 任务在目标机器上以 `setsid bash <shared_dir>/jobs/<id>.sh` 运行。
- 脚本先记下自己的 pid，再执行命令，最后写出 `<id>.exit`。
- 进程已经消失（或成了僵尸进程）却没有写出退出码的任务，记为 `lost`。

**状态存放**
- 任务状态在共享盘的文件里，调度进程重启后能接着管理正在跑的任务。
- 队列数据库是状态目录里的 SQLite，放在本机本地磁盘上。
- 升级 CPU/内存调度时会自动给旧任务补默认申报量，保留任务状态、优先级、日志与进程。重启 daemon 不终止任务；已有的启动采样保存在数据库里，重启后仍用于预留。没有启动采样的旧任务在剩余 `warmup` 内保守预留全额。GPU 利用率历史需要重新积累一个窗口。

**容器里的进程号**

在这些容器里，`nvidia-smi` 报告的是宿主机命名空间的进程号，和 `ps` 看到的对不上。gq 不在两者之间做对应：
- 任务按它自己记下的容器内 pid 跟踪。
- GPU 只看整卡的显存和利用率。

所以 `gq gpus` 能显示卡上有几个 GPU 进程，但分不出这些进程属于谁。

## 测试

```bash
~/.local/share/pipx/venvs/gpu-queue/bin/python -m pytest tests
```
