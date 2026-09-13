# GPU 故障半径收敛与任务重试（2026-09-13）

## 触发原因

在核对 `PROJECT_REVIEW_AND_ROADMAP.md` 的就绪结论时发现，2026-09-13 的全部 pilot
都在一个看不到 NVIDIA 设备节点的受限沙箱中执行，因此 manifest 记录的
`cuda_available=false`、`[DNN] device=cpu` 是沙箱假象。回查历史 GPU 批次后确认了
本机能力，也发现了一个此前未被记录、且尚未修复的故障模式。

## 证据

**本机确有 GPU**（`/proc/driver/nvidia` 不依赖设备节点，可在沙箱内读取）：

- 2 × NVIDIA GeForce RTX 4090（`0000:3b:00.0`、`0000:86:00.0`）；
- 驱动 `535.230.02`，CUDA 工具链 12.2；
- 2026-09-08/09 的 4096 帧 sweep 日志累计出现 233 次 `device=cuda:1`、
  227 次 `device=cuda:0`；`device=cpu` 的 22 次全部来自 9-13 的沙箱内 pilot。

**2026-09-10 的 `Exp6_Bc` 批次大面积失败**：

| 批次 | 提交 | 设备 | 日志 | 完成 | 失败 |
| --- | --- | --- | ---: | ---: | ---: |
| `20260908_171820_895619_Exp6_Bc` | `46ef0ad` | cuda | 128 | 128 | 0 |
| `20260910_010528_175721_Exp6_Bc` | `cc57c0f`（dirty） | cuda | 128 | 32 | **94** |

失败算法分布为 24 AC / 24 COB / 22 LDA / 24 MTD，即**连不使用网络的 COB/MTD 也
全部失败**。堆栈落在：

```text
run_sweeps.py -> set_seed(seed)
utils/reproducibility.py:11 -> torch.manual_seed(seed)
  -> torch.cuda.manual_seed_all -> _lazy_call -> default_generator.manual_seed
RuntimeError: CUDA error: unknown error
```

同一批次还伴随一次未解释的停摆：`MTD B_c=400 MHz` 的 8 个种子**同时**耗时
22.91 h（同组其他算法为 0.03–0.54 h），符合多进程阻塞在 CUDA 上下文初始化上
直到超时的特征。此外 `Exp7_Bsat` 在该批次整体缺失，因此不存在一套完整的
coupled 时代 4096 帧结果。

**根因未随代码演进而消失**：`utils/reproducibility.py` 自 `7d72653` 起未改动，
在 `cc57c0f` 与当前 HEAD 上内容相同。`torch.manual_seed` 会为**每个**调用者转调
`torch.cuda.manual_seed_all`，所以“启发式算法不初始化 CUDA”这一 README 表述与
实现不符，故障半径被扩大到整批任务。

## 修改内容

1. `utils/reproducibility.py`：`set_seed(seed, use_cuda=True)` 新增开关。
   `use_cuda=False` 时只调用 `torch.default_generator.manual_seed(seed)`，完全不触碰
   CUDA 运行时；默认路径保持原行为逐字不变。
2. `core/agents/lda_agent.py` / `core/agents/baselines.py`：新增类属性
   `LDAAgent.uses_dnn = True`、`HeuristicAgent.uses_dnn = False`，作为“是否需要
   CUDA”的单一声明来源。
3. `main.py`：`run_simulation` 按 `getattr(agent_class, 'uses_dnn', True)` 决定
   `set_seed` 的 `use_cuda`，未知 agent 保持历史行为。
4. `run_sweeps.py`：
   - worker 按 `AgentClass.uses_dnn` 播种，并在任务日志写入 `CUDA: ...` 一行；
   - 新增 `_cuda_health_check()`，结果写入 manifest 的 `runtime.cuda_health`；
     预检失败只告警不中止，避免把一次探查失败升级为整批失败；
   - 新增 `--task-retries`（默认 1，`LDA_TASK_RETRIES` 可覆盖）。串行与进程池
     两条路径都会重试失败任务，被替换的失败日志保留为 `*.attemptN`；
   - manifest 新增 `task_retries` 字段。
5. `tests/test_reproducibility.py`（新增 10 项）：CPU-only 播种不触碰 CUDA、
   CPU 随机流与默认路径逐位一致、启发式不建网络、GPU 预检只报告不抛异常、
   重试上界与日志归档、失败结果元组形状、负重试值在写盘前被拒绝。

## 影响边界

- **不改变任何数值**：`use_cuda=False` 只影响是否创建 CUDA 上下文；Python、
  NumPy 与 CPU torch 三条随机流与默认路径逐位一致（由
  `test_cpu_only_seed_leaves_cpu_streams_identical` 固定）。
- **不改变既有产物语义**：失败任务的记录形状、CSV 字段与统计口径均未变化，
  历史 manifest 仍可读。
- 重试面向环境故障。确定性 bug 会在重试后继续失败，日志以 `*.attemptN` 保留，
  不应据此把失败当作已修复。

## 仍未完成

- 上述修改只在无 GPU 沙箱中测试过。GPU 设备分配与故障恢复**尚未在真实设备上
  验证**，必须在宿主机 shell 补做一次 pilot：

  ```bash
  PYTHON_BIN=/home/hp/miniconda3/envs/sagin/bin/python \
    scripts/ldactl sweep --experiments Exp1_J Exp6_Bc \
    --frames 512 --seeds 42 123 --workers 10
  ```

- 2026-09-10 的 CUDA 错误是设备/驱动瞬态故障还是代码可复现问题，尚无定论。
  本次修改只收敛了故障半径，没有证明根因已经消除。
