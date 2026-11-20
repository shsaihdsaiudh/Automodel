# NVIDIA-NeMo/Automodel#4122 实验记录：探索用 FSDP2 取代 DDP

Issue：https://github.com/NVIDIA-NeMo/Automodel/issues/4122

没有 PR。本地分支 `issue-4122-ddp-to-fsdp2` 和 `main` 都在 `384f6b2ae`，没有代码提交。工作区说明在 `local/README.md`。

- `dev/bench/`：`ddp_vs_fsdp2.py`、`grid*.sh`、`summarize.py`，结果在 `runs/` 和 `*.jsonl`。
- `dev/bench/profiles/*.json.gz`：torch profiler trace（每个 40–45 MB，已 gzip），解压后可以在 Perfetto 里打开。

## 目录

| 目录 | 来源 |
|---|---|
| `local/` | WSL 本地 `~/automodel-4122/{README.md,bench,env.sh,am-uv-sync.sh}` |
| `dev/` | 开发机 `/planning-users/yangyeyuan/automodel-4122/{logs,bench,README.md,env.sh,am-uv-sync.sh}（不含 models）` |

`EXCLUDED-<目录>.txt` 列出没有收进来的文件及大小。收集时跳过的内容：Python/CUDA 环境、venv、wheel、模型和 checkpoint、`*.sqlite`（可以从 `*.nsys-rep` 重新导出）、NeMo-RL 的 rollout 样本 `train_data_step*.jsonl`、50 MB 以上的文件、`__pycache__`。

## 脱敏

这个分支在公开 fork 上。内网 IP、主机名、内部域名和镜像地址、集群卷名、队列和资源名都替换成了 `<redacted…>`；日志里的环境变量转储（`KEY=value` 行和 `env.txt`）已删除。`*.nsys-rep` 会把进程的环境变量写进报告（3976 的 trace 里有云服务 access key），所以全部没有收，清单在 `EXCLUDED-nsys-rep.txt`（如果有）。路径 `/planning-users/yangyeyuan/...` 保留，用来对照原始位置。

整理时间：2026-11-20。
