# automodel-4122: NVIDIA-NeMo/Automodel#4122

[RFC] Explore deprecation of DDP in favor of FSDP2
(https://github.com/NVIDIA-NeMo/Automodel/issues/4122).

Built 2026-11-19 from Automodel main `384f6b2ae`.

```
~/automodel-4122/                      local (WSL), source of truth
  Automodel/   git checkout, branch issue-4122-ddp-to-fsdp2
               remotes: origin = NVIDIA-NeMo/Automodel, fork = shsaihdsaiudh/Automodel
  env.sh, am-uv-sync.sh, README.md     workspace files, sent by `cmdev push`

cloudml:/planning-users/yangyeyuan/automodel-4122/   run directory
  Automodel/        same checkout (copied from the local clone, .git included)
  Automodel/.venv   uv venv, Python 3.12.16, torch 2.10.0+cu130, transformers 5.18.0,
                    megatron-fsdp 0.5.0, diffusers 0.39.0
  logs/             `cmdev run -d` logs
```

## Use

```
cd ~/automodel-4122/Automodel
cmdev run 'uv run --locked --no-sync pytest -q tests/unit_tests/distributed/test_ddp_manager.py'
cmdev run 'uv run --locked --no-sync torchrun --nproc-per-node 1 <script>'
cmdev run -d '...'          # long jobs, log in automodel-4122/logs/
```

Use `--no-sync`: a plain `uv run` would try to re-sync from upstream URLs, which the dev
machine cannot reach.

## Environment

`env.sh` reuses the toolchain of the NeMo RL workspaces (`nemo-rl-3976/env`: uv 0.13.6,
managed Python, CUDA 13.0, uv cache, HF cache through Artifactory).

The venv was built with

```
am-uv-sync.sh --extra diffusion --no-install-package onnxruntime-gpu
```

(base deps + default groups build/docs/test + diffusion). `am-uv-sync.sh` rewrites the
uv.lock download URLs to Artifactory for one sync and restores the file afterwards.

Not installed:

- `onnxruntime-gpu` (test group): its index `aiinfra.pkgs.visualstudio.com` is not
  reachable from the dev machine and has no mirror.
- `--extra cuda` (Transformer Engine, mamba-ssm, DeepEP, ...): source builds, not needed
  for DDP vs FSDP2 on HF models. Add it with `am-uv-sync.sh --extra cuda --extra diffusion
  --no-install-package onnxruntime-gpu` if a benchmark needs TE.

Checked on 2026-11-19:

- `tests/unit_tests/distributed/test_ddp_manager.py` + `test_model_parallelizer.py`: 71 passed
- `tests/functional_tests/parallelism/test_ddp_single_rank.py` (GPU): 7 passed, 2 skipped
- `fully_shard` on a (1, 1) dp_replicate x dp_shard mesh and
  `torch.distributed._composable.replicate_with_fsdp.replicate` both run fwd/bwd on 1 GPU

## Limitation

The dev machine has one H20. DDP vs FSDP2 comparisons with world_size > 1 (the case the
RFC cares about) need a multi-GPU node.
