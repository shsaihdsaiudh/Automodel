# Benchmark DDP vs FSDP2 (dp_shard=1) vs replicate() for NVIDIA-NeMo/Automodel#4122.
#
# Launch with torchrun, e.g.
#   torchrun --nproc-per-node 1 ddp_vs_fsdp2.py --model <dir> --mode ddp --out results.jsonl
#
# Modes:
#   none        no wrapper (lower bound; what Automodel does today at world_size == 1)
#   ddp         torch DDP with Automodel's DDPConfig defaults
#   fsdp2       fully_shard on a (dp_replicate=world, dp_shard=1) mesh, Automodel's unit layout
#   replicate   torch.distributed._composable.replicate_with_fsdp.replicate, same unit layout
#   fsdp2_shard fully_shard on a 1D dp_shard=world mesh (reference: plain FSDP2)
#
# Precision:
#   bf16         bf16 weights, bf16 grads, bf16 gradient reduction in every mode
#   fp32_master  fp32 weights; DDP/none use autocast(bf16), FSDP2/replicate use
#                MixedPrecisionPolicy(param=bf16, reduce=fp32)

import argparse
import contextlib
import json
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP

MODES = ("none", "ddp", "fsdp2", "replicate", "fsdp2_shard")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local HF model directory")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--precision", choices=("bf16", "fp32_master"), default="bf16")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--mbs", type=int, default=1, help="micro batch size per rank")
    p.add_argument("--ga", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--reshard-after-forward", type=int, choices=(0, 1), default=0, help="fsdp2 layer units")
    p.add_argument("--bucket-view", type=int, choices=(0, 1), default=0, help="ddp gradient_as_bucket_view")
    p.add_argument("--bucket-cap-mb", type=float, default=None)
    p.add_argument("--profile-dir", default=None, help="if set, profile 3 extra steps into this directory")
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True, help="JSONL file; rank 0 appends one line")
    return p.parse_args()


def decoder_layers(model):
    layers = model.model.layers
    assert isinstance(layers, torch.nn.ModuleList)
    return list(layers)


def untied_embeddings(model):
    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    if out is None or inp.weight is out.weight:
        return []
    return [inp, out]


def wrap(model, args, world, local_rank):
    """Apply the wrapper for args.mode. Returns (model, sync_ctx_fn, uses_autocast)."""
    bf16 = torch.bfloat16
    reduce_dtype = bf16 if args.precision == "bf16" else torch.float32
    mp = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=reduce_dtype, output_dtype=None, cast_forward_inputs=False)
    autocast = args.precision == "fp32_master"

    if args.mode == "none":
        return model, lambda last: contextlib.nullcontext(), autocast

    if args.mode == "ddp":
        kwargs = dict(
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=False,
            gradient_as_bucket_view=bool(args.bucket_view),
        )
        if args.bucket_cap_mb is not None:
            kwargs["bucket_cap_mb"] = args.bucket_cap_mb
        ddp = DDP(model, **kwargs)
        return ddp, (lambda last: contextlib.nullcontext() if last else ddp.no_sync()), autocast

    if args.mode == "replicate":
        from torch.distributed._composable.replicate_with_fsdp import replicate

        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp_replicate",))
        for layer in decoder_layers(model):
            replicate(layer, mesh=mesh, mp_policy=mp)
        for emb in untied_embeddings(model):
            replicate(emb, mesh=mesh, mp_policy=mp)
        replicate(model, mesh=mesh, mp_policy=mp)
    else:
        if args.mode == "fsdp2":
            mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("dp_replicate", "dp_shard"))
        else:
            mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp_shard",))
        raf = bool(args.reshard_after_forward)
        for layer in decoder_layers(model):
            fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=raf)
        for emb, emb_raf in zip(untied_embeddings(model), (raf, False)):
            fully_shard(emb, mesh=mesh, mp_policy=mp, reshard_after_forward=emb_raf)
        fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=False)

    def sync_ctx(last):
        model.set_requires_gradient_sync(last)
        return contextlib.nullcontext()

    return model, sync_ctx, False


def make_batches(vocab, args, rank, device, n=8):
    g = torch.Generator().manual_seed(args.seed * 1000 + rank)
    return [torch.randint(0, vocab, (args.mbs, args.seq_len), generator=g).to(device) for _ in range(n)]


def train_step(model, opt, params, batches, step, args, sync_ctx, autocast):
    loss_sum = None
    for micro in range(args.ga):
        x = batches[(step * args.ga + micro) % len(batches)]
        last = micro == args.ga - 1
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
        with sync_ctx(last):
            with amp:
                loss = model(input_ids=x, labels=x, use_cache=False).loss / args.ga
            loss.backward()
        loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
    return loss_sum


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    vocab = model.config.vocab_size

    model, sync_ctx, autocast = wrap(model, args, world, local_rank)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0, foreach=True)
    batches = make_batches(vocab, args, rank, device)

    total = args.warmup + args.steps
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(5)] for _ in range(total)]
    host_ms, losses, gnorms = [], [], []

    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_start = None
    for step in range(total):
        if step == args.warmup:
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
        e = ev[step]
        t0 = time.perf_counter()
        e[0].record()
        loss = train_step(model, opt, params, batches, step, args, sync_ctx, autocast)
        e[1].record()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
        e[2].record()
        opt.step()
        e[3].record()
        opt.zero_grad(set_to_none=True)
        e[4].record()
        host_ms.append((time.perf_counter() - t0) * 1e3)
        losses.append(loss)
        gnorms.append(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_start

    timed = range(args.warmup, total)
    step_ms = [ev[i][0].elapsed_time(ev[i][4]) for i in timed]
    phase = {
        name: statistics.median(ev[i][a].elapsed_time(ev[i][b]) for i in timed)
        for name, a, b in (("fwd_bwd_ms", 0, 1), ("clip_ms", 1, 2), ("opt_ms", 2, 3), ("zero_ms", 3, 4))
    }
    tokens_per_step = args.mbs * args.seq_len * args.ga
    q = statistics.quantiles(step_ms, n=10)

    prof_path = None
    if args.profile_dir:
        from torch.profiler import ProfilerActivity, profile

        os.makedirs(args.profile_dir, exist_ok=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for step in range(3):
                train_step(model, opt, params, batches, step, args, sync_ctx, autocast)
                torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        base = os.path.join(args.profile_dir, f"{args.tag or args.mode}_rank{rank}")
        if rank == 0:
            prof.export_chrome_trace(base + ".json")
            with open(base + ".txt", "w") as f:
                f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
                f.write("\n\n")
                f.write(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30))
        prof_path = base

    if rank == 0:
        rec = {
            "tag": args.tag,
            "model": os.path.basename(args.model.rstrip("/")),
            "n_params": n_params,
            "mode": args.mode,
            "precision": args.precision,
            "seq_len": args.seq_len,
            "mbs": args.mbs,
            "ga": args.ga,
            "world": world,
            "reshard_after_forward": args.reshard_after_forward,
            "bucket_view": args.bucket_view,
            "steps": args.steps,
            "step_ms_median": statistics.median(step_ms),
            "step_ms_p10": q[0],
            "step_ms_p90": q[-1],
            "wall_step_ms": wall_s * 1e3 / args.steps,
            "host_ms_median": statistics.median(host_ms[args.warmup :]),
            **phase,
            "tokens_per_s_per_gpu": tokens_per_step / (wall_s / args.steps),
            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            "losses": [round(t.item(), 5) for t in losses[:20]],
            "grad_norms": [round(t.item(), 5) for t in gnorms[:20]],
            "profile": prof_path,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "host": socket.gethostname(),
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        brief = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("losses", "grad_norms")}
        print(json.dumps(brief), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
