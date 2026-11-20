# Summarize grid JSONL into markdown tables: step time per variant and ratio to DDP.
import json
import sys
from collections import defaultdict

VARIANTS = ["none", "ddp", "ddp_bv", "fsdp2", "fsdp2_raf", "replicate"]


def variant(r):
    if r["mode"] == "ddp":
        return "ddp_bv" if r["bucket_view"] else "ddp"
    if r["mode"] == "fsdp2" and r["reshard_after_forward"]:
        return "fsdp2_raf"
    return r["mode"]


def main(path):
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if not r["tag"].startswith(("prof_", "smoke"))]
    groups = defaultdict(dict)
    for r in rows:
        key = (r["model"], r["precision"], r["seq_len"], r["mbs"], r["ga"])
        groups[key][variant(r)] = r

    for metric, label in (("step_ms_median", "step time (ms, median)"), ("peak_mem_gib", "peak memory (GiB)")):
        print(f"\n### {label}\n")
        head = ["model", "precision", "seq", "mbs", "ga"] + VARIANTS
        if metric == "step_ms_median":
            head += ["fsdp2/ddp", "replicate/ddp", "fsdp2-none ms"]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for key in sorted(groups, key=lambda k: (k[0], k[1], k[4], k[2], k[3])):
            g = groups[key]
            cells = [str(x) for x in key]
            cells += [f"{g[v][metric]:.1f}" if v in g else "-" for v in VARIANTS]
            if metric == "step_ms_median":
                d = g.get("ddp", {}).get(metric)
                for v in ("fsdp2", "replicate"):
                    cells.append(f"{g[v][metric] / d:.2f}" if d and v in g else "-")
                n, f = g.get("none", {}).get(metric), g.get("fsdp2", {}).get(metric)
                cells.append(f"{f - n:.1f}" if n and f else "-")
            print("| " + " | ".join(cells) + " |")

    print("\n### phase breakdown, median ms (fwd+bwd / clip / optimizer)\n")
    print("| model | precision | seq | mbs | ga | " + " | ".join(VARIANTS) + " |")
    print("|" + "---|" * (5 + len(VARIANTS)))
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[4], k[2], k[3])):
        g = groups[key]
        cells = [str(x) for x in key]
        for v in VARIANTS:
            r = g.get(v)
            cells.append(f"{r['fwd_bwd_ms']:.0f}/{r['clip_ms']:.1f}/{r['opt_ms']:.1f}" if r else "-")
        print("| " + " | ".join(cells) + " |")

    print("\n### loss / grad-norm agreement vs none (first 20 steps, max abs diff)\n")
    for key in sorted(groups):
        g = groups[key]
        ref = g.get("none")
        if not ref:
            continue
        diffs = []
        for v in VARIANTS[1:]:
            if v in g:
                dl = max(abs(a - b) for a, b in zip(ref["losses"], g[v]["losses"]))
                dg = max(abs(a - b) / max(abs(a), 1e-6) for a, b in zip(ref["grad_norms"], g[v]["grad_norms"]))
                diffs.append(f"{v}: dloss={dl:.2e} rel_dgn={dg:.2e}")
        print(f"- {key}: " + "; ".join(diffs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid_1gpu.jsonl")
