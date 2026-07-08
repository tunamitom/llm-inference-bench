#!/usr/bin/env python3
"""Accept-length benchmark for DFlash services on :8001.

Two modes:

  (a) Per-DOMAIN (default): one greedy request per domain at short context,
      parse the container's per-batch acceptance log lines in that window.

  (b) Per-CONTEXT-LENGTH (--contexts): build a long prompt of ~N tokens from a
      corpus file (the SAME text truncated to each depth, so only context length
      varies), generate greedily, and parse acceptance + the engine's pure-decode
      gen-throughput at that depth. Maps the DFlash acceptance cliff vs context.

Supports both sglang and vLLM log formats:
  - sglang: "accept len:" / "accept rate:" / "gen throughput (token/s):" / "#full token:" in "Decode batch" lines
  - vLLM:   "Mean acceptance length:" / "Avg Draft acceptance rate:" in SpecDecoding metrics

Usage:
  python3 accept_bench.py                                  # domain sweep, auto-detect container on :8001
  python3 accept_bench.py coding-webdev math               # specific domains only
  python3 accept_bench.py --container mimo-pro-dflash-vllm
  python3 accept_bench.py --port 8002
  # context-length sweep (the acceptance cliff):
  python3 accept_bench.py --contexts 1000,16000,32000,64000,96000,128000,192000
  python3 accept_bench.py --contexts 64000,128000 --corpus /path/to/text --gen-tokens 1500 --out results.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


PROMPTS = {
    "coding-webdev": (
        "Write a complete single-file HTML page implementing a responsive todo "
        "list app with vanilla JavaScript: add/remove/complete items, filter "
        "tabs (all/active/completed), localStorage persistence, and clean CSS. "
        "Output only the code.",
        1500,
    ),
    "coding-humaneval": (
        "Write a Python function `def longest_balanced_substring(s: str) -> int` "
        "that returns the length of the longest balanced parentheses substring. "
        "Include a docstring, type hints, an O(n) implementation, and 5 unit "
        "tests using pytest.",
        900,
    ),
    "math": (
        "Solve step by step: A rectangular box has integer side lengths. Its "
        "surface area is 286 and its volume is 280. Find the side lengths and "
        "the length of the space diagonal. Show full reasoning.",
        900,
    ),
    "agent": (
        "You are an autonomous agent with tools: search(query), read_file(path), "
        "write_file(path, content), run(cmd). Task: find why unit tests fail in "
        "a Python repo after a dependency bump. Produce a step-by-step plan and "
        "then simulate executing it with tool calls in JSON, one per step, with "
        "observations.",
        900,
    ),
    "chat-mtbench": (
        "Compose an engaging travel blog post about a recent trip to Hawaii, "
        "highlighting cultural experiences and must-see attractions.",
        700,
    ),
}

# Task appended after the long corpus slice in --contexts mode (elicits long, genuine output).
CONTEXT_TASK = (
    "\n\n# TASK\nAnalyze the text/code above in thorough detail: explain what it does, walk "
    "through its structure and control flow, and call out potential bugs or improvements. "
    "Write a long, multi-section analysis."
)

# Output-domain presets for --task. The corpus is the KV filler (held constant); the task sets
# what the model GENERATES, so acceptance is measured on that output domain at each depth.
TASKS = {
    "code": CONTEXT_TASK,
    "reason": (
        "\n\n# TASK\nIgnore the text above. Solve this problem step by step, showing all work: "
        "using inclusion-exclusion, find how many integers from 1 to 100000 are divisible by none "
        "of 2, 3, 5, 7, then compute the sum of those integers. Explain every step, then verify the "
        "result a second independent way."
    ),
    "chat": (
        "\n\n# TASK\nIgnore the text above. Write a long, engaging short story (1500+ words) about a "
        "lighthouse keeper who finds a message in a bottle, with rich description and dialogue."
    ),
}


def detect_container(port):
    """Auto-detect the Docker container serving on the given port."""
    proc = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
        capture_output=True, text=True,
    )
    for line in proc.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        name = parts[0]
        ports = parts[1] if len(parts) > 1 else ""   # host-network containers have empty Ports
        if f"0.0.0.0:{port}->" in ports or f"0.0.0.0:{port}/tcp" in ports:
            return name
    # Fallback: try to find by listening port
    proc2 = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    for name in proc2.stdout.strip().splitlines():
        check = subprocess.run(
            ["docker", "exec", name, "ss", "-tlnp"],
            capture_output=True, text=True,
        )
        if f":{port}" in check.stdout:
            return name
    return None


def chat(port, prompt, max_tokens, model="turin", ignore_eos=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        if ignore_eos and e.code == 400:   # server rejects ignore_eos -> retry without it
            return chat(port, prompt, max_tokens, model, ignore_eos=False)
        raise
    dt = time.time() - t0
    usage = resp.get("usage", {})
    return dt, usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)


def accept_stats(container, since_epoch):
    """Parse acceptance from Docker logs since `since_epoch`.

    Returns (lens, rates, tps, depths). tps (engine gen-throughput tok/s) and
    depths (#full-token decode depth) are sglang-only; empty list for vLLM.
    """
    proc = subprocess.run(
        ["docker", "logs", container, "--since", str(since_epoch)],
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    lens, rates, tps, depths = [], [], [], []
    for line in out.splitlines():
        # SGLang: "Decode batch, ... #full token: N, ... accept len: X, accept rate: Y, ... gen throughput (token/s): Z, ..."
        if "accept len:" in line and "Decode batch" in line:
            try:
                lens.append(float(line.split("accept len:")[1].split(",")[0]))
                rates.append(float(line.split("accept rate:")[1].split(",")[0]))
            except (IndexError, ValueError):
                continue
            try:
                tps.append(float(line.split("gen throughput (token/s):")[1].split(",")[0]))
            except (IndexError, ValueError):
                pass
            try:
                depths.append(int(line.split("#full token:")[1].split(",")[0]))
            except (IndexError, ValueError):
                pass
        # vLLM: "SpecDecoding metrics: Mean acceptance length: X.XX, ... Avg Draft acceptance rate: XX.X%"
        elif "Mean acceptance length:" in line and "Avg Draft acceptance rate:" in line:
            try:
                lens.append(float(line.split("Mean acceptance length:")[1].split(",")[0]))
                rates.append(float(line.split("Avg Draft acceptance rate:")[1].split("%")[0]) / 100.0)
            except (IndexError, ValueError):
                pass
    return lens, rates, tps, depths


def run_domains(args, container):
    domains = args.domains or list(PROMPTS)
    print(f"{'domain':18s} {'tok':>5s} {'tok/s':>7s} {'accept_len':>10s} {'rate':>5s} {'batches':>7s}")
    for name in domains:
        if name not in PROMPTS:
            print(f"Unknown domain: {name}", file=sys.stderr)
            continue
        prompt, max_tokens = PROMPTS[name]
        since = int(time.time())
        time.sleep(1.1)
        dt, ctok, _ptok = chat(args.port, prompt, max_tokens, model=args.model)
        time.sleep(1.0)
        lens, rates, _tps, _depths = accept_stats(container, since)
        if lens:
            al = sum(lens) / len(lens)
            ar = sum(rates) / len(rates)
        else:
            al = ar = float("nan")
        print(f"{name:18s} {ctok:5d} {ctok/dt:7.1f} {al:10.2f} {ar:5.2f} {len(lens):7d}")


def run_contexts(args, container):
    """Sweep context length: same corpus truncated to each depth -> acceptance(depth)."""
    try:
        raw = open(args.corpus, errors="ignore").read()
    except OSError as e:
        print(f"ERROR: cannot read corpus {args.corpus}: {e}", file=sys.stderr)
        sys.exit(1)
    if not raw.strip():
        print(f"ERROR: corpus {args.corpus} is empty", file=sys.stderr)
        sys.exit(1)
    targets = [int(x) for x in args.contexts.split(",") if x.strip()]
    task_text = TASKS.get(args.task, args.task) if args.task else CONTEXT_TASK
    print(f"Container: {container}  corpus: {args.corpus} ({len(raw)} chars)  task={args.task or 'code'}  "
          f"gen_tokens={args.gen_tokens}  cpt={args.chars_per_token}  ignore_eos={args.ignore_eos}",
          file=sys.stderr)
    # Warmup (wake sleep-on-idle; not measured)
    chat(args.port, "Say hi in one word.", 8, model=args.model)
    time.sleep(1.0)
    hdr = f"{'target':>8} {'ctx_tok':>8} {'accept_len':>10} {'rate':>6} {'dec_tok/s':>10} {'samples':>7}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for tgt in targets:
        need = int(tgt * args.chars_per_token)
        corpus = raw * (need // len(raw) + 1)
        prompt = corpus[:need] + task_text
        since = int(time.time())
        time.sleep(1.1)
        try:
            dt, ctok, ptok = chat(args.port, prompt, args.gen_tokens, model=args.model, ignore_eos=args.ignore_eos)
        except Exception as e:
            print(f"{tgt:>8} {'ERR':>8}  {type(e).__name__}: {str(e)[:70]}")
            rows.append({"target": tgt, "error": f"{type(e).__name__}: {e}"})
            continue
        time.sleep(1.3)
        lens, rates, tps, depths = accept_stats(container, since)
        if lens:
            al = sum(lens) / len(lens)
            ar = sum(rates) / len(rates)
            tp = (sum(tps) / len(tps)) if tps else float("nan")
        else:
            al = ar = tp = float("nan")
        # ctx_tok = API prompt_tokens = exact prefill depth (immune to radix/pool accounting)
        print(f"{tgt:>8} {ptok:>8} {al:>10.2f} {ar:>6.2f} {tp:>10.1f} {len(lens):>7}")
        rows.append({
            "target": tgt, "ctx_tok": ptok,
            "accept_len": round(al, 3) if al == al else None,
            "accept_rate": round(ar, 3) if ar == ar else None,
            "decode_tps": round(tp, 1) if tp == tp else None,
            "completion_tokens": ctok, "samples": len(lens),
        })
    if args.out:
        json.dump({"model": args.model, "container": container, "gen_tokens": args.gen_tokens,
                   "corpus": args.corpus, "ignore_eos": args.ignore_eos, "rows": rows},
                  open(args.out, "w"), indent=2)
        print(f"saved {args.out}", file=sys.stderr)
    good = [r for r in rows if r.get("accept_len")]
    if good:
        print("\naccept_len = draft quality at depth; break-even vs no-spec ~ 1.1-1.3", file=sys.stderr)
        for r in good:
            v = "WIN" if r["accept_len"] >= 1.8 else ("marginal" if r["accept_len"] >= 1.25 else "LOSS")
            print(f"  ctx {r['ctx_tok']:>7}: accept_len {r['accept_len']:.2f} -> {v}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Accept rate benchmark for DFlash services")
    parser.add_argument("--port", type=int, default=8001, help="Server port (default: 8001)")
    parser.add_argument("--container", default=None, help="Docker container name (auto-detected if omitted)")
    parser.add_argument("--model", default="turin", help="Model name (default: turin)")
    parser.add_argument("--contexts", default=None,
                        help="Comma-separated context token depths to sweep, e.g. 1000,16000,64000,128000. "
                             "When set, runs the context-length sweep instead of the domain sweep.")
    parser.add_argument("--corpus",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_decode_bench.py"),
                        help="Text file used to build long prompts in --contexts mode "
                             "(default: sibling llm_decode_bench.py).")
    parser.add_argument("--gen-tokens", type=int, default=1500, dest="gen_tokens",
                        help="Tokens to generate per context point (default 1500; should exceed the "
                             "server decode log interval so the per-batch accept_len averages are stable).")
    parser.add_argument("--chars-per-token", type=float, default=3.3, dest="chars_per_token",
                        help="Approx chars/token for sizing prompts (true depth is read from the logs).")
    parser.add_argument("--ignore-eos", action="store_true", dest="ignore_eos",
                        help="Force full gen-tokens via ignore_eos (more samples, but forced text past a "
                             "natural stop can inflate acceptance via repetition). Default off.")
    parser.add_argument("--task", default=None,
                        help="Output-domain task for --contexts mode: preset (code|reason|chat) or literal text. "
                             "Default: code-analysis.")
    parser.add_argument("--out", default=None, help="Optional path to save --contexts results as JSON.")
    parser.add_argument("domains", nargs="*", default=[], help="Domains to run (default: all)")
    args = parser.parse_args()

    # Auto-detect container
    if args.container:
        container = args.container
    else:
        container = detect_container(args.port)
        if not container:
            print(f"ERROR: no container found on port {args.port}", file=sys.stderr)
            sys.exit(1)
    print(f"Container: {container}  Port: {args.port}", file=sys.stderr)

    if args.contexts:
        run_contexts(args, container)
    else:
        run_domains(args, container)


if __name__ == "__main__":
    main()
