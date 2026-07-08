#!/usr/bin/env python3
"""DFlash accept + decode-t/s bench for the vLLM service (port 8001).

Why a vLLM-specific variant of accept_bench.py:
  - accept_bench.py reads accept_len from sglang "Decode batch" log lines; its vLLM
    branch only grabs the 10s-interval "SpecDecoding metrics" log (sparse, and it
    leaves decode-t/s EMPTY for vLLM).
  - Here acceptance comes from /metrics counter DELTAS around each request (exact,
    independent of the log interval) and decode-t/s from client-side STREAMING
    (first-token..last-token), so it's a real pure-decode rate.

Mirrors accept_bench.py exactly otherwise: same corpus filler (accept_corpus.txt
truncated to each depth) + same TASK presets, so it's apples-to-apples vs the
sglang accept_*.json runs. ctx_tok is the API prompt_tokens (true prefill depth).

Acceptance math (vLLM spec-decode counters, k = num_speculative_tokens):
  steps   = d(spec_decode_num_drafts_total)
  acc     = d(spec_decode_num_accepted_tokens_total)
  dtok    = d(spec_decode_num_draft_tokens_total)   (= steps*k)
  accept_len  = 1 + acc/steps      # mean tokens emitted per step incl. the bonus token
  accept_rate = acc/dtok           # vLLM-style: fraction of drafted tokens accepted (= (accept_len-1)/k)
"""
import argparse, json, os, time, urllib.request, urllib.error

CONTEXT_TASK = (
    "\n\n# TASK\nAnalyze the text/code above in thorough detail: explain what it does, walk "
    "through its structure and control flow, and call out potential bugs or improvements. "
    "Write a long, multi-section analysis."
)
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

CTRS = ("vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_drafts_total")


def get_metrics(port):
    with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=30) as r:
        txt = r.read().decode(errors="ignore")
    out = {}
    for m in CTRS:
        tot = None
        for line in txt.splitlines():
            if line.startswith(m + "{") or line.startswith(m + " "):
                try:
                    tot = (tot or 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        out[m] = tot
    return out


def stream_chat(port, prompt, max_tokens, model, ignore_eos):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens,
               "stream": True, "stream_options": {"include_usage": True}}
    if ignore_eos:
        payload["ignore_eos"] = True
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None; n = 0
    with urllib.request.urlopen(req, timeout=2400) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now; n += 1
    span = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    ctok = usage.get("completion_tokens") if usage else n
    ptok = usage.get("prompt_tokens") if usage else None
    dtps = ((ctok - 1) / span) if (span and ctok and ctok > 1) else None
    return {"ttft": (t_first - t0) if t_first else None, "decode_tps": dtps,
            "completion_tokens": ctok, "prompt_tokens": ptok, "span": span}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="mimo-v25-pro-fp4-dflash")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--task", default="code")
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "accept_corpus.txt"))
    ap.add_argument("--gen-tokens", type=int, default=1500, dest="gen_tokens")
    ap.add_argument("--chars-per-token", type=float, default=3.3, dest="cpt")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--ignore-eos", action="store_true", dest="ignore_eos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    raw = open(a.corpus, errors="ignore").read()
    task_text = TASKS.get(a.task, a.task)
    targets = [int(x) for x in a.contexts.split(",") if x.strip()]
    print(f"corpus={a.corpus} ({len(raw)} chars) task={a.task} gen={a.gen_tokens} "
          f"ignore_eos={a.ignore_eos} model={a.model}", flush=True)
    try:
        stream_chat(a.port, "Say hi in one word.", 8, a.model, False)  # wake/warmup
    except Exception as e:
        print("warmup err:", e, flush=True)
    time.sleep(1.0)
    hdr = f"{'target':>9} {'ctx_tok':>8} {'accept_len':>10} {'rate':>6} {'dec_tok/s':>10} {'ttft_s':>8} {'steps':>7} {'ctok':>6}"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    rows = []
    for tgt in targets:
        need = int(tgt * a.cpt)
        prompt = (raw * (need // len(raw) + 1))[:need] + task_text
        m0 = get_metrics(a.port)
        try:
            res = stream_chat(a.port, prompt, a.gen_tokens, a.model, a.ignore_eos)
        except Exception as e:
            print(f"{tgt:>9} {'ERR':>8}  {type(e).__name__}: {str(e)[:60]}", flush=True)
            rows.append({"target": tgt, "error": f"{type(e).__name__}: {e}"}); continue
        time.sleep(1.0)
        m1 = get_metrics(a.port)
        steps = (m1[CTRS[2]] - m0[CTRS[2]]) if m0[CTRS[2]] is not None else 0
        acc = (m1[CTRS[0]] - m0[CTRS[0]]) if m0[CTRS[0]] is not None else 0
        dtok = (m1[CTRS[1]] - m0[CTRS[1]]) if m0[CTRS[1]] is not None else 0
        al = (1 + acc / steps) if steps > 0 else None
        ar = (acc / dtok) if dtok > 0 else None
        nan = float("nan")
        print(f"{tgt:>9} {str(res['prompt_tokens']):>8} {(al or nan):>10.2f} {(ar or nan):>6.3f} "
              f"{(res['decode_tps'] or nan):>10.1f} {(res['ttft'] or nan):>8.1f} {int(steps):>7} "
              f"{str(res['completion_tokens']):>6}", flush=True)
        rows.append({"target": tgt, "ctx_tok": res["prompt_tokens"],
                     "accept_len": round(al, 3) if al else None,
                     "accept_rate": round(ar, 3) if ar else None,
                     "decode_tps": round(res["decode_tps"], 1) if res["decode_tps"] else None,
                     "ttft_s": round(res["ttft"], 2) if res["ttft"] else None,
                     "completion_tokens": res["completion_tokens"],
                     "draft_steps": int(steps), "accepted": int(acc), "drafted": int(dtok),
                     "samples": int(steps)})
    if a.out:
        json.dump({"model": a.model, "container": "mimo-vllm", "engine": "vllm",
                   "gen_tokens": a.gen_tokens, "corpus": a.corpus, "task": a.task,
                   "ignore_eos": a.ignore_eos, "k": a.k, "rows": rows},
                  open(a.out, "w"), indent=2)
        print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
