# Accept Bench

Acceptance rate and decode throughput benchmarks for speculative decoding (DFlash) on vLLM and SGLang.

## Files

| File | Description |
|---|---|
| `accept_bench_vllm.py` | Accept bench driver for vLLM (reads `/metrics` counters) |
| `accept_bench.py` | Accept bench driver for SGLang (reads log lines) |
| `accept_corpus.txt` | 5.2MB filler text, truncated to each context depth |
| `run-codegen-various.sh` | 6-task codegen sweep (Python/TS/Go/C++) at 4 context depths |
| `run-codegen-various-sglang.sh` | Same sweep for SGLang |
| `accept_*.json` | Pre-computed results (chat, code, reason, fp8 variants, 1M code, vs-context) |

## Requirements

- Running vLLM or SGLM server on `localhost:8001`
- Python 3.10+ (stdlib only — `urllib`, `json`, `time`)

## Quick start

```bash
# Edit model name and output path at the top of the script, then:
bash run-codegen-various.sh
```

## Output

Each task produces a JSON file with per-context metrics: acceptance length, acceptance rate, decode tok/s, TTFT, and completion token count.