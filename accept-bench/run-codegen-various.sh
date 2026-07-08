#!/usr/bin/env bash
# Various code-generation DFlash accept + decode-t/s, at 16k/128k/~300k, on the bf16 vLLM service.
# 6 diverse real-codegen prompts (mixed languages) -> robust mean +/- spread per depth.
set -uo pipefail
cd /home/tom/llm-inference-bench/accept-bench
OUT=/home/tom/llm-inference-bench/results/codegen-20260708
mkdir -p "$OUT"
M=ripper
CTX=16000,128000,340000,500000

declare -A P
P[py_lru]=$'\n\n# TASK\nIgnore the text above. Write a complete, self-contained Python module: a thread-safe LRU cache with per-key TTL expiry (get/set/delete/clear, capacity eviction, lock-based), plus a pytest suite. Docstrings and type hints. Output only Python code.'
P[py_dijkstra]=$'\n\n# TASK\nIgnore the text above. Write a complete Python module implementing Dijkstra shortest-path on a weighted directed graph (adjacency-list, binary heap) returning distances and reconstructed paths, with a pytest suite. Type hints and docstrings. Output only Python code.'
P[py_parser]=$'\n\n# TASK\nIgnore the text above. Write a complete Python recursive-descent parser and evaluator for arithmetic expressions supporting + - * /, parentheses, unary minus and floats, with a tokenizer, clear error handling, and a pytest suite. Output only Python code.'
P[ts_util]=$'\n\n# TASK\nIgnore the text above. Write a complete TypeScript utility module implementing strongly-typed debounce and throttle (leading/trailing options, cancel and flush), with JSDoc and a vitest test suite. Output only TypeScript code.'
P[go_mw]=$'\n\n# TASK\nIgnore the text above. Write a complete Go HTTP server using net/http with a composable middleware chain (request logging, panic recovery, bearer-token auth, token-bucket rate limiter) and two example handlers. Output only Go code.'
P[cpp_queue]=$'\n\n# TASK\nIgnore the text above. Write a complete C++17 single-header thread-safe bounded blocking queue (push/pop/try_pop with timeout, condition variables) plus a short main() demonstrating producer/consumer threads. Output only C++ code.'

for name in py_lru py_dijkstra py_parser ts_util go_mw cpp_queue; do
  echo "######## $name ########"
  python3 accept_bench_vllm.py --task "${P[$name]}" --contexts "$CTX" --model "$M" --out "$OUT/cg_${name}.json" || echo "!!${name}_FAILED"
done
echo "######## ALL CODEGEN DONE ########"
