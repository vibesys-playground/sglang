# Qwen3.5-397B-A17B 4xB200 SGLang task

Run from the SGLang repository root:

    vibesys --project . --task qwen3.5-397b-a17b-4xb200-sglang \
      --local --interface service --modal

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `Qwen/Qwen3.5-397B-A17B` (a hybrid
Gated-DeltaNet + sparse-MoE model) through SGLang's OpenAI-compatible API
across 4 B200 GPUs, in FP8 to fit the weights. The optimization target is TP+EP
sharding with all-to-all dispatch, FP8 grouped-GEMM MoE kernels, and SGLang's
RadixAttention prefix cache. This is the SGLang counterpart to the vLLM variant.

Evaluator Python dependencies are isolated from SGLang's project environment by
the manifest's `uv run --no-project` commands.
