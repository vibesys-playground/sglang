# GLM-5.2 8xB200 SGLang task

Run from the SGLang repository root:

    vibesys --project . --task glm-5.2-8xb200-sglang \
      --local --interface service --modal

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `zai-org/GLM-5.2` (a 753B sparse-MoE model
with a sparse-attention mechanism) through SGLang's OpenAI-compatible API
across 8 B200 GPUs, in FP8. The optimization target is large-scale expert
parallelism with all-to-all dispatch, FP8 grouped-GEMM MoE kernels on
Blackwell, and long-context KV-cache management.

Evaluator Python dependencies are isolated from SGLang's project environment by
the manifest's `uv run --no-project` commands.
