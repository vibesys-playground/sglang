# Kimi-K3 8xB200 SGLang task

Run from the SGLang repository root:

    vibesys --project . --task kimi-k3-8xb200-sglang \
      --local --interface service --modal

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `moonshotai/Kimi-K3` (a 2.8T/104B MoE with
Kimi Delta Attention and native MXFP4 weights) through SGLang's
OpenAI-compatible API across 8 B200 GPUs. The optimization target is MXFP4
dequant fused into the MoE grouped-GEMM path, Kimi Delta Attention state
management, routing across an extreme 896-expert pool, and large-scale expert
parallelism. This is a frontier-scale target; multi-node (16xB200) may be
required for long-context serving.

Evaluator Python dependencies are isolated from SGLang's project environment by
the manifest's `uv run --no-project` commands.
