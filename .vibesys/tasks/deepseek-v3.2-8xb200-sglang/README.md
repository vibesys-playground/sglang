# DeepSeek-V3.2 8xB200 SGLang task

Run from the SGLang repository root:

    vibesys --project . --task deepseek-v3.2-8xb200-sglang \
      --local --interface service --modal

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `deepseek-ai/DeepSeek-V3.2` (685B MoE with
Multi-head Latent Attention and DeepSeek Sparse Attention, native FP8) through
SGLang's OpenAI-compatible API across 8 B200 GPUs on a long-context 8k/1k
workload. The optimization target is MLA KV-cache compression, the DSA
indexer, FP8 grouped-GEMM MoE kernels on Blackwell, and large-scale expert
parallelism.

Evaluator Python dependencies are isolated from SGLang's project environment by
the manifest's `uv run --no-project` commands.
