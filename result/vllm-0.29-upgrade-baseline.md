# 阶段 0 基线快照：vLLM 0.27.1 MTP 生产配置

> 采集日期：2026-09-20（WSL 本机时间）
> 对应计划：`result/vllm-0.29-upgrade-plan.md` 阶段 0
> 采集方式：用 `scripts/direct.sh start` 以与 `scripts/start-api-server-mtp.bat` 等价的参数
> 直接启动 0.27.1 MTP 服务（采集时同步服务并未运行，故先起服务再采基线）。

---

## 一、环境快照

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090，32607 MiB |
| Windows 驱动 | 610.88 |
| WSL | Ubuntu，WSL2 |
| venv python | Python 3.14.4（`/home/kami/vllm/venv`） |
| vLLM | 0.27.1 |
| torch | 2.13.0+cu130 |
| transformers | 5.15.0 |
| flashinfer-python | 0.6.16.post3 |
| nvidia-cuda-runtime | 13.3.29 |
| nvidia-cuda-nvrtc | 13.3.33 |
| nvidia-cuda-cupti | 13.3.75 |
| nvidia-nvtx | 13.3.29 |
| humming-kernels | 0.1.10 |
| quack-kernels | 0.6.1 |
| 空闲显存（服务停止前） | 2194 MiB 已用 |
| 服务运行显存 | 30957 MiB 已用（0.90 利用率） |

`~/vllm/env-info.txt` 记录的是旧 prefix（`/home/kami/qwen3-nvfp4-rtx5090`，2026-08-18 创建），
与实际使用的 `/home/kami/vllm/venv` 不一致；本次基线以 `pip freeze` / 运行时查询为准，未改该文件。

---

## 二、0.27.1 服务 argv（回退比对基准）

```
/home/kami/vllm/venv/bin/python /home/kami/vllm/venv/bin/vllm serve \
  --model /home/kami/models/Qwen3.8-27B-NVFP4-VLLM \
  --quantization modelopt --kv-cache-dtype fp8 --enable-prefix-caching \
  --host 127.0.0.1 --port 8192 --served-model-name Qwen3.8-27B-NVFP4-VLLM \
  --max-model-len 180000 --max-num-seqs 16 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90 \
  --spec-method mtp --spec-tokens 3 \
  --override-generation-config.temperature 1.0 \
  --override-generation-config.top_p 0.95 \
  --override-generation-config.top_k 20 \
  --override-generation-config.min_p 0.0 \
  --override-generation-config.presence_penalty 0.0 \
  --override-generation-config.repetition_penalty 1.0 \
  --trust-remote-code
```

日志：`~/vllm/logs/upgrade-0271-baseline.log`。

关键启动事实（供 0.29.0 对比）：

- 引擎：`Initializing a V1 LLM engine (v0.27.1)`；`speculative_config=SpeculativeConfig(method='mtp', num_spec_tokens=3)`；
- 量化：`quantization=modelopt_fp4`，`kv_cache_dtype=fp8`，`enable_prefix_caching=True`，`enable_chunked_prefill=True`；
- 权重加载：FlashInfer CUTLASS NVFP4 GEMM（`Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`）；
- 注意力：`Using FLASHINFER attention backend`；
- GDN prefill：`Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)`；
- `max_num_scheduled_tokens` 因投机解码被自动压到 2048（WARNING，非故障）；
- 权重 checkpoint 17.48 GiB，3 个 shard。

---

## 三、基线耗时（3 短 + 3 长，同一请求体）

请求体固定为 `/tmp/bench-short.json`（128 输出 token）与 `/tmp/bench-long.json`
（重复填充 prompt，`usage.prompt_tokens = 125060`）。

| 轮次 | TTFT (s) | 总耗时 (s) | HTTP |
| --- | --- | --- | --- |
| short #1 | 1.691 | 1.691 | 200 |
| short #2 | 1.347 | 1.347 | 200 |
| short #3 | 1.129 | 1.129 | 200 |
| long #1 | 33.063 | 33.064 | 200 |
| long #2 | 1.705 | 1.705 | 200 |
| long #3 | 1.702 | 1.702 | 200 |

说明：长请求 #2/#3 命中 prefix cache（日志 `Prefix cache hit rate: 65.6%`），
故只有首轮是完整 125k-token prefill（约 33 s）；两版本用同一协议，可比性成立。
`long #2/#3` 的 1.7 s 是缓存命中量级，不单独作为吞吐判据。

---

## 四、0.27.1 功能参考样本（供阶段 5 对比）

| 项目 | 0.27.1 结果 |
| --- | --- |
| `reasoning` 字段 | **存在**，思考内容独立于 `content`（message keys：annotations/audio/content/function_call/reasoning/refusal/role） |
| MTP 投机统计 | Mean acceptance length **2.59**，Avg Draft acceptance rate **53.1%**，Accepted 314 / Drafted 591 tokens，per-position 0.701 / 0.497 / 0.396 |
| 引擎标记 | `Initializing a V1 LLM engine (v0.27.1)` |

阶段 5 需在 0.29.0 上复测同一组：工具调用、`reasoning` 字段名、长上下文、16 并发、MTP 接受率。

---

## 五、Gate 0 结论

- [x] 基线文件含 argv、两组耗时、env-info/组件快照 → 本文件；
- [x] 服务已在采集后停止，显存释放（见 result 文档阶段 0 小节）。
