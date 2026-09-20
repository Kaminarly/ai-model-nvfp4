# 执行结果：WSL vLLM 0.27.1 → 0.29.0 升级（验证 + 晋升）

> 执行日期：2026-09-20（WSL 本机时间）
> 依据计划：[`vllm-0.29-upgrade-plan.md`](./vllm-0.29-upgrade-plan.md)
> 基线快照：[`vllm-0.29-upgrade-baseline.md`](./vllm-0.29-upgrade-baseline.md)
> **结论：阶段 1–4、5（必测项）、6、7 全部通过，0.29.0 已晋升为仓库默认；未触发回退。**
> 唯一失败项是**可选的 DSpark 重试**（计划里明确不作阻塞项），失败形态与 0.27.1 同源。

---

## 一、Gate 总览

| 阶段 | 内容 | 结果 | 关键证据 |
| --- | --- | --- | --- |
| 0 | 前置检查 + 0.27.1 基线 | **PASS** | `vllm-0.29-upgrade-baseline.md`（argv / 两组耗时 / 组件快照） |
| 1 | 构建 0.29.0 运行时 | **PASS** | `verify` 6/6；`pip check` → `No broken requirements found.` |
| 2 | 统一 preflight | **PASS** | 22/22，`preflight result: READY` |
| 3 | 短上下文冒烟 | **PASS** | `/v1/models`=8192；短 chat 200；越界 400 |
| 4 | 长上下文生产配置 | **PASS** | `/v1/models` `max_model_len=200000`；短请求 200 |
| 5 | 功能验收 + MTP 回归 | **PASS**（必测 1–5）；可选 DSpark **FAIL（非阻塞）** | 见第四、五节 |
| 6 | 性能基线对比 | **PASS** | 无 >10% 无解释回归；MTP 解码较无投机提速约 **1.35×** |
| 7 | 仓库版本 pin 晋升 | **PASS** | 五套测试 409 项全绿；launcher 默认路径实测跑 0.29.0 |

---

## 二、阶段 1 · 运行时构建

**命令**：`VLLM_VERSION=0.29.0 bash scripts/wsl2-env.sh create --prefix $HOME/vllm --force`
（完整重建，非原地 `pip install`；日志 `~/vllm/logs/upgrade-stage1-create.log`）

**结果**：exit 0，4 个 section 全部完成；`verify` 全绿（venv / python 3.14.4 / nvcc release 13.3 / vllm 0.29.0 / torch 2.13.0+cu130 / GPU 可见，6/6）。

**安装后关键钉版**（`pip freeze`）：

| 组件 | 0.27.1 | 0.29.0 |
| --- | --- | --- |
| vllm | 0.27.1 | **0.29.0** |
| torch | 2.13.0 | 2.13.0（**未变**，与 dry-run 预判一致） |
| transformers | 5.15.0 | **5.17.0** |
| flashinfer-python | 0.6.16.post3 | **0.6.18** |
| nvidia-cutlass-dsl | — | 4.6.2 |
| humming-kernels | 0.1.10 | 0.1.12 |
| quack-kernels | 0.6.1 | 0.6.4 |
| instanttensor | — | 0.2.0 |
| nvidia-cuda-runtime / nvrtc / cupti / nvtx | 13.3.29 / 13.3.33 / 13.3.75 / 13.3.29 | 同上（脚本在 vllm 之后重新钉回 13.3.x） |

**风险 3（CUDA 组件冲突）结论**：`pip check` 输出 `No broken requirements found.`——0.29.0 元数据要求 nvidia-cuda-{nvrtc,runtime,cupti}/nvtx 13.0.x，脚本钉回 13.3.x **未产生任何 `pip check` 冲突**，FlashInfer JIT（0.6.18）也正常编译（阶段 3 实测）。

---

## 三、阶段 2–4 · 预检与服务启动

- **阶段 2**：`VLLM_VERSION=0.29.0 preflight.sh` → 22/22 通过、`preflight result: READY`。
- **阶段 3**（`serve.sh`，8192/1，端口 8000）：
  - `/v1/models` 返回模型，`max_model_len=8192`；
  - 短 chat 正常返回 `content`；
  - 16384-token 越界 prompt → **HTTP 400**，错误信息 `This model's maximum context length is 8192 tokens ...`（引擎入队阶段拒绝，非 OOM）。
- **阶段 4**（`direct.sh` 默认生产配置，200000/16/0.90，端口 8192）：preflight READY、VRAM gate 通过（free 30760 ≥ 29347 MiB）、`/v1/models` `max_model_len=200000`、短请求 200。

### 3.1 阶段 3 首次启动失败及修复（本次升级最重要的新发现）

**症状**：0.29.0 首次启动，引擎在 `GPUModelRunnerV2` 构造期崩溃：

```
File ".../vllm/v1/worker/gpu/buffer_utils.py", line 47, in __init__
    raise RuntimeError("UVA is not available")
RuntimeError: UVA is not available
```

**根因**：0.29.0 起 **V2 model runner（MRV2）成为默认 runner**，它用 `cudaHostAlloc`（UVA）分配 request-state 缓冲。UVA 依赖固定内存（pinned memory），而 `vllm/platforms/cuda.py::is_pin_memory_available()` 在 WSL2 上**无论内核是否够新都默认关闭固定内存**，只有显式设 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 才返回 true。本机内核 `6.18.33.2-microsoft-standard-WSL2` 远超 vLLM 的 4.19.121 门槛。

**验证**：`torch.zeros(...,pin_memory=True)` 实测成功；`is_uva_available()` 在设该变量后由 `False → True`。

**处置**：在共享启动环境 `scripts/lib/serve-lib.sh::prepare_vllm_env` 中统一导出 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，因此 `serve.sh` / `direct.sh` / `fullcontext.sh` 与全部 bat 启动器都自动生效，无需各调用点分别设置。该修复是 0.29.0 在本机可用的前提。

---

## 四、阶段 5 · 功能验收（必测项 1–4）

在阶段 4 服务（200000 无投机）与实际生产 MTP 服务上各测一遍，结论一致：

| # | 项目 | 0.29.0 结果 |
| --- | --- | --- |
| 1 | 工具调用 | `finish_reason=tool_calls`，返回结构化 `tool_calls`（`calculator`，`arguments={"expression":"128*56+19"}`），`qwen3_xml` parser 正常 |
| 2 | 推理字段 | **字段名仍为 `reasoning`**（与 0.27.1 一致）；`content` 独立。新增 `usage.completion_tokens_details.reasoning_tokens`（0.27.1 无此细分） |
| 3 | 长上下文 | 125060-token prompt 正常 200（无 400 / OOM） |
| 4 | 并发 | 16 路并行短请求 **16/16 HTTP 200**，无 5xx |

**风险 2 结论**：`reasoning_content` 移除这一"破坏性变更"在本项目的 `qwen3` 推理 parser 下**未出现字段改名**——思考内容仍在 `reasoning`。README 4.8/Q13 已据此更新为"两版本实测一致"。

---

## 五、阶段 5 第 5 项 · MTP 回归（必测）

**配置**：`VLLM_SPEC_METHOD=mtp FULL_MAX_MODEL_LEN=180000`，16 并发 / 0.90，端口 8192（等价 `start-api-server-mtp.bat`）。
**日志**：`~/vllm/logs/upgrade-stage5-mtp.log`。

| 检查 | 结果 |
| --- | --- |
| argv 含 `--spec-method mtp --spec-tokens 3` | ✅ |
| `speculative_config=SpeculativeConfig(method='mtp', num_spec_tokens=3)` | ✅ |
| 服务稳定、应答正常 | ✅ |
| 投机接受率非零 | ✅ 见下表 |
| **实际 model runner** | **`Using V2 Model Runner`**（MRV2，未回退 MRV1） |

**风险 1 结论**：0.29.0 下 **MTP 直接跑在 MRV2 上**（日志 `gpu_worker.py:429 Using V2 Model Runner`），**没有**如发布说明所说回退到 MRV1；功能与加速均正常。

**投机统计对比**：

| 指标 | 0.27.1（基线） | 0.29.0（本轮多次采样区间） |
| --- | --- | --- |
| Mean acceptance length | 2.59 | 2.01 – 2.70 |
| Avg Draft acceptance rate | 53.1% | 33.8% – 56.7% |
| Per-position 接受率 | 0.701 / 0.497 / 0.396 | 0.557–0.757 / 0.333–0.529 / 0.123–0.414 |

两者量级相当；接受率随请求内容波动属正常，无系统性退化。

**MTP 相对无投机的提速（同 180000/16/0.90，流式、剔除 prefill）**：

| 配置 | 解码吞吐（3 次） |
| --- | --- |
| 0.29.0 无投机 | 78.0 / 78.7 / 78.3 tok/s |
| 0.29.0 MTP | 88.5 / 105.8 / 106.2 tok/s |

稳态约 **1.35×**（105.8–106.2 vs 78.3–78.7），满足 Gate 5"吞吐高于无 MTP 同配置"。

### 5.1 第 6 项（可选）· DSpark 重试 → **失败**（非阻塞）

**配置**：`VLLM_EXTRA_ARGS="--spec-method dspark --spec-model /home/kami/models/Qwen3.8-27B-DSpark-Acc --spec-tokens 7"`，草稿用 `Qwen3.8-27B-DSpark-Acc`（`Qwen3DSparkModel`、`markov_rank=256`、`block_size=7`）。日志 `~/vllm/logs/upgrade-stage5b-dspark.log`。

**结果**：启动失败，与 0.27.1 **同源、同报错**：

```
File ".../vllm/model_executor/models/qwen3_dspark.py", line 313, in load_weights
  → .../vocab_parallel_embedding.py", line 496, in weight_loader
RuntimeError: The size of tensor a (128) must match the size of tensor b (256) at non-singleton dimension 1
```

**根因定位**：checkpoint 的 `markov_head.markov_w1/w2` 形状是 `[248320, 256]`，而 0.29.0 构建出的对应参数第二维是 **128**。已核对：

- checkpoint `config.json` 顶层 `markov_rank = 256`、`dflash_config.markov_rank = 256`；
- `SpeculativeConfig` 用 vLLM 自身 `get_config` 解析该草稿目录，`markov_rank = 256`（解析正确）；
- `qwen3_dspark.py::Qwen3DSparkModel.__init__` 走 `config.markov_rank` 建头（即 256）。

即 **配置解析层没问题，但 V2 路径下实际建出的 markov 头 rank 是 128**，说明 0.29.0 的 DSpark/MRV2 加载路径对该 checkpoint 仍不兼容。**这不阻塞其他 Gate**（计划第五节明确 DSpark 为可选项）：vLLM 原生 DSpark 路线仍不可用，单流 DSpark 继续沿用已实测的 SparkInfer 引擎（见 README 4.9）。

**上游反馈建议**：0.29.0 的 DSpark markov 头在 MRV2 下与 `markov_rank=256` 的 checkpoint 不匹配（0.27.1 亦有同类问题），值得开 issue。

---

## 六、阶段 6 · 性能基线对比（0.27.1 vs 0.29.0，均为 MTP / 180000 / 16 / 0.90）

同一套协议（3 短 + 3 长，同一请求体；长 prompt = 125060 token）。

| 指标 | 0.27.1 MTP | 0.29.0 MTP | 变化 |
| --- | --- | --- | --- |
| short #1 (s) | 1.691 | 1.156 | **−31.6%** |
| short #2 (s) | 1.347 | 1.055 | **−21.7%** |
| short #3 (s) | 1.129 | 1.031 | **−8.7%** |
| long #1 · 125k prefill (s) | 33.063 | 33.477 | **+1.3%** |
| long #2 · 前缀命中 (s) | 1.705 | 1.624 | −4.7% |
| long #3 · 前缀命中 (s) | 1.702 | 1.556 | −8.6% |

**判定**：

- 短请求（含思考、128 输出 token）**全面变快** 9%–32%；
- 长 prefill 慢 **1.3%**，在噪声范围内，远低于 10% 阈值，**不构成回归**；
- 前缀命中长请求变快 5%–9%；
- 无任何 >10% 的无解释回归 → **Gate 6 PASS**。
- 性能对比只依赖本机两列数据（计划要求；SGLang 路线的 55 s 值不作判据）。

---

## 七、阶段 7 · 仓库版本 pin 晋升（已执行）

**改动清单**（`git diff`，共 7 个文件）：

| 文件 | 改动 |
| --- | --- |
| `scripts/lib/wsl2-env-lib.sh` | `VLLM_VERSION` 默认 `0.27.1 → 0.29.0`；更新 CUDA 栈注释（torch 2.13、cutlass-dsl 4.6.2、13.0→13.3 覆盖说明） |
| `scripts/lib/serve-lib.sh` | **新增 `export VLLM_WSL2_ENABLE_PIN_MEMORY=1`**（0.29.0 必需，见 3.1）；版本相关注释同步 |
| `scripts/wsl2-env.sh` | 两处注释里的 0.27.1 → 0.29.0 |
| `tests/run-tests.sh` | 5 处版本期望串 → 0.29.0 |
| `tests/preflight-tests.sh` | 1 处版本期望串 → 0.29.0 |
| `tests/fakebin/vllm`、`tests/fakebin/python3` | `FAKE_VLLM_VERSION` 默认值 → 0.29.0 |

README 同步：第 1/2/3 节版本表述、4.6 MTP（版本描述 + `--spec-tokens` 说明）、5.4、6.8（**新增 UVA/固定内存说明**）、Q13（`reasoning` 字段两版本一致）。

**测试套件**（晋升后全跑）：

| 套件 | 结果 |
| --- | --- |
| `tests/run-tests.sh` | 51 passed, 0 failed |
| `tests/preflight-tests.sh` | 71 passed, 0 failed |
| `tests/serve-tests.sh` | 88 passed, 0 failed |
| `tests/fullcontext-tests.sh` | 114 passed, 0 failed |
| `tests/sparkinfer-tests.sh` | 85 passed, 0 failed |
| **合计** | **409 passed, 0 failed** |

**晋升后端到端实测**：不带 `VLLM_VERSION` 覆盖、直接走 `direct.sh`（即 bat 启动器的路径）+ MTP 生产配置 → 引擎日志 `Initializing a V1 LLM engine (v0.29.0)`、`/v1/models` `max_model_len=180000`、短 chat 200、argv 含 `--spec-method mtp --spec-tokens 3`。**默认 pin 生效，bat 启动器无需改动即可按 0.29.0 运行。**

> 注意：`VLLM_VERSION` 是 `wsl2-env-lib.sh` 的**运行时默认**，不改动 `tests/` 之外任何环境；显式 `VLLM_VERSION=0.27.1` 仍可重建回旧版本（见回退）。

---

## 八、回退情况

**未触发**。所有 Gate 通过，0.29.0 保留为默认。

若日后需回退：`wsl2-env.sh create --force`（不带 `VLLM_VERSION`）现在会装 **0.29.0**；要回到 0.27.1 需显式 `VLLM_VERSION=0.27.1 bash scripts/wsl2-env.sh create --force`，并 `git revert` 阶段 7 的提交。回退只影响 venv，模型目录 / `~/vllm/logs` / Windows 侧 portproxy 与防火墙规则不受影响。

---

## 九、产物与遗留

- 本文件、`vllm-0.29-upgrade-baseline.md`（阶段 0 快照）、阶段 7 的仓库改动。
- 服务日志留档在 `~/vllm/logs/`：`upgrade-0271-baseline.log`、`upgrade-stage1-create.log`、`upgrade-stage3-smoke.log`、`upgrade-stage4-full.log`、`upgrade-stage5-mtp.log`、`upgrade-stage5b-dspark.log`、`upgrade-stage5-nospec180k.log`、`upgrade-0290-mtp-decode.log`、`upgrade-stage7-default.log`。
- **遗留**：vLLM 原生 DSpark 在 0.29.0 仍不可用（markov rank 128 vs 256），单流 DSpark 继续用 SparkInfer；建议向上游报 issue（见 5.1）。
- **未测项（计划内即无）**：`max_num_batched_tokens` 默认 8192→16384 只体现在批处理行为，本次以吞吐/显存间接观察（长 prefill 变化仅 +1.3%，显存占用与基线同档 ~30.9 GiB），未见异常。
