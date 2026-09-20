# 执行计划：WSL vLLM 0.27.1 → 0.29.0 升级（验证 + 回退）

> 制定日期：2026-09-19（WSL 本机时间）
> 参考资料：[vLLM 0.28.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)、[vLLM 0.29.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)、[PyPI vllm](https://pypi.org/project/vllm/)
> 执行状态：**已于 2026-09-20 执行完毕；0.29.0 已晋升为仓库默认，未触发回退。** Gate 结果、pip check 记录、功能验收明细、性能对比与 DSpark 结论见 [`vllm-0.29-upgrade-result.md`](./vllm-0.29-upgrade-result.md)，阶段 0 基线快照见 [`vllm-0.29-upgrade-baseline.md`](./vllm-0.29-upgrade-baseline.md)。

---

## 一、目标与范围

**目标**：把 WSL Ubuntu 里 `/home/kami/vllm/venv` 的 vLLM 从 0.27.1 升级到 PyPI 最新版 **0.29.0**（2026-09-09 发布；中间还有 0.28.0，2026-08-26），完成验证链，并为任一阶段失败提供一键回退路径。

**范围内**：

1. 重建 0.29.0 运行时（`wsl2-env.sh create --force` + 环境变量覆盖，完整重建 venv 而不是原地 `pip install`，理由见第四节风险 3）；
2. 验证链：统一 preflight → 短上下文冒烟 → 长上下文生产配置 → 功能验收（工具调用 / 推理字段 / MTP 回归）→ 性能基线对比；
3. 回退：验证失败时重建回 0.27.1，服务恢复原 MTP 生产配置；
4. （可选，阶段 7）验收全部通过后，把仓库内的版本 pin 从 0.27.1 提升到 0.29.0。

**范围外**：不改模型权重（`wsl2-env.sh` 从不下载/复制/删除模型文件）；不动 Windows 驱动与 WSL 系统；不做公网/局域网暴露；不动 `~/vllm/logs` 与模型目录。

**关键设计（实验与晋升分离）**：

- 阶段 1–6 全程用环境变量 `VLLM_VERSION=0.29.0` 覆盖运行，**不改仓库任何文件**。脚本里的版本 Gate（`check_vllm_venv` 要求 `vllm 版本 == $VLLM_VERSION`，默认 0.27.1）因此要求每次调用显式带 `VLLM_VERSION=0.29.0`；
- 回退 = 去掉覆盖、按默认 0.27.1 重建，不需要还原任何仓库改动；
- 只有阶段 7（晋升）才改仓库默认 pin 与测试套件。

---

## 二、0.27.1 → 0.29.0 改进要点（与本项目相关部分）

完整改动见两份发布说明（0.28.0 共 584 commits，0.29.0 共 594 commits）。与本项目（Qwen3.8-27B NVFP4 + RTX 5090 + MTP/DSpark + 18/20 万上下文）直接相关的：

| 类别 | 改进 | 对本站点的意义 |
| --- | --- | --- |
| 投机解码 | DSpark 置信度调度验证（#47808）、top-k Markov 投影（#49969）、自适应投机 token 预算（DSpark TTFT 提升约 60%，#51725）、DFlash2、Qwen GDN fused MTP decode kernel（#51674、#52539） | DSpark 路线在 0.27.1 上失败（`weight_loader` 报错，见 `qwen38-nvfp4-dspark-upgrade-result.md`），0.29.0 值得重试（阶段 5 可选项） |
| 引擎 | Model Runner V2 成为所有模型默认 runner（#53183），MRV1 宣布 v0.32 移除；部分投机方法在 MRV2 尚未支持、会回退 MRV1 | MTP/DSpark 是否落在 MRV2 上需实测（风险 1） |
| NVFP4 | batch-invariant NVFP4 MoE（CUTLASS，#40372）、weight-only NVFP4 checkpoint 路由 W4A16（#54427） | 本项目 modelopt NVFP4 权重加载路径的 kernel 选择可能变化 |
| Blackwell | FlashInfer XQA decode 支持 SM12x（#49718）、Blackwell CUDA graph capture 默认提到 1024（#49390）、b12x FP4 MoE 新 kernel 覆盖 SM120/121（#52018） | RTX 5090 直接受益 |
| 默认值 | `max_num_batched_tokens` 8192 → 16384（0.28，#51726） | 长上下文批处理行为变化，阶段 4/6 观察 |
| API/安全 | `reasoning_content` 输出移除被文档标记为破坏性客户端变更（0.28，#50624）；`cache_salt` 限长 1024、`api_key`/`hf_token` 日志脱敏（0.29） | 客户端 `reasoning` 字段名需重新验证（风险 2） |
| 依赖 | Transformers 5.15.0（0.28）、FlashInfer 0.6.18（0.29）；**torch 钉在 2.13.0——与 0.27.1 完全相同**（dry-run 实测） | CUDA 13 栈不变，torch 层无变动 |

---

## 三、执行前状态快照（2026-09-19 实测）

| 项目 | 现状 | 备注 |
| --- | --- | --- |
| WSL | Ubuntu，WSL2（version 2），运行中 | 单发行版 |
| 当前 vLLM | **0.27.1**（`/home/kami/vllm/venv`，Python 3.14.4，torch 2.13.0，transformers 5.15.0） | `venv` 里只有这一个 vllm 安装；旧 `~/qwen3-nvfp4-rtx5090/venv` 已不存在 |
| PyPI 最新 | **0.29.0**（2026-09-09）；中间版 0.28.0（2026-08-26） | `pip index versions vllm` 实测 |
| 可行性预检 | 在现有 3.14.4 venv 里 `pip install --dry-run vllm==0.29.0` **解析成功** | torch 保持 2.13.0；新增 wheel：vllm-0.29.0、flashinfer-python 0.6.18、humming-kernels 0.1.12、instanttensor 0.2.0、quack-kernels 0.6.4、nvidia-cutlass-dsl 4.6.2；另会把 nvidia-cuda-nvrtc/runtime/cupti/nvtx 从 13.3.x 降到 13.0.8x（风险 3） |
| 运行中服务 | MTP 服务（vLLM 0.27.1，PID 1030）：`--spec-method mtp --spec-tokens 3`，180000 上下文 / 16 并发 / 0.90 利用率，端口 8192，loopback，带默认采样参数 | 即 `start-api-server-mtp.bat` 的配置；升级前需先停（先完成阶段 0 基线） |
| 显存 | 30,852 / 32,607 MiB 已用（被上面的服务占用） | 重建与重启前必须释放 |
| 模型目录 | `/home/kami/models/Qwen3.8-27B-NVFP4-VLLM`（默认，含 MTP 头；MTP 路线必须用它） | 其他目录：`-DSpark`（final 构建，无 MTP 头）、`Qwen3.8-27B-DSpark-Acc`、`Qwen3.6-…` |
| 仓库 pin | `scripts/lib/wsl2-env-lib.sh`：`VLLM_VERSION` 默认 0.27.1；`tests/` 全套件钉死 0.27.1 | 晋升前需同步（阶段 7） |

---

## 四、已知风险与待确认项

| # | 风险 | 影响 | 处置 |
| --- | --- | --- | --- |
| 1 | **MRV2 默认化**：0.29.0 里部分投机解码方法 MRV2 尚未支持，配置到这些特性会自动回退 MRV1（发布说明原文） | MTP/DSpark 行为可能变化 | 待确认：阶段 5 看服务日志确认实际 runner 与投机接受率；MTP 回归（阶段 5 第 5 项）为必测 |
| 2 | **`reasoning_content` 移除**被 0.28 文档标记为破坏性客户端变更 | 客户端读取推理内容的字段名可能变化（0.27.1 是 `reasoning`，见 README 4.8） | 待确认：阶段 5 第 2 项记录实际字段名；若变化，同步客户端适配说明 |
| 3 | **CUDA 组件版本冲突**：0.29.0 的依赖 humming-kernels 要求 nvidia-cuda-nvrtc 13.0.88 / nvidia-cuda-runtime 13.0.96 / nvidia-cuda-cupti 13.0.85 / nvidia-nvtx 13.0.85；而 `wsl2-env.sh create` 在安装 vllm **之后**会把这四个组件钉回 13.3.x（FlashInfer JIT 的 CCCL 头检查要求） | `pip check` 可能报冲突 | 用完整重建（`create --force`）而不是原地 pip install；阶段 1 末尾跑 `pip check` 记录结果；实际影响由阶段 3 服务启动（JIT 首建）验证 |
| 4 | **默认值变化**：`max_num_batched_tokens` 8192 → 16384 | 长上下文下的批处理与显存行为变化 | 阶段 4/6 观察吞吐与显存 |
| 5 | **测试套件钉死 0.27.1**：`tests/run-tests.sh`、`tests/preflight-tests.sh`、`tests/fakebin/vllm`、`tests/fakebin/python3` 与 README 多处版本描述 | 阶段 7 晋升后旧测试会失败 | 阶段 7 同步全部引用；晋升前测试套件保持原样 |
| 6 | **网络/下载量**：完整重建要下载数 GB（torch 2.13 cu13 wheel 等） | 耗时 15–30 分钟；断网即失败 | 执行前确认 WSL 网络；失败直接重试（重建幂等，venv 每次全删） |
| 7 | **DSpark 重试**：0.27.1 上 vLLM+DSpark 路线失败（`vocab_parallel_embedding.weight_loader` 报错）；0.29.0 含 DSpark 新改进（#47808、#51310） | 重试成功可换回 vLLM 原生 DSpark；失败维持 SGLang 路线（镜像已于 2026-09-19 删除，需 `docker pull` 约 18 GB） | 阶段 5 可选项，不作为 Gate 阻塞项 |

---

## 五、阶段总览

| 阶段 | 内容 | 预计耗时 | 出口 Gate |
| --- | --- | --- | --- |
| 0 | 前置检查 + 0.27.1 基线采集 | 20–40 min | 基线与快照落盘；服务已停、显存已释放 |
| 1 | 构建 0.29.0 运行时 | 15–30 min | `verify` 全过；`pip check` 已记录 |
| 2 | 统一 preflight | 5–10 min | READY，全部项通过 |
| 3 | 短上下文冒烟（`serve.sh`，8192/1，端口 8000） | 10–20 min（含 FlashInfer 首次 JIT 构建） | 服务启动、应答、越界拒绝 |
| 4 | 长上下文生产配置（`direct.sh`，200000/16/0.90，端口 8192） | 15–30 min | 服务启动，`/v1/models` 报 200000 |
| 5 | 功能验收 + MTP 回归（DSpark 重试可选） | 20–40 min | 必测项 1–5 全部通过 |
| 6 | 性能基线对比 | 10–20 min | 数据落盘，无 >10% 无解释回归 |
| 7 | （可选）仓库版本 pin 晋升 | 30–60 min | 测试套件全过 |
| R | 回退（任一 Gate 失败时） | 20–40 min | 恢复 0.27.1，MTP 服务可正常重启 |

---

## 六、详细步骤

所有命令从 Windows PowerShell 发出（脚本在 WSL 内运行；项目路径经 `/mnt/d` 映射）。`$HOME/vllm` 由 WSL 内的 bash 展开。

### 阶段 0 · 前置检查与基线采集

**前置条件**：当前 MTP 服务（0.27.1）正在运行——先采基线，**之后**才停服务。

1. 记录运行中服务的完整 argv（回退比对的基准）：

   ```powershell
   wsl pgrep -af vllm
   ```

   把输出写入 `result/vllm-0.29-upgrade-baseline.md`。

2. 对运行中的 0.27.1 MTP 服务跑基线协议（第 3 步），数据写入同一文件。

3. 基线协议（0.27.1 与 0.29.0 各跑一遍，同配置才可比；MTP 配置用 `start-api-server-mtp.bat` 的等价参数）：

   ```bash
   # WSL 内构造两个固定请求体
   python3 - <<'EOF'
   import json
   # 短请求：固定提问，128 输出 token
   short = {"model": "Qwen3.8-27B-NVFP4-VLLM",
            "messages": [{"role": "user", "content": "用三句话解释什么是 KV cache 量化。"}],
            "max_tokens": 128}
   # 长请求：约 15 万 token 的 prompt（重复句填充），对齐既有 153k 实测量级
   para = "The quick brown fox jumps over the lazy dog. " * 12500
   long = {"model": "Qwen3.8-27B-NVFP4-VLLM",
           "messages": [{"role": "user", "content": para + "\n用一句话总结上面这段文字。"}],
           "max_tokens": 64}
   open("/tmp/bench-short.json", "w").write(json.dumps(short, ensure_ascii=False))
   open("/tmp/bench-long.json", "w").write(json.dumps(long, ensure_ascii=False))
   EOF

   # 各 3 次：记录 TTFT（time_starttransfer）与总耗时
   for i in 1 2 3; do
     curl -s -o /dev/null -w "short ttft=%{time_starttransfer}s total=%{time_total}s\n" \
       -X POST http://127.0.0.1:8192/v1/chat/completions \
       -H 'Content-Type: application/json' -d @/tmp/bench-short.json
   done
   for i in 1 2 3; do
     curl -s -o /dev/null -w "long ttft=%{time_starttransfer}s total=%{time_total}s\n" \
       -X POST http://127.0.0.1:8192/v1/chat/completions \
       -H 'Content-Type: application/json' -d @/tmp/bench-long.json
   done
   ```

   0.27.1 阶段跑 3 短 + 3 长；0.29.0 阶段（MTP 配置）同样 3 短 + 3 长。跨引擎量级参照（README 4.6，SGLang DSpark 路线）：153k 输入实跑约 55 秒正常返回——vLLM 两版本之间对比只用自己的两列数据，该值不作通过/失败判据。

4. 保存 `~/vllm/env-info.txt` 与 `nvidia-smi` 快照（驱动、显存）到 baseline 文件。

5. 停止运行中服务：

   ```powershell
   wsl pkill -f 'vllm serve' ; wsl --shutdown
   ```

   **期望**：`pgrep -af vllm` 无结果；`nvidia-smi` 显存占用回落到数百 MiB 以下。`wsl --shutdown` 会重启 WSL VM（释放 WSL 侧显存），之后重新 `wsl` 进入。

**Gate 0**：baseline 文件含 argv、两组耗时、env-info；显存已释放。

### 阶段 1 · 构建 0.29.0 运行时

1. 完整重建 venv（删旧 venv，重装 CUDA 13 工具链 + vllm 0.29.0 + 13.3.x 组件钉版 + lib64 符号链接；`~/vllm/logs` 与 `env-info.txt` 不受影响）：

   ```powershell
   wsl bash -lc 'VLLM_VERSION=0.29.0 bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh create --prefix $HOME/vllm --force'
   ```

   **期望**：4 个 section 全部完成，结尾 `runtime created at /home/kami/vllm`。

2. 验证版本：

   ```powershell
   wsl bash -lc 'VLLM_VERSION=0.29.0 bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh verify --prefix $HOME/vllm'
   ```

   **期望**：全项 `[OK]`，其中 `vllm 0.29.0`、`nvcc release 13`。

3. 记录组件一致性（风险 3）：

   ```powershell
   wsl $HOME/vllm/venv/bin/pip check
   ```

   **期望**：无冲突，或冲突仅限 nvidia-cuda-{nvrtc,runtime,cupti,nvtx} 四个 13.0 vs 13.3 的元数据矛盾（记录原文，不算失败）；若出现 vllm 自身或其直接依赖的冲突，视为失败。

**停止条件**：安装/verify 失败 → 按脚本输出的 `fix:` 提示处理（多为网络）；两次重试仍失败 → 走回退 R（此时 venv 已坏，回退本身就是重建 0.27.1）。

**Gate 1**：verify 全过；pip check 结果已记录。

### 阶段 2 · 统一 preflight

```powershell
wsl bash -lc 'VLLM_VERSION=0.29.0 bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/preflight.sh --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-VLLM'
```

**期望**：`exit 0 = READY`，全部项通过（0.27.1 下为 22/22；0.29.0 项数相同，vLLM 版本检查按 `VLLM_VERSION=0.29.0` 校验）。

**注意**：不带 `VLLM_VERSION=0.29.0` 时，preflight 的 vLLM 版本检查会按默认 0.27.1 报 FAIL——这是版本 Gate 的设计行为，不是故障。

**Gate 2**：READY。

### 阶段 3 · 短上下文冒烟

```powershell
wsl bash -lc 'VLLM_VERSION=0.29.0 bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/serve.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-VLLM'
```

配置：8192 上下文 / 1 并发 / 端口 8000 / vLLM 默认 GPU 利用率。首次启动会触发 FlashInfer JIT kernel 首建（`MAX_JOBS=1` 串行），耗时数分钟属正常。

**期望（三项全过才过 Gate）**：

1. `curl -s http://127.0.0.1:8000/v1/models` 返回模型；
2. 短 chat 请求正常返回 `content`；
3. 超过 8192 token 的 prompt 被拒绝（400，引擎在入队阶段拒绝，不是 OOM）。

**日志位置**：`~/vllm/logs/`。

完成后 `Ctrl-C` 停服务。

**Gate 3**：三项全过。

### 阶段 4 · 长上下文生产配置

```powershell
wsl bash -lc 'VLLM_VERSION=0.29.0 bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/direct.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-VLLM'
```

默认即生产配置：200000 上下文 / 16 并发 / 0.90 利用率 / 端口 8192（`direct.sh` 内置默认，无需额外参数）。

**期望**：preflight + VRAM Gate 通过；服务启动；`/v1/models` 的 max model length 为 200000；短请求正常应答；`nvidia-smi` 无 OOM。

**Gate 4**：启动且应答。

### 阶段 5 · 功能验收 + MTP 回归

对阶段 4 的服务执行必测项 1–4；第 5 项需单独重启 MTP 配置服务；第 6 项可选。

1. **工具调用**：发一个带 `tools` 字段的请求（计算器类工具），确认响应含结构化 `tool_calls`（`qwen3_xml` parser）。
2. **推理字段**：发一个需要思考的请求，确认思考内容独立于正文。**记录实际字段名**（0.27.1 为 `reasoning`；0.28 起 `reasoning_content` 移除被标记为破坏性变更——若字段名变化，把结论写进 result 文档并更新 README 4.8 的客户端适配说明）。
3. **长上下文**：重放约 15 万 token 的长 prompt（阶段 0 的 `bench-long.json`），期望正常返回且无 400/OOM；耗时与阶段 6 的 0.27.1 基线对比（不引用 SGLang 路线的 55 秒值作判据）。
4. **并发**：16 个并行短请求，全部成功、无 5xx。
5. **MTP 回归（必测，当前生产配置就是 MTP）**：停掉阶段 4 服务，启动 0.29.0 MTP 配置：

   ```powershell
   wsl bash -lc 'VLLM_VERSION=0.29.0 VLLM_SPEC_METHOD=mtp FULL_MAX_MODEL_LEN=180000 bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/direct.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-VLLM'
   ```

   期望：服务日志出现 `--spec-method mtp --spec-tokens 3`；服务稳定；统计里投机接受率非零且吞吐高于无 MTP 同配置。**同时记录日志中实际使用的 model runner（V1/V2）**（风险 1）。

6. **（可选）DSpark 重试**：`VLLM_SPEC_METHOD` 换成 DSpark 的 spec method，模型目录指向 `Qwen3.8-27B-NVFP4-DSpark`（final 构建，含 DSpark 草稿）。0.27.1 上该路线在 `weight_loader` 处失败；0.29.0 的 DSpark 改进（#47808 置信度调度、#51310 Qwen3.6 dSpark 接受覆盖）可能已修复。成功 → 记录为 vLLM 原生 DSpark 恢复可用；失败 → 维持 SGLang 路线（需先 `docker pull` 镜像，约 18 GB）。**失败不阻塞其他 Gate。**

**停止条件**：必测项 1–5 任一项失败 → 收集 `~/vllm/logs/` 与服务 stdout 的完整日志存入 result 文档，然后走回退 R。

**Gate 5**：必测项 1–5 全过。

### 阶段 6 · 性能基线对比

对 0.29.0 MTP 配置（阶段 5 第 5 项的服务）跑阶段 0 的同一套协议（3 短 + 3 长），数据写入 `result/vllm-0.29-upgrade-result.md`，与 0.27.1 基线并列。

**判定**：

- TTFT / 总耗时 / 吞吐任一指标回归 >10% 且无法定位原因 → 视为回归，建议保持 0.27.1（或记录根因后再定）；
- 发布说明的性能数字（如 DSpark TTFT ~60%）针对 DSpark 路线，本项目默认 MTP，不直接适用——对比只看自己测的两列数据。

**Gate 6**：数据落盘；无 >10% 的无解释回归。

### 阶段 7 · （可选）仓库版本 pin 晋升

**触发条件**：阶段 1–6 全部通过，且决定把 0.29.0 变为默认。改动（完成后跑测试套件）：

1. `scripts/lib/wsl2-env-lib.sh`：`VLLM_VERSION="${VLLM_VERSION:-0.27.1}"` → `0.29.0`，并更新注释里 0.27.1 的 CUDA 栈描述（torch/CUTLASS 版本段落）；
2. `tests/run-tests.sh`、`tests/preflight-tests.sh`：`vllm==0.27.1` / `vllm 0.27.1` 期望串 → 0.29.0；
3. `tests/fakebin/vllm`（`FAKE_VLLM_VERSION` 默认值）与 `tests/fakebin/python3`（`*vllm*` 分支回显值）；
4. `README.md` 中的版本相关表述（含 4.8 节 `reasoning` 字段名说明——若阶段 5 第 2 项发现字段名变化，一并更新）；
5. 跑 `tests/` 下全部测试入口（`tests/run-tests.sh`、`tests/serve-tests.sh`、`tests/preflight-tests.sh`、`tests/fullcontext-tests.sh`），全绿。

晋升完成后，Windows 双击启动器（`start-api-server-vllm.bat` / `-mtp.bat`）自动按 0.29.0 运行（它们不设 `VLLM_VERSION`，脚本默认值即生效）。

**Gate 7**：测试套件全过。

---

## 七、回退计划

**触发条件（任一）**：

- 阶段 1：verify 失败，或 `pip check` 出现 vllm 及其直接依赖的冲突；
- 阶段 2：preflight 失败（VRAM 被占用等环境问题先排除再判断）；
- 阶段 3/4/5：服务起不来（JIT 首建失败、OOM、parser/spec-method 不识别、权重加载报错）；
- 阶段 5：必测功能回归且无法通过客户端适配解决；
- 阶段 6：回归 >10% 且无法定位原因。

**回退步骤（重建回 0.27.1，全部从 PowerShell 执行）**：

1. 停掉一切 0.29.0 服务：

   ```powershell
   wsl pkill -f 'vllm serve' ; wsl --shutdown
   ```

2. 按仓库默认（0.27.1）完整重建——这是与阶段 1 对称的一步：

   ```powershell
   wsl bash -lc 'bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh create --prefix $HOME/vllm --force'
   ```

   不带 `VLLM_VERSION` 覆盖时按默认 0.27.1 安装（含 13.3.x 组件钉版，即 0.27.1 时代的已知良好状态）。需重新下载数 GB，15–30 分钟。

3. 验证回到 0.27.1：

   ```powershell
   wsl $HOME/vllm/venv/bin/pip show vllm
   wsl bash -lc 'bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh verify --prefix $HOME/vllm'
   wsl bash -lc 'bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/preflight.sh --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-VLLM'
   ```

   **期望**：`vllm 0.27.1`；verify 全过；preflight READY。

4. 用原启动器恢复生产服务：双击 `scripts/start-api-server-mtp.bat`（180000/16/0.90 + MTP spec-tokens 3），确认 `/v1/chat/completions` 正常应答。

5. **若阶段 7 已执行**：先 `git revert`（或手工还原）仓库改动——`wsl2-env-lib.sh` 的 pin 回 0.27.1、测试 fixture 与 README 同步还原、重跑测试套件确认全绿——再执行上面第 1–4 步。

**回退范围说明**：回退只影响 venv（删除 + 重建）；`wsl2-env.sh` 从不触碰模型文件，模型目录、`~/vllm/logs`、Windows 侧 portproxy/防火墙规则均不变。回退完成后 MTP 服务恢复 0.27.1 时代的等价生产配置。

**回退后的记录**：在 `result/vllm-0.29-upgrade-result.md` 写清失败阶段、日志摘录、根因判断与是否保留 0.29.0 观察。

---

## 八、产出物

| 文件 | 产出阶段 | 内容 |
| --- | --- | --- |
| `result/vllm-0.29-upgrade-baseline.md` | 阶段 0 | 状态快照、0.27.1 服务 argv、两组基线耗时、env-info/nvidia-smi 快照 |
| `result/vllm-0.29-upgrade-result.md` | 阶段 1–7 执行后 | 各 Gate 结果、pip check 记录、功能验收明细（含 runner V1/V2 与 `reasoning` 字段名）、性能对比表、DSpark 重试结论、回退记录（若有） |
| 仓库改动（仅阶段 7） | 阶段 7 | `wsl2-env-lib.sh` pin、`tests/` fixture、README 版本表述 |

---

## 九、参考

- [vLLM 0.28.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)、[vLLM 0.29.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- 项目脚本：`scripts/wsl2-env.sh`（运行时构建/验证）、`scripts/preflight.sh`（统一预检）、`scripts/serve.sh`（短上下文）、`scripts/direct.sh`（生产配置直启）、`scripts/start-api-server-mtp.bat`（MTP 启动器）
- 既有记录：`README.md` 4.2/4.6/4.8 节、`result/qwen38-nvfp4-dspark-upgrade-result.md`（0.27.1 上 DSpark 失败根因）、`result/sglang-image-removal-result.md`（SGLang 兜底路线）
