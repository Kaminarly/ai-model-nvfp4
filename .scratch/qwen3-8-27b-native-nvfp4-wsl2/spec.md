# Qwen3.8-27B 原生 NVFP4 本地运行规格

## Problem Statement

用户在 Windows 10 电脑上拥有一张 RTX 5090 32GB 显卡，希望运行 `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` 的原生 NVFP4 权重，而不是 GGUF、转换版或重新量化版。现有方案必须避免未经用户明确授权下载模型，并能在启动前判断环境是否满足运行条件。

## Solution

以 Windows 10 为宿主系统，通过 WSL2 运行 Ubuntu、CUDA 13 工具链与 vLLM 0.27.1。该路径直接读取模型发布者提供的 ModelOpt NVFP4 safetensors 权重，使用 FP8 KV 缓存，不执行 GGUF 转换、再量化或自动模型下载。

在真正启动服务前，运行一个统一的运行前检查。检查通过后，用户指定本地原始模型目录，以离线模式启动 OpenAI 兼容接口。先以较短上下文验证服务可用，再按需启用 262,144 token 上下文。

## User Stories

1. 作为 Windows 10 用户，我希望不更换操作系统就能运行原生 NVFP4 模型，以便继续使用现有电脑环境。
2. 作为 RTX 5090 用户，我希望运行路径能使用显卡的 Blackwell FP4 能力，以便获得该模型预期的速度和显存效率。
3. 作为模型使用者，我希望加载发布者的原始 safetensors 权重，以便避免 GGUF 转换带来的格式变化。
4. 作为模型使用者，我希望系统绝不自动下载模型，以便完全掌控网络流量、存储位置和模型版本。
5. 作为模型使用者，我希望启动前能确认本地模型文件完整，以便避免因缺少分片或配置文件导致启动失败。
6. 作为模型使用者，我希望能确认 WSL2 已正确启用，以便避免在不受支持的原生 Windows 环境中安装 vLLM。
7. 作为模型使用者，我希望能确认 Linux 环境可识别 RTX 5090，以便确保推理实际使用 GPU。
8. 作为模型使用者，我希望能确认 CUDA 13 编译工具可用，以便首次启动时可完成所需的 Blackwell FP4 内核编译。
9. 作为模型使用者，我希望使用与模型卡一致的 vLLM 版本和量化设置，以便降低兼容性风险。
10. 作为模型使用者，我希望先以较短上下文启动服务，以便在不占满显存的情况下验证环境。
11. 作为模型使用者，我希望在验证成功后启用原生 262,144 token 上下文，以便使用该模型针对 RTX 5090 优化的长上下文能力。
12. 作为模型使用者，我希望服务只监听本机接口，以便默认不向局域网暴露模型服务。
13. 作为模型使用者，我希望获得一个兼容 OpenAI 接口的本地服务，以便现有客户端可以直接接入。
14. 作为模型使用者，我希望启动时报告显存不足或版本不匹配的明确原因，以便能够针对性修复环境。
15. 作为模型使用者，我希望在全长上下文启动前获知显存接近上限，以便主动关闭占用 GPU 的程序。
16. 作为模型使用者，我希望能在不下载模型的前提下验证整个运行环境，以便将下载行为与环境配置行为分开管理。

## Implementation Decisions

- 运行平台固定为 Windows 10 宿主系统上的 WSL2 Ubuntu，不采用原生 Windows vLLM。vLLM 官方文档将 Linux 列为支持的操作系统，并明确说明 Windows 应通过 WSL 使用。
- 推理引擎固定为 vLLM 0.27.1，CUDA 变体固定为 CUDA 13。模型卡以该组合在 RTX 5090 上验证过 NVFP4 推理。
- 权重格式固定为 ModelOpt NVFP4 safetensors，量化方式为 `modelopt`，KV 缓存格式为 FP8。不得转换为 GGUF、AWQ、GPTQ 或其他格式。
- 运行环境使用独立 Python 环境，避免系统中已有 Python、PyTorch 或 CUDA 依赖影响 vLLM 的二进制兼容性。
- CUDA 13 工具链安装在 WSL Linux 环境中；不得在 WSL 内安装 Linux NVIDIA 驱动。GPU 驱动由 Windows 宿主系统提供。
- 本地模型由用户自行放置并指定目录。启动服务时必须启用 Hugging Face 与 Transformers 离线模式，确保运行程序不会访问模型托管服务。
- 运行前检查是唯一的验证边界，聚合检查 Windows 版本、WSL2 发行版、GPU 可见性、驱动版本、CUDA 编译器、vLLM 版本、可用显存和模型完整性。
- 初次可用性验证使用较短上下文、单请求配置。只有在可用性验证通过后，才启用 262,144 token、16 并发序列与 0.97 显存利用率的完整配置。
- 完整上下文启动前必须检查没有其他显著占用 GPU 显存的程序。32GB RTX 5090 在完整配置下显存余量很小。
- 服务默认仅绑定回环地址。开放局域网或互联网访问需要另行设计认证、访问控制和网络边界。

## Testing Decisions

- 测试只验证用户可感知的外部行为：运行前检查是否清楚报告可运行或不可运行，服务是否可以离线启动，接口是否能返回模型列表和推理结果。
- 运行前检查必须至少覆盖：WSL2 是否存在、Linux 是否识别 RTX 5090、CUDA 版本是否为 13、vLLM 版本是否为 0.27.1、显存是否为约 32GB、模型目录是否包含全部三个权重分片及必要配置文件、离线开关是否启用。
- 冒烟测试以短文本请求为准，预期服务返回正常响应，且不触发任何模型下载。
- 长上下文测试在短上下文冒烟测试通过后进行，逐步提高上下文长度，直到验证完整上下文配置或获得明确的显存边界。
- 失败测试必须模拟缺少模型分片、WSL2 未安装、CUDA 编译器版本不符、GPU 不可见、显存被其他程序占用和离线模型路径无效等情况。
- 当前项目没有可复用的测试模块或既有测试范例；运行前检查将作为最高层、唯一的验证入口。

## Out of Scope

- 下载、复制、缓存、删除或转换模型文件。
- 训练、微调、重新量化、基准测试或质量评估模型。
- 使用 GGUF 或任何其他替代格式。
- 原生 Windows vLLM 支持、社区 Windows 分支、Docker 部署和多 GPU 部署。
- 对外网或局域网公开服务、身份认证、访问控制和生产监控。
- 启用可选的 DSpark 草稿模型，因为它需要额外模型文件并会改变显存与上下文取舍。

## Further Notes

- 已确认宿主机为 Windows 10 专业版 19045、RTX 5090 32GB、驱动 610.88、CUDA 12.8。Windows 侧 CUDA 12.8 不作为 WSL 中 vLLM 的编译环境；WSL 应单独提供 CUDA 13 工具链。
- 当前项目目录没有模型、脚本或 Git 远程仓库，因此本规格无法发布到远程议题系统，改为保存在本地项目中。
- 模型发布者说明其完整 262,144 token 上下文配置需要 0.97 显存利用率，并已在单张 RTX 5090 32GB 上验证。实际可用性仍受桌面程序占用的显存影响。
- 该规格的唯一待确认项是运行前检查的输出形式。建议采用单一命令行检查入口，输出可执行、不可执行及对应修复建议。

## 实施总结（议题 01–04，全部 resolved 且真机闭环）

四个议题按 01→04 顺序实施，全部在用户真机（Windows 10 19045 + WSL2 Ubuntu kernel 6.18.33.2 + RTX 5090 32GB 驱动 610.88）验证通过。脚本位于 `scripts/`，测试位于 `tests/`（四套全部 green：run 51 / preflight 67 / serve 81 / fullcontext 113）。

### 议题 01 — 建立固定 WSL2 推理运行环境（`scripts/wsl2-env.sh`）

- 交付：`prereqs`（前提体检，失败给 reason+fix）/ `create`（建 venv + CUDA 13.3 工具链 + vLLM 0.27.1）/ `verify`（命令行版本核验）。
- 真机结果：prereqs 7/7、verify 6/6 通过。最终环境：venv Python 3.14.4 + `cuda-toolkit[nvcc]==13.3.1` + vLLM 0.27.1 + torch 2.13.0+cu130。
- 关键修正（均已固化）：
  1. PyPI 上 `nvidia-cuda-toolkit-cu13` 不存在 → 改用官方元包 `cuda-toolkit[nvcc]==13.3.1`；vLLM 0.27.1 无 `[cuda-13]` extra → 改 `vllm==0.27.1`（CUDA 13 栈内置在 base wheel）。
  2. nvcc 实际装在 `site-packages/nvidia/cu13/bin/nvcc`，不在 `$venv/bin` → verify 增加 site-packages 查找 + `LD_LIBRARY_PATH` 补兄弟库。
  3. `.wslconfig` 镜像网络等键在 Windows 10 上不受支持（回退 NAT，不影响回环推理）——仅记录，未处理。

### 议题 02 — 统一运行前检查（`scripts/preflight.sh`，22 项聚合）

- 交付：单一边界 `run_preflight`，聚合 01 前提 + 运行环境 + 离线开关 + 模型完整性 + 显存余量；出口 0=READY / 1=NOT READY / 2=用法错误。
- 真机结果：preflight 17/17→22/22 READY；`tests/acceptance-02.sh` 15/15（临时移走文件复现失败路径）。
- 真机暴露的盲区与补强（全部固化进 preflight + 启动器 + 环境脚本）：
  1. **截断分片**（拷贝不完整，vLLM 加载 33% 崩溃）→ `shard-headers` 检查（读 safetensors 头部 data_offsets 与文件大小比对，秒级抓截断）。
  2. **系统 C 编译器缺失**（Triton 编译崩溃）→ `c-compiler` 检查，升级为真实编译探测（`#include <Python.h>` 最小 TU）。
  3. **Python 开发头文件缺失**（`Python.h`）→ 由 `c-compiler` 编译探测覆盖。
  4. **FlashInfer/deep-gemm 找不到 venv 内 CUDA** → 启动器导出 `CUDA_HOME` + PATH；preflight 新增 `cuda-home` 检查。
  5. **ninja 缺失**（FlashInfer JIT 构建工具）→ `build-tools` 检查。
  6. **venv CUDA 树版本错配**（nvcc 13.3 vs 头文件 13000，CCCL 编译期拒绝）→ 钉 4 个 CUDA 包到 13.3.x；preflight 新增 `cuda-toolkit-version` 检查（按 CCCL 规则比较 nvcc release 与 CUDART_VERSION）。
  7. **WSL RAM 上限 OOM 杀进程**（ninja 全并行编译叠加 19 GiB 权重）→ 启动器导出 `MAX_JOBS=1` 串行化一次性 JIT 构建。
  8. **FlashInfer JIT 链接失败**（pip cu13 树无 `lib64`、`libcuda.so` 只在 WSL 侧、缺 tvm_ffi）→ `create` 建 `lib64` 符号链接；启动器导出 `FLASHINFER_EXTRA_LDFLAGS="-L<venv>/tvm_ffi/lib -ltvm_ffi"`。

### 议题 03 — 离线短上下文服务（`scripts/serve.sh`）

- 交付：预检 READY 后前台启动 OpenAI 兼容服务；固定 `--quantization modelopt`、`--kv-cache-dtype fp8`、`--trust-remote-code`；默认回环绑定 `127.0.0.1:8000`；短上下文 8192/1；强制离线。
- 真机结果（start 9）：预检 22 项 READY → `/v1/models` 返回模型（max_model_len 8192）→ 短文本推理正常（FP4 GEMM 运行）。非回环绑定被拒绝（exit 2）。
- 关键点：服务只读本地原始 safetensors；脚本经测试断言不含任何下载/转换命令；回环绑定与离线环境变量均有 fake 测试覆盖。

### 议题 04 — 完整长上下文验证（`scripts/fullcontext.sh`）

- 交付：受控验证序列——预检闸门 → full-config 显存门 → 短上下文冒烟（含超长 4xx 拒绝探测）→ `CONTEXT_LADDER` 逐级核对 `/v1/models` → 完整配置（含真实并发探测、长请求、超界探测、显存采样）前台保持运行；失败报告实际边界 + 较低配置建议。
- 真机结果（两次全流程均 exit 0，`verified and enabled`）：
  - 首跑 0.97 被显存门拦截（free ~30600 < 31629），脚本如实报告余量/原因/建议——验收 3 预期行为。
  - 0.93 全流程通过（131072 梯度 + 16 并发 + 长请求 200 + 超界 400 + VRAM 94%）。
  - **定案配置 0.90 全流程通过**：`FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=131072 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER="131072"`（vLLM 官方默认 0.90，KV 不是瓶颈，启动门槛最低，开 VS Code 也稳）。
- 关键发现：
  1. **本机实际边界 = 131072 token（128k）**；规格目标 262144 需释放显存（`wsl --shutdown`）后按默认梯级重跑才有机会，未在本机验证。
  2. Qwen3.8-27B（config 内部 `model_type: qwen3_5`）是混合注意力架构（64 层仅 15 层标准注意力，其余 linear_attn），实际 KV 占用远小于 vLLM 按全层注意力的悲观估算（0.90 下 KV 池 6.93 GiB / 216,946 token，`Maximum concurrency ... 1.66x` 为悲观值）。
  3. 显存门失败时的"按当前余量动态计算建议配置"增强已提出，用户拍板维持写死档位，不实施。

