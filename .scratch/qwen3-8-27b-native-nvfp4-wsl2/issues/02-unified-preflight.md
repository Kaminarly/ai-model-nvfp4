# 02 — 实现统一运行前检查

**What to build:** 提供唯一的运行前检查入口，在启动服务前聚合验证 WSL2 发行版、Linux 中的 RTX 5090 可见性、驱动与 CUDA 13 编译工具、vLLM 0.27.1、约 32GB 可用显存、用户指定的原始 ModelOpt NVFP4 safetensors 模型完整性，以及离线运行开关，并针对每个失败项给出原因和修复建议。

**Blocked by:** 01 — 建立固定的 WSL2 推理运行环境

**Status:** resolved

- [x] 检查通过时明确报告环境可运行，并列出已验证的版本、GPU、显存和模型文件状态。
- [x] 检查至少能识别缺少任一权重分片、必要配置文件缺失、模型路径无效、GPU 不可见、WSL2 未启用、CUDA 版本不符和 vLLM 版本不符。
- [x] 检查确认 Hugging Face 与 Transformers 离线模式已启用，并能证明检查过程不会触发模型下载。
- [x] 检查发现显存被其他程序明显占用时，报告风险和释放显存的建议。
- [x] 检查结果以单一、可供后续启动流程消费的成功或失败边界结束。

## Comments

### 2026-08-19 — 由 agent 实现（fake 测试通过；真实模型目录待用户验证）

- 交付物：
  - `scripts/preflight.sh` — 入口 CLI：`--model-dir DIR`（必填）、`--prefix DIR`（默认同 01）、`--help`；出口 `0`=READY / `1`=NOT READY / `2`=用法错误
  - `scripts/lib/preflight-lib.sh` — 复用 01 的 `run_prereq_checks`，新增模型完整性、离线模式、显存占用/余量检查
  - `tests/preflight-tests.sh` + 运行时构建的模型 fixtures — fake 环境测试
  - `README.md` — 新增"运行前检查（议题 02）"一节
- 检查序列（单一边界 `run_preflight`）：01 全部前提（wsl2/ubuntu/windows/gpu/gpu-model/python/venv）→ 运行环境（venv / nvcc release 13 / vLLM 版本）→ 离线模式 → 模型路径 → 模型文件 → 分片 index 交叉核对 → 其他进程显存占用（警告）→ 启动可用显存（默认 ≥ 20 GiB）。
- 模型契约（来自模型卡 `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` 实际文件列表，`REQUIRED_MODEL_FILES`）：`model-00001/2/3-of-00003.safetensors`、`model.safetensors.index.json`、`config.json`、`hf_quant_config.json`、`tokenizer.json`、`tokenizer_config.json`、`generation_config.json`、`chat_template.jinja`；缺任一文件 / 空文件 / index 引用缺失分片均失败并给修复建议。
- 离线模式：`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` 默认未设置时按已启用处理并 `export 1`；显式设为非 1 时报 `offline mode disabled` 并给出修复。测试断言脚本不含 `snapshot_download` / `huggingface-cli` / `wget` / `curl ` / `git clone`。
- 显存：`--query-compute-apps` 汇总其他进程占用，超过 `VRAM_OCCUPIED_WARN_MIB`（默认 4096 MiB）时警告并列出进程 + 释放建议（不阻断）；可用显存低于 `MIN_FREE_VRAM_MIB`（默认 20480 MiB）时失败。
- 测试：`bash tests/preflight-tests.sh` **43/43 通过**；01 套件在重构（`run_prereq_checks` / `check_venv_python` / `check_nvcc_venv` / `check_vllm_venv` 抽出共用，`cmd_verify` 改调共享函数）后 **51/51 通过**，无回归。
- 待用户真实验证：把模型目录放到 WSL 内后运行：
  ```powershell
  wsl -d Ubuntu -- bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/preflight.sh --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
  ```
  预期 READY 后即可进入议题 03。

### 2026-08-19 — 真实硬件验收通过，议题关闭

- 模型下载到 `D:\Qwen3.8-27B-NVFP4-RTX5090`（约 20 GB），校验：3 个 safetensors 分片大小与模型卡一致（9,972,777,720 / 9,875,839,416 / 744,532,384 字节），index 引用 3 分片（1312/1074/13 个权重，total_size 20,592,849,504 吻合），配置齐全。
- 拷入 WSL ext4 `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`（大小与源逐项一致）。
- 真实 preflight：**17/17 全部通过，`preflight result: READY`**。覆盖 wsl2/ubuntu/windows/gpu-model/python/venv/nvcc 13.3/vLLM 0.27.1/offline/model-path/model-files(10)/model-index(3 shards)/vram-others/vram（free 29945 MiB）。
- 前置：WSL 已从 C 盘迁移至 `D:\WSL`（导出备份 `D:\WSL\ubuntu-backup.tar` → unregister → import，DefaultUid=1000 恢复 kami，verify 6/6 复验通过）。
- 后续：议题 03（短上下文离线服务）已解除阻塞，可认领。

### 2026-08-19 — 五项验收在真实硬件全部通过，议题关闭

- 真实验收脚本 `tests/acceptance-02.sh`（在 WSL 内运行，临时移走文件再恢复，不下载）：**15/15 通过**。
- 第 2 项（失败识别）：真机临时移走 `model-00002-of-00003.safetensors` → `missing required file` + `index references missing shard`，rc=1；移走 `config.json` → 指名缺失，rc=1；无效路径 → `model directory not found`，rc=1；fake nvidia-smi 缺失/无设备 → `nvidia-smi reports no GPU` + 明确"不要在 WSL 内装 Linux 驱动"，rc=1。（WSL2/CUDA/vLLM 版本不符由 01 的 `run_prereq_checks` 覆盖并在 fake 测试验证。）
- 第 3 项（离线）：`HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0` → `offline mode disabled`，rc=1；脚本不含下载命令（fake 测试断言）。
- 第 4 项（显存占用）：fake 5120 MiB 占用 → WARN + 列出进程 + 释放建议，rc=0 不阻断；fake 可用显存 20319 MiB → `only 20319 MiB VRAM free` + `Close GPU-using programs`，rc=1。
- 第 5 项（单一成功/失败边界）：真机确认 NOT READY rc=1、READY rc=0、缺 `--model-dir` rc=2。
- 备注：经 wsl.exe 命令行传参时 Git Bash 会吞 `$?`/参数，故验收脚本以文件形式在 WSL 内执行，内部取退出码。
- 模型目录在验收后逐项恢复，最终重跑 preflight `preflight result: READY`（17/17），现场完好。

### 2026-08-19 — 议题 03 真机暴露两个盲区，补强 preflight（新增 2 项检查）

议题 03 真机启动两次失败，根因都不是脚本问题，但暴露了 preflight 的两个覆盖盲区，已补强：

- **盲区 1：截断分片**。用户拷贝模型时 `model-00002-of-00003.safetensors` 只拷了 8,159,838,208 / 9,875,839,416 字节，vLLM 加载到 33% 崩：`safetensors._safetensors_rust.SafetensorError: incomplete metadata, file not fully covered`。旧检查只验证文件存在且非空，抓不住"存在但截断"。补齐分片后 3/3 分片加载成功（19.18 GiB，16.97s，显存 18.77 GiB）。
- **盲区 2：系统 C 编译器缺失**。权重加载成功后在 profile 阶段崩：Triton 首次运行需编译自带 C 驱动（`RuntimeError: Failed to find C compiler`）。旧检查只验证了 venv 内 nvcc，未验证系统 cc/gcc。`sudo apt-get install -y build-essential` 可修复。
- **盲区 3（当日追加）：Python 开发头文件缺失**。装上 build-essential 后再次启动，Triton 编译 C 驱动仍在模型导入阶段崩：`fatal error: Python.h: No such file or directory`（gcc 有了，但 `/usr/include/python3.14/Python.h` 不存在）。`sudo apt-get install -y python3-dev` 可修复。`c-compiler` 检查随后升级为**真实编译探测**：用 venv python 的 include 路径实际编译一个 `#include <Python.h>` 的最小程序（与 Triton 编译 C 驱动同路径、毫秒级），cc/gcc 缺失或 Python.h 缺失都拦在启动前。
- **盲区 4（当日追加）：FlashInfer/deep-gemm 找不到 venv 内 CUDA**。python3-dev 装好后再次启动，权重加载、Triton、torch.compile 全部成功，但在 profile 阶段 FlashInfer JIT 编译采样内核时崩：`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`。根因：CUDA 13 工具链装在 venv 内（`site-packages/nvidia/cu13/bin/nvcc`），FlashInfer JIT 与 deep-gemm 在子进程里独立查 `CUDA_HOME` 或 `/usr/local/cuda`，既看不到 venv 也要求系统 CUDA。修法（不装系统 CUDA，符合设计约束）：`serve-lib.sh` 启动前把 venv 内 CUDA 树导出为 `CUDA_HOME` 并加入 PATH；preflight 新增 `cuda-home` 检查验证该树可解析（nvcc + include 存在）。
- **盲区 5（当日追加）：ninja 缺失**。CUDA_HOME 导出后 FlashInfer JIT 已进入内核构建，但崩在构建工具：`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`（`build-essential` 不带 ninja）。`sudo apt-get install -y ninja-build` 可修复；preflight 新增 `build-tools` 检查探测 ninja（`NINJA_BIN` 可覆盖，测试用），缺失提示 `ninja-build`。

新增检查（`scripts/lib/preflight-lib.sh`，测试 61/61 通过，01 套件 51/51、03 套件 78/78 无回归）：

- `shard-headers` — 读每个 safetensors 前 8 字节 LE 头部长度 + 头部 JSON `data_offsets` 最大端点，与文件实际大小比对，毫秒级抓住截断拷贝（不用哈希 10 GB）；失败信息与 vLLM 实际报错一致（`incomplete metadata, file not fully covered`）。
- `c-compiler` — 探测系统 cc/gcc（`CC_BIN` 可覆盖，测试用）并用 venv python include 路径编译 `Python.h` 最小 TU（`CC_FAIL_COMPILE=1` 可强制失败，测试用）；cc 缺失提示 `build-essential`，编译探测失败提示 `python3-dev`。
- `cuda-home` — 解析 venv 内 CUDA 树（`nvcc_in_venv` 的上一级目录），验证 `bin/nvcc` 与 `include/` 存在；配合 `serve-lib.sh` 启动时 `export CUDA_HOME` + PATH，让 FlashInfer JIT / deep-gemm 的采样内核编译能找到 nvcc 与头文件，无需系统 CUDA。
- `build-tools` — 探测 ninja（FlashInfer JIT 用 ninja 构建 CUDA 内核；`NINJA_BIN` 可覆盖，测试用）；缺失提示 `ninja-build`。

测试 fixtures 从"纯文本假分片"升级为**最小合法 safetensors 文件**（8 字节 LE 头 + JSON + 零数据），并新增截断分片（截半）、缺 C 编译器（`CC_BIN=/nonexistent/gcc`）与缺 Python.h（`CC_FAIL_COMPILE=1`）用例；`tests/fakebin/cc`、`tests/fixtures/pyinclude/Python.h` 提供测试用假编译器与头文件；fake-pip 的 site-packages 布局补建 `include/` 与 `lib/` 目录以模拟真实 `cuda-toolkit` 树。


### 2026-08-19 — 议题 03 真机第 6~8 次启动暴露三个盲区，补强 preflight / 启动器 / 环境脚本（新增 1 项检查 + 3 项启动加固）

议题 03 真机排障（start 6→9）又暴露三个环境盲区，均已修复并固化：

- **盲区 6：venv CUDA 树内版本错配（CCCL 编译期拒绝）**。start 6 崩在 FlashInfer JIT 编译 SM120 FP4 GEMM 内核：CCCL `cuda_toolkit.h:41` `#error "CUDA compiler and CUDA toolkit headers are incompatible"`。根因：`cuda-toolkit` meta 包拉出的 `cu13` 树混装 — `nvidia-cuda-nvcc` 13.3.73 与 `nvidia-cuda-runtime` 13.0.96（头文件 `CUDART_VERSION 13000`）；CCCL 要求编译器与头文件主次版本一致（`nvcc --version` release 13.3 vs `CUDART_VERSION` 13000）。修法：把 `nvidia-cuda-runtime/nvrtc/cupti/nvtx` 全部对齐到 13.3.x（`CUDART_VERSION 13030`）。
  - preflight 新增 **`cuda-toolkit-version`**：读 venv nvcc release 与同树 `cuda_runtime_api.h` 的 `CUDART_VERSION`，按 CCCL 的规则比较主次版本（支持 `cuda_runtime.h` 兜底；`FAKE_CUDA_RUNTIME_VERSION` 测试可覆盖头文件值）。失败提示对齐包：`pip install --upgrade nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cuda-cupti nvidia-nvtx`。
- **盲区 7：WSL RAM 上限导致 ninja 并行构建 OOM 杀进程**。start 7 修复版本错配后，FlashInfer JIT 内核编译本身成功，但 ninja 默认全并行（~18 个 TU，每 nvcc 数 GiB），叠加 19.18 GiB 权重常驻后超 `.wslconfig memory=24GB`，内核 OOM-killer 直接杀掉 `VLLM::EngineCore`（dmesg 可查，serve.sh 无任何报错即"静默死"）。修法：serve-lib 启动前导出 **`MAX_JOBS=1`** 串行化一次性 JIT 构建（之后缓存，不影响后续启动速度）。
- **盲区 8：FlashInfer JIT 链接阶段找不到 CUDA 运行库与 tvm_ffi**。start 8 串行化后 16 个 TU 全部编译完成，最终链接 `c++ ... -shared -L$cuda_home/lib64 -lcudart -lcuda` 失败：`cannot find -lcudart` / `cannot find -lcuda`；补 `lib64` 后又出现 `undefined symbol: TVMFFIEnvGetStream`。根因：FlashInfer JIT 硬编码 `-L$cuda_home/lib64`（pip 的 cu13 树只有 `lib/` 的 `libcudart.so.13`），`libcuda.so` 只存在于 WSL2 的 `/usr/lib/wsl/lib`，且内核还引用 tvm_ffi 符号而链接命令未带 `-ltvm_ffi`。修法：(a) 在 `cu13/lib64` 建符号链接 `libcudart.so`→`../lib/libcudart.so.13`、`libcuda.so`→`/usr/lib/wsl/lib/libcuda.so`（已固化进 `wsl2-env.sh create`，`--dry-run` 可见）；(b) serve-lib 导出 `FLASHINFER_EXTRA_LDFLAGS="-L<venv>/tvm_ffi/lib -ltvm_ffi"`（FlashInfer 官方支持的环境变量）。

固化产物与测试（`bash tests/preflight-tests.sh` 67/67、`tests/serve-tests.sh` 80/80、`tests/run-tests.sh` 51/51，全绿无回归）：

- `scripts/lib/preflight-lib.sh` — 新增 `check_cuda_toolkit_versions`（`cuda-toolkit-version` 检查）；`fake-pip` 的假头文件现在写入可控 `CUDART_VERSION`（默认 13000 与假 nvcc release 13.0 匹配）。
- `scripts/lib/serve-lib.sh` — 导出 `MAX_JOBS=1` 与 `FLASHINFER_EXTRA_LDFLAGS`，启动信息打印 `JIT build` / `JIT link` 两行。
- `scripts/wsl2-env.sh` — create 步骤 3 后追加：4 个 CUDA 包钉 13.3.x + 创建 `lib64` 符号链接。
- 测试新增：preflight 的版本错配红/绿用例（`FAKE_CUDA_RELEASE=13.3 FAKE_CUDA_RUNTIME_VERSION=13000` → NOT READY）；serve 的 `MAX_JOBS=1` / `FLASHINFER_EXTRA_LDFLAGS` 断言（后者在 fake venv 无 tvm_ffi 时按跳过处理）。
