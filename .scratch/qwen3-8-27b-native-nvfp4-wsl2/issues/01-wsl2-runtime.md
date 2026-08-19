# 01 — 建立固定的 WSL2 推理运行环境

**What to build:** 在 Windows 10 主机的 WSL2 Ubuntu 中建立与规格一致的独立推理环境，使用户可以在不改变操作系统的情况下准备好 CUDA 13 工具链和 vLLM 0.27.1，并避免系统级 Python、PyTorch 或 CUDA 依赖干扰运行。

**Blocked by:** None — can start immediately

**Status:** resolved

- [x] 用户可以确认 Windows 10、WSL2 Ubuntu 和 RTX 5090 的支持前提；不支持的环境会给出明确原因。
- [x] 用户可以创建独立 Python 环境并使用 CUDA 13 工具链和 vLLM 0.27.1；环境版本可被命令行验证。
- [x] 配置过程不会安装 WSL 内的 Linux NVIDIA 驱动，也不会下载、复制或转换模型文件。
- [x] 失败时输出可执行的修复建议，而不是只显示底层错误。

## Comments

### 2026-08-18 — 由 agent 实现（真实硬件验证待用户机器执行）

- 交付物：
  - `scripts/wsl2-env.sh` — 入口 CLI：`prereqs` / `create` / `verify` / `help`
  - `scripts/lib/wsl2-env-lib.sh` — 检查与输出辅助库（版本配置、ver_ge、检查计数）
  - `tests/run-tests.sh` + `tests/fakebin/` + `tests/fixtures/` — 无 WSL/GPU 环境下的 fake 测试
  - `README.md` — 用法与配置说明
- 用法（WSL2 Ubuntu 内）：`bash scripts/wsl2-env.sh prereqs|create|verify`；从 PowerShell 可 `wsl -d Ubuntu -- bash /mnt/<盘符>/<路径>/ai-model-nvfp4/scripts/wsl2-env.sh ...`
- 默认运行时目录：`~/qwen3-nvfp4-rtx5090`（WSL ext4 内，避免 drvfs 性能与符号链接问题），可用 `--prefix` 覆盖。
- 验收对照：
  - 前提确认：`prereqs` 聚合检查 WSL2 内核、Ubuntu 发行版、Windows 10 build（19041–21999）、RTX 5090、Windows 驱动 ≥ 610.88，每项失败输出 `reason` + `fix`。
  - 独立环境：`create` 用 `python3 -m venv` 建独立环境，以 pip 装入 `nvidia-cuda-toolkit-cu13` 与 `vllm[cuda-13]==0.27.1`；`verify` 从命令行验证 python / nvcc / vLLM 版本与 GPU 可见性。
  - 不装驱动、不碰模型：脚本不含任何 Linux 驱动安装命令；`--model-dir`/`--model` 被显式拒绝；测试断言输出不含模型下载行为。
  - 失败修复：所有检查与安装失败均输出可执行修复建议。
- 测试：`bash tests/run-tests.sh` 在本机（无 WSL、无 GPU）以 fake 工具全部通过（47 项断言）；真实机器上的 `prereqs`/`create`/`verify` 仍需在用户硬件上验证。
- 备注：脚本输出为英文（ASCII），避免 Windows 控制台编码问题；版本号可通过环境变量覆盖（`VLLM_VERSION`、`CUDA_TOOLKIT_PIP_SPEC`、`VLLM_PIP_SPEC` 等，见 README）。

### 2026-08-18 — 真实硬件 prereqs 验证（用户在目标机器执行）

- 环境：Windows 10.0.19045.5854 + WSL2 Ubuntu（kernel `6.18.33.2-microsoft-standard-WSL2`）+ RTX 5090（驱动 610.88，31 GiB VRAM，CUDA UMD 13.3）。
- 结果：`prereqs` **6/7 通过，1 项失败**：
  - `[FAIL] venv` — python3-venv/ensurepip 缺失；python3 为 3.14.4（≥ 3.10，通过）。
  - 修复命令：`sudo apt-get update && sudo apt-get install -y python3-venv`
- 其余全部 OK：wsl2、ubuntu、windows（build 19045，19041–21999 范围内）、gpu、gpu-model（RTX 5090 + driver 610.88）、python。
- WSL 启动警告（不影响 prereqs，供记录）：`C:\Users\Administrator\.wslconfig` 中镜像网络模式（`networkingMode=mirrored`）、DNS 隧道、Hyper-V 防火墙在 Windows 10 19045 上不受支持，已回退 NAT；另报 `.wslconfig` 中键 "10" 未知。NAT 回退不影响本议题（本地回环推理），但若想消除警告，要么删除 `.wslconfig` 中不支持的键，要么安装新版 Store WSL（`wsl --update` 或 GitHub MSI）。
- 风险提示：Ubuntu 默认 Python 3.14 对 vLLM 0.27.1 / `nvidia-cuda-toolkit-cu13` 的支持未验证，需在 `create` + `verify` 阶段确认；若 pip 装不上，考虑改用 python3.12 建 venv。
- 下一步：安装 python3-venv → 重跑 `prereqs`（应 7/7）→ `create` → `verify`。
- 跟进（同日）：安装 `python3-venv` 后重跑 `prereqs`，**7/7 全部通过**，`overall result: OK`。剩余未验证：`create`（CUDA 13 工具链 + vLLM 0.27.1 安装）与 `verify`。特别注意 Python 3.14.4 下 `vllm[cuda-13]==0.27.1` 与 `nvidia-cuda-toolkit-cu13` 能否 pip 安装成功。

### 2026-08-18 — create 失败：pip 包名修正（`nvidia-cuda-toolkit-cu13` 不存在；vLLM 0.27.1 无 `[cuda-13]` extra）

- 现象：`create` 第 3/4 步 `pip install nvidia-cuda-toolkit-cu13` 报 `No matching distribution found`，安装中止。
- 根因（PyPI 核验）：
  1. `nvidia-cuda-toolkit-cu13` 与旧名 `nvidia-cuda-toolkit-cu12` 在 PyPI 均不存在（404）。CUDA 13 官方工具链的元包是 **`cuda-toolkit`**（最新 13.3.1，universal wheel），`nvcc` extra 正好拉入 `nvidia-cuda-nvcc` + crt + runtime + nvvm 等编译链。
  2. vLLM 0.27.1 的 extras 中**没有 `cuda-13`**；其 CUDA 13 栈（`torch==2.13.0`、`nvidia-cutlass-dsl[cu13]`、`humming-kernels[cu13]` 等）已内置在 base wheel 的 `requires_dist` 中，因此 `vllm[cuda-13]==0.27.1` 同样会装不上。`requires_python` 为 `>=3.10,<3.15`，Python 3.14.4 受支持，无需降级 Python。
- 修正（已提交并全测通过）：
  - `scripts/lib/wsl2-env-lib.sh`：默认 `CUDA_TOOLKIT_PIP_SPEC=cuda-toolkit[nvcc]==13.3.1`；默认 `VLLM_PIP_SPEC=vllm==0.27.1`（仍可用环境变量覆盖）。
  - `scripts/wsl2-env.sh`：CUDA/vLLM 安装失败提示语与 verify 修复提示改为新默认串。
  - `README.md`：版本配置表同步。
  - `tests/run-tests.sh`、`tests/fakebin/fake-pip`：断言与 fake 匹配更新；`bash tests/run-tests.sh` **47/47 通过**。
- 风险：`cuda-toolkit[nvcc]==13.3.1`（CUDA 13.3.1 编译链）与 vLLM 0.27.1 的匹配性未在真机验证（vLLM 0.27.1 为未来版本假设）；如 `verify` 出现 nvcc 版本/兼容问题，可用 `CUDA_TOOLKIT_PIP_SPEC=cuda-toolkit[nvcc]`（最新 13.x）重装。
- 下一步（用户执行）：删除不完整 venv 后重跑 `create`（命令见下），再跑 `verify` 并贴回输出。
  ```powershell
  wsl -d Ubuntu -- bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh create --prefix ~/qwen3-nvfp4-rtx5090 --force
  wsl -d Ubuntu -- bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh verify
  ```

### 2026-08-19 — 真机 create 成功，verify 修正 nvcc 位置（site-packages 布局）

- 用户重跑 `create --force`（新包名）成功：venv Python 3.14.4、vLLM 0.27.1、torch 2.13.0+cu130 均安装完成。
- 但 `verify` 报 `nvcc not found in venv`：`cuda-toolkit` wheel 把 nvcc 装在 **`venv/lib/python3.14/site-packages/nvidia/cu13/bin/nvcc`**，不生成 `$venv/bin/nvcc`。
- 修正（已提交，测试 51/51 通过）：
  - `scripts/wsl2-env.sh` `verify`：nvcc 查找顺序改为先 `$venv/bin/nvcc`，再 `site-packages/nvidia/cu*/bin/nvcc`；运行时把同目录 `../lib` 加入 `LD_LIBRARY_PATH`（nvcc 需要 crt/nvvm 等兄弟库）；失败提示改为说明两个查找位置。
  - `tests/fakebin/fake-pip`：支持 `FAKE_NVCC_SITE_PACKAGES=1` 模拟真实 NVIDIA 布局；`tests/run-tests.sh` 新增 create+verify 该布局的回归用例（4 条断言）。
- 下一步（用户执行）：重跑 `verify`，nvcc 应显示 `release 13.3.x`；若报 `nvcc reports '<unknown>'` 则说明 nvcc 运行缺库，需要再把 nvcc 同目录 `lib` 的依赖一起定位。

### 2026-08-19 — 真实硬件全部验收通过，议题关闭

- 用户重跑 `verify`：**6/6 全部通过，overall result: OK**。nvcc release 13.3、vLLM 0.27.1、torch 2.13.0+cu130、GPU 可见。
- 最终环境：WSL2 Ubuntu（kernel 6.18.33.2）+ venv Python 3.14.4 + `cuda-toolkit[nvcc]`（release 13.3）+ vLLM 0.27.1 + RTX 5090（驱动 610.88，31 GiB）。
- 四项目标全部达成：前提可确认（7/7 prereqs）、独立环境可创建并命令行验证（verify 6/6）、不装 Linux 驱动且不碰模型（脚本约束 + 测试断言）、失败均有修复建议。
- 途中修正的两处默认包名问题（`nvidia-cuda-toolkit-cu13` 不存在 → `cuda-toolkit[nvcc]==13.3.1`；vLLM 0.27.1 无 `[cuda-13]` extra → `vllm==0.27.1`）与 nvcc site-packages 布局均已固化进脚本/测试（51/51 通过）。
- 建议清理：`~/qwen3-nvfp4-rtx5090/venv/bin/python -m pip cache purge`（回收约 10 GB wheel 缓存）。
- 后续：议题 02（统一运行前检查）已解除阻塞，可认领。

