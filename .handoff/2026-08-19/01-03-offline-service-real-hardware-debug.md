# 交接：议题 03 真机调试（第 6 次启动失败：FlashInfer CCCL 头文件版本不匹配）

## 项目背景

Windows 10 + WSL2 Ubuntu + RTX 5090（32GB，驱动 610.88）上，用 vLLM 0.27.1 离线跑 `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` 的原生 ModelOpt NVFP4 权重。议题 01（环境）02（统一预检）已 resolved；议题 03（短上下文离线服务）代码已实现、fake 测试全绿，但**真机启动仍在排障中**。议题 04（长上下文）阻塞在 03。

**重要约定**：项目**没有 git 仓库**；用户明确"不需要提交，直接更新 issue"。所有产出记录进 `.scratch/qwen3-8-27b-native-nvfp4-wsl2/issues/*.md` 的 Comments。

## 已实现（不要重复造）

- `scripts/serve.sh` + `scripts/lib/serve-lib.sh` — 离线服务启动器：预检闸门（非 READY 拒绝启动）、固定 vLLM 参数（`--quantization modelopt --kv-cache-dtype fp8 --trust-remote-code`、短上下文 8192/1）、回环绑定（非回环拒绝）、`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` 强制、启动时导出 `CUDA_HOME`=venv 内 CUDA 树 + PATH。
- preflight 新增 5 项检查（议题 02 记录里都有）：`shard-headers`（safetensors 头部覆盖，抓截断）、`c-compiler`（cc/gcc + Python.h 编译探测）、`cuda-home`（venv CUDA 树可解析）、`build-tools`（ninja）。
- 测试三套全绿：`bash tests/preflight-tests.sh` 61/61、`bash tests/serve-tests.sh` 78/78、`bash tests/run-tests.sh` 51/51。fixtures 含最小合法 safetensors、`fakebin/cc`、`fakebin/ninja`、`fixtures/pyinclude/Python.h`。

## 真机排障进度（5 个环境盲区已解决）

按时间顺序，用户每次启动把日志存到 `result/03 issue/serve.sh start N.txt`：

1. **分片截断**（start 1）：`model-00002` 只拷了 8.16/9.88 GB → 补齐 + 预检 `shard-headers`。
2. **缺 gcc**（start 2）：Triton `Failed to find C compiler` → `apt install build-essential` + 预检。
3. **缺 Python.h**（start 3）：`fatal error: Python.h` → `apt install python3-dev` + 预检编译探测。
4. **缺 CUDA_HOME**（start 4）：FlashInfer `Could not find nvcc ... /usr/local/cuda doesn't exist` → serve-lib 导出 venv CUDA_HOME + 预检 `cuda-home`。
5. **缺 ninja**（start 5）：`FileNotFoundError: 'ninja'` → `apt install ninja-build` + 预检 `build-tools`。

真机当前依赖：venv `/home/kami/qwen3-nvfp4-rtx5090`（Python 3.14.4、nvcc 13.3、vLLM 0.27.1）、模型 `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`（3 分片已字节校验完整）。启动命令：
```bash
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/serve.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
```

## 当前阻塞（第 6 次启动，start 6.txt）

预检 21 项全绿（`build-tools ninja: /usr/bin/ninja` OK，READY）。权重加载、Triton、torch.compile 全部成功。**崩在 FlashInfer JIT 用 ninja 编译 SM120 FP4 GEMM 内核**（`fp4_gemm_cutlass_sm120`，位于 `~/.cache/flashinfer/0.6.16.post3/120f/cached_ops/`）：

```
cuda_toolkit.h:41:8: error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
...
RuntimeError: Ninja build failed.
```

每个 `.cuda.o` 目标都报同一行（`serve.sh start 6.txt` 约 887–1110 行）。这是 CCCL（`flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h`）的版本一致性检查：**nvcc 编译器版本 vs 构建时可见的 CUDA 运行时头文件（cuda_runtime.h 的 CUDART_VERSION）不匹配**。此前假设 venv 内 nvcc 13.3 与 `cuda-toolkit[nvcc]` 附带头文件版本一致，现在看并不一致（或 CCCL 支持上限低于 13.3）。

### 下一步调试建议（按顺序试）

1. **读 start 6.txt 里完整的 ninja 构建命令**（`.cuda.o` 目标的 nvcc 命令行）：确认 `-I` include 路径顺序、nvcc 用的是哪个、cuda_runtime.h 从哪来。
2. **核对 venv CUDA 版本一致性**（真机执行）：
   ```bash
   ~/qwen3-nvfp4-rtx5090/venv/bin/nvcc --version   # 13.3.1？
   grep -E "define (CUDART_VERSION|CUDA_VERSION)" ~/qwen3-nvfp4-rtx5090/venv/lib/python3.14/site-packages/nvidia/cu13/include/cuda_runtime.h
   ls ~/qwen3-nvfp4-rtx5090/venv/lib/python3.14/site-packages/nvidia/cu13/include/ | head
   ```
   若 include/cuda_runtime.h 版本 ≠ nvcc 版本（如 13.0 vs 13.3）→ 版本错配，装齐同一版本：`pip install 'nvidia-cuda-runtime-cu13==13.3.*'` 等，或 `create --force`。
3. **查 flashinfer 0.6.16.post3 捆绑 CCCL 支持的 CUDA 上限**：`grep -rn "CUDA_TOOLKIT" venv/lib/python3.14/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h`，看它期望/上限版本。若 13.3 超出 CCCL 支持 → 方案：升/降 nvcc 工具链到 CCCL 支持区间，或换 flashinfer 版本。
4. **备选绕过**：vLLM 的 `flashinfer_mm_fp4` 选了 'cutlass' 后端做 NVFP4 GEMM；可查 vLLM 0.27.1 是否支持禁用 flashinfer 该路径（如 `VLLM_USE_FLASHINFER=0` 或换 attention/gemm backend），但注意这是 NVFP4 权重的核心 GEMM，绕过可能影响性能或直接失败，优先修版本。
5. 修好后让用户重跑 `serve.sh start`，把日志存 `result/03 issue/serve.sh start 7.txt` 再贴回。

## 完成后收尾（务必做）

- 真机启动成功（看到 `Uvicorn running on http://127.0.0.1:8000`）后：让用户 `curl http://127.0.0.1:8000/v1/models` 与短文本 chat 请求验证，把结果更新进议题 03 Comments（交付记录补一段"真机排障：5 个盲区 + 第 6 次 CCCL 错配"），确认 03 真正闭环，解除议题 04 阻塞。
- 若第 6 个问题也属环境类并已加预检，同步更新议题 02 Comments 与 README 聚合检查列表。
- 测试保持全绿（61/61、78/78、51/51）后再更新 issue。

## Suggested skills

- `/diagnosing-bugs` — 当前任务：CCCL 版本错配的根因定位（按上面调试建议走）。
- `/implement`（若确定修复并要加预检/改 serve-lib）+ `/code-review` 走一遍。
- 收尾更新议题时参考 `/implement` 的 issue 更新规范（`docs/agents/issue-tracker.md`）。
