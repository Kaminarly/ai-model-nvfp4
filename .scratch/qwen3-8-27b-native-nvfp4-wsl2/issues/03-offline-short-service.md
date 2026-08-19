# 03 — 以短上下文离线启动本机接口

**What to build:** 在统一运行前检查通过后，用户指定本地原始模型目录即可启动 ModelOpt NVFP4 权重的离线推理服务；服务使用 FP8 KV 缓存、固定的 vLLM 量化设置并只监听本机回环接口，现有 OpenAI 客户端无需改用其他协议即可完成模型发现和短文本推理。

**Blocked by:** 02 — 实现统一运行前检查

**Status:** resolved

- [x] 服务拒绝未通过运行前检查的启动请求，并保留清楚的失败原因。
- [x] 服务只读取用户指定目录中的原始 safetensors 权重，不执行 GGUF、AWQ、GPTQ 转换或重新量化。
- [x] 服务在离线模式下成功启动，模型列表接口返回已加载模型，短文本请求返回正常推理结果。
- [x] 服务默认仅绑定回环地址，不对局域网或互联网开放端口。
- [x] 可验证启动和短文本请求期间没有发生模型下载或访问模型托管服务的行为。

## Comments

### 2026-08-19 — 由 agent 实现（fake 测试通过；真实离线启动待用户验证）

- 交付物：
  - `scripts/serve.sh` — 入口 CLI：`start --model-dir DIR [--prefix DIR] [--host IP] [--port N] [--dry-run]` / `help`；出口 `0`=已启动（前台运行，Ctrl-C 停止）/ `1`=拒绝启动（预检失败，保留每项 reason+fix）/ `2`=用法错误
  - `scripts/lib/serve-lib.sh` — 复用 02 的 `run_preflight` 作为唯一启动闸门；固定 vLLM 启动参数（`--quantization modelopt`、`--kv-cache-dtype fp8`、`--trust-remote-code`、短上下文 `--max-model-len 8192` + `--max-num-seqs 1`）；把 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` 导出到 vLLM 进程
  - `tests/serve-tests.sh` + `tests/fakebin/vllm`（fake OpenAI 兼容 API 服务器，记录实际 argv 与离线环境变量）— 无 WSL/GPU 环境测试
  - `README.md` — 新增"离线短上下文服务（议题 03）"一节
- 验收对照：
  - 拒绝启动：预检非 READY 时输出 `service NOT started` 并保留全部失败 reason+fix（缺分片 / 模型路径无效 / 可用显存不足均有测试覆盖）。
  - 只读原始权重：`--model <dir>` 直接指向用户目录；argv 不含任何转换或再量化标志；测试断言脚本全文无 gguf/awq/gptq。
  - 离线启动：模型列表 `GET /v1/models` 返回已加载模型；`POST /v1/chat/completions` 返回短文本推理（fake 服务器实测，进程侧离线环境变量已断言）。
  - 回环绑定：默认 `127.0.0.1:8000`；非回环 `--host`（含 `SERVE_HOST=0.0.0.0`）直接拒绝并提示规格要求（exit 2）；`::1` 视为回环放行；端口校验 1–65535（防前导零八进制误判）。
  - 无下载行为：启动强制离线开关；测试对完整被引用链（serve.sh / serve-lib.sh / preflight-lib.sh / wsl2-env-lib.sh）断言不含 `snapshot_download` / `huggingface-cli` / `wget` / `curl ` / `git clone`。
- 测试：`bash tests/serve-tests.sh` **77/77 通过**；01 套件 51/51、02 套件 43/43，无回归。
- 代码评审：双轴（Spec / Standards）三轮迭代，最终两轴均无需要改代码的发现；中途修正：拒绝非回环绑定（规格 Out of Scope，改为仅提示不够）、端口范围校验、删除 `print_serve_plan` 中间层（改走共享 `run_or_dry`）、无下载断言扩展到完整 lib 链。
- 待用户真实验证（同 02 流程）：
  ```powershell
  wsl -d Ubuntu -- bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/serve.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
  ```
  预期预检 READY 后前台启动，`GET http://127.0.0.1:8000/v1/models` 返回模型，短文本请求返回正常结果；验证后议题 04（完整长上下文）解除阻塞。


### 2026-08-19 — 真机排障完成：CCCL 版本错配 + JIT 构建 OOM + 链接失败，start 9 启动成功

- 排障过程（6→9 次启动，日志存 `result/03 issue/serve.sh start 6..9.txt`）：
  1. **start 6（CCCL 头文件版本错配）**：FlashInfer JIT 编译 SM120 FP4 GEMM 内核时 CCCL `cuda_toolkit.h:41` 报 `#error "CUDA compiler and CUDA toolkit headers are incompatible"`。根因：venv `cu13` 树混装 — `nvidia-cuda-nvcc` 13.3.73 与 `nvidia-cuda-runtime` 13.0.96（`CUDART_VERSION 13000`）。CCCL 检查要求编译器与头文件主次版本一致。修复：`pip install --upgrade nvidia-cuda-runtime==13.3.29 nvidia-cuda-nvrtc==13.3.33 nvidia-cuda-cupti==13.3.75 nvidia-nvtx==13.3.29` → `CUDART_VERSION 13030` 与 nvcc 13.3 对齐。
  2. **start 7（WSL RAM OOM）**：修复后内核编译本身成功，但 ninja 默认并行编译 ~18 个内核，每个 nvcc 占数 GiB，叠加 19.18 GiB 权重常驻后超出 `.wslconfig memory=24GB` 上限，内核 OOM-killer 杀掉 `VLLM::EngineCore`（dmesg 有记录）。修复：serve-lib 导出 `MAX_JOBS=1` 串行化一次性 JIT 构建（之后缓存）。
  3. **start 8（JIT 链接失败）**：串行化后 16 个 TU 全部编译完成，但最终链接 `c++ ... -shared -L$cuda_home/lib64 -lcudart -lcuda` 失败：pip 的 `cu13` 树只在 `lib/` 提供 `libcudart.so.13`，没有 `lib64`；`libcuda.so` 只在 WSL2 的 `/usr/lib/wsl/lib`。修复：在 `cu13/lib64` 建两个符号链接（`libcudart.so` → `../lib/libcudart.so.13`，`libcuda.so` → `/usr/lib/wsl/lib/libcuda.so`）。随后还发现链接需要 `-ltvm_ffi`（`undefined symbol: TVMFFIEnvGetStream`），用 FlashInfer 的 `FLASHINFER_EXTRA_LDFLAGS` 追加 `-L<venv>/tvm_ffi/lib -ltvm_ffi`。
- 环境修复已固化：`scripts/wsl2-env.sh create` 步骤 3 现会 (a) 把上述 4 个 CUDA 包钉到 13.3.x，(b) 创建 `lib64` 符号链接（`--dry-run` 可见）。
- 启动器加固（serve-lib.sh）：导出 `MAX_JOBS=1` + `FLASHINFER_EXTRA_LDFLAGS`，启动信息里打印 `JIT build` / `JIT link` 两行，方便排障。
- 预检新增 1 项：`cuda-toolkit-version`（比较 nvcc release 与 `CUDART_VERSION`，复现 CCCL 的判定；start 9 日志可见 `[OK] nvcc 13.3 matches toolkit headers 13.3 (CUDART_VERSION 13030)`）。
- 测试：`tests/serve-tests.sh` 79→80、`tests/preflight-tests.sh` 61→67、`tests/run-tests.sh` 51，全绿。
- **真机验收（start 9，2026-08-19 13:23）**：预检 22 项 READY → `Application startup complete`、`HTTP server started`（Uvicorn on 127.0.0.1:8000）；`GET /v1/models` 返回 `Qwen3.8-27B-NVFP4-RTX5090`（max_model_len 8192）；`POST /v1/chat/completions` 短文本返回 `Hello!`（prompt 59 / completion 21 tokens，FP4 GEMM 正常运行）。
- 议题 03 真机闭环；议题 04（长上下文 262144）解除阻塞。注意：`.wslconfig memory=24GB` 下该 27B FP4 模型加载（~18.8 GiB 显存 + ~19 GiB 常驻）余量很小，长上下文前建议把 WSL 内存上限调高（如 28–30GB）并重启 WSL。
