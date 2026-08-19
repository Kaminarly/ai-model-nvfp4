# Qwen3.8-27B 本地推理服务（Windows + WSL2 + RTX 5090）

> 在你自己电脑上，离线运行一个 27B 参数的大语言模型，并提供一套和 OpenAI 一模一样的 API。任何会调用 OpenAI 接口的程序都能直接接上，全程不联网、不上传数据。

本项目的四个开发议题（环境搭建 / 运行前检查 / 短上下文服务 / 完整长上下文验证）均已完成并在你的真机上验证通过。设计细节与实施总结见 [`spec.md`](./.scratch/qwen3-8-27b-native-nvfp4-wsl2/spec.md)。

---

## 1. 这是什么项目

一句话：**把一个大模型跑在你自己的显卡上，并给它一个标准的 API 服务外壳。**

- **模型**：Qwen3.8-27B（约 270 亿参数），使用发布者提供的**原生 NVFP4 量化权重**。NVFP4 是 RTX 50 系（Blackwell）显卡的原生格式，速度和显存占用最优；我们直接加载原版 safetensors 文件，**不做任何格式转换、不重新量化**。
- **引擎**：vLLM 0.27.1，业界主流的推理服务框架，提供 OpenAI 兼容接口。
- **上下文**：单个对话最多 **131072 个 token（128k）**——这是你这台机器实测验证过的边界。
- **安全**：服务**只监听本机**（127.0.0.1），不向局域网或互联网开放端口；启动强制离线模式，绝不联网下载模型或上传你的数据。

### 你不需要做的事

- 不需要把模型转换成 GGUF / AWQ / GPTQ 等格式；
- 不需要手动安装显卡驱动之外的任何系统级软件（脚本全自动装）；
- 不需要联网使用模型——模型文件就在你自己的硬盘上，脚本也禁止联网。

## 2. 用到了哪些技术 / 装了什么环境

| 组件 | 版本 / 说明 |
| --- | --- |
| Windows 10 | 宿主系统（build 19045，已实测） |
| WSL2 + Ubuntu | Windows 内运行的 Linux 子系统，模型服务跑在里面 |
| RTX 5090 32GB | 显卡，Blackwell 架构；驱动 610.88（已实测） |
| CUDA 13.3 | GPU 编译工具链，装在独立环境里，**不污染系统** |
| vLLM 0.27.1 | 推理引擎，服务本体 |
| Python 3.14 | 独立虚拟环境（venv），与系统 Python 隔离 |
| ModelOpt NVFP4 | 模型量化格式；KV 缓存用 FP8 |
| OpenAI 兼容 API | 服务对外的接口格式 |

所有组件都**故意钉死版本**：这是模型发布者验证过的组合。想升级有专门流程（见第 6 节），不建议随意改动。

---

## 3. 第一次使用：装环境（只做一次）

> 前提：你已经把模型目录放到了 WSL 里（本项目路径为 `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`，约 20 GB，包含 3 个权重分片 + 索引 + 配置文件）。

打开一个 **WSL 终端**（Windows 上运行 `wsl` 命令，或在 PowerShell 里用 `wsl -d Ubuntu -- ...`），依次执行三步：

```bash
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh prereqs   # 1. 体检：检查系统/显卡/驱动是否满足
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh create    # 2. 安装：建独立环境 + CUDA 工具链 + vLLM
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh verify    # 3. 验收：确认全部装好、版本正确
```

| 子命令 | 干什么 | 什么时候用 |
| --- | --- | --- |
| `prereqs` | 检查 Windows 版本、WSL2、GPU、驱动等前提，不满足会告诉你**缺什么、怎么修** | 第一次 / 环境怀疑有问题时 |
| `create` | 创建独立 Python 环境，安装 CUDA 13.3 工具链和 vLLM 0.27.1 | 第一次（约需 10+ 分钟，视网速） |
| `verify` | 从命令行验证 python / nvcc / vLLM 版本和 GPU 可见性 | 装完、或升级后 |

> PowerShell 里的写法：`wsl -d Ubuntu -- bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh prereqs`（后面两步同理）。

三个脚本的更多参数（`--prefix` 运行时目录、`--force` 重建、`--dry-run` 只打印不执行）见各脚本 `help`。

---

## 4. 启动服务

### 4.1 两条启动路径

| 路径 | 脚本 | 上下文 | 适用 |
| --- | --- | --- | --- |
| 短上下文服务 | `serve.sh` | 8192 token | 快速验证环境、占用显存小 |
| 完整长上下文服务 | `fullcontext.sh` | **131072 token（128k）** | 日常使用，推荐 |

完整路径会自动做一遍"体检 → 冒烟 → 逐级提升上下文 → 启动完整配置"，每一步都验证生效，任何一步失败都会**明确告诉你为什么、怎么修**，不会静默退出。

### 4.2 推荐启动命令（128k 完整服务）

在 WSL 终端里执行（**本机实测验证过的配置**，开着 VS Code 也稳）：

```bash
FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=131072 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER="131072" \
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/fullcontext.sh start \
  --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
```

看到 **`full-context configuration verified and enabled`** 和 **`full-context service is running: http://127.0.0.1:8000/v1`** 就说明成功了。服务在前台运行，**按 `Ctrl-C` 停止**。

> PowerShell 完整写法（想不带参数用默认值的话，把开头的环境变量整段去掉即可）：
> ```powershell
> wsl -d Ubuntu -- bash -lc "FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=131072 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER='131072' bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/fullcontext.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
> ```

### 4.3 参数是什么意思、有什么效果

**命令行选项（`fullcontext.sh start` / `serve.sh start` 通用）**

| 参数 | 含义 | 默认值 | 说明 / 效果 |
| --- | --- | --- | --- |
| `--model-dir DIR` | 模型目录 | 无（必填） | 指向本地原始 NVFP4 权重文件夹；脚本只读它，不下载、不改动 |
| `--prefix DIR` | 运行时目录 | `~/qwen3-nvfp4-rtx5090` | venv 和日志放哪；一般不用改 |
| `--host IP` | 监听地址 | `127.0.0.1` | 默认只允许本机访问；绑定其它地址会被拒绝（安全设计） |
| `--port N` | 端口 | `8000` | 8000 被占用时改这里 |
| `--dry-run` | 只打印计划 | 关 | 不真正启动，先看脚本会执行哪些命令 |

**环境变量（放在命令前，空格隔开；不写就用默认值）**

| 变量 | 默认值 | 含义 / 效果 |
| --- | --- | --- |
| `FULL_GPU_MEM_UTIL` | `0.97` | 显存利用率：告诉 vLLM 最多占用显卡显存的多少。**越大 KV 缓存越大（上下文潜力越高），但启动时要求空闲显存越多，越容易因桌面程序占用而失败**。本机实测 **0.90** 最稳（推荐） |
| `FULL_MAX_MODEL_LEN` | `262144` | 目标上下文长度（token）。本机实测 **131072**（128k）能过，262144 需要释放显存后才有机会 |
| `FULL_MAX_NUM_SEQS` | `16` | 最大并发序列数（同时处理几个请求）。KV 缓存够大时 16 没问题 |
| `CONTEXT_LADDER` | `32768 65536 131072 262144` | 逐级验证的梯度：从短到长一步步试，试到哪一步失败，就如实报告"实际边界 = 上一级"。**建议写成 `"131072"` 单步直达，省 3 次启动** |
| `SHORT_MAX_MODEL_LEN` | `8192` | （serve.sh）短服务上下文 |
| `SHORT_MAX_NUM_SEQS` | `1` | （serve.sh）短服务并发 |

> 注意 `CONTEXT_LADDER` 要写成 `"131072"` 这样带引号；如果传空串会被当成"没设置"而用回默认四阶梯度。

### 4.4 启动时脚本在做什么（了解即可）

1. **运行前检查**（22 项体检）：环境、GPU、显存、模型文件完整性、离线开关——不过关直接拒绝并告诉你修法；
2. **显存门**：检查当前空闲显存够不够你设定的 `FULL_GPU_MEM_UTIL`，不够就给出可行的降低配置建议；
3. **短上下文冒烟**：先用 8192 token 快速启动验证一遍（包括"超长请求必须被拒绝"）；
4. **逐级提升**：按 `CONTEXT_LADDER` 逐级启动并核对 `/v1/models` 上报的上下文长度；
5. **启动完整配置并保持运行**：最后这次启动做并发验证、长请求验证、边界验证，然后一直跑到你按 Ctrl-C。

每次启动的 vLLM 日志存在 `~/qwen3-nvfp4-rtx5090/logs/`（`fullcontext-smoke.log`、`fullcontext-ramp-<长度>.log`），失败排查用得上。

---

## 5. 启动成功后：在 Windows 上调用模型服务

服务跑在 WSL 里，但**绑定了 127.0.0.1**，WSL2 默认会把 localhost 转发给 Windows，所以你在 **Windows 一侧直接用 `http://127.0.0.1:8000` 访问**即可。

### 5.1 先确认服务活着（Windows PowerShell）

```powershell
curl http://127.0.0.1:8000/v1/models
```

应返回类似（`max_model_len: 131072` 就是 128k 上下文）：

```json
{"object":"list","data":[{"id":"Qwen3.8-27B-NVFP4-RTX5090","max_model_len":131072,...}]}
```

### 5.2 发一条聊天消息（curl）

```powershell
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{\"model\":\"Qwen3.8-27B-NVFP4-RTX5090\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，介绍一下你自己\"}],\"max_tokens\":64}'
```

> `model` 填的是服务显示的模型名（默认 = 模型目录的文件夹名）。

### 5.3 用 Python 调用（推荐，写代码时用这个）

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen3.8-27B-NVFP4-RTX5090",
    messages=[{"role": "user", "content": "用一句话解释什么是 KV 缓存"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

因为接口是 OpenAI 兼容的，所以任何现成的 OpenAI 客户端、框架、工具，把 `base_url` 指到 `http://127.0.0.1:8000/v1` 就能直接用。

### 5.4 关于上下文长度的提醒

- 单个对话的 131072 token 是**总共**的预算：系统提示 + 历史消息 + 当前问题 + 模型回答，全部加起来。
- 一次请求的 token 数 = 输入（你的问题 + 历史） + `max_tokens`（你允许模型输出的长度）。
- 超过 131072 的请求会被拒绝（HTTP 400，错误信息类似 `maximum context length`），这是正常保护，不是故障。

---

## 6. 注意事项

1. **显存是最大的约束**：开服务前尽量关掉占显存的程序（浏览器、视频、其它 AI 工具）。启动时显存不够，脚本会明确告诉你"还差多少、怎么释放、或改用哪个更低配置"。
2. **内存（RAM）也会被占用**：WSL2 有内存上限，模型文件映射 + 进程常驻可达几十 GB，上限设太小会被系统杀掉（OOM）。这是正常现象，不是内存泄漏。详见第 7 节问题 11。
3. **第一次启动特别慢**：vLLM 首次运行要为你的显卡编译内核（FlashInfer JIT，约 5–15 分钟，取决于 CPU）。之后启动就快了，编译结果有缓存。**期间不要关窗口**。
4. **保持配置一致**：本机验证过的组合是 `0.90 / 131072 / 16`。调大 `FULL_GPU_MEM_UTIL` 或上下文到 262144 前，确认显存余量（参考第 7 节问题 1）。
5. **服务只在本机**：默认不开放局域网。这是安全设计，想对外开放需要额外设计，本项目的规格里明确不做。
6. **永远离线**：脚本强制 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，只读你指定的本地模型目录。如果你的目录里没有模型，报错会提示你，而不是偷偷去下载。
7. **改版本要谨慎**：环境版本全部钉死（vLLM 0.27.1 + CUDA 13 + Python 3.14），这是模型卡验证过的组合。升级走 `wsl2-env.sh create --force`（会用环境变量指定的新版本重建），不建议手动 `pip install` 乱改。
8. **测试**：项目自带四套自动化测试（`bash tests/run-tests.sh`、`tests/preflight-tests.sh`、`tests/serve-tests.sh`、`tests/fullcontext-tests.sh`），用假工具模拟环境，可在没有 WSL/GPU 的机器上跑，改代码后跑一遍防回归。

---

## 7. 常见问题与解决方法

**Q1：启动报 `full-config VRAM gate`，提示显存不够，怎么办？**
原因：桌面程序占用了显存，你设定的 `FULL_GPU_MEM_UTIL × 显存总量` 大于当前空闲显存。
解决（按顺序试）：
1. 关掉 Windows 上占 GPU 的程序（浏览器、视频、其它模型），然后执行 `wsl --shutdown` 再重开 WSL（WSL 会缓存显存，必须重启才真正释放）；
2. 或降低配置：`FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=131072 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER="131072"`；
3. 或只跑短服务 `serve.sh start`（8192 token，占用小）。

**Q2：`prereqs` 报 `venv` 失败 / `python3-venv` 缺失？**
```bash
sudo apt-get update && sudo apt-get install -y python3-venv
```
然后重跑 `prereqs`。

**Q3：启动时报"找不到 C 编译器" / `Python.h: No such file or directory` / `ninja` 不存在？**
系统缺编译内核的组件，逐条安装后重试：
```bash
sudo apt-get install -y build-essential python3-dev ninja-build
```

**Q4：加载模型时崩溃，报 `incomplete metadata, file not fully covered`（截断分片）？**
你的模型文件拷了一半。删除后重新完整拷贝模型目录，重跑 `preflight.sh --model-dir ...`（有 22 项检查，其中 `shard-headers` 就是专门抓截断拷贝的）。

**Q5：启动报 `CUDA compiler and CUDA toolkit headers are incompatible`（CCCL 版本错配）？**
这是 CUDA 包版本没对齐。本项目已经把正确的包钉死（`nvidia-cuda-runtime` 等对齐 13.3.x），正常不需要处理。如果出现，说明环境被动过，重跑一次环境修复命令：
```bash
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh create --prefix ~/qwen3-nvfp4-rtx5090 --force
```

**Q6：WSL 内存不足，启动过程中进程被系统杀掉（`OOM`）？**
权重加载约 19 GB 常驻 + 首次内核编译吃内存。脚本已强制 `MAX_JOBS=1` 串行编译来降低峰值；如果还是 OOM，把 `.wslconfig` 里的 `memory` 调大（如 28–30GB）后 `wsl --shutdown` 重启。

**Q7：Windows 上访问 `http://127.0.0.1:8000` 不通？**
先确认 WSL 里服务真的在跑（第 4.2 节的输出）。若在跑仍不通：WSL2 的 localhost 转发偶尔会被镜像网络等配置干扰，检查 `C:\Users\<你>\\.wslconfig` 里是否设置了不受支持的键（Windows 10 上 `networkingMode=mirrored` 等不受支持，删掉后 `wsl --shutdown` 重启）。仍不行就在 WSL 内访问（`curl http://127.0.0.1:8000/v1/models`）。

**Q8：请求返回 HTTP 400 / `maximum context length`？**
你的输入超过了上下文上限（131072 token）。这是预期行为——模型有硬边界，超了就拒绝。缩短输入、或清掉部分历史消息即可。

**Q9：端口 8000 被占用？**
换个端口：加 `--port 8001`（Windows 侧访问地址相应变成 `http://127.0.0.1:8001`）。

**Q10：为什么我设了 `FULL_MAX_MODEL_LEN=262144` 却启动失败？**
262144 是规格目标，但需要更大的 KV 缓存，也就是更高的显存占用。你这台机器在桌面程序占用显存的情况下，实测边界是 131072；`wsl --shutdown` 释放显存后重试有机会成功。脚本会如实报告"实际边界在哪一级"，这本身是设计好的行为。

**Q11：为什么服务跑起来时，电脑的内存（RAM）也被占了几十 GB？**
这是正常现象，不是内存泄漏。原因有三层：
1. **WSL2 本身是一个"轻量虚拟机"**，有内存上限（默认取物理内存的 50% 或 8 GB 中较大者；本项目在 `.wslconfig` 里设了 24 GB）。WSL 里所有进程的内存花销都算在这个额度里。
2. **模型文件会被映射进内存**：约 20 GB 的权重文件，vLLM 加载时用内存映射（mmap）读入，文件内容会留在系统的页缓存里，看起来就像白白占了 20 GB。这部分可回收，但数字很吓人。
3. **加载和推理本就要在内存里过一道**：权重走"硬盘 → 内存 → 显存"的传输；跑起来后 vLLM 还在内存里常驻分词器词表、前缀缓存、调度元数据、CUDA 固定内存缓冲，加上 Python + torch + CUDA 上下文本身的 1–2 GB。

显存管的是"模型和 KV 缓存放在哪推理"，内存管的是"文件映射、加载通道、元数据和运行时本身"，两者独立、都要给够。真机踩过的坑：`.wslconfig memory=24GB` 时，19 GB 权重常驻 + 首次内核编译叠加，超限被系统 OOM-killer 杀掉 vLLM 进程——所以**如果启动过程中进程被无声杀掉，先怀疑内存上限**，把 `.wslconfig` 的 `memory` 调到 28–30 GB 后 `wsl --shutdown` 重启（见问题 6）。
