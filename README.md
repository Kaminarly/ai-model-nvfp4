# Qwen3.8-27B 本地推理服务（Windows + WSL2 + RTX 5090）

> 在你自己电脑上，离线运行一个 27B 参数的大语言模型，并提供一套和 OpenAI 一模一样的 API。任何会调用 OpenAI 接口的程序都能直接接上，全程不联网、不上传数据。

本项目的四个开发议题（环境搭建 / 运行前检查 / 短上下文服务 / 完整长上下文验证）均已完成并在你的真机上验证通过。设计细节与实施总结见 [`spec.md`](./.scratch/qwen3-8-27b-native-nvfp4-wsl2/spec.md)。

---

## 1. 这是什么项目

一句话：**把一个大模型跑在你自己的显卡上，并给它一个标准的 API 服务外壳。**

- **模型**：Qwen3.8-27B（约 270 亿参数），使用发布者提供的**原生 NVFP4 量化权重**。NVFP4 是 RTX 50 系（Blackwell）显卡的原生格式，速度和显存占用最优；我们直接加载原版 safetensors 文件，**不做任何格式转换、不重新量化**。
- **引擎**：vLLM 0.27.1，业界主流的推理服务框架，提供 OpenAI 兼容接口。
- **上下文**：单个对话最多 **200000 个 token（200k）**——这是你这台机器实测验证过的边界。
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
| 直接启动（推荐日常用） | `direct.sh` | **200000 token（200k）** | 预检 + 显存门后直接启动，不做任何验证，端口 8192 |
| 完整长上下文验证 | `fullcontext.sh` | **200000 token（200k）** | 想要一步步验证（冒烟→逐级提升→完整配置）时用 |

完整路径会自动做一遍"体检 → 冒烟 → 逐级提升上下文 → 启动完整配置"，每一步都验证生效，任何一步失败都会**明确告诉你为什么、怎么修**，不会静默退出。

### 4.2 推荐启动命令（直接启动，端口 8192）

日常使用推荐 `direct.sh`：先做运行前检查（22 项体检）和显存门检查，通过后**直接启动 vLLM 和 API 服务**，不做任何测试或验证，最快跑起来。端口默认 **8192**。

在 WSL 终端里执行（**本机实测验证过的配置**，开着 VS Code 也稳）：

```bash
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/direct.sh start \
  --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
```

看到 **`preflight READY`**、**`full-config VRAM gate` 通过**和 **`service booting: http://127.0.0.1:8192/v1`** 就说明成功了。服务在前台运行，**按 `Ctrl-C` 停止**。

- 端口默认 **8192**（不是 8000），可用 `--port N` 或 `SERVE_PORT` 改。
- 内置本机实测定案配置：上下文 **200000**、并发 **16**、显存利用率 **0.90**；可用环境变量 `FULL_MAX_MODEL_LEN` / `FULL_MAX_NUM_SEQS` / `FULL_GPU_MEM_UTIL` 覆盖。
- 想一步步验证（冒烟 → 逐级提升 → 完整配置）用下面的 `fullcontext.sh`；想快速验证环境用 `serve.sh`（8000 端口，8192 上下文）。

### 4.2.1 Windows 上双击启动（start-api-server.bat）

不想开 WSL 终端的话，直接双击 `scripts/start-api-server.bat` 就能启动和上面一模一样的服务（等价于运行 `direct.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`）：

- 自动把脚本路径转成 WSL 路径并调用 `direct.sh`，服务就在弹出的窗口里前台运行，**按 `Ctrl-C` 停止**；停止后脚本会自动执行 `wsl --shutdown` 彻底关闭 WSL（释放 WSL 占用的显存，和问题 1 里推荐的释放方式一致），然后窗口停留显示结果，按任意键关闭；
- **直接启动默认模型**（不再有模型选择菜单）：`/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`，上下文 200000；想换模型就在双击前设置环境变量 `MODEL_DIR`（WSL 路径），或在**命令行**里直接传参数：`start-api-server.bat --model-dir /path/to/model --port 8001`（参数与 `direct.sh` 完全一致）；
- 正式启动前会问一次**局域网访问**：`1` 开启、`2` 关闭、`0` 退出；直接回车等于 `2`（只本机）。选 `1` 会弹 UAC 提权，然后完成端口转发和防火墙（见 5.0 节）；
- 非默认 WSL 发行版可设 `WSL_DISTRO`（默认 `Ubuntu`）。

**想要开 MTP 的版本**：双击 `scripts/start-api-server-mtp.bat`（见 4.6 节）。两个脚本的区别只有两点——MTP 开关和上下文长度（200000 vs 180000），**端口都是 8192**，所以**不要同时启动两个脚本**，会端口冲突。

**默认采样参数**：两个脚本都内置同一组服务端默认采样参数（`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`），作为 vLLM 的服务端默认值，请求里显式传了同名参数会覆盖它（见 5.4 节）。可用环境变量 `VLLM_SAMPLING_JSON` 覆盖整组默认值。

### 4.3 完整长上下文验证启动（fullcontext.sh）

```bash
FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=200000 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER="200000" \
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/fullcontext.sh start \
  --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
```

看到 **`full-context configuration verified and enabled`** 和 **`full-context service is running: http://127.0.0.1:8000/v1`** 就说明成功了。服务在前台运行，**按 `Ctrl-C` 停止**。

> PowerShell 完整写法（想不带参数用默认值的话，把开头的环境变量整段去掉即可）：
> ```powershell
> wsl -d Ubuntu -- bash -lc "FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=200000 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER='200000' bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/fullcontext.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
> ```

### 4.4 参数是什么意思、有什么效果

**命令行选项（`direct.sh` / `fullcontext.sh` / `serve.sh` 的 `start` 通用）**

| 参数 | 含义 | 默认值 | 说明 / 效果 |
| --- | --- | --- | --- |
| `--model-dir DIR` | 模型目录 | 无（必填） | 指向本地原始 NVFP4 权重文件夹；脚本只读它，不下载、不改动 |
| `--prefix DIR` | 运行时目录 | `~/vllm` | venv 和日志放哪；一般不用改 |
| `--host IP` | 监听地址 | `127.0.0.1` | 默认只允许本机访问；绑定其它地址会被拒绝（安全设计）。想开放局域网用 `--lan`（绑定 `0.0.0.0`，仍需 Windows 端口转发 + 防火墙，见 5.0 节） |
| `--port N` | 端口 | `8000`（`direct.sh` 为 `8192`） | 端口被占用时改这里，`SERVE_PORT` 也可覆盖 |
| `--dry-run` | 只打印计划 | 关 | 不真正启动，先看脚本会执行哪些命令 |

**环境变量（放在命令前，空格隔开；不写就用默认值）**

| 变量 | 默认值 | 含义 / 效果 |
| --- | --- | --- |
| `FULL_GPU_MEM_UTIL` | `0.97` | 显存利用率：告诉 vLLM 最多占用显卡显存的多少。**越大 KV 缓存越大（上下文潜力越高），但启动时要求空闲显存越多，越容易因桌面程序占用而失败**。本机实测 **0.90** 最稳（推荐） |
| `FULL_MAX_MODEL_LEN` | `262144`（`direct.sh` 为 `200000`） | 目标上下文长度（token）。本机实测 **200000**（200k）能过，262144 需要释放显存后才有机会 |
| `FULL_MAX_NUM_SEQS` | `16` | 最大并发序列数（同时处理几个请求）。KV 缓存够大时 16 没问题 |
| `CONTEXT_LADDER` | `32768 65536 131072 262144` | （仅 fullcontext.sh）逐级验证的梯度：从短到长一步步试，试到哪一步失败，就如实报告"实际边界 = 上一级"。**建议写成 `"200000"` 单步直达，省 3 次启动** |
| `SERVE_PORT` | `8000` | （direct.sh 为 `8192`）服务端口 |
| `SHORT_MAX_MODEL_LEN` | `8192` | （serve.sh）短服务上下文 |
| `SHORT_MAX_NUM_SEQS` | `1` | （serve.sh）短服务并发 |
| `VLLM_SPEC_METHOD` | 空（不启用） | 启用推测解码。设 `mtp` 开启 MTP（多 token 预测），等效给 vLLM 加 `--spec-method mtp --spec-tokens 3`。`start-api-server-mtp.bat` 默认设了它 |
| `VLLM_SAMPLING_JSON` | 空（vLLM 默认采样） | 服务端默认采样参数的 JSON，等效给 vLLM 加 `--override-generation-config.*`。两个 `start-api-server*.bat` 默认设了 `{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}` |
| `VLLM_EXTRA_ARGS` | 空 | 额外透传给 vLLM 的命令行参数（空格分隔） |

> 注意 `CONTEXT_LADDER` 要写成 `"200000"` 这样带引号；如果传空串会被当成"没设置"而用回默认四阶梯度。

### 4.5 启动时脚本在做什么（了解即可）

**`direct.sh`（直接启动）**：只有两道闸门，通过即启动，不做任何验证——

1. **运行前检查**（22 项体检）：环境、GPU、显存、模型文件完整性、离线开关——不过关直接拒绝并告诉你修法；
2. **显存门**：检查当前空闲显存够不够 `FULL_GPU_MEM_UTIL`，不够就给出可行的降低配置建议；
3. **启动 vLLM + API 服务**并保持前台运行（Ctrl-C 停止）。

三条启动路径（`direct.sh` / `fullcontext.sh` / `serve.sh`）每次都带上固定 vLLM 设置：`--quantization modelopt`（NVFP4 权重）、`--kv-cache-dtype fp8`（KV 缓存用 FP8，比 bf16 减半显存，长上下文才能装下）、`--enable-prefix-caching`（前缀缓存：重复的 prompt 前缀直接命中缓存，多轮对话 / 批量相似请求提速明显）、`--trust-remote-code`。

**`fullcontext.sh`（完整验证）**：在上面的基础上，还会——冒烟（8192 快速启动验证 + 超长请求必须被拒绝）→ 逐级提升（按 `CONTEXT_LADDER` 核对 `/v1/models` 上报的上下文）→ 启动完整配置（并发验证、长请求验证、边界验证）后才保持运行。

每次启动的 vLLM 日志存在 `~/vllm/logs/`（`fullcontext-smoke.log`、`fullcontext-ramp-<长度>.log`），失败排查用得上。

### 4.6 可选加速：开启 MTP（多 token 预测）

**这个模型原生支持 MTP**：`config.json` 里有 `"mtp_num_hidden_layers": 1`，权重分片也带了完整的 MTP 层（`mtp.*` 共 15 个权重键，且量化配置把 `mtp*` 排除在 NVFP4 量化之外、保持 bf16）。vLLM 0.27.1 原生支持 `--spec-method mtp`，**不需要额外草稿模型**——MTP 是模型自带的"多 token 预测"草稿头，推理时每步先草拟接下来几个 token、再一起验证，接受率高时一次前向就能多生成几个 token，**加速的是输出（decoding）阶段**，长输出收益明显。

**最简单的方式：双击 `scripts/start-api-server-mtp.bat`**。它与 `start-api-server.bat` 唯一的区别就是开启了 MTP 并把上下文从 200000 降到 180000（MTP 多占一些显存，留出余量），端口同为 8192。启动前同样会问局域网访问（`1` 开启 / `2` 关闭 / `0` 退出，默认关闭）。它内部通过 `direct.sh` 启动，vLLM 参数自动带上：

```
--spec-method mtp --spec-tokens 3
```

（`--spec-tokens 3` 是 MTP 的草稿 token 数；vLLM 0.27.1 对 MTP 必须显式给这个值，否则启动报 `num_speculative_tokens must be provided`。）

**手动方式**：在 `direct.sh` / `fullcontext.sh` / `serve.sh` 前设环境变量即可，无需改脚本：

```bash
VLLM_SPEC_METHOD=mtp bash scripts/direct.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
```

底层 `scripts/lib/serve-lib.sh` 的 `serve_argv` 已支持三个透传环境变量（`VLLM_SPEC_METHOD` / `VLLM_SAMPLING_JSON` / `VLLM_EXTRA_ARGS`），见 4.4 节参数表。

注意事项：
- **显存**：RTX 5090 32GB 下，NVFP4 权重约 18.8GB + FP8 KV 缓存（比 bf16 减半，这是长上下文能装下的关键）+ 前缀缓存 + MTP 层（bf16），整体放得下但余量不大。开启后若 OOM，把 `FULL_GPU_MEM_UTIL` 从 0.90 降到 0.85 再试，或关掉 MTP。
- **MTP 与 CUDA graph**：vLLM 0.27.1 在 spec-decode + FlashInfer 下会把 CUDA graph 降级为 PIECEWISE 模式（日志会有一条警告），这是正常降级，不影响功能。
- **`min_p` / `logit_bias` 与推测解码**：vLLM 会警告 `min_p and logit_bias parameters won't work with speculative decoding`，即 MTP 开启时服务端默认的 `min_p` 不生效（默认 0.0 本就等于不启用，无实际影响）。
- **首次启动遇到 FlashInfer JIT 内核编译时**，用 bat 脚本启动最省心——它自带完整环境（`CUDA_HOME`、`MAX_JOBS=1` 等），避免手动启动漏掉环境变量。

### 4.7 用 llama.cpp 跑 GGUF 模型（start-api-server-gguf.bat，可选）

上面几节都是 vLLM + NVFP4 原生权重的路线。如果手里是 **GGUF 量化文件**（vLLM 那套引擎读不了 GGUF），本项目另外编译了 **llama.cpp**（源码在 WSL 的 `~/llama.cpp`，复用 vLLM venv 里的 CUDA 13.3 工具链，针对 RTX 5090 的 sm_120 架构编译），双击 `scripts/start-api-server-gguf.bat` 即按本机实测参数启动：

- **模型**：`/home/kami/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q5_K_M.gguf`（Q5_K_M，约 20GB；同目录还有 `mmproj-BF16.gguf` 视觉投影可选）；
- **参数**（本机实测）：上下文 **128000**、KV 缓存 **q8_0**（显存减半）、并发 auto、CPU 线程 8、全量 GPU 卸载 + Flash Attention；默认仅本机监听，启动前同样会问局域网访问（`1` 开启 / `2` 关闭 / `0` 退出，默认关闭）；
- **采样默认值**：temperature 0.7、top-k 20、top-p 0.8、min-p 0、repeat-penalty 1.0、presence-penalty 1.5；**默认关闭思考**（`--reasoning off`，直接回答不输出思考过程）；
- **端口 8192**，与 vLLM 版脚本相同——**不要与 `start-api-server.bat` / `start-api-server-mtp.bat` 同时启动**（端口冲突）。API 同样 OpenAI 兼容，模型名 `Qwen3.6-27B-Fable-Fusion`。

可覆盖的环境变量：`MODEL_GGUF`（GGUF 文件 WSL 路径）、`LLAMA_PORT`、`LLAMA_CTX`、`WSL_DISTRO`。想手动改参数启动，用 WSL 里的 `~/llama.cpp/llama-server.sh`（详见其文件头注释）。

---

## 5. 启动成功后：在 Windows 上调用模型服务

服务跑在 WSL 里，但**绑定了 127.0.0.1**，WSL2 默认会把 localhost 转发给 Windows，所以你在 **Windows 一侧直接用 `http://127.0.0.1:<端口>` 访问**即可。端口看脚本：`direct.sh` 是 **8192**，`serve.sh` / `fullcontext.sh` 是 **8000**。下面以 `direct.sh` 的 8192 为例。

### 5.0 局域网访问（其他设备通过 Wi-Fi / 网线访问这台机器）

默认只在本机监听（`127.0.0.1`）。想让**同一局域网里的其它电脑 / 手机**访问 API，需要三步：**让服务绑定所有网卡 + Windows 端口转发到 WSL + 防火墙放行**。

最简单的方式：双击 `start-api-server.bat` / `start-api-server-mtp.bat` / `start-api-server-gguf.bat` 任一，在启动前的菜单里选 **`1` 开启**（`2` 关闭，回车默认关闭；`0` 退出）。选 `1` 会弹 **UAC 管理员授权**，然后完成端口转发和防火墙。专用脚本 `scripts/start-api-server-lan.bat` 仍可用，等价于默认启动器选 `1`（vLLM 200k，无 MTP）。

开启后会做这三步：

1. **服务绑定 `0.0.0.0`**（WSL 内所有网卡；vLLM 用 `--lan`，GGUF 覆盖 llama-server 的 `--host`）；
2. **Windows 端口转发**：`netsh interface portproxy add v4tov4 listenport=8192 listenaddress=0.0.0.0 connectport=8192 connectaddress=<WSL的IP>`。WSL2 是 NAT 网络，WSL 的 IP（本机 `172.19.28.195`）每次启动都会变，所以脚本每次启动时自动读取当前 IP 并写入转发，退出时自动删除（避免残留指向旧 IP 的转发）；
3. **Windows 防火墙放行**入站 TCP 8192，退出时自动删除。

启动窗口会打印本机局域网 IP（如 `192.168.x.x`）和访问地址。**其它设备访问**：

```text
http://<这台电脑的局域网IP>:8192/v1/models          # 看模型
http://<这台电脑的局域网IP>:8192/v1/chat/completions # 聊天
```

在别的电脑 / 手机上把 `base_url` 指到 `http://<局域网IP>:8192/v1` 即可（OpenAI 兼容，见 5.3）。

也可以**手动**用命令行（管理员 PowerShell）做同样的事：

```powershell
wsl -d Ubuntu -- bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/direct.sh start --lan --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
# 另开一个管理员 PowerShell：
$wslIp = (wsl -d Ubuntu -- hostname -I).Trim().Split(' ')[0]
netsh interface portproxy add v4tov4 listenport=8192 listenaddress=0.0.0.0 connectport=8192 connectaddress=$wslIp
netsh advfirewall firewall add rule name="Qwen API 8192" dir=in action=allow protocol=TCP localport=8192
# 用完删除：
# netsh interface portproxy delete v4tov4 listenport=8192 listenaddress=0.0.0.0
# netsh advfirewall firewall delete rule name="Qwen API 8192"
```

**注意事项（务必读）**：

- **没有鉴权**：`--lan` 会把 API 暴露给局域网里的所有设备，任何人都能调用（无需密钥）。只在**可信网络**（家庭 / 办公室内网）使用；不要在公共 Wi-Fi 或不受信任的网络开。
- **只做端口转发，不做 NAT 外网映射**：访问地址是 `http://<局域网IP>:8192`，路由器外的互联网设备仍无法访问（除非你另外配端口映射，本项目不做）。
- **Windows 防火墙当前是关闭的**（本机 `netsh advfirewall show currentprofile` 显示专用/公用均为关闭），此时防火墙规则其实用不上；但为了你在打开防火墙时也能通，脚本仍会添加规则。
- 想换端口：`set SERVE_PORT=8001` 后再双击，或加 `--port 8001`。
- 三个 bat（`start-api-server.bat` / `-mtp` / `-gguf`）默认端口都是 8192，**不要同时启动**；开启局域网时端口转发也只能对应其中一个。


### 5.1 先确认服务活着（Windows PowerShell）

```powershell
curl http://127.0.0.1:8192/v1/models
```

应返回类似（`max_model_len: 200000` 就是 200k 上下文）：

```json
{"object":"list","data":[{"id":"Qwen3.8-27B-NVFP4-RTX5090","max_model_len":200000,...}]}
```

### 5.2 发一条聊天消息（curl）

```powershell
curl -X POST http://127.0.0.1:8192/v1/chat/completions -H "Content-Type: application/json" -d '{\"model\":\"Qwen3.8-27B-NVFP4-RTX5090\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，介绍一下你自己\"}],\"max_tokens\":64}'
```

> `model` 填的是服务显示的模型名（默认 = 模型目录的文件夹名）。

### 5.3 用 Python 调用（推荐，写代码时用这个）

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8192/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen3.8-27B-NVFP4-RTX5090",
    messages=[{"role": "user", "content": "用一句话解释什么是 KV 缓存"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

因为接口是 OpenAI 兼容的，所以任何现成的 OpenAI 客户端、框架、工具，把 `base_url` 指到 `http://127.0.0.1:8192/v1`（或 8000，看你用哪个脚本）就能直接用。

### 5.4 调整采样参数（温度 / top-p / top-k 等）

这些参数**每次请求单独传**，不用重启服务、不用改启动脚本；不同请求可以给不同值。vLLM 0.27.1 的 `/v1/chat/completions` 接口原生支持：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `temperature` | 1.0 | 温度，越高越随机、越低越确定 |
| `top_p` | 1.0 | 核采样，只从累计概率 top-p 的 token 里采样 |
| `top_k` | 0 | **vLLM 里 `0` 表示不启用 top-k 过滤**（不是取前 0 个） |
| `min_p` | 0.0 | 相对概率过滤 |
| `repetition_penalty` | 1.0 | 大于 1 惩罚重复 |
| `presence_penalty` | 0.0 | 出现过的 token 降权 |
| `frequency_penalty` | 0.0 | 按出现频率降权 |
| `seed` | 无 | 固定随机种子，同输入同参数可复现 |
| `max_tokens` | 无 | 输出长度上限（见 5.5） |

curl 示例（在 5.2 的请求体里加这些字段）：

```powershell
curl -X POST http://127.0.0.1:8192/v1/chat/completions -H "Content-Type: application/json" -d '{\"model\":\"Qwen3.8-27B-NVFP4-RTX5090\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"temperature\":0.6,\"top_p\":0.9,\"top_k\":40,\"repetition_penalty\":1.05,\"max_tokens\":2048}'
```

Python（OpenAI SDK）里直接当参数传：

```python
resp = client.chat.completions.create(
    model="Qwen3.8-27B-NVFP4-RTX5090",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.6,
    top_p=0.9,
    top_k=40,   # OpenAI 标准没有这个字段，vLLM 兼容层直接收
)
```

要点：
- 不传就用默认值（temperature=1.0 / top_p=1.0 / 无 top-k），输出偏"发散"；做固定任务建议 temperature 0.6~0.8、top_p 0.9。
- vLLM 0.27.1 **没有服务端默认采样参数的启动项**，想在网关（OpenWebUI / OneAPI 等）统一默认值，就在网关里配。
- 这是思考类模型：temperature 同时作用于思考 token 和正文，调低会让思考也收敛；`reasoning_effort` 控思考深度、`temperature` 控随机性，两者正交。

### 5.5 上下文上限：max-input 与 max-output

当前配置（`--max-model-len 200000`，200k）下，**输入和输出不是各占一半，而是共用一个 200000 token 的总预算**（系统提示 + 历史消息 + 当前问题 + 模型回答，全部算进去）：

- **max-input（输入硬上限）= 200000 token**。输入一旦超过 200000 就被拒绝（HTTP 400，`maximum context length`），这是正常保护，不是故障。
- **max-output（本次输出上限）= `min(请求的 max_tokens, 200000 − 本次输入)`**：
  - 传了 `max_tokens`：取「你传的值」和「200000 − 输入」中较小的。例：输入 150000 + `max_tokens: 8192` → 输出上限 8192；`max_tokens: 60000` → 被压到 50000。
  - **没传 `max_tokens`**：输出上限 = `200000 − 本次输入`（vLLM 的 `get_max_tokens` 逻辑，见 `entrypoints/serve/utils/api_utils.py`）。输入为空时理论上限就是 200000，实际还受显存、`FULL_MAX_NUM_SEQS` 等约束。

一句话：**输入 + 输出 ≤ 200000 是唯一硬约束**。想拿"尽可能长"的回答就不传 `max_tokens`（或传个大值），让它吃满剩余预算；想控制长度就显式传。

---

## 6. 注意事项

1. **显存是最大的约束**：开服务前尽量关掉占显存的程序（浏览器、视频、其它 AI 工具）。启动时显存不够，脚本会明确告诉你"还差多少、怎么释放、或改用哪个更低配置"。
2. **三个启动脚本的端口和验证不同**：`direct.sh`（日常推荐，端口 8192，预检+显存门后直接启动、不做验证）、`serve.sh`（端口 8000，短上下文 8192）、`fullcontext.sh`（端口 8000，完整验证）。Windows 侧访问的地址要跟脚本一致（8192 或 8000）。
3. **内存（RAM）也会被占用**：WSL2 有内存上限，模型文件映射 + 进程常驻可达几十 GB，上限设太小会被系统杀掉（OOM）。这是正常现象，不是内存泄漏。详见第 7 节问题 11。
4. **第一次启动特别慢**：vLLM 首次运行要为你的显卡编译内核（FlashInfer JIT，约 5–15 分钟，取决于 CPU）。之后启动就快了，编译结果有缓存。**期间不要关窗口**。
5. **保持配置一致**：本机验证过的组合是 `0.90 / 200000 / 16`。调大 `FULL_GPU_MEM_UTIL` 或上下文到 262144 前，确认显存余量（参考第 7 节问题 1）。
6. **默认只在本机**：默认绑定 127.0.0.1，不向局域网或互联网开放端口。想开放局域网访问，在三个启动器的启动前菜单选 `1`（或用 `start-api-server-lan.bat` / `direct.sh --lan`）并配合 Windows 端口转发 + 防火墙，详见 5.0 节；**该模式没有鉴权，请只在可信网络使用**。
7. **永远离线**：脚本强制 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，只读你指定的本地模型目录。如果你的目录里没有模型，报错会提示你，而不是偷偷去下载。
8. **改版本要谨慎**：环境版本全部钉死（vLLM 0.27.1 + CUDA 13 + Python 3.14），这是模型卡验证过的组合。升级走 `wsl2-env.sh create --force`（会用环境变量指定的新版本重建），不建议手动 `pip install` 乱改。
9. **测试**：项目自带四套自动化测试（`bash tests/run-tests.sh`、`tests/preflight-tests.sh`、`tests/serve-tests.sh`、`tests/fullcontext-tests.sh`），用假工具模拟环境，可在没有 WSL/GPU 的机器上跑，改代码后跑一遍防回归。

---

## 7. 常见问题与解决方法

**Q1：启动报 `full-config VRAM gate`，提示显存不够，怎么办？**
原因：桌面程序占用了显存，你设定的 `FULL_GPU_MEM_UTIL × 显存总量` 大于当前空闲显存。
解决（按顺序试）：
1. 关掉 Windows 上占 GPU 的程序（浏览器、视频、其它模型），然后执行 `wsl --shutdown` 再重开 WSL（WSL 会缓存显存，必须重启才真正释放）；
2. 或降低配置：`FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=200000 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER="200000"`；
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
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh create --prefix ~/vllm --force
```

**Q6：WSL 内存不足，启动过程中进程被系统杀掉（`OOM`）？**
权重加载约 19 GB 常驻 + 首次内核编译吃内存。脚本已强制 `MAX_JOBS=1` 串行编译来降低峰值；如果还是 OOM，把 `.wslconfig` 里的 `memory` 调大（当前为 32GB）后 `wsl --shutdown` 重启。

**Q7：Windows 上访问 `http://127.0.0.1:8000` 不通？**
先确认 WSL 里服务真的在跑（第 4.2 节的输出）。若在跑仍不通：WSL2 的 localhost 转发偶尔会被镜像网络等配置干扰，检查 `C:\Users\<你>\\.wslconfig` 里是否设置了不受支持的键（Windows 10 上 `networkingMode=mirrored` 等不受支持，删掉后 `wsl --shutdown` 重启）。仍不行就在 WSL 内访问（`curl http://127.0.0.1:8000/v1/models`）。

**Q8：请求返回 HTTP 400 / `maximum context length`？**
你的输入超过了上下文上限（200000 token）。这是预期行为——模型有硬边界，超了就拒绝。缩短输入、或清掉部分历史消息即可。

**Q9：端口 8000 / 8192 被占用？**
换个端口：加 `--port 8001`（Windows 侧访问地址相应变成 `http://127.0.0.1:8001`）。

**Q10：为什么我设了 `FULL_MAX_MODEL_LEN=262144` 却启动失败？**
262144 是规格目标，但需要更大的 KV 缓存，也就是更高的显存占用。你这台机器在桌面程序占用显存的情况下，实测边界是 200000；`wsl --shutdown` 释放显存后重试有机会成功。脚本会如实报告"实际边界在哪一级"，这本身是设计好的行为。

**Q11：为什么服务跑起来时，电脑的内存（RAM）也被占了几十 GB？**
这是正常现象，不是内存泄漏。原因有三层：
1. **WSL2 本身是一个"轻量虚拟机"**，有内存上限（默认取物理内存的 50% 或 8 GB 中较大者；本项目在 `.wslconfig` 里设了 32 GB）。WSL 里所有进程的内存花销都算在这个额度里。
2. **模型文件会被映射进内存**：约 20 GB 的权重文件，vLLM 加载时用内存映射（mmap）读入，文件内容会留在系统的页缓存里，看起来就像白白占了 20 GB。这部分可回收，但数字很吓人。
3. **加载和推理本就要在内存里过一道**：权重走"硬盘 → 内存 → 显存"的传输；跑起来后 vLLM 还在内存里常驻分词器词表、前缀缓存、调度元数据、CUDA 固定内存缓冲，加上 Python + torch + CUDA 上下文本身的 1–2 GB。

显存管的是"模型和 KV 缓存放在哪推理"，内存管的是"文件映射、加载通道、元数据和运行时本身"，两者独立、都要给够。真机踩过的坑：早期 `.wslconfig memory=24GB` 时，19 GB 权重常驻 + 首次内核编译叠加，超限被系统 OOM-killer 杀掉 vLLM 进程——现已把内存上限上调到 32GB 规避此问题。所以**如果启动过程中进程被无声杀掉，先怀疑内存上限**，把 `.wslconfig` 的 `memory` 调大（当前 32GB）后 `wsl --shutdown` 重启（见问题 6）。

**Q12：`direct.sh` 和 `fullcontext.sh` 有什么区别？**
`direct.sh`（端口 8192）只做两道闸门——运行前检查（22 项体检）+ 显存门——通过后**直接启动 vLLM 和 API 服务**，不做冒烟、逐级提升、并发/边界探测等任何验证，最快跑起来，适合日常使用。`fullcontext.sh`（端口 8000）会先跑短上下文冒烟、按 `CONTEXT_LADDER` 逐级核对上下文、最后做并发/长请求/超界探测并保持运行，适合第一次跑、或想确认环境真实边界时。两者用的 vLLM 配置相同（0.90 / 200000 / 16），区别只在"要不要多花时间做验证"。

**Q13：模型把思考过程直接当正文输出了（没有 `<think>` 标签包裹）？**
这是 vLLM 服务端没启用 Qwen3 推理解析器导致的，已修复（2026-08-19）。原因：这个模型的 chat template 默认开启思考（`enable_thinking=true`），vLLM 只有在带 `--reasoning-parser qwen3` 启动时才会在生成 prompt 里注入 `<think>` 起始标签，引导模型按"思考 + `</think>` + 正文"输出，并把思考提取到独立字段；没有这个参数时，模型会把思考过程当成正文直接输出。
修复：启动参数已在 `scripts/lib/serve-lib.sh` 统一加上 `--reasoning-parser qwen3`（`serve.sh` / `direct.sh` / `fullcontext.sh` / `start-api-server.bat` 都生效），重启服务即可。修复后思考内容在响应的 `reasoning` 字段（这是 vLLM 0.27.1 的字段名；OpenAI 标准叫 `reasoning_content`），正文在 `content`，不再混在一起。如果客户端界面仍不显示思考块，多半是客户端只认 `reasoning_content` 字段名，可联系适配。

**Q14：双击 `scripts` 里的 `start-api-server*.bat` 窗口一闪而过 / 直接闪退？**
已修复（2026-08-21）。闪退是四个问题叠加造成的，每一个都能让窗口在出现后立刻关闭：
1. **换行符和 BOM 错误**：`mtp` / `gguf` / `bat` 三个文件是 LF 换行（其中两个还带 UTF-8 BOM）。cmd.exe 解析 LF 换行的批处理有已知 bug（按 512 字节块读取，行在块边界被截断成乱码），脚本走不到结尾的 `pause` 就崩了；BOM 则让首行 `@echo off` 失效。
2. **`.sh` 脚本被转成了 CRLF**：Windows 上 `git config core.autocrlf=true` 会在 checkout 时把所有 `.sh` 转成 CRLF，而 bash 把行尾的 `\r` 当命令字符，`wsl bash direct.sh` 一执行就语法错误崩溃——这是"立刻"闪退的最直接原因。
3. **UAC 提权空参数**：`start-api-server-lan.bat` 双击时命令行参数为空，PowerShell 的 `-ArgumentList '%*'` 变成空字符串直接报错，UAC 弹窗都不出现。
4. **cmd 的中文解析缺陷**：`chcp 65001` 下 cmd 解析 UTF-8 中文批处理是非确定性失败的（实测多次运行偶发报错）。

修复内容（全部已实测验证）：四个 `.bat` 统一为 **CRLF + 无 BOM + 纯 ASCII**（菜单/提示改为英文）；全部 `.sh` 转回 **LF**；修 UAC 提权（空参数分支 + 拒绝后 `pause` 不闪退）；`--lan` 参数改为逐参数过滤（绕开 cmd 对空变量做字符串替换会输出 `--lan=` 垃圾的陷阱）；`echo` 内容带括号的 `if/else` 块改成 `goto` 分流（绕开括号块解析错误）。同时新增根目录 `.gitattributes`（`*.bat eol=crlf`、`*.sh eol=lf`、`tests/fakebin/vllm eol=lf`）并已 `git add --renormalize` 规范化，防止以后再被 autocrlf 弄乱。
现在的行为：双击任意 `.bat` 会停留显示菜单（1 开 LAN / 2 默认 / 0 退出），提权时正常弹 UAC，服务启动后停在窗口，Ctrl-C 停止后显示结果并等你按键——不再闪退。
