# SparkInfer × 本机 Qwen3.8-27B NVFP4 / DSpark 加速器研究报告

> 研究日期：2026-09-18
> 研究性质：**只读调研**。未修改任何代码、模型、环境、框架；未下载任何模型、加速器或镜像；未构建、未拉取、未启动任何容器。
> 唯一副作用：为读取本机现状，WSL 发行版 Ubuntu 被拉起过一次（`wsl -l -v` 初始显示为 Stopped），全部探测命令均为只读（`ls` / `du` / `cat` / `nvidia-smi` / `docker images` / `docker info` / `curl` 头部探测，以及一次本地模型目录的 sha256 计算）。
> 研究对象：<https://github.com/gittensor-ai-lab/sparkinfer>
> 目标权重：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`、`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`

---

## 一、结论摘要

1. **权重完全对得上。** 本机 `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` 与 `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4` 就是 SparkInfer 官方发布容器默认下载的那两份 checkpoint，也是模型卡明确写着 "Runs on **SparkInfer**, SGLang and vLLM unmodified" 的那份权重。**不需要任何格式转换、重新量化或 GGUF 转换**：`lm_head` 的 NVFP4 由引擎原生加载。
2. **WSL 内跑得通，且不必联网下载权重。** 官方镜像 `ghcr.io/gittensor-ai-lab/sparkinfer-qwen38` 在 WSL2 + Docker + `--gpus all` 下可用（本机已有同架构的 SGLang 容器成功运行先例）；把两个本地目录只读挂进容器的 `/models/qwen38-nvfp4` 与 `/models/qwen38-dspark`，入口脚本会因为 `config.json` 已存在而**跳过下载**。代价是仍需拉取镜像本身（压缩层合计 0.42 GB，模型卡称镜像约 1 GB）。
3. **DSpark 现在可以在 HTTP API 上用，不再只是 bench harness。** 0.5.7（2026-09-14）起 `sparkinfer_server` 支持 `serve-dspark`。但生效条件很窄：**greedy 采样 + 纯文本 + 全服务同一时刻只有这一个请求**。
4. **本仓库既有结论已过期，需要显式纠正。** `result/qwen38-nvfp4-dspark-upgrade-research.md:264` 与 HF 目标模型卡都写着 "DSpark 只在 bench harness，HTTP 服务器还不能 curl"——这两处的依据都早于 sparkinfer 0.5.7（详见第六节时间线）。
5. **三个必须先做的取舍**：上下文（SparkInfer 开 DSpark 默认 131072，低于本项目现有 vLLM 的 200000）、并发（8–16 并发下 SparkInfer 聚合吞吐落后 vLLM 约 1.6×，而本项目现用 16 并发）、以及官方自述的**已知正确性缺陷**（int8 KV 下 ≥2048 token 的预填充注意力不准，默认开启；issue #976 已记录过 ≥4k token 长提示答错且不可复现的现场，见 7.1）。
6. **源码构建路线本机缺依赖**：`BUILD_SERVER=ON` 需要 `rustc ≥ 1.79`（tokenizers-cpp），本机 PATH 中无 `cargo` / `rustc`。CUDA 工具链本身是有的（见 2.4）。

**一句话建议**：以官方容器 + 本地只读挂载的**离线运行**为首选路线；在真正实测前不要动本仓库现有的 vLLM / SGLang 启动链路（端口与显存都只够一个引擎）。

---

## 二、本机现状（本次实测）

### 2.1 宿主与 WSL

| 项 | 实测值 | 来源 |
| --- | --- | --- |
| WSL 发行版 | Ubuntu，WSL2 | `wsl -l -v`、注册表 `HKCU:\...\Lxss` |
| 发行版根文件系统 | `D:\WSL`（`/dev/sdd`，1007G，已用 89G，可用 868G） | `df -h /` |
| 发行版内核 | `6.18.33.2-microsoft-standard-WSL2` | `uname -r` |
| 发行版版本 | Ubuntu 26.04 LTS | `/etc/os-release` |
| 内存上限 | 30 GiB 可见（`.wslconfig` 设 `memory=32GB`），swap 8 GiB | `free -g`、`C:\Users\Administrator\.wslconfig` |
| systemd | 已启用（`systemd=true`，PID 1 为 systemd） | `.wslconfig`、`docker info` 显示 `Cgroup Driver: systemd` |
| Windows 侧磁盘 | `D:\` 604G，可用 131G | `df -h /mnt/d` |

### 2.2 GPU

| 项 | 实测值 |
| --- | --- |
| 型号 / 显存 | NVIDIA GeForce RTX 5090 / 32607 MiB |
| 驱动 | 610.88 |
| 计算能力 | 12.0（即 `sm_120`，Blackwell） |
| 空闲占用 | 2160 MiB / 利用率 9% |

`sm_120` 正是 SparkInfer 的目标架构（README：built for `sm_120` + `sm_121`，容器 "Blackwell (`sm_120`) only"）。

### 2.3 Docker 与容器运行时

| 项 | 实测值 |
| --- | --- |
| Docker | 29.8.1（`docker --version`） |
| 存储驱动 / cgroup | `overlayfs` / cgroup v2（systemd） |
| 默认 runtime | `runc` |
| 已注册 runtime | `nvidia runc io.containerd.runc.v2`（`/etc/docker/daemon.json` 中 `nvidia` → `nvidia-container-runtime`） |
| NVIDIA 工具 | `/usr/bin/nvidia-container-runtime`、`nvidia-container-toolkit`、`nvidia-ctk` 均在 |
| 已有镜像 | `lmsysorg/sglang:qwen38-27b`（41.9 GB），容器 `qwen38-sglang` 处于 Exited |
| 该地址的 SparkInfer 镜像 | **不存在**（本次未拉取） |

**结论**：GPU 直通所需的两件套（`--gpus` + nvidia runtime）已就绪，SparkInfer 容器不需要额外安装运行时组件。

### 2.4 工具链（影响"源码构建"路线）

| 工具 | 状态 |
| --- | --- |
| `cmake` | 4.2.3（`/usr/bin/cmake`） |
| `gcc` / `g++` | 15.2.0（Ubuntu 15.2.0-16ubuntu1） |
| `ninja` | 已安装 |
| CUDA `nvcc` | **在 venv 内，不在 PATH**：`/home/kami/vllm/venv/lib/python3.14/site-packages/nvidia/cu13/bin/nvcc`（pip 包 `nvidia-cuda-nvcc` 13.3.73；`cuda_toolkit` 13.0.3.0） |
| `cargo` / `rustc` | **缺失**（PATH 中无；`command -v` 均未命中） |
| `hf` CLI | 在 venv 内（`/home/kami/vllm/venv/bin/hf`，另有已废弃的 `huggingface-cli`） |
| Python | venv 用系统 `python3`（符号链接），`env-info.txt` 记录 Python 3.14.4 / vLLM 0.27.1 / CUDA 13.3.1 |
| Docker 权限 | 本机 `kami` 不在 docker 组，需 root（既有脚本 `scripts/sglang-dspark.sh:19-20` 已记录此事实） |

### 2.5 网络可达性（仅做连通性探测，未下载）

| 目标 | 结果 |
| --- | --- |
| `https://ghcr.io/v2/` | HTTP 401（正常：匿名未授权即 401，说明**可达**） |
| `https://huggingface.co/api/models/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` | HTTP 200 |
| `https://github.com/gittensor-ai-lab/sparkinfer` | HTTP 200 |

镜像元数据查询（tags / manifest / config blob，均为元数据，不涉及层下载）已完成，见 3.4。

### 2.6 两个模型目录的实测内容

**目标模型 `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`（`du` 17G，15 个文件）**

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `model-00001-of-00002.safetensors` | 9,972,777,720 | 与 HF `main` 同尺寸 |
| `model-00002-of-00002.safetensors` | 7,943,334,864 | 与 HF `main` 同尺寸 |
| `model.safetensors.index.json` | 236,508 | 索引：2387 个张量，`mtp` 命中 0，`visual` 333 个 |
| `tokenizer.json` | 12,809,320 | **必要**：容器入口固定传 `--tokenizer "$MODEL_DIR/tokenizer.json"` |
| `config.json` | 13,167 | 量化配置内联在此 |
| `hf_quant_config.json` | 9,050 | ModelOpt 侧排除清单 |
| `generation_config.json` | 213 | `temperature 1.0 / top_k 20 / top_p 0.95` |
| `chat_template.jinja` / `tokenizer_config.json` / `vocab.json` / `merges.txt` | 14,221 / 1,150 / 6,722,759 / 3,353,259 | 模板与分词资源 |
| `preprocessor_config.json` / `processor_config.json` / `video_preprocessor_config.json` | 390 / 1,191 / 385 | 视觉/视频预处理 |
| `crc32.txt` | 86 | 两行分片 CRC32 |

索引元数据：`total_parameters 15,193,246,960`、`total_size 17,915,815,528` 字节 = **17.92 GB / 2 分片**——与 HF `main`（final build，2 分片、17.92 GB、MTP 已删）一致，且 `config.json` 的 `text_config.mtp_num_hidden_layers = 0`、索引中无 `mtp` 张量，**证实这是 final build，不是 `pre-final`**。

> 补充事实：本仓库早前的升级研究（`result/qwen38-nvfp4-dspark-upgrade-research.md:11`、`:74`）曾记录"本机是 `pre-final`（3 分片 18.77 GB）"。本次实测显示本机**已经升级为 final build**（2 分片、17.92 GB、无 MTP），其附录 A.7（2026-09-16）记录的下载与落位已完成。

**目标模型的关键结构（`config.json` 实测）**

| 键 | 值 |
| --- | --- |
| `architectures` / `model_type` | `Qwen3_5ForConditionalGeneration` / `qwen3_5`（`text_config.model_type = qwen3_5_text`） |
| 层数 / 隐藏维 | 64 / 5120 |
| `layer_types` | 48 × `linear_attention` + 16 × `full_attention`（`full_attention_interval = 4`，混合 Gated-DeltaNet） |
| 注意力 | `num_attention_heads 24` / `num_key_value_heads 4` / `head_dim 256` |
| 上下文 / 词表 | `max_position_embeddings 262144` / `vocab_size 248320` |
| 视觉塔 | `vision_config` 存在（`qwen3_5_vision`，depth 27），`language_model_only = false` |
| 量化 | `quantization_config.quant_method = "modelopt"`、`quant_algo = "NVFP4"`、权重与激活均 4-bit / group_size 16（W4A4）、`kv_cache_scheme` = 8-bit float（FP8） |
| 排除（BF16 保留） | `model.language_model.embed_tokens`、48 层 × {`linear_attn.conv1d`, `in_proj_a`, `in_proj_b`}、`model.visual*` |
| `lm_head` | **未在排除清单中 → 已量化为 NVFP4**（模型卡：0.72 GB vs BF16 的 2.54 GB） |

**DSpark 草稿 `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4`（`du` 1.4G，3 个文件）**

| 文件 | 大小 |
| --- | --- |
| `model.safetensors` | 1,399,670,058 |
| `config.json` | 2,828 |
| `hf_quant_config.json` | 937 |

草稿自身**没有** tokenizer / chat template / `generation_config.json`——它复用目标模型的，这正是容器入口只对目标模型传 `--tokenizer` 的原因。草稿结构：`Qwen3DSparkModel`、5 层全注意力、`block_size 7`、`target_layer_ids [4,16,28,40,52]`、Markov head（vanilla, rank 256）+ confidence head、`draft_vocab_size 248320`、RoPE 为 YaRN（factor 32）、`modelopt` NVFP4 W4A4。

> 与 HF 侧目录的差异（影响离线校验，见 4.3）：本机目标目录**缺少** HF `main` 上的 `README.md`、`LICENSE`、`README.qwen-upstream.md`、`assets/*.png`（13 个）；草稿目录缺少 `README.md`。这些在容器自带的 manifest 校验算法里**不是**点文件，会计入哈希。

---

## 三、SparkInfer 是什么（一手来源）

### 3.1 定位

- 一个**原生 C++/CUDA 推理运行时**，面向消费级/边缘 Blackwell（`sm_120` + `sm_121`），无 Python 栈，官方称二进制 2.5 MB（不含权重），MIT 许可。
- 仓库活跃度（GitHub API，本次查询）：创建于 2026-06-22，`pushed_at` 2026-09-18T06:23:01Z，80 stars / 74 forks，语言 C++，开放 issue 3 个。
- 版本节奏：CHANGELOG 显示 0.5.5 起连续发版，最新 **0.5.10（2026-09-17）**。
- 该项目自称由公开竞赛（SN74 / Gittensor）持续优化，每个 PR 在 RTX 5090 上跑正确性与速度门禁；README 中的基准表由机器人自动刷新，因此不会落后于代码。

### 3.2 与本模型直接相关的官方数据（均来自仓库 README / server/README）

| 项 | 数值 / 说明 |
| --- | --- |
| 官方基准（ModelOpt NVFP4 目标，RTX 5090） | 128 上下文 95.7 tok/s、4k 93.6、16k 90.2（decode）；prefill 6,942 / 14,364 / 13,794 |
| DSpark 相对 AR 均值加速 | 4k 4.01×、16k 2.97×、32k 2.63× |
| DSpark 门禁用例（16k 长文） | 130.4 vs 88.6 tok/s = **1.472×**，平均接受 τ 1.730，且要求与关闭草稿的输出**逐字节一致**（lossless） |
| 上下文与显存 | AR 默认 `--ctx 262144`（27.9 GB）；`serve-dspark` 默认 `131072`（32 GB 卡上全上下文 KV 池装不下草稿）；360,000 是上限（31.2 GB，decode 降到约 74 tok/s） |
| 草稿自身的注意力窗口 | `SPARKINFER_DSPARK_MAX_CTX` 默认 **16384**，上限为 `--ctx` |
| 服务接口 | OpenAI（`/v1/chat/completions`、`/v1/completions`、`/v1/responses`）、Anthropic（`/v1/messages`）、Ollama、LM Studio；另有 `/v1/score`（教师强制打分）、`/v1/tokenize`、`/v1/capacity`、`/metrics` |
| 输入模态 | NVFP4 目标支持文本 / 图片 / 视频；GGUF 目标不支持图片视频（视觉塔来自 HF 目录） |
| 运行时体积对比 | sparkinfer 2.5 MB vs llama.cpp CUDA 80 MB vs vLLM 605 MB |

### 3.3 模型加载：这份权重为什么能直接吃

- 目标权重是 **NVIDIA ModelOpt** 量化的 NVFP4（`quant_method: "modelopt"`），不是 `compressed-tensors` 打包格式。模型卡明确写 "**The NVFP4 `lm_head` loads natively — nothing needs converting**"，且容器入口直接 `-m "$MODEL_DIR"`。
- 模型卡还给出一个对工具作者有用的坑：被排除的模块必须**同时**列在 `hf_quant_config.json` 的 `exclude_modules` 与 `config.json` 的 `quantization_config.ignore` 两处；只写其一会在加载时报 `Parameter lm_head.input_scale not found`。本机两个目录**两处都存在且内容对应**（已实测比对），所以这个风险在本机权重上不成立。
- 对照组：模型卡提到 RadixArk 的 DSpark 草稿 "**does not load on SparkInfer at all**"，原因是它的 Gated-DeltaNet 投影被 compressed-tensors loader 判为畸形 FP8。这从反面说明 loader 对"格式与排除清单是否规范"是敏感的——而本机这两份是作者认证能加载的。

**代码级证据（本次阅读 `main` 分支源码，2026-09-18）**

- 入口分发逻辑（`server/src/model_engine.cpp:224-247`）：`-m` 指向**目录**时，要求目录内存在 `config.json`，然后按 `config.json` 是否含 `quantization_config` 选择 `LoadKind::CompressedTensors`（含）或 `LoadKind::PlainSafetensors`（不含）；随后调用 `load_compressed_tensors()`。本机目标的 `config.json` **含** `quantization_config`（`quant_method: "modelopt"`），因此走 ModelOpt / compressed-tensors NVFP4 路径。
- `config.json` 必须带 `text_config` 块（`runtime/examples/qwen38_hf_config.h:37-38` 的报错文案为 `config.json missing text_config block`），RoPE 参数从 `text_config.rope_parameters` 读取。本机目标**满足**（`text_config.model_type = qwen3_5_text`）。
- 权重布局：`model.safetensors.index.json` + 分片，或单文件 `model.safetensors`，两者皆可；索引里引用但缺失的分片会被跳过而非致命错误（`runtime/src/safetensors.cpp:362-421`）。本机目标用索引+2 分片，草稿用单文件——**两种都被支持**。
- 草稿的期望布局被写死为 `dir/config.json` + `dir/model.safetensors`（单文件，`runtime/src/models/dflash_draft.cpp:719-720`），`block_size` / `mask_token_id` / `target_layer_ids` 取自 `config.json` 的 `dflash_config`。本机草稿目录**完全符合**。
- **纯 bf16/fp16 safetensors 不受支持**（`server/src/model_engine.cpp:350-357`，需先用 `runtime/tools/convert_qwen35.py` 转换）——与本机无关，因为本机是 NVFP4。
- **chat template 不读 checkpoint 里的文件**：模板以 `apply_qwen36_chat_template` 形式编译进二进制（`server/src/chat_tokenizer.cpp:335`），未发现读取 `chat_template.jinja` / `tokenizer_config.json` 的代码。0.5.9 CHANGELOG 说明其规则与 checkpoint 自带模板一致（`reasoning and (preserve_thinking or index0 > last_user_index)`），但**渲染仍由引擎自己那份模板负责**，与本项目 vLLM / SGLang 路线使用 checkpoint 模板渲染不能默认划等号。
- NVFP4 的数值布局：E2M1 权重 + block_size 16 的 UE4M3 组缩放 + 单个 tensor 级 F32 全局缩放（`kernels/include/sparkinfer/kernels/compressed_tensors.h:21-25`）；子格式按张量探测而非硬编码，源码注释明确区分了两类 checkpoint：compressed-tensors 把第 0–55 层放 NVFP4、56–63 层放 FP8，而 **ModelOpt 把全部 64 层放 NVFP4**（`runtime/src/models/qwen35.cpp:6555-6569`）——本机属于后者。
- 构建架构：`CMakeLists.txt:12` 默认 `CMAKE_CUDA_ARCHITECTURES "89;90;100;120"`；`sm_121` 因当前 CUDA 12.8 工具链不支持而被排除（`kernels/CMakeLists.txt:8`）。本机 `sm_120` 在支持范围内，SM120 的 FP4 kernel 以 `120a` 编译。

**第三方现场日志（issue #1086，2026-09-15，RTX 5090 / Ubuntu 26.04）**：该 issue 报告 `serve-dspark` 显存不足，但日志恰好完整记录了这份 checkpoint 被加载的过程，可作为"引擎确实认识它"的第一手旁证：

```text
[sparkinfer-server] arch Qwen3.8-27B dense hybrid, layers=64, experts=1 top-1, max_seq=262144
[sparkinfer-server] kv_cache: int8=1 slots=16/64 blocks=16392 resident=8.0 GiB
[sparkinfer-server] loading compressed-tensors checkpoint ...
[compressed-tensors] NVFP4 lm_head kept for wide packed decode (0.81 GB)
[compressed-tensors] loaded 64 layers, native NVFP4 prefill FFN 64/64, decode FFN NVFP4
[sparkinfer-server] prefix cache: on (32 entries, 8192 MiB host, ...)
[sparkinfer-server] vision tower ready: 27 blocks, out_hidden=5120
[sparkinfer-server] model ready: /models/qwen38-nvfp4
[dflash] YaRN: factor=32.0 orig_max=8192 ...
[dflash] malloc: out of memory
```

这段日志同时说明三件事：① 目标是按 `compressed-tensors` 路径加载的 ModelOpt NVFP4（`lm_head` 0.81 GB 保留、64/64 层原生 NVFP4）；② 视觉塔从该目录正常装载（27 blocks）；③ **草稿（`[dflash]`）在 `ctx 262144` 下会 OOM**——这正是 0.5.8 把 `serve-dspark` 默认上下文降到 131072 并让"装不下就启动失败"的原因（注意该用户当时用的是 `latest`，即早于 0.5.8 的版本）。

### 3.4 官方发布容器（本次通过镜像元数据实测，未拉取）

| 项 | 实测值 |
| --- | --- |
| 镜像 | `ghcr.io/gittensor-ai-lab/sparkinfer-qwen38` |
| 可用标签 | `0.5.5`、`0.5`、`0.5.6`、`0.5.7`、`0.5.8`、`0.5.9`、`0.5.10`、`latest`，以及 `sha-*` / `sha256-*` 标签 |
| 构建时间 / 版本 | 2026-09-17T20:56:55Z / `org.opencontainers.image.version = 0.5.10`，revision `ab936d96a3…` |
| 描述 | "SparkInfer serving Qwen3.8-27B-NVFP4 (**AR + image + video + DSpark speculative decode**)" |
| 压缩层合计 | 419,355,498 字节 ≈ **0.42 GB**（模型卡称镜像约 1 GB） |
| 基础镜像 | `nvidia/cuda:12.8.1-base-ubuntu24.04`（构建阶段为 `12.8.1-devel`）；`CUDA_VERSION=12.8.1` |
| 容器内运行时要求 | `NVIDIA_REQUIRE_CUDA=cuda>=12.8 ...`（枚举驱动分支最高到 565.x；本机 610.88 高于其下限） |
| 依赖 | 仅 `ffmpeg`（视频取帧必需）、`python3`、`huggingface_hub[cli]`、`python3-yaml`；**无 Python 推理栈** |
| 入口 | `ENTRYPOINT ["/opt/sparkinfer/gittensor-entrypoint.sh"]` → 转交 `/opt/sparkinfer/entrypoint.sh` → `sparkinfer_server` |
| 默认环境 | `MODEL_REPO=gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`、`MODEL_DIR=/models/qwen38-nvfp4`、`DRAFT_REPO=gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`、`DRAFT_DIR=/models/qwen38-dspark`、`MODEL_NAME=qwen38-nvfp4`、`HOST=0.0.0.0`、`PORT=8080`、`SPARKINFER_MAX_OUTPUT_TOKENS=16384` |
| 其它 | `VOLUME ["/models"]`、`EXPOSE 8080`、`HEALTHCHECK` 打 `/health`、无 `USER`（以 root 运行） |

镜像里的 `docker/gittensor-manifest.yaml`（仓库同源文件）把本模型描述为 "Qwen3.8-27B (NVFP4) chat with the DSpark drafter on sparkinfer v0.5.8's release container, one RTX 5090 per instance"，`min_vram_gb: 32`，`max_load_s: 90`，并记录 "30.1 GB peak at CTX 65536 with the drafter（30,835 MiB under 4 × 16K prompts）"。

### 3.5 WSL 支持情况

**官方文档没有任何 WSL / Windows 支持声明。** 对 `README.md`、`server/README.md`、`CHANGELOG.md`、`CONTRIBUTING.md`、`EVAL-TRUST.md`、`docs/*.md`、`docker/*`、`runtime/README.md`、`eval/README.md` 及 `.github/workflows/*` 检索 `wsl` / `Windows Subsystem` 均无命中；`Windows` 只出现在三处无关位置（README 里指向 NVIDIA "RTX Spark" Windows PC 新闻的链接、CHANGELOG 提到 GitHub 证明的 Windows bench 二进制、`docs/release-v0.4.4-draft.md` 里的 `sparkinfer-v0.4.4-windows-amd64-cuda13-sm120.zip` 产物名）。

**唯一一条真实的 WSL 记录**在 issue #713（closed）的一条验证备注里，作者披露自己的环境是 "RTX 5070 Ti 16GB, **WSL2**, CUDA 12.8, sm_120"，并提到 "compute-sanitizer unavailable under WSL/WDDM"。这是仓库内**仅有**的 WSL 证据：它说明有贡献者确实在 WSL2 + `sm_120` 下跑过这个引擎（但不是在 RTX 5090 上、也不是发布容器）。issue 全文检索 `WSL` 仅此一条。

因此 WSL 路线的可行性依据不是官方声明，而是三条间接证据：① 容器是标准 Linux + CUDA 12.8 基础镜像，只依赖 `--gpus` 与 nvidia runtime（该 runtime 本机已装并已注册，见 2.3）；② **本机已有同架构（`lmsysorg/sglang:qwen38-27b`，CUDA 13 基础镜像）在 WSL2 中成功运行的先例**；③ 上述 issue #713 的 WSL2 现场记录。这一条整体标记为"结构上可行、未实测"。

---

## 四、怎么把本机模型接到 SparkInfer 上

### 4.1 路线 A（推荐）：官方容器 + 本地权重挂载

镜像已预置 `MODEL_DIR` / `DRAFT_DIR` 两个默认路径，把本机目录挂到这两个路径即可，**不需要改镜像、不需要重新下载权重**：

| 本机（WSL 内）路径 | 容器内路径 | 挂载方式 |
| --- | --- | --- |
| `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` | `/models/qwen38-nvfp4` | 只读（`:ro`）——源码层面已确认可行：权重分片以 `O_RDONLY` 打开，模型目录旁不写任何缓存或索引（见 3.3） |
| `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4` | `/models/qwen38-dspark` | 只读（`:ro`），同上 |

**为什么这样就不会触发下载**：`docker/entrypoint.sh` 的 `fetch()` 逻辑是 `if [ ! -f "$2/config.json" ]` 才下载。两个本地目录都有 `config.json`，所以下载分支根本不会进入。

**带 DSpark 的启动命令（在 WSL 内以 root 执行；本次未执行，仅作为研究结论）**

```bash
docker run --rm --gpus all \
  -p 8192:8080 \
  -v /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090:/models/qwen38-nvfp4:ro \
  -v /home/kami/models/Qwen3.8-27B-DSpark-NVFP4:/models/qwen38-dspark:ro \
  -e CTX=131072 \
  -e SPARKINFER_SAMPLING_DEFAULTS=greedy \
  -e MODEL_NAME=Qwen3.8-27B-NVFP4-DSpark \
  ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10 serve-dspark
```

要点：

- **端口映射成 8192**：本项目现有全部启动器（vLLM / GGUF / SGLang）都固定在 8192，客户端 base_url 不用改；容器内部始终是 8080。
- **不加 `-d`、不加 `--restart`**：WSL 在最后一个会话结束时回收发行版，容器会被一起杀掉。既有 `scripts/sglang-dspark.sh:9-17` 已经把这个行为记录清楚了——前台运行就是保活，Ctrl-C 就是停止按钮。
- **`SPARKINFER_SAMPLING_DEFAULTS=greedy`**：默认值 `generation_config` 会让不带 `temperature` 的请求按 checkpoint 的 `temperature 1.0 / top_k 20 / top_p 0.95` 采样，而**DSpark 只对 greedy 请求生效**。要么设这个环境变量，要么每个请求显式带 `temperature: 0`。
- **不建议设 `SPARKINFER_NO_DOWNLOAD=1`**：原因见 4.3（实测会因 manifest 哈希不匹配而拒绝启动）。
- **与现有引擎互斥**：显存只够一个引擎，且端口同为 8192；启动前请确认 vLLM / SGLang 容器已停止。
- **按 digest 固定版本**：镜像标签与 CHANGELOG 已到 0.5.10，但 GitHub Releases 列表只到 0.5.6（见 7.7 第 7 条）；用 `0.5.10` 标签或直接按 manifest digest 固定，避免 `latest` 漂移。

**不带 DSpark（最大上下文）的对照启动**

```bash
docker run --rm --gpus all -p 8192:8080 \
  -v /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090:/models/qwen38-nvfp4:ro \
  -e CTX=262144 -e MODEL_NAME=Qwen3.8-27B-NVFP4 \
  ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10
```

（`serve` 是默认模式，AR 默认 `CTX=262144`。）

### 4.2 路线 B：源码构建（本机当前缺依赖）

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer && cd sparkinfer
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SERVER=ON && cmake --build build -j
./build/server/sparkinfer_server -m <weights> --tokenizer <weights>/tokenizer.json \
  --ctx 262144 --host 0.0.0.0 --port 8080
```

- 官方要求：**CUDA Toolkit 12.8+**，且 `BUILD_SERVER=ON` 需要 **rustc ≥ 1.79**（`tokenizers-cpp` 在 configure 阶段 clone 并编译；Dockerfile 显式把 `cargo`/`rustc` 软链到 `/usr/bin/`）。
- 本机差距：`cargo` / `rustc` **缺失**；`nvcc` 存在但不在 PATH（venv 内 CUDA 13.3.73）。`cmake` / `gcc` / `ninja` 具备。
- 结论：源码路线在**不安装 Rust 工具链**的前提下不可行（而安装属于环境改动，不在本次研究范围）。

### 4.3 离线运行的三道闸门（重要，含本次实测结论）

容器入口有两层脚本，离线行为由三个变量/文件决定：

| 机制 | 触发条件 | 行为 |
| --- | --- | --- |
| `fetch()` 跳过下载 | `$MODEL_DIR/config.json` 已存在 | 完全不发起下载（**挂载本地权重即可命中**） |
| `SPARKINFER_NO_DOWNLOAD=1` | 显式设置 | 明确拒绝下载：目录里没有 `config.json` 就打印"该挂什么"并 `exit 1` |
| manifest 哈希校验 | `SPARKINFER_NO_DOWNLOAD=1` **且** `/manifest.yaml` 存在（镜像已内置该文件） | 用 `/opt/sparkinfer/gittensor-entrypoint.sh` 对 manifest 中每个 artifact 做 sha256 校验，**不匹配就拒绝启动**（exit 1） |

**本次实测（决定性）**：我用与入口脚本**完全相同的目录哈希算法**（对非点文件按 `相对路径\0文件sha256\n` 排序拼接后再 sha256，跳过顶层点文件/点目录）计算了本机两个目录：

| 目录 | 磁盘文件数 | 计算得到的目录 sha256 | 镜像内置 manifest 期望值 | 结果 |
| --- | --- | --- | --- | --- |
| `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4` | 3 | `991a9c241b2d40f5…` | `506d2ebdc91de82b…` | **不匹配** |
| `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` | 15 | `9566495800b9d58a…` | `d4382c5ebc8c9ece…` | **不匹配** |

不匹配的可解释原因有两个，均不需要下载即可判断：
1. 本机目录只保留了推理必需文件，HF 修订里还有 `README.md`、`LICENSE`、`README.qwen-upstream.md`、`assets/*.png` 等**非点文件**会计入哈希；
2. manifest 里目标模型记录的是 `revision: 8e2c0cd25468ac5c1f85f621c2b8edb15ea1f03a`，而 HF `main` 当前 commit 是 `5b7a687f…`（草稿侧 manifest 记 `eba1ac5a…`，与 HF `main` 一致）。

**操作结论**：走"本地挂载 + 不设 `SPARKINFER_NO_DOWNLOAD`"这条路径，既不触发下载也不触发校验；若确实需要"拒绝联网"的强保证，应改为 `-e SPARKINFER_NO_DOWNLOAD=1 -e MANIFEST_PATH=/nonexistent`（入口脚本读 `MANIFEST_PATH`，指向不存在的文件即可跳过校验，同时保留"缺文件就报错、绝不下载"的行为）。**这两个做法都是研究结论，本次未执行。**

---

## 五、DSpark 在 HTTP 服务上的真实边界

### 5.1 什么时候生效

来自 `server/README.md` 与 0.5.7 CHANGELOG：DSpark **只**在"greedy + 纯文本 + 全服务只有这一个活跃请求"时启用；以下情形走普通自回归或中途交接：

- 带采样参数、penalties、`logit_bias`、logprobs 的请求；
- 图片 / 视频输入；
- 工具调用（`tools`）与 `response_format` 结构化输出；
- 命中前缀缓存（cached prefix）的请求；
- **有第二个请求到来时**：正在投机的那个请求会在某个已确认 token 处交接，然后以批处理方式继续。

另外 0.5.7 记录了一条物理限制：**投机会在下一个 attention 分档边界停止**（第一档是 512 token 上下文），之后由普通 decode 完成请求——这就是"371 token 提示、回答跨过 512 token"那一行只有 1.01× 的原因。可观测性靠 `/metrics` 的 `sparkinfer_speculative_runs_total`、`_tokens_total`、`_handoffs_total`、`_tier_stops_total`。

### 5.2 官方给的速度量级（需按口径阅读）

0.5.7 在 `SPARKINFER_DETERMINISTIC=1`、单请求、与"同进程无草稿"逐字节比对的前提下测得：

| 请求类型 | 相对无草稿的解码加速 |
| --- | --- |
| JSON | 3.45× |
| 代码 | 2.86× |
| 长请求（3,201 token 提示） | 2.54× |
| 数学 | 2.45× |
| 聊天 | 1.47× |
| 371 token 提示、回答跨 512 token | 1.01× |

仓库 README 另有按上下文统计的一张表（4k 4.01× / 16k 2.97× / 32k 2.63×）与门禁用例（16k 长文 1.472×）。**两组数字的语料不同，都应引用区间而不是单点**（README 自己就是这么要求的）。

HF 模型卡给的是 bench harness 口径：整体 97.1 → 264.8 tok/s（2.73×），代码类最高 420.2 tok/s（4.32×）；模型卡同时强调"Read these as their own harness, not as server numbers"，并写明该 harness 用的是几百 token 上下文、每类 1 条 prompt。

### 5.3 与 SGLang 路线的对照（本项目已实测的那条）

| 维度 | SparkInfer 容器（本文） | SGLang 镜像（本项目已跑通） |
| --- | --- | --- |
| DSpark 上下文 | `serve-dspark` 默认 `CTX=131072`；草稿自身注意力默认 16384 | 实测 163840（KV 池上限 166793 @0.90） |
| 投机并发 | 仅单活跃请求；第二个请求发生交接 | 单请求（`max-running-requests 1`） |
| 投机生效条件 | greedy、纯文本、无工具、无结构化输出、无缓存命中 | 由 `--speculative-*` 参数与接受阈值控制 |
| 启动体量 | 镜像约 1 GB（压缩层 0.42 GB），无 Python 栈 | 镜像 41.9 GB（已在盘上） |
| 需要下载 | 需要拉取镜像（本次未拉） | 已就绪 |
| 权重来源 | 本地挂载，无需下载 | 本地只读挂载 |

---

## 六、与既有结论的冲突（必须显式说明）

按 `docs/agents/domain.md` 的要求，输出与既有记录冲突时必须明确指出，不能静默覆盖。

**冲突点**：`result/qwen38-nvfp4-dspark-upgrade-research.md:264` 的结论是——

> "SparkInfer（…）：作者称其 DSpark 目前**只在 benchmark harness 中**，HTTP 服务器仍是纯自回归——'不是今天能 `curl` 的东西'。因此**不能**用它做带 DSpark 的 API 服务。"

**该结论已被上游推翻**。时间线（全部取自一手来源）：

| 日期 | 事件 | 来源 |
| --- | --- | --- |
| 2026-09-04 | 0.5.4：DSpark 在 bench 上补测 16K/32K | CHANGELOG.md |
| **2026-09-10** | **HF 目标模型卡最后修改**，其中写 "its bench harness; the HTTP server is autoregressive-only today"、"If you want speculation over HTTP today, serve with SGLang" | HF `.../raw/main/README.md` |
| **2026-09-14** | **0.5.7：`sparkinfer_server` 支持 DSpark（#1076），发布容器可用 `serve-dspark`** | CHANGELOG.md |
| 2026-09-16 | 0.5.8：`serve-dspark` 默认 `--ctx 131072`；草稿装不下就启动失败；修 8 个真实 agent 会话命中的问题 | CHANGELOG.md |
| 2026-09-17 | 0.5.9：入口支持 `SPARKINFER_MODE=serve-dspark` 与 pre-staged 权重哈希校验；0.5.10 发布 | CHANGELOG.md、镜像 labels |

也就是说：**模型卡（09-10）与既有研究（09-13/09-16）都比 0.5.7（09-14）早**，所以两处写"HTTP 上不能用 DSpark"在当时是对的，现在不再成立。本仓库现有 README（4.8 节）选择 SGLang 作为 DSpark 的 API 路线，**这个选择本身仍然成立**（它已被本机实测验证），只是"SparkInfer 不能通过 API 用 DSpark"这条排除理由已失效——SparkInfer 现在是一个**新增的、未实测的**候选路线。

**建议的处置**（本次研究不修改任何文件，仅提出）：在 `result/qwen38-nvfp4-dspark-upgrade-research.md` 的第 264 行与第 472–473 行的资料清单处追加一条勘误注记，指向本文档与 sparkinfer 0.5.7 CHANGELOG。

---

## 七、风险与注意事项

### 7.1 官方自述的正确性缺陷（最高优先级）

`server/README.md` 以显著位置写明一个**默认开启**的已知缺陷：

> 在 int8 KV 下（服务器在 `--ctx ≥ 4096` 时会启用 int8 KV）、提示词达到 **2048 token 或更长**时，GQA-fused MMA 预填充注意力与逐 token 参考实现**严重不一致且不可复现**。

官方给出的 KL 实测（16 个教师强制位置，Qwen3.6-35B-A3B / RTX 5090）：

| prefix | 1500 | 2000 | **2100** | 3000 | 4000 |
| --- | --- | --- | --- | --- | --- |
| 默认（fused） | 0.00043 | 0.00022 | **0.18672** | 0.20657 | 0.23978 |
| `SPARKINFER_PREFILL_ATTN_GQA_RQH=1` | ~0.0001 | ~0.0001 | ~0.0001 | ~0.0001 | 0.00008 |

官方解释：该分支默认打开而不是静默关掉，是因为关掉会改变被评测挂钩的长上下文预填充吞吐。**对长提示词场景，这意味着默认路径可能给出与参考实现明显不同的结果。** 本项目主打的正是长上下文（现配置 200k），因此实测前必须把这一条列为第一验收项；可用的规避手段是设置 `SPARKINFER_PREFILL_ATTN_GQA_RQH=1`（代价是长上下文预填充变慢）。

这条缺陷有对应的现场记录：issue **#976**（2026-09-06 建、同日关闭）标题为 "Long-context output degrades and becomes nondeterministic from ~4k tokens (vLLM on the same checkpoint does not)"，在**发布镜像 0.5.5**、`--ctx 40960`、greedy、无前缀缓存、无投机的条件下实测：约 1,955 token 提示仍正确稳定（`391` ×3），从 4,076 token 起开始 flaky，7,643 token 起"稳定但答错"，24,869 与 31,263 token 三次给三个不同答案；而 **vLLM v0.28 在同一 checkpoint、同一批提示、同一台机器上稳定答对**（30,498 token 与 32,044 token 两例均为 `391`）。也就是说长提示词下的正确性与可复现性问题是**引擎侧、且已在更早版本上被真实观测到**的。该 issue 已关闭但**没有留下任何评论说明修复提交**，因此"0.5.10 是否已彻底解决"仍需以实测回答（见 7.7）。

同族的另一条（0.5.8 已默认修好，但值得知道）：`SPARKINFER_SPARSE_GQA6=1` 会让长上下文 decode 只看第一个 KV 块与最近 4096 token，中间内容对模型不可见（0.5.8 起默认关闭，精确注意力为默认）。0.5.8 为此给出的现场例子是：44,000 token 提示里 20,000 token 前的一行 `ERROR`，此前答"没有含 ERROR 的行"，修复后能正确引用该行。

### 7.2 采样默认值会让 DSpark 悄悄失效

`SPARKINFER_SAMPLING_DEFAULTS` 默认为 `generation_config`，而本机 `generation_config.json` 是 `temperature 1.0 / top_k 20 / top_p 0.95`——**不带 `temperature` 的请求属于采样请求，DSpark 不生效**。要观察到投机行为，需 `SPARKINFER_SAMPLING_DEFAULTS=greedy` 或请求显式 `temperature: 0`，并用 `/metrics` 的 `sparkinfer_speculative_runs_total` 验证（README 明确建议这样做）。

### 7.3 并发：这条路线在本项目当前用法下可能更慢

模型卡实测（同一权重、同一卡、流式 API、客户端计时、`--ctx 40960`、128 token/请求）：

| 并发请求数 | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| SparkInfer 聚合 tok/s | 88.9 | 162.1 | 249.6 | **344.3** | **345.1** | 318.2 |
| 每请求 tok/s | 88.9 | 81.1 | 63.0 | 43.7 | 22.0 | 13.4 |

同页给出的对照：vLLM 0.28 在 2/4/8/16 并发下为 134.5 / 266.6 / 545.8 / 546.6 tok/s——**8–16 并发时 SparkInfer 落后约 1.6×**。本项目现有 vLLM 配置是 `FULL_MAX_NUM_SEQS=16`，因此"换引擎"不能只看单流吞吐。模型卡的建议是：单流或少数流用 SparkInfer，多用户就用 vLLM。

### 7.4 上下文取舍

- SparkInfer AR：262144（27.9 GB）可用，360000 是上限（31.2 GB，decode 约 74 tok/s）。
- SparkInfer + DSpark：默认 131072，**低于**本项目 vLLM 实测的 200000。
- 且 DSpark 草稿自身只关注 16384 token 上下文（`SPARKINFER_DSPARK_MAX_CTX`）；0.5.8 修掉了"提示长于草稿上下文就报 `speculative decode failed`"的问题。
- 参考：本项目 SGLang 路线实测带 DSpark 可用 163840。

### 7.5 显存与启动

- 启动前需要约 30 GB 空闲显存（manifest：CTX 65536 + 草稿峰值 30.1 GB；本机空闲时已占 2160 MiB）。
- WSL 会缓存显存，加配置前需 `wsl --shutdown` 释放（本项目 README 问题 1 的处理方式）。
- 镜像 manifest 给出 `max_load_s: 90`，并注明"v5 热启动 11–17 s，草稿是第二次加载"。

### 7.6 WSL 特有的运行方式

- 前台 `docker run`，不要 `-d` + `restart`：WSL 在最后一个会话结束时终止发行版（既有 `scripts/sglang-dspark.sh` 文件头已记录日志中的 `InitTerminateInstanceInternal ... reboot(RB_POWER_OFF)`）。
- 本机 `kami` 不在 docker 组，需 root（`wsl -d Ubuntu -u root`）。
- 容器内部绑 `0.0.0.0`，局域网访问仍需 Windows 侧 `portproxy` + 防火墙（与本项目 5.0 节机制相同，但现有 bat 脚本启动的是 vLLM，不能直接复用）。
- 端口冲突：容器内 8080，请映射到本项目统一使用的 8192，并与 vLLM / GGUF / SGLang 启动器互斥。

### 7.7 待确认清单（本次未实测，不应视为已知）

1. **0.5.10 与模型卡所测 v0.5.5 的性能差异**：模型卡的数字来自 v0.5.5（AR 92.9 tok/s、int8 KV 默认），README 的自动刷新表来自当前 main，两者语境不同，不能混引。
2. **镜像内置 manifest 的 `revision 8e2c0cd2` 与本机 `main` 的关系**：本机权重文件与 `main` 同名同尺寸；manifest 期望的目录哈希与本地目录不一致（原因已定位为缺非必需文件与可能的修订差异），但"本机权重是否 100% 等于 manifest 所指修订"未验证。
3. **int8 KV 的 ≥2048-token 预填充缺陷在 0.5.10 上是否仍存在**：`server/README.md`（当前 main）仍将其列为 "known defect, left ON by default"，故按存在处理；issue #976 记录的 4k+ 长提示正确性/复现性问题在关闭时**没有留下修复说明**，是否已彻底解决必须实测（见 7.1 与第八节第 6 项）。
4. **WSL2 下的实际启动、加载耗时与显存占用**：全部未实测（本次不下载镜像、不启动容器）。
5. **图像/视频输入在本机的实际可用性**：容器内置 ffmpeg、本机目录含视觉塔与 preprocessor 配置，但未实测。
6. **chat template 的渲染差异**：sparkinfer 使用**编译进二进制**的模板而非 checkpoint 的 `chat_template.jinja`。0.5.9 说明其规则与 checkpoint 模板一致，但同一提示在本项目 vLLM / SGLang 路线与 SparkInfer 下是否逐 token 等价，未验证——这会直接影响"同提示同采样下输出可对比"这一前提。
7. **GitHub Releases 与镜像标签不同步**：本次观测到镜像标签与 CHANGELOG 已到 0.5.10（2026-09-17），而 GitHub Releases 列表最新只到 0.5.6（2026-09-14）。因此**不要用 `latest` 作为可复现基准，改按 digest 固定**（本条为观测事实，差异原因未追查）。

---

## 八、后续实测（若决定推进）的验收清单

> 以下为研究结论形式的清单，本次**未执行任何一项**。

前置（零成本，可先做）：

1. 确认本地权重文件名无多余后缀（历史上出现过 `model-XXXX.safetensors_.safetensors` 的命名陷阱，见本仓库既有研究附录 A.7）；核对 `crc32.txt` 两行与实际 CRC32。
2. 确认引擎互斥：`docker ps` 无 vLLM / SGLang 容器在跑；端口 8192 空闲。
3. 关闭 Windows 侧占显存的程序，必要时 `wsl --shutdown` 释放显存后再进入 WSL。

实测阶段（需要拉取镜像，本次未做）：

4. 拉取并固定镜像摘要：用 `0.5.10`（或按 digest 固定）而不是 `latest`，避免"今天能跑、明天变化"。
5. 先跑**无 DSpark** 的 `serve`，`-e CTX=262144`，确认：`/health` 返回 ok、`/v1/info` 报出 `max_context`、日志无下载动作（证明走的是挂载的本地权重）。
6. 长提示词正确性专项（最高优先级）：按 7.1 与 issue #976 的复现方式（自然散文系统提示 + 一个有唯一答案的问题、`temperature=0`、`enable_thinking=false`），在 int8 KV 默认路径与 `SPARKINFER_PREFILL_ATTN_GQA_RQH=1` 下各测一次 ≥4k token 提示，确认答案正确且多次运行可复现（`SPARKINFER_DETERMINISTIC=1` 可作对照）。
7. 切换到 `serve-dspark`，请求带 `temperature: 0`、不带 tools/图片，然后查 `/metrics` 的 `sparkinfer_speculative_runs_total` 与 `_tokens_total`，确认投机**确实生效**而不是静默走了 AR。
8. 与既有两条基线做同口径对比：vLLM 200k（现日常路线）、SGLang + DSpark 163840（现加速路线）——同一批 prompt、同一输出长度、同一采样设置，记录 decode tok/s、TTFT、显存峰值。
9. 并发验证：按 8 与 16 并发复测，确认在**本项目实际使用形态**（多客户端 / 16 并发）下是否真的更好；若并发场景更多，按 7.3 的结论，vLLM 仍可能是更优解。
10. 终止与重入：Ctrl-C 后确认容器与 WSL 会话干净退出（避免残留容器占住显存）。

---

## 九、证据来源

**SparkInfer 仓库（一手）**

- README：<https://github.com/gittensor-ai-lab/sparkinfer>、<https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/README.md>
- 服务端文档：<https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/server/README.md>（DSpark 加载、`SPARKINFER_DSPARK_MAX_CTX`、已知 GQA 缺陷、全套环境变量表）
- CHANGELOG（0.5.4 / 0.5.7 / 0.5.8 / 0.5.9 / 0.5.10）：<https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/CHANGELOG.md>
- 容器入口脚本：<https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/docker/entrypoint.sh>
- 计算池入口（哈希校验）：<https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/docker/gittensor-entrypoint.sh>
- 计算池 manifest：<https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/docker/gittensor-manifest.yaml>
- Dockerfile：<https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/docker/Dockerfile>
- 仓库元数据与提交时间线：`https://api.github.com/repos/gittensor-ai-lab/sparkinfer`、`.../commits`（观测时 `main` 最新提交 `4962fc0a7`，2026-09-18T06:23:01Z）
- 源码（按需读取 `raw.githubusercontent.com`，未 clone）：
  - `server/src/model_engine.cpp`（`-m` 目录分发、compressed-tensors 选择、纯 safetensors 不支持）
  - `server/src/chat_tokenizer.cpp`（模板编译进二进制）
  - `server/src/sparkinfer_server.cpp`（`--draft-model` / 草稿加载失败即 `return 1`）
  - `runtime/src/safetensors.cpp`（分片 `O_RDONLY`、缺分片跳过而非致命）
  - `runtime/src/models/qwen35.cpp`、`runtime/src/models/dflash_draft.cpp`（NVFP4/FP8 分格式探测、草稿布局与 `DSPARK_MAX_CTX`）
  - `runtime/examples/qwen38_hf_config.h`（`text_config` 必需）
  - `kernels/include/sparkinfer/kernels/compressed_tensors.h`、`kernels/CMakeLists.txt`、`CMakeLists.txt`
  - `eval/pr_qwen38_bot.py`（用**本地目录**加载 NVFP4 checkpoint 的评测脚本，环境变量 `QWEN38_MODEL_DIR`）
- 相关 issue：
  - #976 "Long-context output degrades and becomes nondeterministic from ~4k tokens (vLLM on the same checkpoint does not)"（2026-09-06 建并关闭，无评论）
  - #1086 "serve-dspark demo [sparkinfer-server] device out of memory"（2026-09-15 建并关闭；正文含完整加载日志）
  - #713（closed，含 WSL2 + `sm_120` 环境披露）
  - #983（`/v1/models` 未声明 video 模态的已知问题）
  - 开放 issue 仅 3 个：#1111、#1107（性能）、#1093（GGUF Vision / mmproj）——**没有任何开放 issue 涉及 NVFP4 加载失败或 WSL**

**镜像（GHCR 元数据，未拉取层）**

- Tags：`https://ghcr.io/v2/gittensor-ai-lab/sparkinfer-qwen38/tags/list`
- Manifest 与 config blob（基础镜像、CUDA 版本、入口、默认环境、labels）

**权重与模型卡（HF 一手）**

- 目标：<https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090>（README、config.json、hf_quant_config.json、generation_config.json、文件清单 API）
- 草稿：<https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4>（README、config.json、hf_quant_config.json）
- 对照：<https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4>（`compressed-tensors` 混合精度，与本 checkpoint 的 `modelopt` NVFP4 不是同一格式）

**本仓库既有文档**

- `README.md`（4.8 节：SGLang + DSpark 现状，本机实测 1.80×）
- `result/qwen38-nvfp4-dspark-upgrade-research.md`（第六节冲突点所在；附录 A.7 记录了 2026-09-16 的文件落位与 sha256 校验）
- `result/wsl-docker-image-software-versions.md`（本机 WSL 与 SGLang 镜像的软件版本清单）
- `scripts/sglang-dspark.sh`（WSL 容器生命周期与 docker 权限的既有结论）

**本次实测命令（只读）**

```text
wsl -l -v                                      # 发行版清单（初始 Stopped）
cat /etc/os-release / uname -r / free -g / df -h
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
docker --version / docker images / docker info / docker ps -a / cat /etc/docker/daemon.json
ls -la /home/kami/models/...                   # 两个 checkpoint 目录
python3 -c "…json.load…"                       # config.json / index.json 关键字段
sha256 目录哈希（复刻 gittensor-entrypoint.sh 的算法）  # 与 manifest 期望值比对
curl -I https://ghcr.io/v2/ 、HF API、GitHub      # 连通性
curl ghcr token/manifest/config（元数据）          # 镜像信息
```

---

## 十、本次研究未做的事（边界声明）

- 未修改、新增或删除本仓库的任何代码、脚本、配置、文档（本文档除外）。
- 未修改 WSL 内的模型、环境、venv、Docker 配置、`.wslconfig`。
- 未拉取任何镜像、未下载任何模型或加速器权重、未构建任何二进制、未克隆任何仓库（源码通过 `raw.githubusercontent.com` 按需读取）。
- 未启动任何容器（现有 `qwen38-sglang` 容器保持 Exited 状态不变）。
- WSL 发行版由"Stopped"被拉起一次用于只读探测；未安装、未升级、未卸载任何包。