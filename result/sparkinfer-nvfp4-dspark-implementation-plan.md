# 执行计划：在 SparkInfer 上复用现有 NVFP4 权重与 DSpark 草稿（WSL2，零模型下载）

> 计划日期：2026-09-18
> 依据：`result/sparkinfer-nvfp4-dspark-research.md`（同日只读调研，含本机实测数据与源码级证据）
> 状态：**计划文档，未执行**。本文件不修改任何代码、模型、环境；执行阶段才产生改动。
> 与既有计划的关系：`result/qwen38-nvfp4-dspark-upgrade-plan.md` 处理的是 vLLM / SGLang 路线；本计划新增**第三条引擎路线（SparkInfer）**，不改动前两条。

---

## 一、目标与范围

### 1.1 目标

1. 在 **WSL2** 内把 SparkInfer 跑起来，对外提供与现有服务同形的 OpenAI 兼容 API。
2. **复用本机已有的两份权重**，不做任何下载、转换、重新量化：
   - 目标：`/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`（final build，2 分片，17.92 GB）
   - 草稿：`/home/kami/models/Qwen3.8-27B-DSpark-NVFP4`（单文件，1.40 GB）
3. DSpark 走 SparkInfer 自带的 `serve-dspark`，而不是 SGLang（后者作为既有已验证路线保留）。
4. 最终把这条路线**固化为本项目的启动器 + 文档 + 测试**，与现有 `direct.sh` / `start-api-server-*.bat` 并列。

### 1.2 唯一允许的网络动作

| 需要的东西 | 状态 | 是否需要下载 |
| --- | --- | --- |
| 目标模型权重 | 已在 `/home/kami/models` | **不需要**（挂载复用） |
| DSpark 草稿权重 | 已在 `/home/kami/models` | **不需要**（挂载复用） |
| vLLM / SGLang 镜像 | 已在本地 | 不需要 |
| **SparkInfer 引擎镜像** | 本地**不存在** | **需要拉取**（压缩层 0.42 GB，模型卡称镜像约 1 GB） |

这是全计划中**唯一**的下载动作，且下载的是**引擎镜像本身**，不是模型、不是加速器权重。若要求"连镜像也不拉"，则本计划只能停在阶段 0 与静态评审（见 5.1 的"零下载分支"）。

### 1.3 明确不做

- 不重新量化、不转 GGUF、不转 bf16、不修改 `config.json` / `hf_quant_config.json` / safetensors。
- 不安装 Rust 工具链，不走源码构建路线（本机缺 `cargo`/`rustc`）。
- 不修改 `.wslconfig`、不动 Windows 驱动、不改动现有 `direct.sh` / `serve.sh` / `fullcontext.sh` / `sglang-dspark.sh` 及其 `.bat`。
- 不删除 SGLang 镜像（既有 SGLang + DSpark 路线保持可用）。
- 不设 `SPARKINFER_NO_DOWNLOAD=1`（理由见 3.3）。

---

## 二、执行前状态快照（2026-09-18 实测，来自研究报告）

| 项 | 值 | 对计划的影响 |
| --- | --- | --- |
| WSL | Ubuntu 26.04、`6.18.33.2-microsoft-standard-WSL2`、systemd | 容器可正常前台运行 |
| GPU | RTX 5090 / 32607 MiB / 驱动 610.88 / `sm_120` | 镜像要求 `cuda>=12.8`、仅 `sm_120`，满足 |
| 空闲显存 | 2160 MiB 已占（约 30 GB 可用） | AR `--ctx 262144` 需 27.9 GB，需先释放占用 |
| Docker | 29.8.1，runtime `nvidia` 已注册，nvidia-container-toolkit 已装 | `--gpus all` 无需额外配置 |
| 已有镜像 | `lmsysorg/sglang:qwen38-27b`（41.9 GB） | 与 SparkInfer 镜像互不影响 |
| 已有容器 | `qwen38-sglang`（Exited） | 不冲突；新容器用独立名字 |
| 磁盘 | `/` 可用 868 GB | 拉镜像无压力 |
| 端口约定 | 现有全部启动器用 8192 | SparkInfer 容器内 8080 → 映射 8192:8080 |
| 工具链 | `cmake` 4.2.3 / `gcc` 15.2.0 / `ninja` 有；`cargo`、`rustc` **无** | 排除源码构建路线 |
| 权重指纹 | 目标 2387 张量、无 `mtp`、2 分片、`total_size 17,915,815,528` | 与 HF `main` final build 一致，可直接复用 |

**复用成立的技术前提（源码级已验证）**：`-m <目录>` 要求目录有 `config.json` 且带 `text_config`，含 `quantization_config` 则走 ModelOpt/compressed-tensors NVFP4 加载路径；草稿要求 `dir/config.json` + 单文件 `dir/model.safetensors`。本机两个目录**全部满足**，因此无需任何预处理。

---

## 三、设计决策（先定死，再动手）

### 3.1 走官方容器，不走源码构建

- 决策：`ghcr.io/gittensor-ai-lab/sparkinfer-qwen38`。
- 理由：容器开箱即用（CUDA 12.8.1 基础镜像 + 预编译二进制），本机缺 `rustc ≥ 1.79`，源码路线不可行且属于环境改动。
- 版本：固定 **`0.5.10`** 或直接按 digest 固定；**不用 `latest`**（GitHub Releases 与镜像标签不同步，`latest` 会漂移）。
  - 截至 2026-09-18 查询：GHCR 上最新语义版本为 **`0.5.10`**，构建于 2026-09-17T20:56:55Z，revision `ab936d96a`；`latest` 与 `0.5` 都指向与它相同的 digest，即没有更新的镜像。
  - 精确固定串（镜像索引 digest）：`ghcr.io/gittensor-ai-lab/sparkinfer-qwen38@sha256:d519d6ed995cf24f4a00c224733082166cfcb56ebaefe45ff94f80e9f518bbe1`（其中 amd64 平台 manifest 为 `sha256:ce8b1bb3471191d7bdeaa29d1a8a46d871bba4189661273b54fd69aae3c402ed`）。
  - 该仓库只有 `sparkinfer-qwen38` 一个公开镜像名（`sparkinfer` / `sparkinfer-server` / `sparkinfer-gguf` 等均不可用）。

### 3.2 本地只读挂载，权重零下载

| 本机路径（WSL 内） | 容器内路径 | 挂载 |
| --- | --- | --- |
| `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` | `/models/qwen38-nvfp4` | `:ro` |
| `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4` | `/models/qwen38-dspark` | `:ro` |

- 只读安全：分片以 `O_RDONLY` 打开，模型目录旁不写缓存/索引（仅两个默认关闭的调试 dump 变量会写 `/tmp` 或指定目录）。
- 容器入口 `fetch()` 的判据是 `$MODEL_DIR/config.json` 是否存在——两个目录都有，因此**下载分支不可达**。

### 3.3 不设 `SPARKINFER_NO_DOWNLOAD=1`，改用"三道围栏"

- 实测事实：设了它且镜像内置 `/manifest.yaml` 存在时，入口会对挂载目录做 sha256 校验；本机两目录的目录哈希与 manifest 期望值**均不匹配**（缺 `README.md`/`assets` 等非必需文件，且 manifest 记录的目标修订为 `8e2c0cd2…`），结果是**拒绝启动**。
- 围栏设计（按强度递增）：
  1. **启动器 preflight 强制检查**：两个目录存在、目标含 `config.json` + `tokenizer.json`、草稿含 `model.safetensors`；任一缺失直接拒绝并说明原因 → 下载分支不可达。
  2. **`-e HF_HUB_OFFLINE=1`**：即便将来目录不完整，`hf download` 也只会报错而不会出网（依据 huggingface_hub 的通用行为，**本项目未实测**，阶段 3 用日志确认）。
  3. **日志判据**：容器启动日志若出现 `[sparkinfer] downloading …` 即视为**验收失败**并立即 Ctrl-C。
- 备选（若要"硬拒绝"语义）：`-e SPARKINFER_NO_DOWNLOAD=1 -e MANIFEST_PATH=/nonexistent`——保留"缺文件就报错、绝不下载"，同时跳过 manifest 哈希门。

### 3.4 端口与生命周期

- 端口：容器内 8080 → 宿主 `-p 8192:8080`，与现有客户端 base_url 保持一致。
- 生命周期：**前台 `docker run`，不 `-d`、不 `--restart`、不 `--rm`**。WSL 在最后一个会话结束时回收发行版，后台容器必被杀；前台窗口就是保活，Ctrl-C 就是停止键，容器保留以便 `docker logs` 回看。
- 互斥：与 vLLM / GGUF / SGLang 三条路线**不可同时启动**（端口相同、显存只够一个引擎）。

### 3.5 上下文与采样的取值策略

| 场景 | `CTX` | 采样默认 | 说明 |
| --- | --- | --- | --- |
| 阶段 3 冒烟（AR） | `32768` | checkpoint 默认 | 先小 KV 池，快速确认加载与接口 |
| 阶段 3 基线（AR） | `262144` | checkpoint 默认 | 原生窗口；需约 30 GB 空闲显存 |
| 阶段 5 DSpark | `131072`（`serve-dspark` 默认） | `SPARKINFER_SAMPLING_DEFAULTS=greedy` | 便于验证投机生效 |
| 阶段 7 固化默认 | 见 6.2 决策点 | 与 vLLM 线路对齐（`generation_config`） | 客户端显式传 `temperature: 0` 才走 DSpark |

- 采样默认值走 `generation_config` 时与 checkpoint 的 `temperature 1.0 / top_k 20 / top_p 0.95` 一致——**正好与本项目 vLLM 线路的服务端默认采样相同**，因此固化时保持默认、把"DSpark 需 `temperature: 0`"写进文档，胜过把整个服务改成 greedy。
- 草稿自身注意力窗口默认 16384（`SPARKINFER_DSPARK_MAX_CTX`，上限为 `CTX`）；阶段 6 可试提高，但需记录显存代价。

### 3.6 正确性优先于速度

`server/README.md` 自述 int8 KV 下 ≥2048 token 的 GQA 预填充注意力存在默认开启的缺陷，issue #976 还记录过 ≥4k token 长提示答错且不可复现的现场。因此：

- 阶段 4 是**强制门**：默认路径 vs `SPARKINFER_PREFILL_ATTN_GQA_RQH=1` 的 A/B，必须先过门再谈性能。
- 若默认路径不可靠，启动器默认带上 `SPARKINFER_PREFILL_ATTN_GQA_RQH=1`（代价是长上下文预填充变慢），并在 README 说明理由。

---

## 四、阶段总览

| 阶段 | 内容 | 需要下载 | 需要 GPU | 产物 / 通过判据 |
| --- | --- | --- | --- | --- |
| 0 | 前置检查与状态冻结 | 否 | 否（仅查询） | 前置检查全绿；记录 digest 与权重指纹 |
| 1 | 拉取并固定镜像 | **是（唯一）** | 否 | 镜像存在，digest 已记录 |
| 2 | `--dry-run` 验证 argv | 否 | 否 | 打印的命令含正确的挂载/端口/模式 |
| 3 | AR 基线启动 + 冒烟 | 否 | 是 | `/health` ok、`/v1/info` 有值、日志无 `downloading` |
| 4 | **长上下文正确性门** | 否 | 是 | 4k/16k 提示答案正确且 3 次可复现；RQH A/B 有结论 |
| 5 | DSpark 启动与生效验证 | 否 | 是 | `/metrics` 投机计数 > 0；抽检输出与 AR 一致 |
| 6 | 性能与并发对比 | 否 | 是 | 与 vLLM 200k、SGLang 163840 同口径成表 |
| 7 | 固化为启动器 + 文档 + 测试 | 否 | 否 | 脚本/文档/测试落地，`tests/` 全绿 |
| 8 | 回滚与收尾 | 否 | 否 | 环境回到阶段 0 状态可复现 |

---

## 五、详细步骤

### 阶段 0 · 前置检查与状态冻结（零成本，先做）

```bash
# 全部在 WSL 内执行（docker 需要 root）
wsl -d Ubuntu -u root -- bash -lc '
  echo "== 端口占用 =="; ss -ltnp | grep -E ":(8192|8080)" || echo "8192/8080 空闲"
  echo "== 现有容器 =="; docker ps -a --format "{{.Names}}\t{{.Status}}"
  echo "== 显存 ==";     nvidia-smi --query-gpu=memory.total,memory.used --format=csv
  echo "== 磁盘 ==";     df -h / | tail -1
  echo "== 权重就位 =="
  for f in /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090/config.json \
           /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090/tokenizer.json \
           /home/kami/models/Qwen3.8-27B-DSpark-NVFP4/config.json \
           /home/kami/models/Qwen3.8-27B-DSpark-NVFP4/model.safetensors; do
    [ -f "$f" ] && echo "OK   $f" || echo "MISS $f"
  done
'
```

**通过判据**：8192 空闲、无其它引擎容器在跑、显存占用接近空闲（约 2 GB）、四项权重文件全 OK。
**失败处置**：有容器在跑 → 先停；显存被占 → 关闭 Windows 侧占 GPU 的程序，必要时 `wsl --shutdown` 后重进 WSL。

**同时冻结指纹**（用于事后确认"确实是同一份权重"）：

```bash
python3 -c "
import json;d=json.load(open('/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090/model.safetensors.index.json'))
print('tensors',len(d['weight_map']),'mtp',sum('mtp' in k.lower() for k in d['weight_map']),'size',d['metadata']['total_size'])"
# 期望：tensors 2387 mtp 0 size 17915815528
```

**零下载分支**：若决定"连引擎镜像也不拉"，到此为止；后续阶段全部标记为"未执行"，本计划转为待批状态。

### 阶段 1 · 拉取并固定镜像（唯一网络动作）

```bash
docker pull ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10
docker image inspect ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10 \
  --format '{{index .RepoDigests 0}} size={{.Size}}'
```

**通过判据**：`RepoDigests` 输出形如 `ghcr.io/...@sha256:…`；`size` 约 4.2e8 量级（压缩层 0.42 GB，解包后约 1 GB）。
**记录**：把 digest 写入 `result/` 的实测结果文档，后续启动器一律用 digest 或 `0.5.10`，禁用 `latest`。
**注意**：这一步拉的是**引擎镜像**，不含任何模型权重（权重由阶段 2 的挂载提供）。

### 阶段 2 · `--dry-run` 验证 argv（不启动容器）

本阶段依赖阶段 7 产出的启动器；若尚未写脚本，可先用等价的手工命令自检一次（打印而非执行）：

```bash
cat <<'EOF'
docker run --rm --gpus all -p 8192:8080 \
  -v /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090:/models/qwen38-nvfp4:ro \
  -v /home/kami/models/Qwen3.8-27B-DSpark-NVFP4:/models/qwen38-dspark:ro \
  -e HF_HUB_OFFLINE=1 \
  -e CTX=131072 \
  -e SPARKINFER_SAMPLING_DEFAULTS=greedy \
  -e MODEL_NAME=Qwen3.8-27B-NVFP4-DSpark \
  ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10 serve-dspark
EOF
```

**通过判据**：命令中同时出现——两个 `:ro` 挂载到正确容器路径、`-p 8192:8080`、`serve-dspark`、无 `SPARKINFER_NO_DOWNLOAD`。

### 阶段 3 · AR 基线启动 + 冒烟（先不带 DSpark）

```bash
docker run --rm --gpus all -p 8192:8080 \
  -v /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090:/models/qwen38-nvfp4:ro \
  -e HF_HUB_OFFLINE=1 -e CTX=262144 \
  -e MODEL_NAME=Qwen3.8-27B-NVFP4 \
  ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10
```

冒烟（另开一个 WSL 会话或从 Windows 侧）：

```bash
curl -s http://127.0.0.1:8192/health
curl -s http://127.0.0.1:8192/v1/info
curl -s http://127.0.0.1:8192/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3.8-27B-NVFP4",
  "messages":[{"role":"user","content":"What is 17 × 23? Reply with only the number."}],
  "temperature":0,"max_tokens":32,
  "chat_template_kwargs":{"enable_thinking":false}}'
```

**通过判据**（四条全中才算过）：
1. `/health` 返回 `{"status":"ok"}`；
2. `/v1/info` 报出与 `CTX` 相符的 `max_context`；
3. 启动日志里**没有** `[sparkinfer] downloading …` 这一行（证明走的是挂载权重）；
4. 冒烟返回 `391`。

**失败处置**：显存不足 → 先降到 `CTX=32768` 确认能起，再逐级加；`config.json` 报错 → 回到阶段 0 检查权重。

### 阶段 4 · 长上下文正确性门（强制，未过不得继续）

三组实验，全部 `temperature: 0`、`enable_thinking: false`、每个条件跑 3 次：

| 实验 | 提示构造 | 期望 |
| --- | --- | --- |
| A 短提示对照 | 约 30 token | `391` ×3 |
| B 中长提示（默认路径） | 约 4,000 token 自然散文 + 同一问题 | 答案正确且 3 次一致 |
| C 长提示（默认路径） | 约 16,000 token 自然散文，在靠前位置（约 8,000 token 之前）埋一行 `ERROR`，末尾提同一问题 | 能正确引用该行 |
| D 与 B 同提示，改 `-e SPARKINFER_PREFILL_ATTN_GQA_RQH=1` | 同上 | 答案正确且 3 次一致 |

**通过判据**：B/C 与 D 的结果一致且正确。
**可选加强**：上游 0.5.8 的复现场景是"44,000 token 提示 + 20,000 token 之前的一行 `ERROR`"；AR 档 `CTX=262144` 容得下，可作为 A/B 的加强用例。
**失败处置**：
- B/C 不可靠、D 正确 → 启动器默认带 `SPARKINFER_PREFILL_ATTN_GQA_RQH=1`，并在 README 标注"长上下文正确性优先，预填充变慢"。
- B/C、D 都不可靠 → **停止固化**，把结论写进 `result/` 实测结果文档，SparkInfer 路线降级为"仅短上下文可用"，默认路线仍为 vLLM。
- 对照参考：同提示在本项目 vLLM 路线下必须稳定答对（研究报告里 vLLM 在同权重的对照中表现稳定）。

### 阶段 5 · DSpark 启动与生效验证

```bash
docker run --rm --gpus all -p 8192:8080 \
  -v /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090:/models/qwen38-nvfp4:ro \
  -v /home/kami/models/Qwen3.8-27B-DSpark-NVFP4:/models/qwen38-dspark:ro \
  -e HF_HUB_OFFLINE=1 -e CTX=131072 \
  -e SPARKINFER_SAMPLING_DEFAULTS=greedy \
  -e MODEL_NAME=Qwen3.8-27B-NVFP4-DSpark \
  ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10 serve-dspark
```

**通过判据**（缺一不可）：
1. 启动日志出现 `serving … DSpark (/models/qwen38-dspark)`，且**没有** `speculative decoding off`（草稿加载失败会直接退出）；
2. 发 3 个满足条件的请求（`temperature: 0`、纯文本、无 tools、无图片）后，`/metrics` 的 `sparkinfer_speculative_runs_total` **大于 0**；
3. 反例确认：带 `temperature: 0.7` 或带 `tools` 的请求**不增加**投机计数（证明边界与文档一致）；
4. 抽检 1 个请求：同一提示在阶段 3 的 AR 服务与阶段 5 的 DSpark 服务下输出一致（"lossless" 的本地抽样验证，不替代官方逐字节门禁）。

**注意**：同时只能有 1 个活跃请求；第二个请求会让投机请求发生交接，这是设计行为，不是异常。

### 阶段 6 · 性能与并发对比（决定默认路线）

1. **单流对比**（同提示集、同输出长度、同采样）：SparkInfer AR / SparkInfer + DSpark / vLLM 200k / SGLang + DSpark 163840，记录 decode tok/s、TTFT、显存峰值。
2. **并发阶梯**：1 / 2 / 4 / 8 / 16，按模型卡的聚合吞吐口径复测（客户端计时、流式），与 vLLM 对照。
3. **验收记录**：把四组数字写进 `result/sparkinfer-nvfp4-dspark-result.md`，并明确"引用区间而不是单点"。

**决策规则**：
- 若日常形态是**多客户端 / 高并发** → 默认路线保持 vLLM，SparkInfer 只作长上下文与单流加速的可选档。
- 若日常形态是**单流或少数流、且长上下文收益明显** → 可把 SparkInfer 提为默认，并在 README 说明取舍。

### 阶段 7 · 固化为项目启动器 + 文档 + 测试

| 文件 | 类型 | 内容要点 |
| --- | --- | --- |
| `scripts/sparkinfer-serve.sh` | 新建（LF） | 前台运行；`start` / `help`；参数 `--model-dir` `--draft-dir` `--port` `--context-length` `--no-spec` `--image` `--name` `--lan` `--dry-run`；环境变量 `SPARKINFER_IMAGE` `SPARKINFER_NAME` `SPARKINFER_CTX` `SPARKINFER_GQA_RQH` `SPARKINFER_SAMPLING_DEFAULTS`；复用 `scripts/lib/wsl2-env-lib.sh` 的 `info/ok/warn/fail/section` |
| 同上 · preflight | — | docker 存在且 daemon 可达 → 镜像存在（否则提示 `docker pull`）→ 两个目录与 `config.json`/`tokenizer.json`/`model.safetensors` 存在 → `nvidia-smi` 可见 → 端口空闲；**任一失败给出可执行的修法**（沿用 `fail_hint` 风格） |
| 同上 · 启动 | — | 同名容器先 `docker rm -f`；`exec docker run` 前台；打印服务地址、模式、上下文、模型 ID、"Ctrl-C 停止" |
| `scripts/start-api-server-sparkinfer.bat` | 新建（CRLF + 无 BOM + 纯 ASCII） | 复制 `start-api-server-dspark.bat` 的结构：LAN 菜单（1/2/0）、UAC 提权、退出时清理 portproxy + 防火墙、结束提示 `docker logs qwen38-sparkinfer`；默认容器名 `qwen38-sparkinfer`，端口 8192 |
| `README.md` | 修改 | 新增 4.9 节：路线说明、参数表、DSpark 生效条件（greedy + 纯文本 + 单请求）、与其它启动器互斥、上下文取舍表；在 4.8 之后、第 5 节之前 |
| `tests/sparkinfer-tests.sh` + `tests/fakebin/docker` | 新建 | 沿用 `tests/` 的 fakebin 模式：假 `docker` 记录 argv；断言 `--dry-run` 输出含两个 `:ro` 挂载、`-p <port>:8080`、`serve-dspark`/无 `serve-dspark`；断言镜像缺失时退出 1 且提示 pull；断言缺 `config.json` 时拒绝启动 |
| `result/sparkinfer-nvfp4-dspark-result.md` | 新建 | 阶段 3–6 的实测结果与结论 |

**注意**：`.gitattributes` 已统一 `*.bat eol=crlf`、`*.sh eol=lf`，新文件**不要再手工引入 BOM 或反转换行**（历史上这正是 `.bat` 闪退的根因之一）。

### 阶段 8 · 回滚与收尾

```bash
# 1) 停服务并删容器（前台运行时 Ctrl-C 即可；兜底：）
docker rm -f qwen38-sparkinfer
# 2) 如需回收镜像空间（约 1 GB）
docker rmi ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10
# 3) 撤销本项目改动（脚本/文档/测试）
git -C /mnt/d/Code/MJ-Project/ai-model-nvfp4 status
git -C /mnt/d/Code/MJ-Project/ai-model-nvfp4 diff
```

**回滚不涉及**：模型文件、草稿、vLLM venv、SGLang 镜像、`.wslconfig`、Windows 驱动——本计划从未修改它们。

---

## 六、决策点与风险

### 6.1 风险表

| 风险 | 触发条件 | 处置 |
| --- | --- | --- |
| manifest 哈希门拒绝启动 | 设了 `SPARKINFER_NO_DOWNLOAD=1` | 不设该变量；或用 `MANIFEST_PATH=/nonexistent`；见 3.3 |
| 长上下文答错/不可复现 | int8 KV + ≥2048 token，默认路径（issue #976） | 阶段 4 强制门；必要时默认带 `SPARKINFER_PREFILL_ATTN_GQA_RQH=1` |
| 启动 OOM | `serve-dspark` + 全上下文 KV 池 | `CTX=131072`（默认）；草稿装不下会**启动失败**而不是静默降级 |
| DSpark 静默不生效 | 请求带采样参数 / tools / 图片 / 命中前缀缓存 / 存在并发 | 请求显式 `temperature: 0`；用 `/metrics` 计数验证，**计数为 0 时不得声称已启用** |
| 并发下反而更慢 | 8–16 并发（模型卡实测落后 vLLM 约 1.6×） | 阶段 6 决策规则；默认路线不轻改 |
| 端口/显存冲突 | 与 vLLM / SGLang 同时启动 | 启动器 preflight 检查端口与容器；文档标注互斥 |
| 显存被 WSL 缓存占住 | 反复启动后 | `wsl --shutdown` 释放；`.wslconfig` 不动 |
| 镜像漂移 | 用 `latest` | 固定 `0.5.10` 或 digest |
| 容器被 WSL 回收 | 用 `-d` 后台运行 | 一律前台；不设 restart 策略 |
| 模板渲染差异 | SparkInfer 用编译内置模板，不读 checkpoint 的 `chat_template.jinja` | 阶段 3/5 的冒烟与阶段 6 的对比中观察；若发现与 vLLM/SGLang 提示渲染不一致，在结果文档中如实标注 |

### 6.2 需要拍板的决策点

| # | 决策 | 建议默认 |
| --- | --- | --- |
| D1 | 是否固化并默认带 `SPARKINFER_PREFILL_ATTN_GQA_RQH=1` | 阶段 4 结果为准；若默认路径不可靠则**带**，并接受预填充变慢 |
| D2 | 固化后的服务端采样默认 | 保持 `generation_config`（与 vLLM 线路默认一致），文档要求 `temperature: 0` 才走 DSpark |
| D3 | 固化后的 `CTX` | DSpark 档 131072；AR 档 262144（同一次只能取一个） |
| D4 | 是否把 SparkInfer 提为日常默认 | 阶段 6 数据说话；当前倾向"多客户端继续 vLLM，长上下文/单流用 SparkInfer" |
| D5 | 是否保留 SGLang 镜像 | 保留（既有已验证路线）；本计划不删 |

---

## 七、验收标准（执行后逐条核对）

**必须全部满足，缺一不算通过：**

1. 阶段 0 前置检查全绿，权重指纹与快照一致（2387 张量 / 0 个 mtp）。
2. 镜像按 digest 固定并记录；**未使用 `latest`**。
3. 启动日志**没有**任何 `downloading` 行，可证明权重来自挂载。
4. `/health`、`/v1/info`、`/v1/chat/completions` 三者可用，返回 OpenAI 兼容结构。
5. 阶段 4 的长上下文 A/B 有明确结论（默认路径是否可靠），并按结论固化。
6. DSpark 阶段：`sparkinfer_speculative_runs_total > 0`，且反例请求不增加该计数。
7. 阶段 6 有与 vLLM / SGLang 同口径的对比数据，并据此给出默认路线结论。
8. 新启动器 `--dry-run` 可打印完整命令；`tests/` 全绿；README 4.9 节落地。
9. 回滚路径可执行：删容器、可选删镜像、`git` 可还原本项目改动。

**明确不算通过的情形：**

- 投机计数为 0 却声称"DSpark 已启用"。
- 长提示答案错误或三次不可复现，却因速度快而放行。
- 用 `latest` 跑通后未记录 digest（不可复现）。
- 为了让 DSpark 生效而把服务端整体改成 greedy 却未在文档说明对采样行为的影响。

---

## 八、执行检查清单

```text
[ ] 0.1 端口 8192 空闲、无其它引擎容器在跑
[ ] 0.2 显存接近空闲（必要时 wsl --shutdown 后重进）
[ ] 0.3 四个必需文件存在（目标 config.json/tokenizer.json、草稿 config.json/model.safetensors）
[ ] 0.4 权重指纹记录（2387 张量 / mtp 0）
[ ] 1.1 docker pull ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10
[ ] 1.2 记录 RepoDigests 与镜像大小
[ ] 2.1 dry-run 命令含两个 :ro 挂载 + 8192:8080 + serve-dspark
[ ] 3.1 AR 启动（CTX=262144），日志无 downloading
[ ] 3.2 /health、/v1/info、冒烟 17×23=391
[ ] 4.1 短提示对照 3/3 一致
[ ] 4.2 4k 提示默认路径 3/3 一致且正确
[ ] 4.3 16k 提示里"ERROR 行"召回正确（可选 44k/20k 加强用例）
[ ] 4.4 RQH=1 对照完成，形成 D1 决策
[ ] 5.1 serve-dspark 启动，日志显示 DSpark 模式
[ ] 5.2 投机计数 > 0
[ ] 5.3 反例（temperature 0.7 / 带 tools）不增加计数
[ ] 5.4 抽检输出与 AR 一致
[ ] 6.1 单流四组对比成表
[ ] 6.2 并发 1/2/4/8/16 阶梯成表
[ ] 6.3 写出 D4 结论
[ ] 7.1 scripts/sparkinfer-serve.sh（LF，preflight + 前台 docker run）
[ ] 7.2 scripts/start-api-server-sparkinfer.bat（CRLF、无 BOM、纯 ASCII、LAN 菜单）
[ ] 7.3 README 4.9 节
[ ] 7.4 tests/sparkinfer-tests.sh + tests/fakebin/docker，全绿
[ ] 7.5 result/sparkinfer-nvfp4-dspark-result.md
[ ] 8.1 删容器；必要时删镜像
[ ] 8.2 git status 干净或仅含预期改动
```

---

## 九、附：与既有路线的定位对照（固化后写进 README）

| 路线 | 脚本 | 上下文 | 加速方式 | 并发表现 | 定位 |
| --- | --- | --- | --- | --- | --- |
| vLLM + NVFP4 | `direct.sh` / `start-api-server-vllm.bat` | 200000（实测边界） | MTP（可选） | 16 并发已实测良好 | 日常多客户端默认 |
| SGLang + DSpark | `sglang-dspark.sh` / `start-api-server-dspark.bat` | 163840 | DSpark（认证路线） | 单请求 | 已实测的单流加速档 |
| SparkInfer（本计划） | `sparkinfer-serve.sh` / `start-api-server-sparkinfer.bat` | AR 262144 / DSpark 131072 | DSpark（引擎内） | 8–16 并发弱于 vLLM | 长上下文 + 单流加速候选 |
| llama.cpp + GGUF | `start-api-server-gguf.bat` | 128000 | — | auto | GGUF 权重专用 |

> 三条 NVFP4 路线**端口相同、显存只够一个**，任何时候只启一条。