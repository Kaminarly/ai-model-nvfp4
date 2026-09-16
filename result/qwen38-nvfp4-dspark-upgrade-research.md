# Qwen3.8-27B-NVFP4-RTX5090 升级到最新版并配置 DSpark 加速：可行性研究报告

> 研究日期：2026-09-13；2026-09-16 更新（所需文件已下载并完成 sha256 校验，补充落位与校验记录，见附录 A.7）
> 研究范围：仅调研与论证。**未修改本机任何代码、模型、环境或框架**（本机指 WSL2 Ubuntu 及其中的 venv / 容器）。
> 结论先行：升级可行且代价很小（只增量下载约 7.9 GB），DSpark 加速有两条可落地路线——作者认证的 SGLang 容器，以及**本机已装好的 vLLM 0.27.1 直接跑**（后者不需要装 Docker）。

---

## 结论（TL;DR）

1. **本机模型不是最新版。** WSL 中的 `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` 是 Hugging Face 仓库的 `pre-final` 构建（3 分片、18.77 GB、仍带 MTP 头）。HF `main` 已是 2026-09-10 的"final build"（2 分片、17.92 GB、MTP 头已删除）。

2. **升级只需替换约 7.9 GB 的文件，不是 17.9 GB。** 两个构建的第一个权重分片是**字节完全相同**的同一个 HF LFS 对象（sha256 `cdd37b0e61eccc8a…`），本机该分片的 CRC32 也与最新版 `crc32.txt` 一致。所以升级 = 保留第 1 分片 + 替换第 2 分片（7,943,334,864 B ≈ 7.94 GB）+ 刷新索引/配置，旧的第 2、3 分片可删。

3. **最新版删掉了 MTP 头，这会打断本项目现有的加速路线。** 本仓库当前用 vLLM 的 `--spec-method mtp`（`scripts/start-api-server-mtp.bat`）依赖权重里自带的 `mtp.*` 张量；final build 里这些张量已不存在。**升级到最新版后必须改用 DSpark 草稿模型，`mtp` 路线会失效。**

4. **DSpark 本质是一个约 1.3 GB 的独立草稿模型**，不是驱动、不是独立服务，只加速解码（输出）阶段，不加速预填充；目标模型会验证每个草稿 token，正常配置下不改变最终输出。

5. **框架不是只能选 vLLM，也不是只能选 SGLang**：
   - **SGLang**：模型作者认证的组合（草稿模型卡明确要求专用镜像 `lmsysorg/sglang:qwen38-27b`），但本机**尚未安装 Docker 和 NVIDIA Container Toolkit**，需要先补装。
   - **vLLM**：本机 `~/vllm/venv` 已装 vLLM **0.27.1**，且其源码里**已包含 DSpark 支持**（`vllm/model_executor/models/qwen3_dspark.py`、`SpeculativeMethod` 含 `"dspark"`、`--spec-method dspark` 可解析）。这条路**不需要 Docker**，但作者未把 vLLM+DSpark 列为认证组合，属于"可试但需实测"。
   - **SparkInfer**：作者的 DSpark 只存在于其 benchmark 工具里，HTTP 服务目前是纯自回归，**不能通过 API 使用 DSpark**，不适合本项目目标。
   - **llama.cpp**：读不了 ModelOpt NVFP4 权重，与本模型无关。

6. **单卡 32 GB 上 DSpark 的硬取舍**：开 DSpark 后上下文从原生 262,144 降到约 165,169（SGLang）/ 更保守配置约 122,880；解码吞吐约 1.9×（SGLang 服务器口径）～2.04×（作者规格表口径）。**长上下文与投机加速二选一**，不能同时拿满。

---

## 一、本机现状（已实测）

在 WSL 中实测得到的当前环境（不是文档抄录）：

| 项目 | 实测值 |
| --- | --- |
| 发行版 | Ubuntu 26.04 (Resolute Raccoon)，`VERSION_ID=26.04` |
| WSL | 2.7.12.0，内核 `6.18.33.2-microsoft-standard-WSL2` |
| Windows | 10.0.19045.5854 |
| GPU | NVIDIA GeForce RTX 5090，32607 MiB，驱动 610.88 |
| WSL 资源 | 约 30 GB 内存、16 核 CPU、根分区剩余约 903 GB |
| PID 1 | `systemd`（可直接 `systemctl`） |
| 系统 Python | 3.14.4（`/usr/bin/python3`，未装 vllm/sglang） |
| 本项目 vLLM | **0.27.1**，位于 `/home/kami/vllm/venv`（`python3.14`） |
| SGLang | **未安装**（venv 内无 `sglang`） |
| Docker | **未安装**（`docker: command not found`） |
| NVIDIA Container Toolkit | **未安装**（无 `nvidia-ctk` / `nvidia-container-cli`） |
| llama.cpp | 源码在 `/home/kami/llama.cpp` |
| HuggingFace CLI | venv 内有 `hf` / `huggingface-cli`，`huggingface_hub 1.28.0` |
| DSpark 草稿模型 | **未下载**（`/home/kami/models/` 下无 DSpark 目录） |

### 1.1 本机目标模型的实际内容

目录 `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`：

- 总大小约 **18.77 GB**，**3 个** 权重分片：
  - `model-00001-of-00003.safetensors` — 9,972,777,720 B
  - `model-00002-of-00003.safetensors` — 8,048,202,912 B
  - `model-00003-of-00003.safetensors` — 744,532,384 B
- 索引共 **2402** 个张量，其中 **15 个 `mtp.*` 张量**（`mtp.fc.weight`、`mtp.layers.0.*` 等）。
- `config.json`：`architectures=["Qwen3_5ForConditionalGeneration"]`、`model_type="qwen3_5"`、`quant_method="modelopt"`；`ignore` 列表把 `linear_attn.conv1d` / `in_proj_a` / `in_proj_b` 等 Gated-DeltaNet 投影排除在 NVFP4 之外。
- 本地 `crc32.txt`：分片 1 = `617cc98f`，分片 2 = `c159e96c`，分片 3 = `35bbe170`。

判断依据：该构建带完整 MTP 头、且 `lm_head` 已是 NVFP4、3 分片 18.77 GB —— 与 HF 的 `pre-final` 分支特征逐项吻合（见下节）。

---

## 二、"最新版"是什么，以及如何升级

### 2.1 HF 仓库的三个构建

`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` 当前有 3 个分支：

| 分支 | commit | 最后修改 | 分片 | 权重大小 | MTP 头 | `lm_head` |
| --- | --- | --- | --- | --- | --- | --- |
| **`main`（最新）** | `5b7a687f…` | **2026-09-10** | **2** | **17.92 GB** | **已删除** | NVFP4 |
| `pre-final` | `35fd99fb…` | 2026-09-01 | 3 | 18.77 GB | 保留 | NVFP4 |
| `pre-lmhead4` | `2ed84505…` | 2026-08-19 | 3 | 20.59 GB | 保留 | BF16 |

`main` 相对本机构建的**实际差异**：删掉 15 个 `mtp.*` 张量（索引 2402 → 2387）、重新分片、刷新 `config.json`（13247 → 13167 B）与 `hf_quant_config.json`（9087 → 9050 B）。仓库 README 明确说明这次改动"不触碰任何被计算的值"（*without touching a single computed value*），即**计算语义不变**，final build 的收益主要是更小的下载与加载足迹。

> 仓库另有被取代的独立仓库 `-No-MTP` 与 `-LMHead4`，README 说明已合并进本仓库，无需单独使用。

### 2.2 本机处于 `pre-final`

- 分片数 3、总大小 18.77 GB、15 个 `mtp.*` 张量、NVFP4 `lm_head` —— 与 `pre-final` 完全对应。
- 本机 `config.json` 大小 13247 B，与 `pre-final` 的 13247 B 一致；与 `main` 的 13167 B 不同。
- 本机文件时间戳为 08-19/08-20，`pre-final` 分支更新于 09-01。

### 2.3 增量升级路径（关键收益：只需替换约 7.94 GB）

对 HF 元数据逐项比对后可以确认：

- **分片 1 是同一个对象。** `main` 的 `model-00001-of-00002.safetensors` 与 `pre-final` 的 `model-00001-of-00003.safetensors`，大小同为 9,972,777,720 B，LFS sha256 同为 `cdd37b0e61eccc8a…`。
- 本机分片 1 的 CRC32 = `617cc98f`，与 `main` 的 `crc32.txt` 第一行 `617cc98f model-00001-of-00002.safetensors` **完全一致**。
- 因此：**分片 1 无需替换**，原地改名即可复用（改名/硬链接为新的文件名）。
- 需要替换的只有 `main` 的分片 2：7,943,334,864 B ≈ **7.94 GB**。
- 旧的分片 2（8.05 GB）、分片 3（0.74 GB）在新构建中不再被引用，可在确认后删除。

净效果：**需替换的文件 ≈ 7.94 GB（原本全量 17.92 GB），磁盘净增 ≈ 负 0.85 GB**。

需要替换/新增的文件清单见附录 A.1，文件由使用者自行准备；落位与校验步骤如下：

```bash
cd /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090

# 1) 备份旧构建（可逆，pre-final 分支也长期保留）
mkdir -p .pre-final-backup
cp config.json hf_quant_config.json crc32.txt model.safetensors.index.json .pre-final-backup/

# 2) 复用字节完全相同的分片 1：改名即可（同盘 mv 瞬时完成，不占额外空间）
mv model-00001-of-00003.safetensors model-00001-of-00002.safetensors

# 3) 将 5 个已校验文件放入本目录（暂存于 /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/，
#    落位命令与校验记录见附录 A.7）：
#    config.json / hf_quant_config.json / crc32.txt /
#    model.safetensors.index.json / model-00002-of-00002.safetensors

# 4) 校验（应只剩 2 个分片，且关键哈希与索引都要对得上）
sha256sum model-00001-of-00002.safetensors   # 期望 cdd37b0e61eccc8a3d7d08f9d1a4f52856a9d88e4e8b42089bd18a970e3a01ec（= main 的 LFS oid）
ls -l model-00002-of-00002.safetensors       # 期望大小 7,943,334,864 B
grep -c . crc32.txt                           # 期望 2（每行对应新构建的一个分片）
python3 -c "import json;from collections import Counter;wm=json.load(open('model.safetensors.index.json'))['weight_map'];print(len(wm), Counter(wm.values()))"
# 期望：2387  Counter({'model-00001-of-00002.safetensors': 1312, 'model-00002-of-00002.safetensors': 1075})

# 5) 确认无误后再删旧分片（务必最后做）
rm -f model-00002-of-00003.safetensors model-00003-of-00003.safetensors
```

> 校验要点：分片 1 的 sha256 必须等于 `cdd37b0e61eccc8a3d7d08f9d1a4f52856a9d88e4e8b42089bd18a970e3a01ec`；索引张量总数必须是 2387 且两个分片分别为 1312 / 1075 个张量。任一项不符都不要删除旧分片。
> 回滚：把 `.pre-final-backup/` 里的 4 个小文件移回、分片 1 改回原名即可；若旧分片已删，再从 `pre-final` / `pre-lmhead4` 分支取回旧构建。

### 2.4 升级的副作用：现有 MTP 加速路线会失效

这是本次升级**最重要的一条影响**：

- 本仓库当前的可选加速是 vLLM MTP（`scripts/start-api-server-mtp.bat` → `VLLM_SPEC_METHOD=mtp` → `--spec-method mtp --spec-tokens 3`），依赖 `config.json` 的 MTP 配置与权重里的 `mtp.*` 张量。
- final build **删除了全部 15 个 `mtp.*` 张量**。升级后 `--spec-method mtp` 无权重可加载，要么报错，要么退化为无投机。
- 因此：**升级到最新版 ⇔ 必须把加速方案换成 DSpark**（或接受无加速）。两者不可兼得于同一份权重。
- 反过来看，DSpark NVFP4 草稿只有 1.41 GB 运行占用、接受率 2.904，作者给出的对比是**比内置 MTP 头快 31.7%、内存只用其约四分之一**，所以对"要加速"这个目标而言，换 DSpark 是升级而不是损失。

---

## 三、DSpark 加速方案对比（三种框架）

### 3.1 DSpark 是什么

目标：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
草稿（推荐）：`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`

草稿仓库实测元数据：

| 项目 | 值 |
| --- | --- |
| commit | `eba1ac5a…`，最后修改 2026-08-19 |
| 大小 | 1,399,670,058 B ≈ **1.30 GB**（单文件 `model.safetensors`） |
| 架构 | `Qwen3DSparkModel`（5 层全注意力 + Markov 头） |
| block size | **7** |
| 量化 | ModelOpt **NVFP4**（W4A4，group 16）；`fc`/QKV/Markov/confidence 保持 BF16 |
| target_layer_ids | `[4, 16, 28, 40, 52]` |
| 许可 / 访问 | Apache-2.0，公开非 gated，**匿名可下载，无需 HF token** |

仓库主页与文件清单：

- https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4
- https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4/tree/main

**需要准备的文件（使用者自行准备；已就绪并校验，见附录 A.7）**，目标目录 `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4`（与 3.2、3.3 节启动命令中的路径一致）：

| 文件 | 大小 | sha256 | 是否必需 |
| --- | --- | --- | --- |
| `model.safetensors` | 1,399,670,058 B ≈ **1.30 GB** | `212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05` | ✅ |
| `config.json` | 2,828 B | `82fd961b632c629736902d9d4fdd3258dee1080f557cf86298cac063a514a0cf` | ✅ |
| `hf_quant_config.json` | 937 B | `cda90695e8c4a5eaed7ce7220afbc8bbe18e7624a167466ec7768c603e756a09` | ✅ |
| `README.md` | 10,793 B | — | 可选 |
| `.gitattributes` | 1,519 B | — | 可选 |

该仓库只有一个 `main` 分支（commit `eba1ac5a…`），无历史分支需要挑选；也没有分片索引文件，单个 `model.safetensors` 即全部草稿权重。

校验要点：`model.safetensors` 的 sha256 必须等于 `212fd1b8…`；`config.json` 应为 2,828 B，`architectures` 为 `["Qwen3DSparkModel"]`、`block_size` 为 `7`。这个 `block_size` 决定了 vLLM 路线的 `--spec-tokens` 不得小于 7。

配套还有 BF16 源草稿 `Qwen3.8-27B-NVFP4-RTX5090-DSpark`（作者标注 2.72 GB，列在推荐项之后）。实测该仓库返回 **HTTP 401（需要认证，不能匿名获取）**，而推荐的 NVFP4 版是公开非 gated 的；因此实际可取用的就是 NVFP4 版，本文后续命令均按 NVFP4 版编写。

关键结论：DSpark 是**目标模型的辅助草稿**，目标模型逐 token 验证，"构造上无损"（lossless by construction）。

### 3.2 路线 A：SGLang（作者认证，推荐但需补装 Docker）

- 草稿模型卡明确：**"Requires the Qwen3.8 SGLang build. Pin the image."**，认证运行时为 `lmsysorg/sglang:qwen38-27b`（内部版本串 `0.0.0.dev0+qwen38.27b.g561c8f3`）。
- 目标模型 README（09-10 更新）给出的 SGLang 命令改用 `lmsysorg/sglang:latest`，数值也调高（`ctx 262144` / `mem-fraction-static 0.90` / `max-running-requests 2` / `max-mamba-cache-size 12`）。
- 本机 Docker Hub 现状：`qwen38-27b` 停留在 2026-08-14（digest `febfb971…`）；`latest` 已滚到 2026-09-04（digest `d6e72886…`）；另有更新的 `qwen38flashnext`（09-03）、`dev-cu13-qwen38-next-local`（09-07）等标签。综合两份卡片的措辞，**建议以 `qwen38-27b` 为认证基线，`latest` 作为目标卡推荐的可选升级，并按 digest 固定以保证可复现**。

已核实：SGLang 上游源码**确实支持 DSPARK**——`python/sglang/srt/speculative/` 下有 `dspark_components/`、`dspark_disaggregation.py`，`SpeculativeAlgorithm` 枚举含 `DFLASH`、`UNO`、`DSPARK`，并提供 `--speculative-dspark-block-size` 等参数。它不是第三方魔改。

**本机前置条件未满足**：Docker 与 NVIDIA Container Toolkit 都未安装。需要先补装（可 `systemd` 已就绪、WSL 内**不要**装 Linux 显卡驱动）。配置步骤与前一份报告 `result/sglang-qwen38-dspark-wsl2.md` 第二、三节一致。

32 GB 单卡稳定配置（来自草稿模型卡，并发 1）：

```bash
docker run --rm --name qwen38-sglang \
  --gpus all --ipc=host --shm-size 32g \
  -p 30000:30000 \
  -v /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090:/models/target:ro \
  -v /home/kami/models/Qwen3.8-27B-DSpark-NVFP4:/models/dspark:ro \
  lmsysorg/sglang:qwen38-27b \
  sglang serve \
    --model-path /models/target \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path /models/dspark \
    --speculative-draft-model-quantization modelopt_fp4 \
    --speculative-dspark-block-size 7 \
    --trust-remote-code --tp-size 1 \
    --context-length 122880 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend flashinfer \
    --chunked-prefill-size 1024 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 8 \
    --mm-feature-transport cpu \
    --cuda-graph-max-bs-decode 1 \
    --mem-fraction-static 0.86 \
    --max-running-requests 1 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port 30000
```

32 GB 上不要删的参数（模型卡原文）：`--mamba-ssm-dtype bfloat16`、`--max-mamba-cache-size 8`、`--speculative-draft-model-quantization modelopt_fp4`、`--mem-fraction-static 0.86`、`--max-running-requests 1`。
其中 `--max-mamba-cache-size` 的经验公式：**≥ 4 × `--max-running-requests` + 4**（每请求占 4 个 Gated-DeltaNet 状态槽）。

若要拿满原生 262K：**删除全部 4 个 `--speculative-*` 参数**，不加载草稿模型，并按无投机配置重估显存；两者不能并存于 32 GB。

### 3.3 路线 B：vLLM 0.27.1（本机已具备，无需 Docker）

这是本次研究最重要的**新发现**，也修正了"只能 SGLang"的旧判断：

实测本机 `/home/kami/vllm/venv/lib/python3.14/site-packages/vllm`：

- `config/speculative.py` 中 `DSparkModelTypes = Literal["dspark"]`，且 `SpeculativeMethod` 联合类型包含它 → **`--spec-method dspark` 是合法取值**。
- 草稿模型自动识别分支同时匹配 `"dspark" in model_name` 与 `architectures` 含 `Qwen3DSparkModel` → 我们的草稿会被正确识别。
- `model_executor/models/qwen3_dspark.py` **已存在**，实现 `Qwen3DSparkModel`（DFlash 骨干 + DSpark Markov 头）；`model_executor/models/registry.py` 有注册项。
- 目标侧 `qwen3_5.py` 暴露 `SupportsEagle3` 与 `set_aux_hidden_state_layers`，可向草稿提供辅助隐状态。
- v1 引擎有 `mamba_hybrid` 模型态的投机解码机制（`MambaBuffers.create` 明确以 "spec-decode + hybrid" 为条件），说明**混合 Mamba 目标 + 投机是可被支持的**。
- 约束（源码注释）：`if self.method == "dspark"` 时要求 **`num_speculative_tokens >= dspark_block_size`**，否则"会产出乱码而非仅降低接受率"。本模型 block size = 7，故 `--spec-tokens` 至少为 7。

可用启动形式（研究结论，未执行；`--spec-model` 指向独立草稿目录）：

```bash
/home/kami/vllm/venv/bin/vllm serve \
  /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090 \
  --served-model-name Qwen3.8-27B-NVFP4-RTX5090 \
  --quantization modelopt --kv-cache-dtype fp8 \
  --spec-method dspark \
  --spec-model /home/kami/models/Qwen3.8-27B-DSpark-NVFP4 \
  --spec-tokens 7 \
  --max-model-len 122880 --max-num-seqs 1 \
  --gpu-memory-utilization 0.86 \
  --enable-prefix-caching --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
```

**但必须如实标注其不确定性**：目标模型 README 的 vLLM 命令**不含任何 `--spec-*` 参数**，其引擎对比表也只给 "SGLang (+DSpark)"；即**作者没有把 vLLM+DSpark 列为认证组合**。vLLM 里的 DSpark 实现主要面向 DeepSeek-V4 / Qwen3-Omni 等场景，能否与 **qwen3_5 混合 GatedDeltaNet 目标 + NVFP4 ModelOpt 草稿** 完全兼容，需要实测（见第七节）。因此路线 B 的定位是"低门槛可试"，路线 A 是"认证兜底"。

### 3.4 其他框架

- **SparkInfer**（`ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:latest`，Blackwell sm_120）：作者称其 DSpark 目前**只在 benchmark harness 中**，HTTP 服务器仍是纯自回归——"不是今天能 `curl` 的东西"。因此**不能**用它做带 DSpark 的 API 服务。其价值是无投机下最大上下文（360,000）与较高基线吞吐（92.9 tok/s）。
- **llama.cpp**（本机 `~/llama.cpp`）：无法加载 ModelOpt NVFP4 压缩权重，与本模型无关。

### 3.5 框架选型建议

| 维度 | SGLang（路线 A） | vLLM 0.27.1（路线 B） |
| --- | --- | --- |
| 作者认证 | ✅ 草稿卡明确要求并给出命令 | ❌ 未认证，需实测 |
| 本机就绪度 | ❌ 需装 Docker + NVIDIA NCT | ✅ 已装，开箱可试 |
| DSpark 支持 | ✅ 上游含 DSPARK 与专用参数 | ⚠️ 上游含 dspark 代码路径，但组合未验证 |
| 上下文 | 无投机 320,960；带 DSpark 约 165,169 | 无投机 262,144；带 DSpark 待实测 |
| 并发 | 草稿卡限 1（GN 状态槽约束） | 待实测 |
| 风险 | 低（照抄认证参数） | 中（可能加载失败或退化） |

**推荐**：先按**路线 B 做一次快速可行性试跑**（零安装成本，几十分钟即可证伪）；若加载/输出异常，再按**路线 A** 补装 Docker 走认证组合。两者都以本节命令为准。

---

## 四、关键取舍与风险

1. **上下文 vs 解码速度二选一。** 32 GB 单卡上，开 DSpark 后可用上下文约 122,880～165,169；关掉可到 262,144（vLLM/SGLang 原生）甚至 320,960（SGLang 无投机）。DSpark 只加速解码，**不加速长提示词预填充**；250K 冷启动 TTFT 约 121 s。
2. **并发受限。** SGLang 草稿卡把并发钉在 1（每请求 4 个 GDN 状态槽，32 GB 放不下第二个）。高并发场景 vLLM 无投机反而更强（作者数据：8～16 客户端时 vLLM 聚合领先）。
3. **升级会删掉 MTP。** 若现有工作流依赖 `start-api-server-mtp.bat`，升级后该路线失效；需同步改为 DSpark 或接受无加速。这是可回滚的（`pre-final` 分支保留）。
4. **vLLM 路线的兼容性未经认证。** `--spec-tokens` 必须 ≥ 7，否则源码注释明示会产出乱码；NVFP4 草稿在 vLLM 下的量化加载、以及 qwen3_5 混合目标抽取辅助隐状态是否完整，都需实测。
5. **认证镜像与滚动标签。** 草稿卡要求 pin `qwen38-27b`，目标卡改推 `latest`；`latest` 每几天滚动。建议按 digest 固定，避免"今天能跑、明天拉新版就挂"。
6. **长上下文质量回退。** 草稿卡自述其长上下文域接受率 **−6.1%**（训练语料上限 2,048 token）；短/中上下文（数学、JSON、工具调用）为正收益。若主打超长文档，需权衡。
7. **Docker 相关注意。** WSL 内**不要**安装 Linux NVIDIA 显卡驱动（用 Windows 驱动的 WSL 接口）；Docker 依赖 `systemd`，本机 PID 1 已是 `systemd`，无需额外处理（历史上 `.wslconfig` 里那条 `systemd=true` 是无效键，会打印警告）。
8. **端口/网络。** SGLang 认证用例走 30000，vLLM 现有脚本走 8192；两者不可同时启动。Windows 10 的 WSL2 是 NAT，局域网需 Windows `portproxy` + 限定 Private/LocalSubnet 的防火墙规则（机制与本仓库 `start-api-server-lan.bat` 相同，但该脚本启动的是 vLLM，不能直接用于 SGLang 容器）。

---

## 五、验收清单（实测时逐条核对）

升级验收：

1. `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` 只剩 **2** 个分片，`crc32.txt` 两行都能对上。
2. `model-00001-of-00002.safetensors` 的 sha256 与 HF LFS 对象一致（证明复用的分片正确）。
3. `model.safetensors.index.json` 张量总数为 **2387**，且 `mtp` 命中数为 **0**（可用 `python3 -c` 读 `weight_map` 统计）。
4. `config.json` 为最新版（约 13167 B），加载后无 `Parameter ... not found` 报错。

DSpark 加速验收：

5. 服务日志明确显示加载了 DSpark 草稿，**未回退**到 MTP 或关闭投机。
6. 启动无 OOM；`nvidia-smi` 显存余量符合预期（SGLang 配置约留 3.5 GB）。
7. 连续发送 ≥ 3 个短请求全部 HTTP 200；无 API key 时（SGLang 不传 `--api-key`）不带认证头也成功。
8. 与无投机基线对比解码吞吐，确认确有提升（SGLang 口径约 1.9×；若持平或更慢，多为接受率/验证成本问题）。
9. 长提示词测试输入+输出总 token 不超过所用上下文上限（保守配置 122,880；SGLang 实测上限约 165,169）。
10. 若走 vLLM：确认 `--spec-tokens ≥ 7`，且输出与无投机基线一致（DSpark 应为无损）。

---

## 六、未核实项 / 后续实测建议

本研究未修改环境，故以下只能在真机实测阶段确认：

1. **vLLM 0.27.1 + DSpark + qwen3_5 混合目标**能否成功加载并加速（最大不确定性）。建议先用最短上下文 + 无投机跑通基线，再加 `--spec-method dspark --spec-model … --spec-tokens 7`，观察日志与吞吐。
2. **vLLM 是否需要额外指定草稿量化**（草稿是 ModelOpt NVFP4）；若自动识别失败，改用 `--speculative-config` JSON 显式给字段。
3. **SGLang 具体镜像版本**：认证基线 `qwen38-27b`（08-14）与目标卡 `latest`（09-04）哪个与 final build + DSpark v2 组合最佳；建议两者都按 digest 固定后对比。
4. **两套加速在 32 GB 上的实测上下文/吞吐/接受率**，以替换本报告引用的作者口径数字。
5. **是否需要升级 vLLM 到 0.28/0.29**：本机 0.27.1 已含 dspark，暂不必升级；若真机实测遇 bug，0.28.0 / 0.29.0 也都包含 dspark 支持（已核实源码），可作为升级候选。

---

## 附录 A：本机 vs HF `main` 逐文件比对（2026-09-13 实测，sha256 级校验）

对 HF `main` 的 33 个条目逐个取 sha256 / LFS oid，与本机 16 个文件逐个比对，结果如下。

> 本附录只涉及目标模型。DSpark 草稿模型是另一个仓库（本机尚未有任何副本），其文件清单与校验值见 3.1 节。

### A.1 需要替换/新增的文件（共 5 个，合计约 7.94 GB，由使用者自行准备）

| # | 文件 | 大小 | HF main sha256（前 8 位） | 本机现状 |
| --- | --- | --- | --- | --- |
| 1 | `model-00002-of-00002.safetensors` | **7,943,334,864 B ≈ 7.94 GB** | `713b84b8` | 不存在（新增） |
| 2 | `model.safetensors.index.json` | 236,508 B | `28b9c74b` | 本机为 `4f0c8847`（旧） |
| 3 | `config.json` | 13,167 B | `62dad33f` | 本机为 `78f65e03`（旧） |
| 4 | `hf_quant_config.json` | 9,050 B | `8e7b3602` | 本机为 `2c30a0d7`（旧） |
| 5 | `crc32.txt` | 86 B | `c7841b92` | 本机为 `7c6967ae`（旧） |

### A.2 无需替换：本机已有且内容完全一致（sha256 相同）

| 文件 | 说明 |
| --- | --- |
| `model-00001-of-00003.safetensors`（9,972,777,720 B） | sha256 = `cdd37b0e61eccc8a3d7d08f9d1a4f52856a9d88e4e8b42089bd18a970e3a01ec`，**与 main 的 `model-00001-of-00002.safetensors` 完全相同**；CRC32 `617cc98f` 也与 main `crc32.txt` 第 1 行吻合。**只需改名为 `model-00001-of-00002.safetensors` 即可复用。** |
| `tokenizer.json`（12,809,320 B） | sha256 = `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`，与 main 的 LFS oid 相同 |
| `vocab.json` / `merges.txt` | sha256 相同 |
| `chat_template.jinja` / `generation_config.json` | sha256 相同 |
| `preprocessor_config.json` / `processor_config.json` | sha256 相同 |
| `tokenizer_config.json` / `video_preprocessor_config.json` | sha256 相同 |

### A.3 需要删除的文件（main 中已不存在）

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `model-00002-of-00003.safetensors` | 8,048,202,912 B | 新构建不再引用 |
| `model-00003-of-00003.safetensors` | 744,532,384 B | 新构建不再引用（旧的第 2 分片里含 2 个 `mtp.*` 张量，第 3 分片是纯 MTP 分片） |

净磁盘变化：**+7.94 GB（新分片 2）− 8.79 GB（旧分片 2、3）≈ 减少 0.85 GB**。

### A.4 本机缺失但可选的仓库元数据

`main` 中还有本机目录没有的 `.gitattributes`、`LICENSE`、`README.md`、`README.qwen-upstream.md`、`assets/*.png`（15 张图）。这些与推理无关；其中 `LICENSE`（Apache-2.0）如需合规留存可自行补齐。

### A.5 语义差异确认（已 diff 实际内容）

- `config.json`：`mtp_num_hidden_layers` **1 → 0**；`mtp_use_dedicated_embeddings` 被删除；`quantization_config.ignore` 移除 `"mtp*"` 与 `"mtp.layers.0*"`（保留 `"model.visual*"`）。
- `hf_quant_config.json`：`exclude_modules` **148 → 146**（少的就是上面两条 `mtp*`）。
- `model.safetensors.index.json`：张量 **2402 → 2387**（少 15 个 `mtp.*`）；分片 **3（1312/1077/13）→ 2（1312/1075）**。两边第 1 分片都恰好是 1312 个张量，与"分片 1 字节相同"一致。
- 被删除的 15 个 MTP 张量：`mtp.fc.weight`、`mtp.norm.weight`、`mtp.pre_fc_norm_embedding.weight`、`mtp.pre_fc_norm_hidden.weight`，以及 `mtp.layers.0.{input_layernorm, post_attention_layernorm}.weight`、`mtp.layers.0.mlp.{gate,up,down}_proj.weight`、`mtp.layers.0.self_attn.{q,k,v,o}_proj.weight`、`mtp.layers.0.self_attn.{q_norm,k_norm}.weight`。

### A.6 落位与校验步骤

```bash
cd /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090

# 1) 备份 4 个小文件（KB 级，保证可秒回滚）
mkdir -p .pre-final-backup
cp config.json hf_quant_config.json crc32.txt model.safetensors.index.json .pre-final-backup/

# 2) 复用字节相同的分片 1：改名即可（同盘 mv 瞬时完成，不占额外空间）
mv model-00001-of-00003.safetensors model-00001-of-00002.safetensors

# 3) 将已校验的 5 个文件从暂存目录放入本目录（落位命令见 A.7）

# 4) 校验：分片数应为 2，关键哈希与索引都要对得上
sha256sum model-00001-of-00002.safetensors    # 期望 cdd37b0e61eccc8a…（与 main 相同）
ls -l model-00002-of-00002.safetensors        # 期望大小 7,943,334,864 B
grep -c . crc32.txt                            # 期望 2
python3 -c "import json;from collections import Counter;wm=json.load(open('model.safetensors.index.json'))['weight_map'];print(len(wm), Counter(wm.values()))"
# 期望：2387  Counter({'model-00001-of-00002.safetensors': 1312, 'model-00002-of-00002.safetensors': 1075})

# 5) 确认第 4 步全部通过后，再删旧分片（删掉后回滚需重取 pre-final 分片）
rm -f model-00002-of-00003.safetensors model-00003-of-00003.safetensors
```

只要第 1 步的备份还在，回滚就是：`mv model-00001-of-00002.safetensors model-00001-of-00003.safetensors && cp .pre-final-backup/* .`（若已删旧分片，再配合 `pre-final` 分支取回分片 2、3）。

### A.7 已准备文件的校验记录与落位（2026-09-16）

文件已下载并暂存于 Windows 侧 `D:\WSL\models\`，在 WSL 内对应 `/mnt/d/WSL/models/`，与两个仓库各占一个子目录：

| 目标位置 | 暂存来源 | 文件数 |
| --- | --- | --- |
| `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` | `/mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/` | 5 |
| `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4`（尚不存在，需新建） | `/mnt/d/WSL/models/Qwen3.8-27B-DSpark-NVFP4/` | 3 |

**校验结果：8/8 全部通过**（大小与 sha256 均与 HF 侧权威值一致；权重在改名后复核仍通过）。

| 文件 | 大小 | sha256 |
| --- | --- | --- |
| `Qwen3.8-27B-NVFP4-RTX5090/model-00002-of-00002.safetensors` | 7,943,334,864 | `713b84b8287e2193290214766c5384fa6475c5626a628e4021c2c9ca90aa61df` |
| `Qwen3.8-27B-NVFP4-RTX5090/config.json` | 13,167 | `62dad33f5c64fedc1826a39f248d91b3ab4734ebdcd0a9981b548073e996d0b2` |
| `Qwen3.8-27B-NVFP4-RTX5090/hf_quant_config.json` | 9,050 | `8e7b36023a5a1184f5e0537bd8a30d15a980afaf0c6b3d7b8eb5dfd10d195938` |
| `Qwen3.8-27B-NVFP4-RTX5090/crc32.txt` | 86 | `c7841b92ffe3951e2274665a43af62964811d653d962f39901112405424d8fca` |
| `Qwen3.8-27B-NVFP4-RTX5090/model.safetensors.index.json` | 236,508 | `28b9c74b60f4e75914845e119e5b6d0f74574373e8d0f99f8af227583cbbf78c` |
| `Qwen3.8-27B-DSpark-NVFP4/model.safetensors` | 1,399,670,058 | `212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05` |
| `Qwen3.8-27B-DSpark-NVFP4/config.json` | 2,828 | `82fd961b632c629736902d9d4fdd3258dee1080f557cf86298cac063a514a0cf` |
| `Qwen3.8-27B-DSpark-NVFP4/hf_quant_config.json` | 937 | `cda90695e8c4a5eaed7ce7220afbc8bbe18e7624a167466ec7768c603e756a09` |

期望值的来源（决定可核验强度）：

- 权重文件的期望值取自 HF 的 **LFS oid**（即 HF 以此 sha256 标识该对象）：目标模型的 `model-00002-of-00002.safetensors`、草稿的 `model.safetensors`。两者均为体积最大的文件，LFS oid 是可获得的最强凭据。
- 非 LFS 的小文件 HF 不提供 LFS oid，改以 `raw/main` 的**实际下载内容哈希**作为基准：目标的 `config.json`、`hf_quant_config.json`、`crc32.txt`、`model.safetensors.index.json`；草稿的 `config.json`、`hf_quant_config.json`（这两个另外与 HF 同路径文件做了逐字节 `diff`，结果 IDENTICAL）。

⚠️ **命名陷阱（下载后必须检查）**：本次下载中，下载工具给两个权重文件各追加了 `_.safetensors` 后缀，得到 `model-00002-of-00002.safetensors_.safetensors` 与 `model.safetensors_.safetensors`。**内容完全正确（sha256 已证明），但名字会导致加载失败**——`model.safetensors.index.json` 按精确文件名索引权重，草稿加载器也固定查找 `model.safetensors`。已改名修正；日后重新下载时同样要确认文件名没有多余后缀。

落位命令（在 WSL 中执行；未执行，供参考）：

```bash
# --- 目标模型：先备份、再复用分片 1、最后覆盖 5 个文件 ---
cd /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
mkdir -p .pre-final-backup
cp config.json hf_quant_config.json crc32.txt model.safetensors.index.json .pre-final-backup/
mv model-00001-of-00003.safetensors model-00001-of-00002.safetensors
cp /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/config.json \
   /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/hf_quant_config.json \
   /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/crc32.txt \
   /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/model.safetensors.index.json \
   /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/model-00002-of-00002.safetensors .

# --- DSpark 草稿：目录尚不存在，需先建 ---
mkdir -p /home/kami/models/Qwen3.8-27B-DSpark-NVFP4
cp /mnt/d/WSL/models/Qwen3.8-27B-DSpark-NVFP4/config.json \
   /mnt/d/WSL/models/Qwen3.8-27B-DSpark-NVFP4/hf_quant_config.json \
   /mnt/d/WSL/models/Qwen3.8-27B-DSpark-NVFP4/model.safetensors \
   /home/kami/models/Qwen3.8-27B-DSpark-NVFP4/
```

落位后按 A.6 第 4 步校验；旧分片 2、3 确认无误后再删。

> 提醒：更新完成后 `config.json` 的 `mtp_num_hidden_layers` 为 `0`，本项目现有的 `--spec-method mtp` 加速路线（`scripts/start-api-server-mtp.bat`）随之失效，需改用 DSpark。

---

## 七、资料来源

- 目标模型卡与 README（最后修改 2026-09-10，commit `5b7a687f…`）
  https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
- 目标模型分支与文件元数据（HF API：`refs` / `tree` / `raw`）
  `.../api/models/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/{refs,tree/main,tree/pre-final}`
- DSpark 草稿模型卡（commit `eba1ac5a…`）
  https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4
- SGLang 上游 DSpark 支持（`speculative/spec_info.py`、`spec_registry.py`、`dspark_components/`）
  https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/speculative
- SGLang 投机解码文档（列出 UNO/DFLASH/EAGLE/EAGLE3/STANDALONE/NGRAM 等）
  https://docs.sglang.io/advanced_features/speculative_decoding.html
- vLLM 上游 dspark 支持（`config/speculative.py` 的 `DSparkModelTypes`/`SpeculativeMethod`、`model_executor/models/qwen3_dspark.py`；v0.27.1 / v0.28.0 / v0.29.0 均已核实含 dspark）
  https://github.com/vllm-project/vllm
- SparkInfer 项目（DSpark 目前仅 bench harness）
  https://github.com/gittensor-ai-lab/sparkinfer
- SGLang 镜像标签与摘要（Docker Hub API）
  https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/
- 本仓库既有 WSL2 + SGLang + DSpark 部署研究
  `result/sglang-qwen38-dspark-wsl2.md`
- 本仓库 vLLM/MTP 现状说明
  `README.md` 第 4.6 节、`scripts/lib/serve-lib.sh`、`scripts/start-api-server-mtp.bat`
