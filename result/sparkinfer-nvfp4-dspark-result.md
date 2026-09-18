# 实测结果：SparkInfer 复用本机 NVFP4 权重 + DSpark 草稿（WSL2，阶段 0–8）

> 执行日期：2026-09-18
> 依据：`result/sparkinfer-nvfp4-dspark-implementation-plan.md`（执行计划）、`result/sparkinfer-nvfp4-dspark-research.md`（只读调研）
> 环境：WSL2 Ubuntu 26.04 / 内核 6.18.33.2-microsoft-standard-WSL2 / Docker 29.8.1 / RTX 5090 32607 MiB / 驱动 610.88 / sm_120
> 性质：**只读复用本机权重**。全程未下载、未转换、未重新量化任何模型权重；唯一网络动作是拉取引擎镜像本身。

---

## 一、结论摘要

1. **路线跑通了，且权重零下载。** 镜像按 digest 固定，两个本地目录只读挂载，启动日志没有任何 `[sparkinfer] downloading` 行。
2. **单流 DSpark 实测 1.65×**（AR 89.7 → DSpark 147.6 tok/s，中位数，客户端流式计时），JSON 类输出最高 354 tok/s（约 4×）。
3. **并发是硬短板：聚合吞吐完全不随并发增长。** 1→16 并发，AR 聚合 87.5→90.8 tok/s；请求确实并行（服务端 `active_requests=8`、`sum(generation_ms)/耗时 ≈ 8`），但每请求速率被等比例摊薄。**厂商模型卡"8 并发 344 tok/s"在本机没有复现。**
4. **长上下文正确性门通过。** 30/1.5k/4k/16k/31k/44k 提示 × 3 次 × 两种探针，默认路径与 `DETERMINISTIC=1` 路径**各 33/33、加固 6/6，共 72 次全部正确且一致**——上游记录的 int8-KV ≥2048 token 预填充缺陷在 0.5.10 上**没有复现**。
5. **执行偏差（必须知道）**：计划里的规避变量 `SPARKINFER_PREFILL_ATTN_GQA_RQH` **在 0.5.10 的二进制里不存在**，该 A/B 无法按原计划执行（详见 8.1）。
6. **一项计划之外的重要发现**：该引擎**默认模式不是逐位可复现的**，开放式提示下两次相同请求会给出不同回答；`SPARKINFER_DETERMINISTIC=1` 才是逐位可复现模式。在该模式下 **DSpark 与 AR 输出逐字节一致（8/8）**，即投机是无损的。

---

## 二、阶段 0 · 前置检查与状态冻结

全部通过，与计划快照一致：

| 项 | 实测值 | 判据 |
| --- | --- | --- |
| 端口 | 8192 / 8080 空闲 | ✅ |
| 容器 | 仅 `qwen38-sglang`（Exited 6 小时前），无引擎在跑 | ✅ |
| 显存 | 2194 MiB 已用 / 29994 MiB 可用 | ✅ |
| 磁盘 | `/` 可用 868 GB | ✅ |
| 必需文件 | 目标 `config.json` + `tokenizer.json`、草稿 `config.json` + `model.safetensors` 四项全 OK | ✅ |
| **权重指纹** | `tensors 2387 mtp 0 size 17915815528` | ✅ 与计划期望值逐字一致 |
| SparkInfer 镜像 | ABSENT（需拉取） | ✅ 符合预期 |

---

## 三、阶段 1 · 拉取并固定镜像

```
Digest: sha256:d519d6ed995cf24f4a00c224733082166cfcb56ebaefe45ff94f80e9f518bbe1
Size:   1467233780 字节 (1.37 GiB)
Created: 2026-09-17T21:05:32Z
revision: ab936d96a306c1febd38c2640c28024c5e3dc905
version:  0.5.10
```

- **digest 与调研文档预测的完全一致**，说明拉到的就是调研时分析的那份镜像。
- **对计划数值的一处更正**：计划写的通过判据是"`size` 约 4.2e8 量级"。`docker image inspect .Size` 报的是**解包后**大小（1.47e9），4.19e8 是**压缩层合计**（调研文档里的 419,355,498 字节）。两个数都对，但计划的判据把两者混为一谈了；本计划实际按 digest 固定，不受影响。
- 镜像入口与默认环境已确认：`ENTRYPOINT ["/opt/sparkinfer/gittensor-entrypoint.sh"]` → `/opt/sparkinfer/entrypoint.sh` → `/opt/sparkinfer/bin/sparkinfer_server`；`MODEL_DIR=/models/qwen38-nvfp4`、`DRAFT_DIR=/models/qwen38-dspark`、`PORT=8080`。**未使用 `latest`。**

---

## 四、阶段 2 · `--dry-run` 验证 argv

启动器产出的真实命令（节选）：

```
docker run --name qwen38-sparkinfer --gpus all -p 8192:8080 \
  -v /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090:/models/qwen38-nvfp4:ro \
  -v /home/kami/models/Qwen3.8-27B-DSpark-NVFP4:/models/qwen38-dspark:ro \
  -e HF_HUB_OFFLINE=1 -e CTX=131072 -e MODEL_NAME=Qwen3.8-27B-NVFP4-DSpark \
  ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10 serve-dspark
```

通过判据逐条核对：两个 `:ro` 挂载到正确容器路径 ✅、`-p 8192:8080` ✅、`serve-dspark` ✅、无 `SPARKINFER_NO_DOWNLOAD` ✅。

护栏也逐条验证过（镜像缺失 → exit 1 并给出 `docker pull` 命令；目标目录缺 `config.json`/`tokenizer.json` → exit 1；草稿不完整 → exit 1 并提示 `--no-spec`；端口被占 → exit 1；端口非法/未知选项 → exit 2）。

> **一处刻意偏离**：计划 3.4 明确"不 `-d`、不 `--restart`、**不 `--rm`**"（理由是保留容器以便 `docker logs` 回看），但阶段 2/3/5 的示例命令里都带了 `--rm`。本执行按 **3.4 的设计决策**做（不带 `--rm`），与既有 `sglang-dspark.sh` 一致；阶段 8 的 `docker rm -f` 也印证了这个选择。

---

## 五、阶段 3 · AR 基线启动与冒烟

| CTX | `/health` | `/v1/info` max_context | 日志有无 `downloading` | 17×23 |
| --- | --- | --- | --- | --- |
| 32768 | `{"status":"ok"}` | 32768 | **无** | `391` ✅ |
| 65536 | ok | 65536 | 无 | ✅ |
| 131072 | ok | 131072 | 无 | ✅ |
| 262144 | ok | 262144 | 无 | ✅ |

**四条通过判据全部满足**（`/health` ok、`/v1/info` 与 CTX 相符、日志无下载、冒烟返回 `391`）。

加载日志（32768 档）：

```
[sparkinfer] serving /models/qwen38-nvfp4 as 'Qwen3.8-27B-NVFP4' on 0.0.0.0:8080 (ctx 32768, ..., autoregressive)
[sparkinfer-server] kv_cache: int8=1 slots=16/64 blocks=2056 resident=1.0 GiB
[sparkinfer-server] loading compressed-tensors checkpoint ...
[compressed-tensors] NVFP4 lm_head kept for wide packed decode (0.81 GB)
[compressed-tensors] loaded 64 layers, native NVFP4 prefill FFN 64/64, decode FFN NVFP4
[sparkinfer-server] prefix cache: on (32 entries, 8192 MiB host, ...)
[sparkinfer-server] vision tower ready: 27 blocks, out_hidden=5120
[sparkinfer-server] model ready: /models/qwen38-nvfp4
```

两个对后续判断很关键的实测事实：

1. **`int8=1`**——int8 KV 在 `--ctx ≥ 4096` 时默认开启，即"缺陷相关路径"是活的。
2. **采样默认值来自 `generation_config`**：
   `sampling defaults from generation_config.json: temperature=1.00 top_k=20 top_p=0.95 ... (SPARKINFER_SAMPLING_DEFAULTS=greedy to decode those greedily)`——证实计划 3.5/D2 的判断：**不带 `temperature` 的请求属于采样请求，DSpark 不会生效。**

### 5.1 `CTX=262144` 能启动但没有余量（重要的计划外发现）

`--ctx 262144` 下实测占用 **31733 MiB / 仅剩 455 MiB**。余量不足导致引擎在**仅 4k 提示**时就被迫降级：

```
[compressed-tensors] NVFP4 lm_head released for a 1472-token batched prefill (the scratch arena needs the VRAM more)
[prefill] ffn chunk 3754 -> 1877 (ctx=3754, free=681 MB) to keep the batched pass
```

即：会**释放 NVFP4 `lm_head`**（正是 "wide packed decode" 用的那份）并把 prefill 按块切开。计划说"AR `--ctx 262144` 需 27.9 GB"，实测约 31.0 GiB，**比计划估计的更吃紧**。结论：262144 可以"起得来"，但**不适合日常长提示**；65536（余 8.0 GB）与 131072（余 5.4 GB）才是可用区间。

---

## 六、阶段 4 · 长上下文正确性门（强制）

### 6.1 探针本身的校验（先证工具，再证引擎）

第一版"埋针"问题写成了 `...in the form X-0000`，模型**照抄占位符**返回 `X-0000`；且在下限 1.5k 提示上也失败——说明**是工具坏，不是引擎坏**。因此在 890 token（低于 2048 悬崖）上先做了措辞筛选：

| 候选问法 | 短长度结果 |
| --- | --- |
| `...reply with only the value that follows "code=" on that line` | **3/3 正确** |
| `Quote verbatim the one line above that begins with ERROR` | 0/3（截断） |
| `What is the error code on it?` | 3/3（但较长） |
| `What authorization token is given above for the Aurora-9 rollout?` | **3/3 正确** |

只有通过短长度校验的问法才被用于长上下文门。最终使用两类探针：**算术（唯一答案 `391`）** 与 **埋针召回（`E-7741` / `ZQ7K4`）**。

> 复现要求：所有请求 `temperature: 0`、`enable_thinking: false`；**前缀缓存关闭**（`SPARKINFER_PREFIX_CACHE=0`）。理由是命中前缀缓存会让第 2/3 次跳过"正在被测的那一步 prefill 注意力"，从而掩盖缺陷而不是测量缺陷（issue #976 也是在无前缀缓存下测的）。

### 6.2 结果：72 次探针全部通过

| 实验（提示规模） | 默认路径（int8 KV 开） | `DETERMINISTIC=1` 路径 |
| --- | --- | --- |
| A ~30 tok 算术 | 3/3 ✅ 一致 | 3/3 ✅ 一致 |
| A2 ~1.5k 埋针 | 3/3 ✅ | 3/3 ✅ |
| A3 ~1.5k 算术 | 3/3 ✅ | 3/3 ✅ |
| B ~4k 算术（#976 的 flaky 区） | 3/3 ✅ | 3/3 ✅ |
| B2 ~4k 埋针 | 3/3 ✅ | 3/3 ✅ |
| C ~16k 埋针 | 3/3 ✅ | 3/3 ✅ |
| C2 ~16k 算术 | 3/3 ✅ | 3/3 ✅ |
| D ~16k 事实埋针 | 3/3 ✅ | 3/3 ✅ |
| E ~31k 算术（#976 的非确定区） | 3/3 ✅ | 3/3 ✅ |
| E2 ~31k 埋针 | 3/3 ✅ | 3/3 ✅ |
| E3 ~31k 事实埋针 | 3/3 ✅ | 3/3 ✅ |
| **小计** | **33/33** | **33/33** |
| F/F2 ~44k 埋针 @20k（计划可选加强，即 0.5.8 修的那个场景） | **6/6 ✅** | — |

**结论（D1 决策）**：默认路径通过，"埋针召回"在 31k 与 44k 都正确——**因此启动器默认不带任何正确性规避开关**（计划 D1 的规则是"默认路径不可靠才带"）。两条路径的 TTFT 差异也在噪声内（16k 算术：1193.9 ms vs 1190.6 ms；31k：2616.8 ms vs 2609.2 ms）。

### 6.3 与上游记录的差异

上游 `server/README.md` 至今仍把"int8 KV 下 ≥2048 token 的 GQA 融合 prefill 注意力不准且不可复现"列为 **known defect, left ON by default**，issue #976 也记录过 4k+ 长提示答错。**本机在 0.5.10 上未复现**：4k、7.6k、16k、24k+、31k、44k 各区间都正确且 3 次一致。这不构成"缺陷已修"的结论（我们只测了一种提示族、两种探针），只说明**在本项目的使用形态下没有挡住路线**。

---

## 七、阶段 5 · DSpark 启动与生效验证

### 7.1 加载与模式（判据 1）

```
[sparkinfer] serving /models/qwen38-nvfp4 as 'Qwen3.8-27B-NVFP4-DSpark' on 0.0.0.0:8080 (ctx 131072, ..., DSpark (/models/qwen38-dspark))
[dflash] YaRN: factor=32.0 orig_max=8192 att_scale=1.3466
[dflash] loaded draft: layers=5 H=5120 B=7 n_cap=5 mask=248077 markov_rank=256 confidence=1
[sparkinfer-server] speculative decoding: DSpark draft /models/qwen38-dspark (block 7, draft context 16384)
```

✅ 出现 DSpark 模式且指向正确挂载路径；**没有** `speculative decoding off`。草稿自身注意力窗口实测默认 16384，与调研一致。

### 7.2 投机生效与边界（判据 2、3）

| 请求类型 | `speculative_runs_total` 增量 | 判定 |
| --- | --- | --- |
| 3 个合格请求（greedy + 纯文本 + 无 tools，提示各不相同） | **+3**（投机 token +487） | ✅ 生效 |
| `temperature: 0.7` | **+0** | ✅ 按文档拒绝 |
| 带 `tools` | **+0** | ✅ 按文档拒绝 |
| **命中前缀缓存**（确认 `prefix_cache_hits_total` +1、复用 1216 token） | **+0** | ✅ 按文档拒绝 |

> 命中前缀缓存那条一开始测错了：短提示（<1024 token）根本进不了缓存（`checkpoints from 1024 tokens`），且同一容器内没发过就谈不上"命中"。改用 1196 token 长提示、先用**非合格**请求（`temperature 0.9`）喂出缓存条目，再 greedy 重放同一提示，才拿到"确认命中且不投机"的干净结果。这也解释了为什么投机请求自己**不会**产生缓存条目（`entries` 长期为 0）：投机路径不写前缀缓存。

**判据 3 通过**：反例请求确实不增加投机计数。

### 7.3 抽检输出一致性（判据 4）——需要正确的前置条件

这一条**第一版测出来是 4/8 不一致**，追查后发现两个测量错误，值得记下来：

1. **采样参数没钉住**：AR 服务端从 `generation_config` 取 `top_k=20/top_p=0.95`，DSpark 服务端处于 `greedy` 默认，两边"省略的参数"取到了不同值。
2. **更关键：该引擎默认模式不是逐位可复现的。** 在**同一台服务**上重复发同一请求：
   - DSpark 服务、钉住 `temperature:0/top_k:1/top_p:1`：两次只有 3/8 相同；
   - AR 服务、同样钉住：两次只有 5/8 相同；
   - AR 服务、只给 `temperature: 0`：6/8 相同；
   - AR 服务、`SAMPLING_DEFAULTS=greedy` 且请求**不带任何采样参数**：4/8 相同。
   - 只有 **`SPARKINFER_DETERMINISTIC=1`** 下：**8/8 相同**（服务端自报 `DETERMINISTIC=1 (bit-reproducible)`）。

   也就是说：**"同样输入必须同样输出"要求 `DETERMINISTIC=1`，默认模式不提供这个保证**。这不是 SparkInfer 独有的问题（`generation_config` 里 temperature=1.0 本身就是采样），但它是**做评测/对拍时必须先想到的前提**。上游自己的 lossless 门禁也是在 `SPARKINFER_DETERMINISTIC=1` 下测的。

在正确前置条件（两侧都 `SPARKINFER_DETERMINISTIC=1`，请求 `temperature: 0`）下重测：

| 比较 | 结果 |
| --- | --- |
| DSpark 两次运行自比 | **8/8 逐字节一致** |
| **DSpark vs AR（无损判据）** | **8/8 逐字节一致** ✅ |

该轮两个 pass 期间 `speculative_runs_total` 从 0 → **16**（8 条提示 × 2 轮，每条都真的走了投机），所以这个"一致"是对**真正在用投机**的请求测出来的，不是投机没生效的假一致。

**判据 4 通过**（在 `DETERMINISTIC=1` 前提下，本机抽样验证；不替代官方逐字节门禁）。

### 7.4 并发下的交接行为

调研与上游都说"第二个请求到来时，正在投机的请求会交接"。**在并发阶梯中确实观察到了**：`sparkinfer_speculative_handoffs_total` 从 0 → 3 → 6 递增。但单独构造的"两个请求重叠"小实验里没有触发（0 次交接，两个请求各自都投机了）——说明交接取决于交叠时机，不是"只要有第二个请求就必然交接"。`tier_stops_total` 全程为 0。

---

## 八、阶段 6 · 性能与并发对比

测量口径：客户端流式计时（`decode_tok/s` 只算首 token 之后），AR 与 DSpark 均在 `--ctx 131072`、**前缀缓存关闭**下测，避免缓存命中污染。

### 8.1 单流

| 指标 | SparkInfer AR | SparkInfer + DSpark | 比值 |
| --- | --- | --- | --- |
| decode tok/s（中位数，8 条提示） | **89.7** | **147.6** | **1.65×** |
| decode tok/s（均值） | ~86 | 159.2 | 1.85× |
| TTFT 中位数 | 95.7 ms | 123.2 ms | 1.29× |
| 最快单条（JSON 输出） | 86.9 | **354.3** | **4.08×** |

与上游区间一致（聊天 1.47×、JSON 3.45×、4k 4.01×）；**引用区间而不是单点**。

### 8.2 并发阶梯

AR（192 token/请求）：

| 并发 | 耗时 s | 聚合 tok/s | 每请求 tok/s |
| --- | --- | --- | --- |
| 1 | 2.20 | 87.5 | 87.5 |
| 2 | 4.41 | 87.0 | 43.5 |
| 4 | 8.60 | 89.3 | 22.3 |
| 8 | 17.05 | 90.1 | 11.3 |
| 16 | 33.81 | **90.8** | 5.7 |

DSpark（96 token/请求；16 并发档未测，见 10.4）：

| 并发 | 耗时 s | 聚合 tok/s | 每请求 tok/s |
| --- | --- | --- | --- |
| 1 | 1.06 | 90.7 | 90.7 |
| 2 | 2.29 | 83.7 | 41.9 |
| 4 | 4.45 | 86.3 | 21.6 |
| 8 | 8.70 | 88.3 | 11.0 |

**两条线在并发下都是平的**（约 84–91 tok/s），DSpark 在并发下没有额外收益。

注意 DSpark 阶梯里 `conc=1` 只有 90.7 tok/s，而 8.1 的单流中位数是 147.6 tok/s——差异来自**提示类型**：阶梯用的是散文式摘要提示，接受率低；单流那 8 条里包含 JSON/代码类提示（最高 354 tok/s）。这正是上游"按区间引用、不要引用单点"的原因，也说明**DSpark 的收益是强提示相关的**。

**聚合吞吐完全不涨。** 为排除"是不是我的客户端串行发了请求"，做了服务端自证实验：8 个并发**非流式**请求，服务端自报 `active_requests=8`、`waiting_requests=0`，`sum(generation_ms)/耗时 = 7.98`——请求确实在并行处理，不是排队。也就是说**并行度没问题，但批量没有带来吞吐收益**：单流本身已接近显存带宽上限（约 18 GB 权重 × 89 tok/s ≈ 1.6 TB/s，RTX 5090 约 1.79 TB/s），而引擎没有把一次权重读取摊薄到多条序列上。

**与厂商数据的差异**：模型卡 `--ctx 40960` 口径给的是 1/2/4/8/16 = 88.9/162.1/249.6/344.3/345.1 tok/s。本机在**同口径 `--ctx 40960`** 下复测 8 并发 = **88.9 tok/s**——恰好等于模型卡**1 并发**的数值；16 并发 = 22.49 s / 16×128 token ≈ 91 tok/s。**344 tok/s 没有复现**，差异原因未定位（可能是 0.5.10 与模型卡所用版本的差异、WSL2、或 `SPARKINFER_BATCH_TOKENS`/调度策略默认值；`continuous batching enabled (policy=0, batch=64)` 是开的）。

### 8.3 与 vLLM / SGLang 的同口径对比——**经确认不做**

计划 6.1 要求"与 vLLM 200k、SGLang + DSpark 163840 同口径成表"。本次**只测了 SparkInfer 两条**；vLLM 与 SGLang 的**同口径重测在执行收尾时经用户确认取消**（各自需要再起一次引擎，vLLM 首次启动还可能触发 FlashInfer JIT 编译，时间不可控）。因此下表中的 vLLM / SGLang 数字来自本项目既有记录，**口径不同，不可直接与上面两行划等号**：

| 路线 | 上下文 | 加速方式 | 并发表现 | 数据来源 |
| --- | --- | --- | --- | --- |
| vLLM + NVFP4 | 200000 | MTP（可选） | 16 并发已实测良好 | 既有 `README` / 本仓库记录 |
| SGLang + DSpark | 163840 | DSpark | 单请求；本机实测 1.80×（85.75→154.12 tok/s） | `result/qwen38-nvfp4-dspark-upgrade-result.md` |
| **SparkInfer AR** | 131072 | — | **聚合吞吐不随并发增长** | 本次实测 |
| **SparkInfer + DSpark** | 131072 | DSpark（引擎内） | 单流 1.65×；并发下投机按设计交接 | 本次实测 |

**为什么这不影响 D4 结论**：D4 的证据是自足的——SparkInfer **自身**在 1→16 并发下聚合吞吐持平（8.2 节，含服务端 `active_requests`/`sum(generation_ms)` 自证不是排队），所以"并发形态下换到这条线不会有收益"这个判断不需要与 vLLM 对比即可成立。反过来说，如果将来要把 SparkInfer 提为默认路线，**那时才必须**补做这份同口径对比。

未取到的另一档：SparkInfer DSpark 的 16 并发（1/2/4/8 已完成，见 8.2）。

### 8.4 D4 决策：**不把 SparkInfer 提为日常默认**

依据：
- 并发形态下**没有任何收益**（聚合吞吐持平，每请求速率被摊薄）；本项目日常是 16 并发（`FULL_MAX_NUM_SEQS=16`），换成这条线会**显著变差**。
- 单流确有收益（1.65×），但 SGLang + DSpark 这条**已验证**的快档已经覆盖单流加速需求，且上下文更高（163840 vs 131072）。
- 上下文档位不如现有 vLLM（200000）与 SGLang（163840）；262144 档实际无余量、长提示降级。

**定位**：SparkInfer 作为**长上下文 / 单流的可选档**保留，不进默认路径；日常多客户端继续 vLLM。

---

## 九、阶段 7 · 固化的产物

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `scripts/sparkinfer-serve.sh` | 新建（LF，纯 ASCII） | 前台运行；`start`/`help`；参数 `--model-dir` `--draft-dir` `--port` `--context-length` `--no-spec` `--model-name` `--image` `--name` `--no-download` `--lan` `--dry-run`；环境变量 `SPARKINFER_IMAGE`/`_NAME`/`_CTX`/`_MODEL_NAME`/`_SAMPLING_DEFAULTS`/`_KV_INT8`/`_DSPARK_MAX_CTX`/`_EXTRA_ARGS`；preflight 覆盖 docker/daemon/镜像/两个目录必需文件/nvidia-smi/端口，任一失败都给出可执行修法 |
| `scripts/start-api-server-sparkinfer.bat` | 新建（CRLF、无 BOM、纯 ASCII，实测 305 CRLF / 0 裸 LF / 0 非 ASCII） | LAN 菜单 1/2/0、UAC 提权、退出清理 portproxy + 防火墙、结束提示 `docker logs qwen38-sparkinfer`；默认容器名 `qwen38-sparkinfer`、端口 8192 |
| `README.md` | 修改 | 新增 **4.9 节**（路线说明、实测数据、上下文取舍表、DSpark 生效条件、可复现性前提、四路线定位对照表）；测试清单"四套"→"五套" |
| `tests/sparkinfer-tests.sh` + `tests/fakebin/docker` + `tests/fakebin/ss` | 新建 | fakebin 模式；**85 项断言全绿**（argv 契约、三道人造围栏、前台生命周期 `-d`/`--restart`/`--rm` 三禁、每条 preflight 失败路径、LF/ASCII 静态检查、以及"不得传递 0.5.10 不认识的 `SPARKINFER_PREFILL_ATTN_GQA_RQH`"） |
| `tests/fakebin/nvidia-smi` | 修改 | 新增 `--query-gpu=name*` 分支（启动器只用名字查询）；不影响既有用例 |
| `.gitattributes` | 修改 | 把只覆盖 `tests/fakebin/vllm` 的单行规则推广为 `/tests/fakebin/* text eol=lf`（详见 9.1） |
| `tests/preflight-tests.sh`、`serve-tests.sh`、`fullcontext-tests.sh` | 工作区换行修正（内容与提交版本逐字节一致，git 不显示改动） | 详见 9.2 |
| `result/sparkinfer-prefill-attn-gqa-rqh-research.md` | 新增（子代理产出） | 8.1 偏差的一手来源证据 |
| `result/sparkinfer-nvfp4-dspark-result.md` | 新建 | 本文件 |

**五套测试全绿**（全部用 `bash tests/<name>.sh` 跑）：

| 套件 | 结果 |
| --- | --- |
| `run-tests.sh` | 51 passed, 0 failed |
| `preflight-tests.sh` | 71 passed, 0 failed |
| `serve-tests.sh` | 88 passed, 0 failed |
| `fullcontext-tests.sh` | 114 passed, 0 failed |
| `sparkinfer-tests.sh`（本次新增） | 85 passed, 0 failed |
| **合计** | **409 passed, 0 failed** |

### 9.1 计划外修正：`tests/fakebin/` 的换行属性

本机 `core.autocrlf=true`。`tests/fakebin/` 下的假工具**没有扩展名**，因此不匹配 `.gitattributes` 里的 `*.sh eol=lf`；仓库只对 `vllm` 一个文件写了例外。实测 `git` 明确警告：

```
warning: in the working copy of 'tests/fakebin/nvidia-smi', LF will be replaced by CRLF the next time Git touches it
```

也就是说下次 checkout 会把这些 fixture 改成 CRLF，而 **CRLF 会让 bash 把 CR 当成命令的一部分**——测试会在换台机器/重新 clone 后莫名其妙地失败。因为本次新增了 `docker` 与 `ss` 两个同类型 fixture，且仓库已有 `vllm` 这个先例，故把该规则推广到整个目录。验证（`git check-attr`）：`docker`/`ss`/`nvidia-smi`/`vllm`/`python3` 等全部为 `text: set, eol: lf`；`.bat` 仍为 `eol: crlf`，`.sh` 仍为 `eol: lf`。

### 9.2 计划外发现：三套既有测试在本机本来就是坏的

跑全量回归时发现 `preflight-tests.sh`、`serve-tests.sh`、`fullcontext-tests.sh` **根本无法启动**：

```
tests/preflight-tests.sh: line 5: set: -: invalid option
tests/preflight-tests.sh: line 6: $'\r': command not found
```

原因是这三个文件的**工作区副本是 CRLF**（实测 CRLF 行数 252 / 297 / 421），而 **git 里存的 blob 是 LF**（`storedCR=0`）。由于 `core.autocrlf=true`，`git status` 把"工作区 CRLF + 索引 LF"当成干净，**所以这个损坏是隐形的**——README 让用户执行的 `bash tests/preflight-tests.sh` 在这台机器上一直是失败的，只是没人跑到。

本次把三个工作区文件规范化为 LF（内容与提交 blob 逐字节一致：`git hash-object` 与 `rev-parse HEAD:<file>` 相等，`git diff` 为空；再用 `git add --renormalize` 清掉索引 stat 缓存造成的假 `M` 标记，`git diff --cached` 为空，未暂存任何内容）。规范化后三套测试分别 71/88/114 全绿。

**这与 SparkInfer 无关**，属于既有仓库缺陷；但由于执行计划的验收标准 8 要求"`tests/` 全绿"，且它就在本次新增测试套件的同一目录，故一并修掉并如实记录。

---

## 十、执行偏差与未完成项（必须对照）

| # | 计划条目 | 实际处理 | 原因 / 影响 |
| --- | --- | --- | --- |
| 1 | 阶段 4 的 D 组用 `SPARKINFER_PREFILL_ATTN_GQA_RQH=1` 做 A/B | **改用 `SPARKINFER_DETERMINISTIC=1`** | 该变量在 0.5.10 二进制里**不存在**（对 `SPARKINFER_[A-Z0-9_]+` 全字符串扫描，40 个变量里没有它）；上游文档仍写着它，CHANGELOG 里 `RQH`/`#976` 出现 0 次。`DETERMINISTIC=1` 是二进制实际支持、且被上游文档描述为"同样会关掉 ~2048 token 以上的融合 prefill"的等价手段。**影响**：A/B 结论基于 DETERMINISTIC 路径而非 RQH 路径 |
| 2 | 阶段 2/3/5 示例命令带 `--rm` | 按 3.4 不带 `--rm` | 计划内部自相矛盾；取更权威、更有理由的设计决策（保留容器以便 `docker logs`） |
| 3 | 阶段 6 与 vLLM / SGLang 同口径成表 | **经用户确认不做**（收尾时取消） | 需各自再起一次引擎、vLLM 可能触发 JIT 编译，时间不可控；D4 结论不依赖它（见 8.3）。**若将来要把 SparkInfer 提为默认路线，必须先补做** |
| 4 | 阶段 6 并发阶梯 1/2/4/8/16 | AR 完整；DSpark 做了 1/2/4/8 | 16 并发投机下 `/metrics` 一度超时（服务端在高并发投机场景响应退化），未取到该档数据；不改变结论 |
| 5 | 阶段 1 通过判据 `.Size ≈ 4.2e8` | 实测 1.47e9 | 计划把"压缩层合计"当成了 `.Size`（解包后大小）；按 digest 固定，不影响验收 |
| 6 | 计划未预见：默认模式非逐位可复现 | 记录为独立发现，写入 README 4.9 | 影响任何"同输入同输出"的对拍/评测；`DETERMINISTIC=1` 是前置条件 |
| 7 | 计划未预见：`--ctx 262144` 无余量、长提示降级 | 记录并写入 README 4.9 | 计划称需 27.9 GB，实测约 31.0 GiB 且 4k 提示即触发 `lm_head` 释放与 prefill 切块 |
| 8 | 计划未预见：`tests/fakebin/` 无扩展名脚本会被 `core.autocrlf=true` 改成 CRLF | 修 `.gitattributes`（把只覆盖 `vllm` 的单行规则推广到整个目录） | 本机 `core.autocrlf=true`，git 明确警告会把 `nvidia-smi` 等改写为 CRLF；CRLF 会让 bash 把 CR 当成命令的一部分。属计划外修正，见 9.1 |
| 9 | 计划未预见：三套既有测试在本机**本来就是坏的**（工作区 CRLF、索引 LF，`git status` 还显示干净） | 规范化为 LF（内容与 blob 逐字节一致，未暂存任何改动） | README 让用户跑的 `bash tests/preflight-tests.sh` 一直是失败的。属既有仓库缺陷，见 9.2 |

---

## 十一、验收标准逐条核对

| # | 标准 | 结果 |
| --- | --- | --- |
| 1 | 阶段 0 全绿，指纹 2387 张量 / 0 mtp | ✅ 完全一致 |
| 2 | 镜像按 digest 固定并记录，未用 `latest` | ✅ digest 已记录，与调研预测一致 |
| 3 | 启动日志无任何 `downloading` 行 | ✅ 四个 CTX 档、AR 与 DSpark 均无 |
| 4 | `/health`、`/v1/info`、`/v1/chat/completions` 可用且 OpenAI 兼容 | ✅ |
| 5 | 阶段 4 长上下文 A/B 有明确结论并按结论固化 | ✅ 72 次探针全过，默认路径无需规避开关 |
| 6 | 投机计数 > 0 且反例不增加 | ✅ +3 生效；0.7 / tools / 缓存命中三条反例均 +0 |
| 7 | 阶段 6 与 vLLM / SGLang 同口径对比并给出默认路线结论 | ⚠️ **对比经确认不做**（收尾时取消）；**默认路线结论已给出**（D4：不把 SparkInfer 提为默认） |
| 8 | 启动器 `--dry-run` 可打印完整命令；`tests/` 全绿；README 4.9 落地 | ✅ 五套 **409/409** 全绿，README 4.9 已落地 |
| 9 | 回滚路径可执行 | ✅ 见第十二节 |

**明确不算通过的情形，逐条对照**：
- 投机计数为 0 却声称已启用 → **未发生**（计数 +3，且反例为 0 已单独验证）。
- 长提示答错却因速度快而放行 → **未发生**（长提示全对；且速度结论并未用于放行）。
- 用 `latest` 跑通后未记录 digest → **未发生**（全程 `0.5.10`，digest 已记录）。
- 为让 DSpark 生效把服务端整体改 greedy 却未说明影响 → **未发生**：固化默认保持 `generation_config`，文档明确要求 `temperature: 0`，并把"`SPARKINFER_SAMPLING_DEFAULTS=greedy` 只用于验证"写进 README 4.9 与启动器帮助。

---

## 十二、阶段 8 · 回滚与收尾

```bash
# 1) 停服务并删容器（前台运行时 Ctrl-C 即可；兜底：）
docker rm -f qwen38-sparkinfer
# 2) 如需回收镜像空间（约 1.4 GB）
docker rmi ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10
# 3) 撤销本项目改动（脚本/文档/测试）
git -C /mnt/d/Code/MJ-Project/ai-model-nvfp4 status
git -C /mnt/d/Code/MJ-Project/ai-model-nvfp4 diff
```

**回滚不涉及**：模型文件、草稿、vLLM venv、SGLang 镜像、`.wslconfig`、Windows 驱动——本次从未修改它们。现有 `qwen38-sglang` 容器全程保持 `Exited` 状态未动。

---

## 十三、给后续的一句话建议

SparkInfer 这条路**可以留着当长上下文/单流可选档**（权重零下载、镜像仅 1.4 GB、单流 1.65×、长上下文正确性已验），但**不要用于多客户端并发**；DSpark 的"无损"结论只在 `SPARKINFER_DETERMINISTIC=1` 下成立；需要同输入同输出时，任何对比实验都必须先设这个变量，否则测到的是引擎的默认非确定性。

---

## 十四、复现说明

本文档 6/7/8 节记录了每个实验的**请求构造**（提示规模与埋针位置、`temperature`、`enable_thinking`、`max_tokens`、以及必须关闭前缀缓存），据此可以重写测量脚本。本次使用的实测脚本是 `.scratch/` 下的临时脚本（`stage4-gate.py` 正确性门、`harness.py` 计数/对拍/性能、`concurrency-diag.py` 并发自证、`dspark-*.py` 边界控制），**执行完毕后已清理**，未作为固化产物保留——固化的产物是第九节列出的启动器、文档与测试套件。唯一需要人工构造的判据是"埋针"，其坑已记录在 6.1 节（问法里不能出现任何看起来像答案的占位符）。