# 实测结果：SparkInfer DSpark 的上下文甜点值与上限 + 图片识别

> 执行日期：2026-09-18 晚（主结果 `result/sparkinfer-nvfp4-dspark-result.md` 同日续测）
> 依据：`result/sparkinfer-nvfp4-dspark-result.md`（阶段 0–8 主结果）、`result/sparkinfer-nvfp4-dspark-research.md`（只读调研）
> 环境：同主结果——WSL2 Ubuntu / Docker 29.8.1 / RTX 5090 32607 MiB / 镜像 `ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10`（digest `sha256:d519d6ed995cf24f4a00c224733082166cfcb56ebaefe45ff94f80e9f518bbe1`）
> 性质：只读复用本机权重，未下载任何模型权重。新增产物只有测试图片与测量脚本（`.scratch/sparkinfer-ctx163840/`）。

---

## 一、结论摘要

1. **DSpark @ ctx 163840 在这台机器上不可用——不是装不下，而是装得下却性能塌陷。** 引擎正常加载、正常投机（8 条提示 → `sparkinfer_speculative_runs_total` **+8**），但：
   - 长输出（512 token/请求）下 decode 只有 **22.8 / 43.9 tok/s**，而同一台机器同一批提示在 ctx 131072 是 **125.9 / 188.9 tok/s**；
   - **比同档位的 AR 还慢**：水文摘要 22.8 vs AR 50.8（**0.45×**），BST 代码 43.9 vs AR 50.4（**0.87×**）。也就是说投机在这一档不但没有收益，反而拖慢；
   - 连 AR 本身也腰斩：95.1 → 50.4–50.8 tok/s（512 token 输出），短输出档 95.4 → 70.5 tok/s。
   - **根因已由 AR-only 对照定位：不是 ctx 太高，而是草稿。** 同一 ctx 163840、同一个引擎，去掉草稿（`--no-spec`）后 AR 回到 **95.2 tok/s**、功耗回到 **445 W**，与 131072 档几乎相同（详见 4.1）。因此结论应表述为：**ctx 163840 本身可用，但"ctx 163840 + DSpark 草稿"在这台 32 GB 卡上不可用**——草稿多占 **3637 MiB**，把加载后余量从 ~5.2 GB 压到 ~1.5 GB（请求期峰值只剩 ~0.7 GB）。
   - **附带确认：KV 不是元凶。** 引擎的 KV 默认就是 8-bit（日志 `kv_cache: int8=1`，`SPARKINFER_KV_INT8` 在 `--ctx ≥ 4096` 时默认开启），实测成本恒为 **32 KB/token**（16 个全注意力层 × 2 × 4 KV head × 256 head_dim × 1 B），163840 档 5.0 GiB、131072 档 4.0 GiB，只差 1 GiB。
2. **ctx 131072 复测正常，是这台机器上 DSpark 的可用档。** 纯文本 greedy（DSpark 生效）decode 中位数 **198.2 tok/s**（区间 134.8–300.0），纯文本采样（AR）**95.4 tok/s**，投机收益按提示类型 1.4–3.1×。与主结果记录的 1.65× 同向；本次数值偏高是因为提示集偏 JSON／代码，**不可与主结果的中位数直接比单点**。
3. **图片识别可用且第一手验证正确**（本机首次实测）：合成图上的埋针 `E-7741` 逐字正确、三个形状与颜色全对、图像描述正确。图片请求**不投机**（`speculative_runs_total` 增量 **0**，与上游文档一致），也**不进前缀缓存**。
4. **图片的代价在 TTFT 与 prompt token，不在 decode。** ctx 131072 档：图片请求 decode **92.4 tok/s ≈ 纯文本 AR 95.4 tok/s**（视觉塔只影响预填充）；但 TTFT 中位数 **726.5 ms vs 93.2 ms**（+633 ms），每张 1024×640 PNG ≈ **630 个 prompt token**（662–674 vs 纯文本 24–44）。相对 greedy 纯文本（198.2 tok/s），图片请求的输出速度慢约 **2.1×**，因为图片只能走 AR。
5. **甜点值（本次扫描结论）：推荐 `--context-length 153600`（150K token）。** 实测干净的档位上限是 **158720（155K）**，但阈值取决于启动瞬间的**绝对空闲显存**，而这台机器的桌面占用在 1.2–2.0 GB 之间波动，所以推荐留 ~2.4 GB 余量的 153600，而不是贴着上限跑。**161280 起出现"512 token 请求偶发落慢路径"（256 token 请求仍 10/10 干净），163840 全面塌陷**。完整扫描见第五节。
6. **定位**：SparkInfer + DSpark 继续按主结果定位在**长上下文/单流可选档**，可用上下文从主结果的 131072 提升到 **153600（推荐）/ 158720（实测上限）**；`--no-spec` 档在 163840 仍正常。中间未被本次覆盖的点（如 156160、160000）未测。

---

## 二、实验设计

三个臂，**单请求串行**（DSpark 只在唯一活跃请求时生效）：

| 臂 | 请求 | 采样 | 预期路径 |
| --- | --- | --- | --- |
| `text-dspark` | 8 条纯文本提示 | `temperature: 0`（greedy） | 投机 |
| `text-ar` | 前 4 条提示 | `temperature: 0.7` | 采样 → AR（同提示对照） |
| `image` | 同一张图 + 3 个问题 | `temperature: 0` | 图片 → AR |

公共设置：`enable_thinking: false`、`max_tokens: 192`（定点 A/B 为 512）、`stream: true`、**前缀缓存关闭**（`SPARKINFER_EXTRA_ARGS=SPARKINFER_PREFIX_CACHE=0`）、不带 `tools`。

计时口径：客户端流式计时，`decode tok/s = (completion_tokens − 1) / (末 token 时刻 − 首 token 时刻)`，即只算首 token 之后，与主结果一致。每个臂前后各读一次 `/metrics`，记录投机计数增量；另记录服务端 `usage` 的 `prompt_tokens`。

两个 ctx 档各跑一轮完整三臂，再各跑一轮 512 token 定点 A/B（同两条提示、greedy 与 `temperature: 0.7` 各一次）。**163840 档的三臂跑在刚加载完的引擎上**（排除"引擎跑了很久才变慢"的解释），512 token A/B 随后复现同样结论。

**第四个对照是事后追加的**：为了把"ctx 高"与"带草稿"两个变量分开，又在 **ctx 163840 下用 `--no-spec` 起了一次纯 AR 引擎**（无草稿、无草稿挂载），跑同一套 512 token 定点 A/B 与 8 条短输出提示。这一轮是本次唯一能定因的实验，见 4.1。

提示词（可原样复现）：`chat` 解释天空为何呈蓝色、`json` 生成商品 JSON、`code` 反转单链表、`math` 火车到站时间、`list` TCP/UDP 五条区别、`prose` 约 120 词水循环、`json2` 日本城市 JSON 数组、`code2` 找 /var 下最大文件的 bash 一行命令；定点 A/B 用 `prose`（约 400 词水循环）与 `code`（BST 完整实现）。

图片：`make-image.ps1` 生成 1024×640 白底 PNG，含 `AURORA-9 CLEARANCE`、`CODE: E-7741`、`issued 2026-09-18 / station 42`，以及红色圆、绿色方、蓝色三角。

第五节的上下文扫描复用同一套脚本，另加 `probe.py` 重复性探针（同一提示重复 6 次 code / 4 次 prose，256 token）来判断某个 ctx 是"干净快"还是"间歇落慢路径"。

---

## 三、131072 与 163840 两档完整对照（max_tokens 192，单请求串行）

| 臂 | ctx 131072 decode 中位数（区间） | TTFT 中位数 | 投机计数增量 | ctx 163840 decode 中位数（区间） | TTFT 中位数 | 投机计数增量 |
| --- | --- | --- | --- | --- | --- | --- |
| 纯文本 greedy | **198.2**（134.8–300.0） | 122.0 ms | +8 次 / +1022 token | **65.0**（55.8–109.0） | 161.9 ms | +8 次 / +1016 token |
| 纯文本 `temperature 0.7` | **95.4**（95.1–95.6） | 93.2 ms | +0 | **70.5**（70.3–70.6） | 98.5 ms | +0 |
| 图片 greedy | **92.4**（90.6–93.8） | 726.5 ms | **+0** | **66.9**（66.4–69.4） | 776.6 ms | **+0** |

逐条数值（ctx 131072，纯文本 greedy）：chat 134.8、json 209.8、code 300.0、math 291.1、list 157.5、prose 150.8、json2 263.8、code2 186.7 tok/s。
逐条数值（ctx 163840，纯文本 greedy）：chat 59.6、json 60.1、code 109.0、math 104.6、list 61.9、prose 55.8、json2 90.9、code2 68.2 tok/s。

两条可直接对照的结论：

- **贪婪文本的投机收益是强提示相关的**：131072 档下 JSON／代码类 2.0–3.1×（json2 263.8、code 300.0），散文类约 1.4–1.6×（prose 150.8、chat 134.8）。引用时按区间，不要引用单点。
- **DSpark 让 TTFT 略升**：131072 档 122.0 ms vs AR 93.2 ms（+29 ms），与主结果记录的 95.7 → 123.2 ms 一致。

---

## 四、定点 A/B 与 AR-only 对照：草稿才是分水岭

### 4.1 带草稿 vs 不带草稿（同一 ctx 163840）

| 配置 | 加载后显存 | 运行峰值 | 水循环摘要 decode | BST 代码 decode | 功耗均值（SM 均值） |
| --- | --- | --- | --- | --- | --- |
| ctx 163840 + 草稿 | 31052 MiB | 31907 MiB | 22.8 | 43.9 | 203 W（2885 MHz） |
| **ctx 163840，`--no-spec`（无草稿）** | **27415 MiB** | **27929 MiB** | **94.0** | **95.0** | **445 W（2905 MHz）** |
| ctx 131072 + 草稿 | 29211 MiB | 30850 MiB | 125.9 | 188.9 | 455 W（2889 MHz） |

短输出档（8 条提示 / 4 条提示，192 token）：

| 配置 | 纯文本 greedy | 纯文本 `temperature 0.7` |
| --- | --- | --- |
| ctx 163840 + 草稿 | 65.0（DSpark） | 70.5（AR） |
| ctx 163840，无草稿 | **95.2（AR）** | — |
| ctx 131072 + 草稿 | 198.2（DSpark） | 95.4（AR） |

**读法**：同一个 ctx 163840、同一个引擎、同一批提示，**只差一个草稿**：AR 从 95.2 → 70.5（短输出）／94.0 → 50.8（512 token），功耗从 445 W → 203 W。所以：

- **ctx 163840 本身没有问题**（无草稿时与 131072 档的 AR 速率、功耗、分段平直度都一致）；
- **草稿的 3637 MiB 占用把加载后余量从 ~5.2 GB 压到 ~1.5 GB**，引擎随即换入慢路径，连非投机的 AR 请求一起变慢——这正是上游把 `serve-dspark` 默认压在 131072、并要求"装不下就启动失败"的原因（0.5.10 这里介于"装得下"与"跑得动"之间的灰区没有报错，只表现为变慢）。

### 4.2 512 token 定点 A/B（带草稿）

同两条提示、同样 512 token 输出、单请求串行；括号内为流内分段速率（第 1–128 / 129–256 / 257–384 / 385–512 个 chunk）。

| 提示 | 模式 | ctx 131072 | ctx 163840 |
| --- | --- | --- | --- |
| 水循环摘要 | DSpark greedy | **125.9**（155.0/152.4/132.2/90.5） | **22.8**（20.8/23.1/21.9/27.4） |
| 水循环摘要 | AR `temperature 0.7` | 95.1（96.3/96.1/95.7/94.4） | 50.8（42.9/45.9/60.7/60.0） |
| BST 代码 | DSpark greedy | **188.9**（310.2/383.5/210.4/95.1） | **43.9**（37.1/60.6/43.0/43.5） |
| BST 代码 | AR `temperature 0.7` | 95.1（96.2/96.2/95.7/94.6） | 50.4（46.1/47.0/53.4/58.1） |

投机相对同档 AR 的比值：

| 提示 | ctx 131072 | ctx 163840 |
| --- | --- | --- |
| 水循环摘要 | **1.32×** | **0.45×** |
| BST 代码 | **1.99×** | **0.87×** |

131072 档四个分段基本平直（94–96），说明 AR 速率稳定；163840 档 AR 从 42.9 爬到 60.7，整条曲线仍远低于 131072。

---

## 五、上下文扫描：DSpark 的甜点值（131072 → 163840）

做法：每个 ctx 起一次引擎（DSpark、前缀缓存关闭、单请求串行），记录启动后显存与 KV 池，跑一遍 512 token 定点 A/B、一遍 8 条短输出臂并采样 GPU 功耗；对候选档再加一轮**重复性探针**（同一提示重复 6 次 code / 4 次 prose，256 token，greedy），用来区分"干净快"与"间歇落慢路径"。

| ctx | KV 池 | 启动后显存 | 启动余量 | DSpark 512（code / prose） | AR 512 | DSpark 中位数（192 token） | 功耗均值 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 131072 | 4.0 GiB | 29211 MiB | ~3.4 GB ⚠️ | 188.9 / 125.9 | 95.1 | 198.2 | 455 W | 健康（主结果档） |
| **153600（推荐）** | 4.7 GiB | 30144 MiB | ~2.4 GB | 189.0 / 125.7 | 95.0 | **223.8** | 459 W | **健康** |
| 158720（实测上限） | 4.8 GiB | 30320 MiB | ~2.2 GB | 188.9 / 123.8 | 94.9 | 205.0 | — | 健康：6/6 + 4/4 完全一致 |
| 161280 | 4.9 GiB | 30410 MiB | ~2.1 GB | **122.7** / 120.9 | **90.6** | 201.3 | 433 W | **临界**：256 token 探针 10/10 干净，512 token 请求偶发落慢路径 |
| 163840 | 5.0 GiB | 31052 MiB | ~1.5 GB | 43.9 / 22.8 | 50.4–50.8 | 65.0 | 203 W | **塌陷**（连 AR 一起） |
| 163840，`--no-spec` | 5.0 GiB | 27415 MiB | ~5.1 GB | 95.0 / 94.0 | — | 95.2 | 445 W | 健康（对照） |

> ⚠️ 131072 那一行的启动显存取自**桌面占用约 1950 MiB** 的早先一轮，其余各行取自约 1200 MiB 的一轮，因此它的"启动余量"偏高约 750 MiB。横向比较只用同一轮内测得的 153600–163840 各行。

重复性探针（同一提示、greedy、256 token、前缀缓存关闭）：

| ctx | code × 6 | prose × 4 | 投机计数 |
| --- | --- | --- | --- |
| 158720 | 338.3 / 338.3 / 339.1 / 339.6 / 340.1 / 338.3 | 152.3 / 146.2 / 157.5 / 150.9 | 每次请求 +1 |
| 161280 | 351.3 / 337.7 / 337.4 / 338.3 / 350.0 / 339.0 | 150.3 / 148.4 / 143.6 / 152.0 | 每次请求 +1 |

**结论与推荐**

- **推荐 `--context-length 153600`（150K token）**：比主结果的 131072 多 17% 上下文，速度与 131072 相同（512 token 臂 189.0/125.7 vs 188.9/125.9，AR 95.0 vs 95.1），且留 ~2.4 GB 启动余量。
- **实测干净上限 158720（155K）**：6/6 与 4/4 完全一致，但启动余量只剩 ~2.2 GB。
- **不要停在 161280 及以上**：161280 的 256 token 探针虽然 10/10 干净，但同一引擎在 512 token 序列里出现过一次 code 掉到 **122.7**（同提示正常 188.9）、AR 首段掉到 **80**（正常 96）——已经踩在阈值上；163840 则在所有测试形态下全面塌陷。
- **为什么推荐值不贴着 158720/161280**：阈值取决于**启动瞬间的绝对空闲显存**，不是 ctx 本身。这台机器的桌面占用在 1.2–2.0 GB 之间波动过（本次实测到 1950 MiB），阈值会随之上下移动。为多 1.6% 的上下文丢掉这点余量不划算。
- **引用速度必须带输出长度**：同一 code 提示，256 token 输出是 **338 tok/s**，512 token 输出是 **188.9**（末段掉到 95）——投机会在下一个注意力分档边界停止，长输出的尾部回到 AR 速率。因此"198.2 tok/s"这类中位数只对 192 token 输出成立。

## 六、图片识别（ctx 131072 档，三条问答原文）

| 问题 | prompt_tokens | completion_tokens | decode tok/s | 回答 |
| --- | --- | --- | --- | --- |
| `img_code` 读出图中的 clearance code | 669 | 7 | 90.6 | `E-7741` |
| `img_shapes` 说出三个形状与颜色 | 674 | 12 | 92.4 | `circle: red` / `square: green` / `triangle: blue` |
| `img_describe` 一句话描述 | 662 | 24 | 93.8 | `The image displays a clearance code with three colored geometric shapes: a red circle, green square, and blue triangle.` |

ctx 163840 档同图同问回答一致（`E-7741`、三色全对、描述同义），只是速度随整档一起下降（按同序 90.6/92.4/93.8 → 66.9/66.4/69.4，见第三节）。

三条可操作事实：

1. **图片请求永远不投机**：两次运行三个图片请求期间 `sparkinfer_speculative_runs_total` 增量均为 **0**，与上游"vision 请求停留在 AR 或交接给 AR"的说明一致。这是我方首次一手验证。
2. **图片不进前缀缓存**（上游说明：缓存按 token id 建键，而所有图片的占位 token id 相同）。多轮图片对话每轮都要重新预填充。
3. **显存／上下文代价**：一张 1024×640 PNG ≈ 630 个 prompt token，在 ctx 131072 下约占 0.5% 预算；TTFT 从 93.2 ms 升到 726.5 ms（+633 ms），这是视觉编码 + 图片预填充的成本，与输出长度无关。

---

## 七、显存与 GPU 证据（为什么 163840 + 草稿会塌陷）

| 观测项 | ctx 131072 + 草稿 | ctx 163840 + 草稿 | **ctx 163840，无草稿** |
| --- | --- | --- | --- |
| 加载时的 KV 池（容器日志） | `blocks=8200 resident=4.0 GiB` | `blocks=10248 resident=5.0 GiB` | `blocks=10248 resident=5.0 GiB` |
| 加载后显存占用 | 29211 MiB / 32607 | 31052 MiB / 32607 | **27415 MiB / 32607** |
| 运行期间显存峰值 | 30850 MiB（余 ~1.7 GB） | **31907 MiB（余 ~0.7 GB）** | **27929 MiB（余 ~4.6 GB）** |
| SM 频率（利用率 ≥90% 的采样） | 均值 2889 MHz | 均值 2885 MHz | 均值 2905 MHz |
| 功耗（同上采样） | 均值 **455 W** | 均值 **203 W** | 均值 **445 W** |
| GPU 利用率 | 95–100% | 99–100% | 95–100% |

**读法**：三档 SM 频率基本相同、利用率都接近满载，但只有"163840 + 草稿"这一档功耗腰斩（203 W vs 445–455 W）——同一利用率下算力在等内存，属于显存压力下的慢路径表现，不是降频，也不是外部负载。同一 ctx 下"有草稿 / 无草稿"的对照把变量收敛到唯一一个：**草稿的 3637 MiB**。

这是主结果里 ctx 262144 AR 现象的同一类问题（那一档日志明确打了 `NVFP4 lm_head released` 与 `[prefill] ffn chunk 3754 -> 1877`）。**本档没有对应的日志行**：`docker logs qwen38-sparkinfer` 在 163840 档除启动行外没有任何按请求的降级输出，因此**引擎内部的具体回退分支仍未定位**——已确认的是触发条件（草稿占用使余量不足），不是代码路径。

### 7.1 为什么 vLLM / SGLang 在同样上下文不会这样

同样的权重、同样的 32 GB 卡，vLLM 跑 200000、SGLang 跑 163840（且 SGLang 那条还带 DSpark）都很快。差别不在 KV 精度，而在三条结构性策略：

| | vLLM / SGLang | SparkInfer |
| --- | --- | --- |
| KV 池分配 | 先装权重，剩下的按 `gpu_memory_utilization 0.90` / `mem-fraction-static 0.90` **自适应给 KV**（vLLM 0.90 可放 ~205k KV token，所以 200k 跑得动；SGLang 0.90 时 KV 上限 166793 token ≈ 刚好覆盖 163840） | **KV 池按整个 `--ctx` 先定死**，再加载草稿（上游原文：*The KV pool is sized for the whole `--ctx` before the drafter loads*），定死之后没有回旋余地 |
| 投机开销 | 本次对比的 vLLM 200k 档无独立草稿；SGLang 的投机是引擎内融合 | 目标模型 **+ 独立 1.41 GB 草稿 + 草稿 KV + 草稿 scratch = 实测 +3637 MiB** |
| 显存不够时 | **拒绝请求**（400/429）或抢占换出，主 kernel 路径不变 | **降级**：换小 chunk、释放 `lm_head`、退回非打包 kernel（262144 档日志有原文），表现为"能跑但每条请求都慢" |

所以"vLLM/SGLang 在 163840/200k 很快"与"SparkInfer 在 163840 + 草稿很慢"并不矛盾：前者是**自适应分配 + 拒绝策略**，后者是**按 ctx 定死 + 降级策略**。同样的显存紧张，前者表现为"接不了更多请求"，后者表现为"每条请求都慢"。另外，KV 已经是 8-bit（int8，32 KB/token），这个方向没有余量可挤——`SPARKINFER_KV_INT8=0` 只会让 KV 更大。

---

## 八、执行偏差与未做项（必须对照）

| # | 项 | 处理 | 影响 |
| --- | --- | --- | --- |
| 1 | 163840 塌陷的触发条件与机制 | **触发条件已定位**：AR-only（`--no-spec`）对照证明是草稿占用（+3637 MiB）而非 ctx 本身。引擎内部的回退分支**未定位**（无相关日志，未做 kernel 级 profiling） | 结论从"163840 不可用"改写为"163840 不带草稿可用、带草稿不可用" |
| 2 | 上下文扫描的粒度 | 测了 131072 / 153600 / 158720 / 161280 / 163840 五个点，**未测** 156160、160000 等更细的点 | 阈值只收敛到 158720–161280 之间（约 2.5K token 宽），给不出精确边界 |
| 3 | 161280 的间歇退化未完全复现 | 512 token 序列里出现过一次 code 122.7，但 256 token 探针 10/10 干净、未再复现 | 只能表述为"已踩阈值、可能出现慢路径"，不能断言必现条件 |
| 4 | 阈值与桌面显存占用耦合 | 阈值取决于启动瞬间的绝对空闲显存；131072 那一轮的桌面占用（~1950 MiB）比其余各轮（~1200 MiB）高约 750 MiB | 跨轮比较启动余量时带偏差；推荐值因此留了 ~2.4 GB 余量而不是贴着上限 |
| 5 | 图片样本 | 只测 1 张合成图（大字 + 几何图形），未测真实照片、小字 OCR、多图、视频 | "支持图片识别"结论的可信范围限于清晰、单图、文字与几何类任务 |
| 6 | 并发 | 全部为单请求串行，未测并发下的图片请求与扫描各档 | 并发结论沿用主结果（聚合吞吐不随并发增长） |
| 7 | 可复现性 | 本次为速度测量，未设 `SPARKINFER_DETERMINISTIC=1` | 回答文本可能逐次不同（默认模式非逐位可复现），速度不受影响 |
| 8 | 结果口径差异 | 本次 `text-dspark` 中位数 198.2（192 token 输出）高于主结果的 147.6 | 提示集与输出长度不同（投机会在分档边界停止），属口径差异，不是引擎变更 |

---

## 九、复现

脚本保存在 `.scratch/sparkinfer-ctx163840/`（临时区，未作为固化产物）：

| 文件 | 用途 |
| --- | --- |
| `make-image.ps1` | 生成测试图片 `image-text.png`（Windows PowerShell + System.Drawing） |
| `harness.py` | 三臂测量（纯文本 DSpark / 纯文本 AR / 图片），含 `/metrics` 投机计数增量 |
| `diag.py` | 512 token 定点 A/B，含流内分段速率 |
| `probe.py` | 重复性探针（同一提示 N 次，区分干净快与间歇落慢路径） |
| `results.json`、`results-ctx131072.json` | 两档三臂原始结果（含回答原文） |
| `results-ctx153600.json`、`results-ctx158720.json`、`results-ctx161280.json`、`results-aronly-ctx163840.json` | 扫描各档的短输出臂原始结果 |
| `diag-ctx163840.json`、`diag-ctx131072.json`、`diag-ctx153600.json`、`diag-ctx158720.json`、`diag-ctx161280.json`、`diag-aronly-ctx163840.json` | 各档定点 A/B 原始结果 |
| `probe-ctx158720-code.json`、`probe-ctx158720-prose.json`、`probe-ctx161280-code.json`、`probe-ctx161280-prose.json` | 两个候选档的重复性探针原始结果 |
| `clocks-*.log` | 各档运行期间每秒 GPU 频率／功耗／利用率采样（ctx131072 / ctx153600 / ctx161280 / ctx163840 / aronly-ctx163840） |

启动（每档一条命令；注意 `SPARKINFER_EXTRA_ARGS` 用于关闭前缀缓存）：

```bash
# 推荐档 153600（150K）
sudo env SPARKINFER_EXTRA_ARGS=SPARKINFER_PREFIX_CACHE=0 \
  bash scripts/sparkinfer-serve.sh start --context-length 153600

# 实测干净上限 158720（155K）
sudo env SPARKINFER_EXTRA_ARGS=SPARKINFER_PREFIX_CACHE=0 \
  bash scripts/sparkinfer-serve.sh start --context-length 158720

# 主结果档 131072
sudo env SPARKINFER_EXTRA_ARGS=SPARKINFER_PREFIX_CACHE=0 \
  bash scripts/sparkinfer-serve.sh start --context-length 131072

# 塌陷档 163840（本次证明：带草稿不可用）
sudo env SPARKINFER_EXTRA_ARGS=SPARKINFER_PREFIX_CACHE=0 \
  bash scripts/sparkinfer-serve.sh start --context-length 163840

# ctx 163840 不带草稿（本次证明：AR 正常，95 tok/s）
sudo env SPARKINFER_EXTRA_ARGS=SPARKINFER_PREFIX_CACHE=0 \
  bash scripts/sparkinfer-serve.sh start --no-spec --context-length 163840
```

测量（WSL 内；`--no-spec` 档的模型 id 是 `Qwen3.8-27B-NVFP4`，故脚本都带 `--model` 参数）：

```bash
python3 .scratch/sparkinfer-ctx163840/harness.py \
  --base-url http://127.0.0.1:8192 \
  --image /mnt/d/Code/MJ-Project/ai-model-nvfp4/.scratch/sparkinfer-ctx163840/image-text.png \
  --max-tokens 192 --out results.json
python3 .scratch/sparkinfer-ctx163840/diag.py --label ctx153600 --max-tokens 512 --out diag-ctx153600.json
python3 .scratch/sparkinfer-ctx163840/probe.py --prompt code --repeats 6 --max-tokens 256 --out probe-ctx153600-code.json
```

判据：投机是否真的生效，看 `/metrics` 的 `sparkinfer_speculative_runs_total` 增量（健康档的纯文本 greedy 每条 +1、8 条 +8；图片与 `temperature 0.7` 均为 +0）。**起步前先看启动后显存**：余量低于 ~2.3 GB 就该降 ctx 或先 `wsl --shutdown` 释放显存。

---

## 十、随本次结论落地的改动

| 文件 | 改动 |
| --- | --- |
| `scripts/sparkinfer-serve.sh` | DSpark 默认上下文 **131072 → 153600**；`--context-length` 帮助文本补上扫描结论（153600 推荐 / 158720 干净上限 / 161280 间歇退化 / 163840 落慢路径）。AR（`--no-spec`）默认仍是镜像原生的 262144，本次结论没有推翻它 |
| `scripts/start-api-server-sparkinfer.bat` | 文件头注释同步：`--no-spec` 会把上下文默认值从 153600 提到原生 262144 |
| `tests/sparkinfer-tests.sh` | argv 断言 `-e CTX=131072` → `-e CTX=153600`，断言文案同步 |
| `README.md` 4.9 | 上下文取舍表从 3 行扩到 6 行（推荐 153600 / 干净上限 158720 / 临界 161280 / 不可用 163840 / AR 65536 / AR 262144）；新增三条要点（上下文与草稿的硬边界、KV 默认 int8 且 32 KB/token、图片识别的实效与代价）；四条路线定位表里 SparkInfer 一行改为 **DSpark 153600 / AR 262144** |

验证：

- **五套测试全绿 409 passed, 0 failed**（`run-tests` 51 / `preflight-tests` 71 / `serve-tests` 88 / `fullcontext-tests` 114 / `sparkinfer-tests` 85），与改动前总数一致；`--dry-run` 现在打印 `-e CTX=153600`，由 `tests/sparkinfer-tests.sh` 的 argv 断言覆盖；
- 静态检查：`start-api-server-sparkinfer.bat` 仍为 **305 CRLF / 0 裸 LF / 0 非 ASCII**；`scripts/sparkinfer-serve.sh` 与 `tests/sparkinfer-tests.sh` 仍为**纯 LF / 纯 ASCII**。

## 十一、清理

```bash
docker rm -f qwen38-sparkinfer        # 容器已删除
wsl --shutdown                        # 已执行，显存回到 1053 MiB
```

`qwen38-sglang` 容器全程保持 `Exited` 未动；权重、草稿、镜像、`.wslconfig`、Windows 驱动均未修改。