# 实测结果：SparkInfer 输出上限（max output tokens）与 DSpark 互斥

> 执行日期：2026-09-18 深夜（同日晚间 `result/sparkinfer-ctx163840-image-result.md` 上下文扫描的续测）
> 依据：上游 `server/README.md` 与 HF 模型卡（联网只读）、本仓库既有实测记录
> 环境：同前——WSL2 Ubuntu / Docker 29.8.1 / RTX 5090 32607 MiB / 镜像 `ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10`（digest `sha256:d519d6ed…`）
> 起因：把 `SPARKINFER_MAX_OUTPUT_TOKENS` 从默认 16384 提到 32768，验证长输出的速度代价
> 性质：只读复用本机权重；未下载任何模型权重

---

## 一、结论摘要

1. **上限可以改，32K 也确实能生成**：实测一次长生成跑到 **17467 token** 后由模型自己结束（`finish_reason=stop`），另一次请求在 8192 处命中 `finish_reason=length`。默认 16384 会把请求的 `max_tokens` 截断到 16384（上游文档），所以想要长输出必须改这个变量。
2. **但改成 32768 会把 DSpark 整体关掉。** 实测门槛为 **`prompt + max_tokens ≤ SPARKINFER_DSPARK_MAX_CTX`（默认 16384）** 才允许投机；上限一改成 32768，每个请求都预留 ≥32K → `sparkinfer_speculative_runs_total` 增量 **0**，短输出 decode 从 **338.3 → ~94 tok/s**（AR 水平），**慢约 3.6×**。
3. **把草稿窗口一起提到 32768 并不能救。** `prompt + max_tokens` 依旧大于窗口，`max_tokens=32768` 仍然 `spec_runs=0`；同时启动显存 +332 MiB（余量掉到 2131 MiB），短请求 decode 也从 338.3 掉到 238.6（ctx 与窗口同时变化，未单独归因）。要开这道门，窗口必须大于"上限 + 提示长度"（如 36864），本次未测。
4. **更本质的事实：DSpark 的收益只存在于短输出。** 同一条提示：256 token 输出 **338 tok/s**；而把上限放到 8192 做长生成时，整个 8192 token 里只投机了 **421** 个 token，整体 **81.7 tok/s**——**比完全不投机的 AR（~94）还低**。所以"要长输出"与"要 DSpark 加速"本质互斥，不只是那道门槛造成的。
5. **KV 预留被一手证实**：每次请求按 `prompt + max_tokens` 预留。`max_tokens=32768` 时空闲块由 9608 → **7553**（2055 块 = 32880 token = **1.0 GiB**，占 4.7 GiB KV 池的 **21%**）；`max_tokens=8192` 时为 9608 → 9089（519 块）。
6. **落地建议**：要 DSpark 就保持镜像默认上限 16384，并把客户端 `max_tokens` 控制在 **~16000 以内**（门槛算的是 `prompt + max_tokens`）；要 32K 长输出就接受 AR 速度（短 ~94、长 ~76–91 tok/s），此时这条路线相对 vLLM 200k / SGLang 163840+DSpark 已无优势。

---

## 二、门槛扫描（上限 32768，草稿窗口 16384）

同一条短提示（约 30 token，输出都约 700 token 就 EOS），**唯一变量是请求的 `max_tokens`**：

| 请求 max_tokens | 预估 prompt+max_tokens | 投机计数增量 | decode tok/s |
| --- | --- | --- | --- |
| 4096 | ~4126 | **+1** | 134.2 |
| 8192 | ~8222 | **+1** | 91.6 |
| 16384 | ~16414 | **0** | 89.0 |
| 16385 | ~16415 | **0** | 93.9 |
| 20480 | ~20510 | **0** | 93.8 |
| 32768 | ~32798 | **0** | 94.1 |

门槛落在 8192 与 16384 之间，且与草稿窗口默认值 **16384** 重合：`prompt + max_tokens > 16384` → 不投机。**注意 16384 这一档就已经被拒**（因为还要加上提示长度）。

---

## 三、32K 长输出实测（上限 32768，草稿窗口 16384）

| 项 | 值 |
| --- | --- |
| prompt / completion | 84 / **17467** |
| finish_reason | **stop**（模型自己结束，不是被上限截断） |
| 分段速率 | 1–4096：**90.8**；4097–8192：84.2；8193–12288：79.4；12289–16384：76.3；16385–17459：73.9 |
| 整体 | **81.7 tok/s**，wall 213.9 s |
| 投机 | runs **0**，tokens **0** |
| KV 预留 | 空闲块 9608 → 7553（2055 块 = 1.0 GiB） |
| 功耗均值 | 423 W（SM 2905 MHz） |

对照组（同一引擎，上限 8192，投机被允许）：长提示生成 8192 token（`finish_reason=length`），投机 runs **+1** 但 tokens **仅 421**，整体 **81.7 tok/s**（分段 78.9 / 84.8）。
再对照（完全不投机的 AR 档，上限 32768 的 700 token 短请求）：**94.1 tok/s**。

**三组数字放在一起**：长输出的投机几乎不产生收益（421 个投机 token / 8192 生成），速率与 AR 同级甚至略低。

---

## 四、草稿窗口一起提到 32768（上限 32768）

| 配置 | 启动后显存 | 短输出探针 code×6 @256 | `max_tokens=32768` 的短请求 |
| --- | --- | --- | --- |
| ctx 158720，上限 16384，窗口 16384 | 30320 MiB | **338.3–340.1**（6/6 有投机） | — |
| ctx 153600，上限 32768，窗口 16384 | 30196 MiB | 该组合未跑此探针 | 94.1 tok/s，投机 **0** |
| ctx 153600，上限 32768，**窗口 32768** | **30476 MiB** | **223.7–243.6**（6/6 有投机） | **61.4 tok/s**，投机仍 **0** |

读法：

- 提高草稿窗口**没有**打开 32K 请求的门（`prompt + max_tokens` 仍 > 32768），但让短请求重新有投机；
- 代价是 **+332 MiB** 启动显存，余量掉到 2131 MiB（本次上下文扫描里，余量 2197 MiB 的 161280 档已经出现间歇退化，所以这个配置本身也更靠近悬崖）；
- 短请求从 338.3 掉到 238.6：**该对比同时改变了 ctx（158720→153600）与草稿窗口（16384→32768），而且 6/6 都检出投机，因此不能把差距单独归因于窗口**——未做分离实验。
- `max_tokens=32768` 的请求反而更慢（61.4 vs 94.1）：既没投机、又因草稿窗口变大而少了余量。

---

## 五、怎么解释

- **门槛机制**：引擎在准入时把 `prompt + max_tokens` 当作该请求的序列预算；草稿只能关注 `SPARKINFER_DSPARK_MAX_CTX` 个 token，预算超出就不让草稿参与。依据是本次的台阶式实测（8192 有、16384 起没有）与上游"草稿自身注意力窗口上限为 `--ctx`"的说明。**引擎这一档没有对应日志，机制未从源码确认**。
- **收益消失机制**：上游 0.5.7 记录"投机会在下一个 attention 分档边界停止（第一档 512 token 上下文）"。与本次观察一致——256 token 输出 338 tok/s，512 token 输出现已实测末段掉到 95（见上下文扫描文档），8192 token 生成里只剩 421 个投机 token。

---

## 六、未做与未验证（必须对照）

| # | 项 | 说明 |
| --- | --- | --- |
| 1 | `SPARKINFER_DSPARK_MAX_CTX=36864` 或更大 | 未测。这是"同时要 32K 上限与投机"的唯一未验证方向，但按第三节结论，长输出本来就没有多少投机收益 |
| 2 | 上限 16384（默认）下的门槛逐点扫描 | 未复测。第二节的扫描是在上限 32768 下做的，机制上与该上限无关，但没有直接验证 |
| 3 | 238.6 vs 338.3 的差距归因 | 未分离（ctx 与草稿窗口同时变） |
| 4 | 并发 | 全部为单请求串行。每请求 1 GiB 的预留会把并发能力压得更低，未测 |
| 5 | 门槛的代码依据 | 引擎无相关日志，未读源码确认判断条件 |
| 6 | 客户端请求超过上限时的行为 | 只验证了"上限内按请求值执行"；`max_tokens` 超过服务端上限时的截断行为取自上游文档（"a larger max_tokens is clamped to this cap"），本次未逐点验证截断后的 `finish_reason`（8192 那次命中的是 length，属于上限内正常截断） |

---

## 七、复现

脚本与原始数据都在 `.scratch/sparkinfer-ctx163840/`（临时区，未作为固化产物）：

| 文件 | 用途 |
| --- | --- |
| `longrun.py` | 长输出测量：分段速率、`finish_reason`、`/v1/capacity` 采样（可读 KV 预留） |
| `gate-scan.sh` | 门槛扫描（同一短提示，只变 `max_tokens`） |
| `probe.py` | 重复性探针 |
| `start-153600-maxtok32k.sh` | ctx 153600 + 上限 32768（窗口默认 16384） |
| `start-153600-32k-both.sh` | ctx 153600 + 上限 32768 + 窗口 32768 |
| `dryrun-maxtok32k.sh` | 只打印 argv，验证透传写法 |
| `longrun-maxtok32k.json`、`gate-A-code-32768.json`、`gate-B-long-8192.json`、`both-code-32768.json`、`probe-both-code256.json`、`clocks-longrun-maxtok32k.log` | 本次原始结果与 GPU 采样 |

启动（`SPARKINFER_EXTRA_ARGS` 按空格切成多个 `-e`，所以变量写在同一个引号串里）：

```bash
# 32K 上限（本次证明会关掉 DSpark）
sudo env SPARKINFER_EXTRA_ARGS="SPARKINFER_PREFIX_CACHE=0 SPARKINFER_MAX_OUTPUT_TOKENS=32768" \
  bash scripts/sparkinfer-serve.sh start --context-length 153600

# 32K 上限 + 32K 草稿窗口（本次证明仍打不开 32K 请求的门）
sudo env SPARKINFER_EXTRA_ARGS="SPARKINFER_PREFIX_CACHE=0 SPARKINFER_MAX_OUTPUT_TOKENS=32768 SPARKINFER_DSPARK_MAX_CTX=32768" \
  bash scripts/sparkinfer-serve.sh start --context-length 153600
```

判据：`/v1/info` 的 `max_output_tokens` 与启动日志的 `max output N` 确认上限生效；门槛看 `/metrics` 的 `sparkinfer_speculative_runs_total` 增量（每次请求 +1 才有投机）。

---

## 八、清理

```bash
docker rm -f qwen38-sparkinfer        # 容器已删除
wsl --shutdown                        # 已执行，显存回到 981 MiB
```

`qwen38-sglang` 容器全程保持 `Exited` 未动；权重、草稿、镜像、`.wslconfig`、Windows 驱动均未修改。