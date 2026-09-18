# 执行结果：Qwen3.8-27B-NVFP4 升级 final build 并启用 DSpark 加速

> 依据文档：[`qwen38-nvfp4-dspark-upgrade-plan.md`](./qwen38-nvfp4-dspark-upgrade-plan.md)（执行计划）、[`qwen38-nvfp4-dspark-upgrade-research.md`](./qwen38-nvfp4-dspark-upgrade-research.md)（研究报告）
> 执行日期：2026-09-16
> 结论：**升级与 DSpark 均已落地并在跑**；计划里的 vLLM 路线（阶段 5）失败，改走 SGLang 认证路线（阶段 7）成功。

---

## 一、结果摘要

| 目标 | 结果 |
| --- | --- |
| 目标模型升级到 final build | ✅ 2 分片、索引 2387 张量、`mtp_num_hidden_layers = 0`，作者自带 CRC32 清单双分片复核通过 |
| DSpark 草稿落位 | ✅ 3 个文件哈希匹配，`block_size = 7` |
| 启用 DSpark 投机解码 | ✅ 走 SGLang 路线；`gamma=7, verify_num_draft_tokens=8, markov_head=VanillaMarkov`，无回退 |
| 吞吐提升 | ✅ 同引擎对比 **1.80×**（85.75 → 154.12 tok/s） |
| 输出无损 | ✅ 低熵任务上逐字节一致；开放式推理文本有分叉（见第四节，非配置错误） |

**当前状态**：`qwen38-sglang` 容器当前不在运行（排障期的临时容器已清理，端口 30000 与显存均已释放）。双击 `scripts/start-api-server-dspark.bat` 即按本文实测的参数启动；服务监听 `http://127.0.0.1:30000`，模型 ID `/models/target`，`max_model_len = 122880`，`/v1/models` 与聊天请求均已验证返回 200。

---

## 二、吞吐与延迟（实测）

| 配置 | 解码中位数 | TTFT | 显存占用 |
| --- | --- | --- | --- |
| 阶段 4 · vLLM 0.27.1，无投机 | 104.42 tok/s | 2.2–2.4 s | 28,653 MiB |
| 阶段 7 对照 · SGLang，无投机 | 85.75 tok/s | 0.09–0.48 s | 31,156 MiB |
| 阶段 7 · SGLang + DSpark | **154.12 tok/s** | 0.09–0.47 s | 31,494 MiB |

同一固定长输出任务（800 完成 token，`temperature=0`，3 次取中位数）。

- **DSpark 自身收益（同引擎，唯一变量是那 4 个 `--speculative-*` 参数）：154.12 ÷ 85.75 = 1.80×**
- 按计划口径（对比阶段 4 的 vLLM 基线）：154.12 ÷ 104.42 = 1.48×。这个数**低估**了 DSpark 的贡献，因为它把"换引擎"的差异也算进去了——SGLang 无投机比 vLLM 无投机慢约 18%。
- 低熵任务（关闭思考，连续输出 300 token）：DSpark 0.903 s vs 无投机 3.578 s，约 **3.96×**。

---

## 三、阶段 5（vLLM 路线 B）失败的两层原因

失败日志保留在 WSL 内：`/home/kami/logs/dspark-fail-uva.log`、`/home/kami/logs/dspark.log`。

**第一层：`RuntimeError: UVA is not available`。**
投机解码走 vLLM 的 V2 model runner（日志 `Using V2 Model Runner`），其 `RequestState` 用 `uva_instead_of_gpu=True` 建 `StagedWriteTensor`，需要 UVA；而 `is_uva_available()` 依赖 `is_pin_memory_available()`，后者在 WSL2 下由 `VLLM_WSL2_ENABLE_PIN_MEMORY` 控制，**默认 0**（vLLM 源码注释就写着"v2 model runner 需要时设为 1"）。本机内核 6.18.33.2 远高于 4.19.121 门槛，加 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 后该错误消失。

**第二层：草稿权重形状不匹配（真正的阻断点）。**
UVA 解决后报：

```
File ".../vllm/model_executor/layers/vocab_parallel_embedding.py", line 484, in weight_loader
  param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
RuntimeError: The size of tensor a (128) must match the size of tensor b (256) at non-singleton dimension 1
```

调用链为 `qwen3_dspark.py:196 load_weights` → `AutoWeightsLoader._load_param`。checkpoint 里词表类权重只有 `markov_head.markov_w1/w2` 是 `[248320, 256]`（`markov_rank = 256`，已用 vLLM 自己的 `get_config` 确认解析结果就是 256），而模型侧对应参数第二维是 128。即 **vLLM 0.27.1 内置的 DSpark Markov 头构建尺寸与这份 checkpoint 不一致**，属于该版本 vLLM 的 DSpark 加载器不兼容，不是本地配置问题。计划中"参数反复试错无意义、失败即转阶段 7"的判断成立。

---

## 四、输出一致性

按计划第六节做的贪婪解码比对，结论需要分两种情况说：

- **低熵任务（关闭思考，连续输出 1–300）：逐字节完全一致。**
  两侧 sha256 均为 `ab33e67d7845de9fd3944c2ce99b154b8f97f6a73650ffbda39a87ec10735f13`，300/300 字符相同。
  **这才是判断草稿配置是否正确的那一项**，它证明 `--speculative-dspark-block-size 7` 配对了、DSpark 不是"输出乱码"。
- **开放式推理任务（800 token 长文）：两侧在第 33 个字符处分叉**，此后各自继续输出通顺中文。
  原因是投机解码的**批量验证**：无投机时每步 1 token，开投机时一次验证 7 个 token，注意力归约顺序不同，浮点结果在小间距处翻转 argmax。这属于"数学上无损但不必逐位相同"，不是配置错误。

另外需说明：计划里"同一 prompt 在阶段 4（vLLM）与阶段 5 下贪婪结果一致"这条**在本机无法作为判据**——vLLM 与 SGLang 的推理内容走不同响应字段（`reasoning` vs `reasoning_content`），且两个引擎的采样/注意力实现不同，跨引擎比对混杂了太多变量。有效判据只能是同引擎开/关投机。

---

## 五、长上下文

`max_model_len = 122880` 已在 `/v1/models` 确认。实测两次长提示词请求，输入+输出均远在窗口内：

| 输入 token | 输出 token | 合计 | 耗时 |
| --- | --- | --- | --- |
| 17,741 | 64 | 17,805 | 2.88 s |
| 66,413 | 64 | 66,477 | 11.82 s |

未压到 122,880 边界（计划也未要求压满）。

---

## 六、本机环境改动（需要知情）

1. **模型目录已按计划改造**（可完整回滚，见下）。
2. **安装了 Docker Engine 29.8.1 与 NVIDIA Container Toolkit 1.20.0**（systemd 服务，`docker.service` 已 enable）。
   - `sudo` 需要密码，本次全部通过 `wsl -u root` 以 root 执行。
   - 用户 `kami` **不在 `docker` 组**，直接跑 `docker` 会 `permission denied`。计划阶段 7 里的 `usermod -aG docker "$USER"` 未执行，因为它需要重启 WSL 才生效，而当时服务正在跑。需要的话补执行后重启一次 WSL。
3. **拉取了 `lmsysorg/sglang:qwen38-27b`**，解压后占盘 59.8 GB（比计划预估的 17.9 GB 下载体积大不少）。
4. **项目里新增了 SGLang + DSpark 启动器**（计划原本写"不修改本项目代码与脚本"，实际落地时新增了两个文件，这是有意的扩展）：
   - `scripts/sglang-dspark.sh` — WSL 侧启动脚本，前台运行容器，支持 `--no-spec`（无投机对照）、`--dry-run`、`--port`、`--context-length` 等
   - `scripts/start-api-server-dspark.bat` — Windows 侧双击启动器，含局域网访问菜单（portproxy + 防火墙，退出时清理）、停止后 `wsl --shutdown` 释放显存
   - `README.md` 4.8 节 — 使用说明
5. **新增辅助脚本**（都在 `D:\WSL\`，是排障期的临时工具，已被上面的项目启动器取代，保留备查）：
   - `install-docker.sh` — Docker + NVIDIA Toolkit 安装（root 执行）
   - `bench_serve.py` — 流式吞吐测量 + 贪婪输出抓取
   - `longctx.py` — 长提示词测试
   - `run-sglang-dspark.sh` / `run-sglang-dspark-attached.sh` / `run-sglang-nospec-attached.sh` — 早期启动脚本
   - `keepalive.cmd` — 备用的计划任务保活脚本（未启用）

---

## 七、重要运维坑：WSL 会回收发行版，服务生命周期得跟着会话走

第一次启动容器（`docker run -d`）在 2 分钟后收到 `Exited (255)`，日志无 traceback，SGLang 是**收到 SIGTERM 后优雅退出**的（`SIGTERM received... Draining requests and shutting down`）。根因不在 SGLang，在 WSL 自己：

```
WSL (2 - init-systemd(Ubuntu)) ERROR: InitTerminateInstanceInternal:2763:
systemctl poweroff did not terminate the instance in 10000 ms, calling reboot(RB_POWER_OFF)
```

**WSL 认为实例空闲就回收整个发行版**，容器不是它认得的"会话"，于是被一起带走。实测下来回收在静置约 5–8 分钟后发生，与有没有容器在跑无关。

### 试过但**无效**的办法

- **`docker run` 前台附着**（靠常驻会话钉住）：会话活着时确实有效（服务连续跑了约 28 分钟），但会话一旦被回收，容器照死。
- **静默常驻进程**（`wsl -e sleep infinity` 分离启动）：进程后来消失（`Get-Process wsl` 计数归零），没能钉住。
- **`.wslconfig` 里加 `vmIdleTimeout=604800000`**（7 天）：加载后静置 7 分钟**仍被回收**，只是间隔略长。既然无效已**撤回**，没在你配置里留这个改动。

### 最终采用方案：项目内的前台启动器（已落地）

本项目已新增 `scripts/sglang-dspark.sh` + `scripts/start-api-server-dspark.bat`，把上面的结论固化成了方案：**容器前台运行、不设重启策略，控制台窗口就是保活，Ctrl-C 就是停止按钮**。这也是 README 4.8 节的做法。

**为什么不用 `--restart` 策略**：我确实验证过 `--restart always` 能在整机重启后自动恢复（三次实测，50–60 秒内服务可用），但在 WSL 反复回收实例的这台机器上，它的实际效果变成"发行版每次启动就重新加载 17 GB 权重、加载到一半又被杀"的循环——服务大部分时间在加载而不是在服务。前台运行把服务生命周期精确绑定到一个会话，反而是正确的取舍。

另外两条路也走过，都不通：静默的常驻 `sleep` 进程会被回收带走；用计划任务每 2 分钟触碰 WSL 需要管理员权限，当前上下文被系统拒绝（`拒绝访问`），而且它同样是在和 WSL 的回收机制对抗。

### 排除项：不是容器的问题

容器每次退出都是**优雅 SIGTERM**（`SIGTERM received... Draining requests and shutting down` / `Gracefully exiting... Remaining number of requests 0`），不是崩溃、不是 OOM（`OOMKilled=false`、`dmesg` 无 OOM 记录）、也不是输出乱码。时间点又都落在 WSL 回收实例的窗口内，所以可以确定：**容器每次都是被外部停掉的**，SGLang 和模型本身没有问题。

---

## 八、验收清单核对

| # | 验收项 | 结果 |
| --- | --- | --- |
| 1 | `/v1/models` 返回 200，模型 ID 正确 | ✅ 200，`/models/target` |
| 2 | 不带认证头的聊天请求返回 200 | ✅ 3/3 返回 200，正文 `服务正常` |
| 3 | 日志显示 DSpark 已加载、未回退 | ✅ `Initialized DSpark draft runner ... gamma=7 ... markov_head=VanillaMarkov` |
| 4 | 连续 ≥3 个短请求全部成功 | ✅ 3/3 |
| 5 | `nvidia-smi` 无 OOM，显存余量符合预期 | ✅ 31,494 / 32,607 MiB，余 694 MiB（`mem-fraction-static 0.86` 下的预期结果） |
| 6 | 长提示词：输入+输出不超过 122,880 | ✅ 实测至 66,477 |
| 7 | 有/无投机贪婪输出一致 | ⚠️ 低熵任务逐字节一致；开放式推理文本有分叉（浮点批量验证所致，非配置错误） |
| 8 | 吞吐提升已量化 | ✅ 同引擎 **1.80×**；按计划口径 1.48× |

---

## 九、回滚

备份完整保留在 `D:\WSL\models\backup\Qwen3.8-27B-NVFP4-RTX5090-pre-final\`（6 个文件，sha256 已逐个复核）。按计划阶段 8 执行即可回到 pre-final，无需重新下载。DSpark 草稿目录可独立保留。

---

## 十、建议的后续动作

1. **报告上游**：vLLM 0.27.1 的 DSpark 加载器与这份 checkpoint 的 `markov_rank` 不匹配（第三节第二层原因），值得开 issue。
2. **补 `usermod -aG docker kami`** 并重启一次 WSL，之后不必再用 root 跑 docker。
3. **保持服务在线**：双击 `scripts/start-api-server-dspark.bat`，窗口开着服务就在（第七节）。
4. **常用命令**：
   - 起服务：双击 `scripts/start-api-server-dspark.bat`（或 WSL 内 `sudo bash scripts/sglang-dspark.sh start`）；加 `--no-spec` 可跑无投机对照
   - 看日志：服务控制台窗口里直接看；停止后可用 `wsl -d Ubuntu -u root -- docker logs qwen38-sglang`
   - 停服务：控制台里 `Ctrl-C`
5. **注意：SGLang 路线与 vLLM 路线不能同时运行。** SGLang + DSpark 实测占用 31,494 MiB，机动余量只剩 694 MiB，装不下第二个实例的权重（约 17 GB）。README 4.8 节里"所以可以和 `start-api-server-vllm.bat` 同时启动"只表达端口不冲突，**显存上是冲突的**，这句建议改成"端口不冲突，但显存只够跑其中一个"。
6. 如需**满 262K 上下文**，按计划关掉 DSpark 单独跑；32 GB 单卡装不下"262K + DSpark"。
7. 若要验证第 7 项在开放式文本上也能一致，需要换成长为低熵任务，或接受"无损但不逐位相同"。