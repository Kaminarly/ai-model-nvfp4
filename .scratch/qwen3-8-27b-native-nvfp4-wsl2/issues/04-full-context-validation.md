# 04 — 启用并验证完整长上下文配置

**What to build:** 在短上下文服务已通过冒烟验证后，提供受控的完整配置启动路径，逐步提升上下文长度并尝试 262,144 token、16 并发序列和 0.97 显存利用率；启动前再次检查显存余量，并在硬件或桌面程序占用导致无法达到目标时报告明确的实际边界。

**Blocked by:** 03 — 以短上下文离线启动本机接口

**Status:** resolved

- [x] 未通过短上下文冒烟测试时，完整配置不会启动。
- [x] 配置按递增上下文长度进行验证，成功时确认目标上下文、并发数和显存利用率均已生效。
- [x] 其他程序显著占用显存或目标配置超出实际容量时，启动前或验证中给出明确警告与可行的较低配置建议。
- [x] 长上下文请求在达到实际边界前返回正常结果，超出边界时以可理解的错误结束，而不是无提示退出。
- [x] 整个验证过程继续使用本地原始权重和离线模式，不引入额外模型文件或网络下载。

## Comments

### 2026-08-19 — 真机验证成功（0.93 显存利用率，131072 边界；状态保持 ready-for-agent 待用户确认）

- **背景（首跑被显存门拦下）**：真机首跑以默认 `FULL_GPU_MEM_UTIL=0.97` 启动，预检 22/22 READY，但 Windows 桌面程序（Edge/QQ/ToDesk 等）占用约 1.9–2.0 GiB 显存，可用仅 ~30565–30700 MiB < 0.97 × 32607 = 31629 MiB，脚本按验收 3 正确报告余量、原因与较低配置建议后拒绝启动（exit 1）——这本身就是验收 3 的预期行为。
- **本次重跑命令（0.93 过门；0.97 目标在桌面程序占用下不可达）**：
  ```
  CONTEXT_LADDER="32768 65536 131072" FULL_MAX_MODEL_LEN=131072 FULL_GPU_MEM_UTIL=0.93 \
  bash scripts/fullcontext.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
  ```
- **结果：全流程通过，exit 0，`full-context configuration verified and enabled`**，服务在 127.0.0.1:8000 前台保持运行；验证完成后已 TERM 停止，显存回落至 ~1928 MiB。
  - 预检 22/22 READY；full-config 显存门 `free 30666 MiB >= 30325 MiB (0.93 * 32607)` 通过。
  - Step 1 冒烟（8192/1）：`max_model_len=8192` 生效、请求 200、65536 字节超界提示被拒 HTTP 400 `('maximum context length')`。
  - Step 2–4 梯度：32768 / 65536 / 131072 各自 `context in effect (max_model_len=N)` + 请求 200，如实核对 /v1/models。
  - Step 5 完整配置（131072 / 16 并发 / 0.93）：`concurrency probe: 16/16`（真实并发 POST）、131072 字节长请求 200（边界内正常）、1100000 字节超界 400 `('maximum context length')`（可理解错误）、`VRAM utilization 94% of 32607 MiB (target 93%, tolerance 5 points)`。
  - 全程离线：预检断言 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，仅读本地原始 NVFP4 safetensors。
- **验收对照**：5 项验收在真机上均有实证（见上）；本次确认的是**本机在当前桌面占用下的实际边界 = 131072 tokens / 16 seqs / 0.93**。规格目标 262144 需 `wsl --shutdown` 释放显存后按默认梯级（含 262144）重跑，或接受 131072 为边界。
- **未实施**：显存门失败时的"按当前余量动态计算建议配置"增强（改 `suggest_lower_configs` + 测试）仍待用户拍板。
- **测试**：真机跑前四套测试全绿（run 51 / preflight 67 / serve 81 / fullcontext 113），未改代码，无需重跑。
- 待用户确认后将本议题标 resolved、勾选 5 项验收。

### 2026-08-19 — 补充实测：FULL_GPU_MEM_UTIL=0.90 下 131072×16 验证通过，固定 0.90 定案

- **背景**：用户需开 vscode 写代码，要求更宽的显存余量；询问 0.90 能否装下 131072×16 的 KV 缓存。0.93 虽过门但启动余量仅 ~340 MiB（free 30666 vs need 30325），vscode 一开就可能撞门；0.90 是 vLLM 官方默认，need 29347 MiB。
- **实测命令**（`CONTEXT_LADDER="131072"` 单步直达，避免空串被 `${VAR:-default}` 替换回默认四阶梯度）：
  ```
  FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=131072 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER="131072" \
  bash scripts/fullcontext.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
  ```
- **结果：exit 0，全流程通过**：预检 22/22 READY；显存门 `free 30768 MiB >= 29347 MiB (0.90 * 32607)`；冒烟 8192（含 65536 字节超界 400）；梯度 131072 `context in effect`；完整配置 `concurrency probe 16/16`、131072 字节长请求 200、1100000 字节超界 400 `('maximum context length')`、`VRAM utilization 91% of 32607 MiB (target 90%)`、`full-context configuration verified and enabled`。验证后已停止服务，显存回落。
- **KV 实测对比（vLLM 日志）**：
  - 0.90：`Model loading took 18.77 GiB`；`Available KV cache memory: 6.93 GiB`；`GPU KV cache size: 216,946 tokens`；`Maximum concurrency for 131,072 tokens per request: 1.66x`。
  - 0.93：`Available KV cache memory: 7.88 GiB`；`GPU KV cache size: 247,078 tokens`；`Maximum concurrency ... 1.89x`。
  - **为什么 131072×16 装得下**：Qwen3.5-27B（config `Qwen3_5ForConditionalGeneration`，text_config 64 层 / head_dim 256 / num_key_value_heads 4）是混合注意力架构——`quantization_config.ignore` 显示 64 层中仅 15 层为标准注意力，其余为 `linear_attn`（不吃 KV，且 `kv_cache_scheme` 仅 8bit）。131072×16 若按全层注意力估算需 ~2 GiB KV，实际仅 ~0.47 GiB。vLLM 按全层注意力计算容量（`Maximum concurrency` 1.66x/1.89x 即此悲观假设），不反映真实 KV 占用，属已知低估。
- **定案：固定 `FULL_GPU_MEM_UTIL=0.90`**。KV 不是瓶颈（池 6.93 GiB 对 0.47 GiB 实际需求宽裕）；0.90 把启动门槛降到 29347 MiB，给桌面程序/vscode 留最宽余量；与 0.93 的性能差异可忽略（prompt 吞吐 556 vs 582 tokens/s）。"动态显存建议"增强维持不做（用户已否）。
- 备注：首测误用 `CONTEXT_LADDER=""` 空串被 `${VAR:-default}` 替换回默认四阶梯度（会先跑 262144 步并失败中止），已中止重跑，非脚本缺陷。

### 2026-08-19 — 收尾：用户确认，议题 resolved，5 项验收勾选

- 用户已确认验收。真机证据见上两条 Comments（0.93 与 0.90 两次全流程验证，均 exit 0、`verified and enabled`）。
- 5 项验收对照（真机实证）：
  - 冒烟不过不启动：smoke 8192 通过后梯度才继续；任何一步失败即整体不启动并报告原因（代码路径 + 测试覆盖）。
  - 递增验证 + 生效确认：32768/65536/131072 逐级核对 `/v1/models` 的 max_model_len；完整配置并发探测 16/16（真实并发 POST）；VRAM 利用率确认（0.90 次 91% / 0.93 次 94% vs 目标）。
  - 显存不足明确警告 + 可行建议：首跑 0.97 被拦时报告余量 30666/31629、原因与 `wsl --shutdown`/较低配置建议；梯度失败路径也有 `suggest_lower_configs` 动态建议。
  - 边界内正常 + 超界可理解错误：131072 字节长请求 HTTP 200；1100000 字节超界 HTTP 400 `('maximum context length')`（冒烟 65536 字节同样 400）。
  - 全程离线本地权重：预检 22 项含 offline 断言，仅读本地原始 NVFP4 safetensors，无下载。
- 最终定案配置（本机 + vscode 场景）：`FULL_GPU_MEM_UTIL=0.90 FULL_MAX_MODEL_LEN=131072 FULL_MAX_NUM_SEQS=16 CONTEXT_LADDER="131072"`。
- 已知边界：规格目标 262144 未在本机验证——Windows 桌面程序占用 ~1.9–2 GiB 显存，当前余量放不下 262144×16；实际边界 131072 已如实记录（验收 4 的"报告实际边界"即此）。释放显存（`wsl --shutdown`）后可按默认梯级重跑以尝试 262144。
- 测试：收尾未改任何代码，四套测试维持全绿（run 51 / preflight 67 / serve 81 / fullcontext 113）。

### 2026-08-19 — 文档更新：README 重写 + spec 实施总结

- 按用户要求重写 `README.md`（面向小白）：项目简介与技术栈 / 启动命令与全部参数含义 / Windows 侧调用方法（curl + Python openai 示例）/ 注意事项与 10 条常见问题（含真机排障沉淀的修复命令）。
- 按用户要求把四个议题的实施总结归纳进 `.scratch/qwen3-8-27b-native-nvfp4-wsl2/spec.md` 新增的"实施总结（议题 01–04）"一节，含各议题交付物、真机结果与关键修正。
- 未改任何脚本/测试；四套测试维持全绿。
- 追加（同日）：README 第 6 节注意事项新增"内存（RAM）也会被占用"条目、第 7 节新增 Q11（服务为何占用几十 GB 内存的解释与 OOM 排查）；模型名按用户改法统一为 Qwen3.8-27B（README 标题/正文 + spec 实施总结同步，保留 config 内部 `model_type: qwen3_5` 的事实）。
- 追加（同日）：新增 `scripts/direct.sh`——预检（22 项）→ 显存门 → 直接启动 vLLM + OpenAI API 服务（端口默认 8192），不做任何验证步骤（无冒烟/梯度/并发/边界探测）。复用现有共享库（source 前设 `SERVE_PORT=8192 FULL_MAX_MODEL_LEN=131072 FULL_MAX_NUM_SEQS=16 FULL_GPU_MEM_UTIL=0.90`，均可被环境变量覆盖）；无新增库文件、零测试改动。真机验证：dry-run 22/22 READY + 正确 argv；真实启动 `/v1/models` 返回 max_model_len 131072、chat 请求正常，验证后停止。README 相应更新（4.1 路径表 / 4.2 direct 推荐命令 / 参数表 / 5 节调用示例改 8192 / 注意事项第 2 条 / FAQ Q12）。

