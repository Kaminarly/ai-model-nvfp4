# 交接：议题 04 完整长上下文验证——真机测试收尾

## 项目背景

Windows 10 + WSL2 Ubuntu + RTX 5090（32GB，驱动 610.88）上，用 vLLM 0.27.1 离线跑 `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` 的原生 ModelOpt NVFP4 权重。议题 01（环境）、02（统一预检）、03（短上下文服务）均已 **resolved 且真机闭环**（03 真机 start 9 成功：预检 READY、/v1/models 返回、短文本推理正常）。**议题 04（完整长上下文验证）代码已实现并通过两轮代码评审，fake 测试全绿，真机首跑在显存门被拦，待释放显存后重跑收尾。**

**重要约定**：项目**没有 git 仓库**；用户明确"不需要提交，直接更新 issue"。所有产出记录进 `.scratch/qwen3-8-27b-native-nvfp4-wsl2/issues/*.md` 的 Comments（遵循 `docs/agents/issue-tracker.md`）。议题 04 文件：`.scratch/qwen3-8-27b-native-nvfp4-wsl2/issues/04-full-context-validation.md`（Status 仍为 ready-for-agent，5 项验收待勾选）。

## 已实现（不要重复造）

议题 04 交付物（fake 测试四套全绿：`tests/run-tests.sh` 51、`preflight-tests.sh` 67、`serve-tests.sh` 81、`fullcontext-tests.sh` 113）：

- `scripts/fullcontext.sh` — 入口 CLI：`start --model-dir DIR [--prefix] [--host] [--port] [--dry-run]` / `help`；出口 0=完整配置已验证并保持运行 / 1=拒绝或失败 / 2=用法错误。
- `scripts/lib/fullcontext-lib.sh` — 验证序列：预检闸门（同 03）→ **full-config 显存门**（可用显存 ≥ `FULL_GPU_MEM_UTIL × 总量`，否则报告余量 + 动态较低配置建议）→ 短上下文冒烟（8192/1，含超长 65536 字节 4xx 拒绝探测，失败则整体不启动）→ `CONTEXT_LADDER` 逐级提升（默认 32768 65536 131072 262144，每步核对 /v1/models 的 max_model_len，失败报"实际边界=上一级"+ 动态建议）→ 完整配置（262144/16/0.97）保持运行（核对上下文、CONCURRENT_PROOF 并发请求、131072 字节长请求、1100000 字节超界 4xx 探测、显存采样）。
- `scripts/lib/serve-lib.sh` / `scripts/serve.sh` — 重构抽出共享 `parse_start_options` / `validate_bind` / `prepare_vllm_env` / `serve_argv`（第 8 参 util 可选：03 不带、04 带 0.97）。
- `scripts/lib/wsl2-env-lib.sh` — 输出函数（info/ok/warn/fail/check_*）改 `env printf` 防重定向缓冲丢失（04 多次启动、测试 kill 场景必需）。
- `tests/fakebin/vllm` — 扩展：上报 `max_model_len`、超限 4xx 拒绝、boot 失败开关（`FAKE_VLLM_BOOT_FAIL`/`FAIL_BOOT_AT`）、`FAKE_VLLM_MAX_LEN_OFFSET`、`FAKE_VLLM_NO_REJECT`、按端口 argv 日志。
- `README.md` — 新增"完整长上下文验证（议题 04）"一节 + env 变量表。

代码评审：按 /code-review 双轴两轮，Spec 轴 6 项 + Standards 轴 6 项 + 第二轮 3 项全部修复，最终两轴均无需要改代码的发现（详见下方"评审记录"）。

## 当前状态：真机首跑（2026-08-19，用户粘贴输出）

真机执行 `bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/fullcontext.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090`：

- **预检 22/22 全绿 READY**（WSL 内核 6.18.33.2、Windows 10 19045、RTX 5090 驱动 610.88 31GiB、Python 3.14.4、nvcc 13.3、vLLM 0.27.1、CCCL 13.3↔13030 一致、模型 3 分片完整、显存 free 30644 MiB）。
- **被 full-config VRAM 门拦下**：总显存 32607 MiB，**1963 MiB 已被其他程序占用**（可用 30644），0.97 × 32607 = **31629 MiB**，差 **985 MiB**。脚本正确报告了余量、原因、wsl --shutdown 修复建议与较低配置，无无提示退出——这本身就是验收 3 的预期行为。

## 下一步（真机收尾，按顺序）

1. **释放显存重跑**（首选，保住 0.97 目标）：Windows 侧关 GPU 程序（先 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 看占 1963 MiB 的是谁，常见浏览器/桌面合成），然后**必须** `wsl --shutdown` 重启 WSL 让显存真正释放，重跑 `fullcontext.sh start`。可用回到 ~32607 即可过门（需 ≥ 31629）。
2. **若系统进程（DWM 等）关不掉**：`FULL_GPU_MEM_UTIL=0.93 bash .../fullcontext.sh start ...`（0.93×32607≈30325 ≤ 30644 过门）。注意：KV cache 缩小后 262144 若装不下，ramp 到 262144 步会失败并如实报告实际边界（可能 131072 或 65536）——这也是验收 4 的正常流程。
3. **或直接接受较低边界**：`CONTEXT_LADDER="32768 65536 131072" FULL_MAX_MODEL_LEN=131072 bash .../fullcontext.sh start ...`。
4. 成功路径预期输出：`preflight READY` → `full-config VRAM gate: free ... >= ...` → smoke 8192（含边界 4xx 拒绝）→ ramp 32768/65536/131072/262144 各自 `context in effect` → `concurrency probe: N/N` → `long-context request ... 200` → `boundary probe ... 1100000 bytes > 262144 ... rejected` → `full-context configuration verified and enabled` → `full-context service is running: http://127.0.0.1:8000/v1 (Ctrl-C to stop)`。
5. **验收对照**：每步日志核对 5 项验收（预检闸门 / 上下文+并发+显存生效确认 / 显存不足时明确警告与可行建议 / 边界内正常返回+超界可理解错误 / 全程离线本地权重）。真机如果暴露新问题（如 FlashInfer JIT 首次编译 262144 的 KV cache 超时、OOM），走 `/diagnosing-bugs`，日志在各步 `$prefix/logs/`（`fullcontext-smoke.log`、`fullcontext-ramp-<len>.log`）与用户终端。

## 待用户决策的增强（可选，未实施）

显存门失败时当前建议写死 0.90/0.85；曾向用户提议改为**按当前可用显存动态计算**（如"当前余量 30644 MiB 下最高可用 `FULL_GPU_MEM_UTIL=0.93`，或需释放 ≥ 985 MiB"）。用户未拍板。若用户点头再实施（改 `suggest_lower_configs` + 测试断言），否则不动。

## 完成后收尾（务必做）

- 真机完整配置成功（看到 `verified and enabled` + 服务保持运行）后，向用户确认，然后按 `/implement` 收尾规范更新议题 04：`Status` → resolved、勾选 5 项验收、Comments 追加交付记录（含真机显存门拦截与释放后的成功日志、代码评审两轮结论）。测试保持四套全绿后再更新。
- 若真机暴露环境/预检盲区，同步更新议题 02 Comments 与 README 聚合检查列表（参考 03 的做法）。
- 本次会话已向用户汇报过实现与评审结果，用户未明确确认前**不要**把 issue 04 标 resolved。

## 评审记录（两轮 /code-review，均已修复，勿重复）

- 第一轮 Spec：并发"假确认"（回显参数）→ 新增 `confirm_concurrency` 真实并发 POST；full 配置边界未验证 → 新增 `FULL_OVERFLOW_BYTES=1100000` 探测；冒烟误用 0.97 与 03 配置不符 → `serve_argv` util 可选；写死建议档 → `suggest_lower_configs "$last_ok"` 动态；`BOOT_WAIT_MAX` 300→600（首次 JIT 编译）；`util_percent` 静默截断 → 拒绝 3 位小数。
- 第一轮 Standards：`cmd_start` 重复 → `parse_start_options`；`serve_argv_full` 同形 → 复用 `serve_argv`；VRAM 解析重复 → `gpu_mem_tot_used`；cygpath 重复 → `win_path`；残留裸 printf → `env printf`；注释失真 → 修注释。
- 第二轮：6 项修复确认；再修 3 点（`wait_http_ok` 死代码 pid 参数删除、`serve_argv_full` 冗余实参、`run_full` 失败路径改 `suggest_lower_configs "$len"`）。最终两轴无必须改代码的发现。

## Suggested skills

- `/diagnosing-bugs` — 真机收尾若暴露新问题（JIT 编译超时、OOM、边界不符）先定位根因。
- `/implement` + `/code-review` — 若用户批准"动态显存建议"增强，或真机暴露需改代码的盲区；改完走双轴评审再收尾。
- 收尾更新议题时按 `/implement` 的 issue 更新规范（`docs/agents/issue-tracker.md`：Status 行、验收勾选、Comments 追加、不得合并议题文件）。
