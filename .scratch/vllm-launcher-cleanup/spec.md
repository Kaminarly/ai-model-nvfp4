# spec: vLLM 启动器改名 + MTP 路线失效确认

Status: done

## 背景

模型目录 `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` 在 **2026-09-16** 被更新过（`config.json`、`hf_quant_config.json`、`model.safetensors.index.json`、第二个分片的 mtime 都是这一天）。更新后的 checkpoint 不再带 MTP 头，于是 `scripts/start-api-server-mtp.bat` 依赖的加速路线失效；同时把 vLLM 路线的启动器改名，与 `-mtp` / `-gguf` / `-dspark` / `-sparkinfer` 的命名对齐。

## 一、MTP 失效的验证（对着硬盘上的文件，不是文档）

| 检查项 | 结果 |
| --- | --- |
| `config.json` → `text_config.mtp_num_hidden_layers` | `0`（更新前是 `1`） |
| `model.safetensors.index.json` | 2387 个张量，`mtp.*` = 0 |
| 两个 safetensors 文件头（元数据，权威来源） | 1312 + 1075 = 2387 个张量，`mtp.*` = 0；索引与文件头逐分片一致 |
| `hf_quant_config.json` | 全文不含 `mtp`（更新前有"`mtp*` 不量化、保持 bf16"的规则） |
| vLLM 0.27.1 代码路径 | `model_executor/models/qwen3_5_mtp.py:80` 用 `mtp_num_hidden_layers` 决定草稿头层数 → 0 层空列表；`load_weights` 只接收 `mtp.` 前缀权重 → 无权重可加载 |

复现命令（WSL 内，只读）：

```bash
wsl -d Ubuntu -- python3 /mnt/d/Code/MJ-Project/ai-model-nvfp4/.scratch/vllm-launcher-cleanup/check-mtp.py \
  /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
```

vLLM 侧依据：

```bash
grep -n 'num_mtp_layers\|spec_step_idx %' ~/vllm/venv/lib/python3.14/site-packages/vllm/model_executor/models/qwen3_5_mtp.py
```

## 二、本次改动

1. **改名**：`scripts/start-api-server.bat` → `scripts/start-api-server-vllm.bat`（`git mv`，行为不变，局域网菜单原样保留）。全仓库引用同步更新（README、其他 `.bat` 注释、`result/` 记录）。
2. **MTP 启动器标注失效**：`scripts/start-api-server-mtp.bat` 保留（按需求不删除），文件头加"已失效"说明、窗口标题与启动时打印警告，并指向 `-dspark` / `-vllm`。
3. **README**：4.2.1（改名说明）、4.4（`VLLM_SPEC_METHOD` 行）、**4.6（改写为"模型更新后已失效"，附验证表 + 历史记录）**、4.7 / 4.8 / 5.0 / 6 的交叉引用、路线对照表的加速方式一列，新增 Q15（MTP 起不来）与 Q16（旧文件名找不到）。
4. **`.bat` 字节契约修复（顺带）**：`scripts/start-api-server-dspark.bat` 结尾原有一个**游离的裸 CR**（HEAD 里就有，导致 git 把它当 `-text`、每次改动都整文件 diff）。已按其余五个启动器的口径规范化并 `git add --renormalize`，现在六个 `.bat` 全部满足 **CRLF / 无 BOM / 纯 ASCII**。

## 三、验证方式

```powershell
# 六个启动器的字节契约（CRLF、无 BOM、纯 ASCII）
powershell -File .scratch\vllm-launcher-cleanup\bat-check.ps1 -Path (Get-ChildItem scripts\*.bat).FullName
```

```bash
bash tests/run-tests.sh        # 项目自带回归测试（假工具，无需 GPU）
bash .scratch/vllm-launcher-cleanup/run-suites.sh   # 五套一次跑完并逐套给结论
```

本次结果（改动后重跑）：`run-tests.sh` 51 / `preflight` 71 / `serve` 88 / `fullcontext` 114 / `sparkinfer` 85，**共 409 项，0 失败**。

另外用 `--dry-run` 端到端实跑过两个启动器（不加载模型，只走预检 + 打印 vLLM argv）：

- `start-api-server-vllm.bat --dry-run`（菜单选 2）：预检 22/22 READY，argv 为 `--max-model-len 200000 --max-num-seqs 16 --host 127.0.0.1 --port 8192`，**无** `--spec-method`；
- `start-api-server-mtp.bat --dry-run`（菜单选 2）：警告正常打印，argv 带 `--spec-method mtp --spec-tokens 3 --max-model-len 180000`——即脚本本身没坏，坏的是这份权重没有 MTP 头。

## Comments

- 2026-09-16 之后，任何"MTP 可选加速"的说法都应改成"已失效，改用 SGLang + DSpark"；`VLLM_SPEC_METHOD` 环境变量本身仍可用于换成别的带 MTP 头的 checkpoint。