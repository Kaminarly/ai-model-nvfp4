# 05 - 局域网访问（LAN access）

Status: resolved

## 需求

用户：在 WSL + Ubuntu 里部署了 AI 模型的 API 服务，希望局域网里能访问。

当前实现：vLLM 只绑定 `127.0.0.1`，且脚本 `validate_bind` 强制拒绝一切非回环绑定（安全设计）。Windows 10 + WSL2 为 NAT 网络，WSL 的 IP（本机 `172.19.28.195`）每次启动都会变，局域网设备无法直接路由到它。

## 方案

三步：让 vLLM 在 WSL 内绑定 `0.0.0.0` → Windows `netsh interface portproxy` 把端口转发到 WSL IP → Windows 防火墙放行入站端口。

实现要点：

- **`scripts/lib/serve-lib.sh`**
  - `validate_bind` 增加第三个参数 `allow-lan`；只有字面 `"1"`（即显式 `--lan`）才放行 `0.0.0.0` / `::`，否则非回环绑定仍被拒绝。
  - `parse_start_options` 增加 `--lan` 选项，置 `START_LAN=1` 并把 `START_HOST` 覆盖为 `0.0.0.0`（`--host` 显式值会被覆盖）。
  - 新增 `lan_hint`：LAN 模式下打印访问地址 + portproxy 提示。
  - `run_offline_serve` 调用 `lan_hint`。
- **`scripts/direct.sh` / `serve.sh` / `fullcontext.sh`**：usage 增加 `--lan`；`direct.sh` 的 dry-run 和实际启动、`fullcontext-lib.sh` 的 `run_full` 与 dry-run 计划都在 LAN 模式打印 `lan_hint`。
- **`scripts/start-api-server-lan.bat`（新）**：一键完成三步。非管理员时用 UAC 提权重启自己；启动时读取 WSL IP（`wsl hostname -I`）和本机局域网 IP（PowerShell `Get-NetIPAddress`，排除回环 / APIPA / WSL vEthernet 网段）；先删旧 portproxy 再添加新转发；添加/删除防火墙规则；服务停止后清理转发 + 规则 + `wsl --shutdown` 释放显存。
- **测试**：`tests/serve-tests.sh` / `tests/fullcontext-tests.sh` 增加 `--lan` 用例（dry-run 绑定 `0.0.0.0`、打印 LAN 提示、`--lan` 覆盖显式 `--host` 和 `SERVE_HOST` 环境变量）。
- **README**：新增 5.0 节"局域网访问"，更新 `--host` 参数表和注意事项第 6 条。

## 安全边界

- 默认仍是回环绑定，`--lan` 是显式 opt-in。
- LAN 模式无鉴权，只应在可信网络使用；README 已加醒目提示。
- 只做局域网端口转发，不做互联网 NAT 映射。

## 测试结果

- `bash tests/serve-tests.sh` → 88 passed, 0 failed
- `bash tests/fullcontext-tests.sh` → 114 passed, 0 failed
- `bash tests/run-tests.sh` → 51 passed, 0 failed
- `bash tests/preflight-tests.sh` → 71 passed, 0 failed
- 手工冒烟：`direct.sh start --dry-run --lan` → `preflight READY` + `--host 0.0.0.0` + LAN 提示 + 完整 vLLM 命令。

（顺带修复：`tests/fullcontext-tests.sh` 的 shard 错误断言与 `preflight-lib.sh` 已改的消息文案不同步，已对齐。）
