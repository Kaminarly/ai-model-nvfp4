# 删除 SGLang 镜像（lmsysorg/sglang:qwen38-27b）并回收 D 盘空间：执行记录

执行时间：2026-09-19（UTC+8）。动机：该镜像落盘 **41.9 GB**，是这台机器上最大的一块可回收磁盘占用。目标是删掉它，同时保证 **vLLM** 与 **SparkInfer** 两条路线不受影响、且不产生 18 GB 的意外重新下载。

## 一、删除前的依赖关系（只读勘察）

| 项目 | 事实 |
| --- | --- |
| 镜像 | `lmsysorg/sglang:qwen38-27b`，image ID `sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1` |
| RepoDigest（删前记录） | `lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1` |
| 创建时间 / 大小 | `2026-08-14T08:46:12Z`；docker 报告 41908291973 B（41.9 GB）；`SHARED SIZE 0B`（与 SparkInfer 镜像不共享层） |
| 标签数量 | 1（`qwen38-27b`），无 `<none>` 悬空镜像，build cache 0B |
| **唯一引用者** | 容器 `qwen38-sglang`（`3a65c31ef1d1`），`Exited (0)`，可写层 101 MB（含日志）。整机只有这一个容器 |
| 容器的挂载 | 仅两个**只读 bind mount**（两个模型目录），`Volumes=null` → 删除容器**不可能**碰到模型文件 |
| Docker 存储位置 | `/var/lib/docker`（overlayfs），位于 WSL 发行版 `Ubuntu` 的 `/dev/sdd`，宿主文件 `D:\WSL\ext4.vhdx` |

## 二、执行的删除步骤

```bash
# 1. 先把容器日志留档（删容器会连带删掉那个 101 MB 可写层）
docker logs qwen38-sglang > /home/kami/logs/qwen38-sglang-2026-09-19.log 2>&1   # 66114 字节
# 2. 再删容器（镜像的唯一引用者；不删容器，docker image rm 会报 conflict）
docker rm qwen38-sglang
# 3. 最后删镜像
docker image rm lmsysorg/sglang:qwen38-27b
#    -> Untagged: lmsysorg/sglang:qwen38-27b
#    -> Deleted: sha256:febfb971c735...
```

删除后的状态：`docker images` 只剩 `ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10`（1.47 GB）；`docker ps -a` 为空；`docker system df` 中 Images = 1.467 GB、Containers = 0B、Build Cache = 0B。另有 25 个匿名卷，全部 0B，未处理（`docker volume prune` 也回收不到空间，且对未运行容器的卷做 prune 有误删风险）。

## 三、两条保命路线的事后验证（都实测过）

| 路线 | 验证方式 | 结果 |
| --- | --- | --- |
| vLLM（`direct.sh` / `start-api-server-vllm.bat`） | `bash scripts/preflight.sh --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` | **22/22 通过，READY**。这条线全程不碰 docker（venv 进程），与镜像无关 |
| SparkInfer（`sparkinfer-serve.sh` / `start-api-server-sparkinfer.bat`） | `sudo bash scripts/sparkinfer-serve.sh start --dry-run` | daemon 可达、`image present: ghcr.io/...:0.5.10`、模型/草稿/GPU/端口全部 OK，dry-run 正常打印 `docker run` 命令 |
| SGLang + DSpark（预期失效） | `sudo bash scripts/sglang-dspark.sh start --dry-run` | **按设计 fail closed**：`[FAIL] image 'lmsysorg/sglang:qwen38-27b' is not present locally.` + `fix: pull it (about 18 GB): docker pull lmsysorg/sglang:qwen38-27b`。**脚本不会自动 pull**，因此不存在"删了又被偷偷下回来"的风险 |

脚本侧依据：`sglang-dspark.sh` 与 `sparkinfer-serve.sh` 各自只 `docker image inspect` **自己**的镜像（缺了就退出并给 pull 提示），全程只有 `docker rm -f "$NAME"` 删**自己**的容器；全仓库没有任何脚本会执行 `docker rmi` / `image rm` / `prune`（`docker rmi` 只出现在历史 `result/` 文档的文字说明里）。

## 四、D 盘空间回收（这一步有坑，已解决）

删镜像只释放了发行版**内部**的空间（`/` 使用量 90G → 51G），宿主侧 `D:` 可用空间**当时一点没变**（129.35 → 129.32 GiB）。原因：

1. WSL 的 `D:\WSL\ext4.vhdx` 是**非稀疏**动态 VHDX，不会自动把空闲块交还 Windows；`fstrim -v /`（实测 trimmed 955.6 GiB）只把块清零，不缩文件；
2. 官方提供的自动回收开关对**已存在的发行版**被禁用：`wsl --manage Ubuntu --set-sparse true` 返回 `Wsl/Service/E_INVALIDARG`，原文 *"由于潜在的数据损坏，目前已禁用稀疏 VHD 支持。要强制发行版使用稀疏 VHD，请运行 wsl.exe --manage <DistributionName> --set-sparse true --allow-unsafe"* —— 考虑到这个发行版里是模型、venv、llama.cpp，**没有使用 `--allow-unsafe`**；
3. 采用的方案是官方文档路径：**离线压缩已停止的 VHDX**（`scripts/reclaim-wsl-space.bat`，管理员权限，diskpart `attach vdisk readonly` → `compact vdisk` → `detach vdisk`）。

实测效果：

| 指标 | 压缩前 | 压缩后 | 变化 |
| --- | --- | --- | --- |
| `D:` 可用空间 | 129.33 GiB | **171.71 GiB** | **+42.38 GiB** |
| `D:\WSL\ext4.vhdx` 逻辑大小 | 95.21 GiB | **52.83 GiB** | −42.38 GiB |

压缩后复查：WSL 正常启动，`/` 51G used / 905G avail，三个模型目录与留档日志都在，docker 里 SparkInfer 镜像完好。

## 五、恢复这条路线（如果以后还想跑 SGLang + DSpark）

```bash
# 约 18 GB 下载 / 41.9 GB 落盘；想复现同一份镜像可按 digest 拉
wsl -d Ubuntu -u root -- docker pull lmsysorg/sglang:qwen38-27b
wsl -d Ubuntu -u root -- docker pull lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1
```

注意标签 `qwen38-27b` 是上游作者维护的，未来可能漂移；要严格复现本次实测过的那份，请用 digest。回滚不涉及模型文件、草稿、vLLM venv、llama.cpp、SparkInfer 镜像与 Windows 驱动——本次从未修改它们。

## 六、本文改动过的文件

| 文件 | 改动 |
| --- | --- |
| `scripts/reclaim-wsl-space.bat` | 新增（管理员离线压缩 WSL VHDX；含 UAC 提权、前后对比、自动重启 WSL） |
| `README.md` | 4.8 节镜像条目加"已删除 + digest + 需手动 pull"；路线对照表加提示；新增 Q17（删镜像后 D 盘为何不变大 + 压缩实测数据）；Q15 指向 dspark 时补上"需先 pull" |
| `scripts/start-api-server-dspark.bat` | 文件头加"镜像已不在本机"说明与 pull 命令 |
| `scripts/sglang-dspark.sh` | 文件头加同样的说明（含 digest） |
| `result/sglang-image-removal-result.md` | 本文 |