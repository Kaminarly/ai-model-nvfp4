# 执行计划：Qwen3.8-27B-NVFP4 升级 final build 并启用 DSpark 加速

> 依据文档：[`qwen38-nvfp4-dspark-upgrade-research.md`](./qwen38-nvfp4-dspark-upgrade-research.md)（研究报告）
> 制定日期：2026-09-16
> 备份路径：**`D:\WSL\models\backup`**（WSL 内为 `/mnt/d/WSL/models/backup`）
> 执行状态：**已于 2026-09-16 执行完毕**。阶段 1–4 通过；阶段 5（vLLM 路线 B）失败，转阶段 7（SGLang 认证路线）成功。逐项实测结果、失败根因与运维坑见 [`qwen38-nvfp4-dspark-upgrade-result.md`](./qwen38-nvfp4-dspark-upgrade-result.md)。

---

## 一、目标与范围

三个目标，按顺序推进，每个阶段都有独立 Gate，任一 Gate 不通过就停在该阶段处理：

1. 把目标模型从 `pre-final` 构建升级到 HF `main`（final build）；
2. 落位 DSpark 草稿模型；
3. 启用 DSpark 投机解码并通过验收。

**范围外**：不修改本项目代码与脚本；不重新量化、不转换权重格式；不做公网部署；不追求"262K 上下文 + DSpark"并存（单卡 32 GB 做不到，见第五节）。

---

## 二、执行前状态快照（2026-09-16 实测）

| 项目 | 现状 | 是否满足 |
| --- | --- | --- |
| WSL | Ubuntu 26.04，PID 1 = systemd，根分区可用 903 GB | ✅ |
| Windows 备份盘 | `D:` 可用 162 GB（已用 74%） | ✅ 备份约需 8.8 GB |
| GPU | RTX 5090 32,607 MiB，当前空闲 29,585 MiB（已有 2.6 GB 被占用） | ✅ |
| 端口占用 | 8000 / 8192 / 30000 均未被监听 | ✅ 无需先停服务 |
| 运行中的推理进程 | 无 vLLM / Docker / SGLang 进程 | ✅ |
| 目标模型 | `/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090` = pre-final（3 分片、18.77 GB、含 15 个 `mtp.*` 张量） | 待升级 |
| 待落位文件 | `/mnt/d/WSL/models/` 下 8 个文件已下载并完成 sha256 校验（8/8 通过） | ✅ |
| 备份目录 | `/mnt/d/WSL/models/backup` 存在且为空 | ✅ |
| vLLM | 0.27.1（`/home/kami/vllm/venv`），源码含 `dspark` 支持 | ✅ 路线 B 可用 |
| Docker / NVIDIA Container Toolkit | 均未安装 | ❌ 路线 A 需补装 |
| DSpark 草稿 | WSL 内尚未落位 | 待落位 |

---

## 三、阶段总览

| 阶段 | 内容 | 预计耗时 | 出口 Gate |
| --- | --- | --- | --- |
| 0 | 前置检查与状态冻结 | 10 min | 无服务占用、GPU 空闲、磁盘充足 |
| 1 | **备份旧构建到 `D:\WSL\models\backup`** | 10–30 min | 备份文件 sha256 全部匹配 |
| 2 | 目标模型升级落位 | 5–10 min | 2 分片、索引 2387 张量、哈希匹配 |
| 3 | DSpark 草稿落位 | 2–5 min | 3 个文件哈希匹配 |
| 4 | 无投机基线试跑 | 15–30 min | 服务起来、输出正常、记录基线吞吐 |
| 5 | 启用 DSpark（vLLM 路线 B） | 15–30 min | 日志确认 dspark 生效、无 OOM |
| 6 | 验收 | 20–30 min | 第六节清单全部通过 |
| 7 | 兜底：SGLang 路线 A（仅当阶段 5 失败） | 40–70 min | 认证组合可用 |

不含阶段 7 的总耗时：**约 1.5–3 小时**。

---

## 四、详细步骤

### 阶段 0 · 前置检查与状态冻结

**目的**：确认没有服务占用显存和端口，并留一份可回溯的状态快照。

```bash
# 1) 确认无服务与端口占用
ss -ltnp | grep -E ':(8000|8192|30000)' || echo "端口空闲"
pgrep -af 'vllm|sglang' || echo "无推理进程"

# 2) 确认 GPU 与磁盘
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
df -h / /mnt/d

# 3) 记录当前（pre-final）模型文件指纹，作为回滚比对基准
cd /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
sha256sum config.json hf_quant_config.json crc32.txt model.safetensors.index.json
```

**期望**：端口空闲、无推理进程、GPU 空闲 ≥ 29 GB、根分区 ≥ 20 GB 可用。
**失败处理**：若端口被占用，先停掉既有服务（本项目 vLLM 用 `Ctrl-C`，之后 `wsl --shutdown` 释放显存）。

**Gate 0**：以上全部满足。

---

### 阶段 1 · 备份旧构建到 `D:\WSL\models\backup`

**目的**：把 pre-final 构建完整留档到用户指定的备份路径，使后续所有改动都可无下载地回滚。

**关键设计：用"搬走"代替"删除"。** 旧分片 2、3 不删，而是移到备份目录；这样回滚时不需要重新下载 8.79 GB。

备份目录结构：

```
D:\WSL\models\backup\Qwen3.8-27B-NVFP4-RTX5090-pre-final\
├── model-00002-of-00003.safetensors    8,048,202,912 B
├── model-00003-of-00003.safetensors      744,532,384 B
├── config.json                                13,247 B
├── hf_quant_config.json                        9,087 B
├── crc32.txt                                     129 B
└── model.safetensors.index.json              237,649 B
```

```bash
# 1) 建备份目录
mkdir -p /mnt/d/WSL/models/backup/Qwen3.8-27B-NVFP4-RTX5090-pre-final

# 2) 记录被搬走文件的指纹（搬之前先算，用于搬后比对）
cd /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
sha256sum model-00002-of-00003.safetensors model-00003-of-00003.safetensors

# 3) 把旧分片 2、3 搬到备份目录（mv，不是 rm；同盘内 mv 瞬时，跨到 /mnt/d 需实际拷贝）
mv model-00002-of-00003.safetensors model-00003-of-00003.safetensors \
   /mnt/d/WSL/models/backup/Qwen3.8-27B-NVFP4-RTX5090-pre-final/

# 4) 复制 4 个小文件的旧版本
cp config.json hf_quant_config.json crc32.txt model.safetensors.index.json \
   /mnt/d/WSL/models/backup/Qwen3.8-27B-NVFP4-RTX5090-pre-final/

# 5) 校验备份内容
cd /mnt/d/WSL/models/backup/Qwen3.8-27B-NVFP4-RTX5090-pre-final
sha256sum *
```

**备份校验期望值**（pre-final 构建的实测值）：

| 文件 | 大小 | sha256 |
| --- | --- | --- |
| `model-00002-of-00003.safetensors` | 8,048,202,912 | `4b547449a2b23c6cd414da0cf65ff9d7e17ad9aa2b119beedcbba14f649eb1dd` |
| `model-00003-of-00003.safetensors` | 744,532,384 | `9ce944d534eabdd493076a3a52c7ebd31f41c135b340a1ea95c5a695e6f1f6b2` |
| `config.json` | 13,247 | `78f65e03f2ac08a39320bf4a2633f1ae1526144da0fba1904b7371e682c304ea` |
| `hf_quant_config.json` | 9,087 | `2c30a0d7e08c5eede4a273c9862aa90f49adfda1cd661dd564742749de9c1a2b` |
| `crc32.txt` | 129 | `7c6967ae0d609135f8f08f70b9e052ba974dc56a490ecd1504fc5943da9784f3` |
| `model.safetensors.index.json` | 237,649 | `4f0c8847dd549636c873737a4703ff1f215a98ec6d5e90b082b31e9e26f4e765` |

**关于分片 1**：`model-00001-of-00003.safetensors`（9,972,777,720 B，sha256 `cdd37b0e61eccc8a3d7d08f9d1a4f52856a9d88e4e8b42089bd18a970e3a01ec`）**不需要备份副本**——它的内容与新版分片 1 逐字节相同，阶段 2 只是原地改名；回滚时改回原名即可。但请把它上面这行 sha256 记下来，回滚前用它确认内容未被改动。

**注意**：跨 `/mnt/d`（9p 文件系统）的 8.79 GB 拷贝速度取决于 9p 吞吐，通常 10–30 分钟。拷贝期间不要中断；`mv` 跨文件系统时只有全部复制成功才会删除源文件。

**Gate 1**：备份目录 6 个文件的 sha256 全部匹配上表。

---

### 阶段 2 · 目标模型升级落位

**目的**：用已校验的 5 个文件替换 pre-final，复用分片 1。

```bash
cd /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090

# 1) 复用字节完全相同的分片 1：原地改名（同盘 mv 瞬时完成）
mv model-00001-of-00003.safetensors model-00001-of-00002.safetensors

# 2) 从暂存目录放入 5 个已校验文件
cp /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/config.json \
   /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/hf_quant_config.json \
   /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/crc32.txt \
   /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/model.safetensors.index.json \
   /mnt/d/WSL/models/Qwen3.8-27B-NVFP4-RTX5090/model-00002-of-00002.safetensors .

# 3) 校验
ls -l model-*.safetensors                      # 期望恰好 2 个分片
sha256sum model-00001-of-00002.safetensors     # 期望 cdd37b0e61eccc8a…（与 final build 的分片 1 一致）
grep -c . crc32.txt                            # 期望 2
python3 -c "import json;from collections import Counter;wm=json.load(open('model.safetensors.index.json'))['weight_map'];print(len(wm), Counter(wm.values()))"
# 期望：2387 Counter({'model-00001-of-00002.safetensors': 1312, 'model-00002-of-00002.safetensors': 1075})
python3 -c "import json;c=json.load(open('config.json'));t=c.get('text_config',c);print('mtp_num_hidden_layers =',t.get('mtp_num_hidden_layers'))"
# 期望：0（确认已切到 final build）
```

**注意**：此时旧分片 2、3 已经不在本目录（阶段 1 已搬走），所以这里不需要再删任何文件。目录里也不应再有 `mtp.*` 相关权重。

**Gate 2**：分片数 = 2、分片 1 sha256 匹配、`crc32.txt` 两行、索引 2387 张量（1312/1075）、`mtp_num_hidden_layers = 0`。

---

### 阶段 3 · DSpark 草稿落位

**目的**：把草稿模型放到启动命令引用的路径。

```bash
# 1) 目录尚不存在，需新建
mkdir -p /home/kami/models/Qwen3.8-27B-DSpark-NVFP4

# 2) 放入 3 个已校验文件
cp /mnt/d/WSL/models/Qwen3.8-27B-DSpark-NVFP4/config.json \
   /mnt/d/WSL/models/Qwen3.8-27B-DSpark-NVFP4/hf_quant_config.json \
   /mnt/d/WSL/models/Qwen3.8-27B-DSpark-NVFP4/model.safetensors \
   /home/kami/models/Qwen3.8-27B-DSpark-NVFP4/

# 3) 校验
cd /home/kami/models/Qwen3.8-27B-DSpark-NVFP4
sha256sum *
ls -l    # 期望仅 3 个文件：config.json(2828) / hf_quant_config.json(937) / model.safetensors(1399670058)
```

**校验期望值**：

| 文件 | 大小 | sha256 |
| --- | --- | --- |
| `model.safetensors` | 1,399,670,058 | `212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05` |
| `config.json` | 2,828 | `82fd961b632c629736902d9d4fdd3258dee1080f557cf86298cac063a514a0cf` |
| `hf_quant_config.json` | 937 | `cda90695e8c4a5eaed7ce7220afbc8bbe18e7624a167466ec7768c603e756a09` |

同时确认 `config.json` 里 `architectures` = `["Qwen3DSparkModel"]`、`block_size` = `7`。

> ⚠️ 文件名必须精确：草稿加载器固定查找 `model.safetensors`。若下载工具曾追加 `_.safetensors` 后缀（本次出现过），必须先改名。

**Gate 3**：3 个文件哈希匹配，`block_size = 7`。

---

### 阶段 4 · 无投机基线试跑

**目的**：先证明 final build 本身能正常加载与推理，并取得**与阶段 5 完全相同的上下文/并发/显存设置**下的基线吞吐，供对比。

**为什么先跑基线**：DSpark 必须开在"已经能正常工作"的模型上。若阶段 5 失败，基线数据能立刻区分"是升级问题"还是"是 DSpark 问题"。

```bash
# 用项目脚本启动（自动带上 modelopt / fp8 KV / 前缀缓存 / trust-remote-code，
# 并在服务端 default 采样参数上复用既有配置）。上下文等参数与阶段 5 保持一致。
FULL_MAX_MODEL_LEN=122880 FULL_MAX_NUM_SEQS=1 FULL_GPU_MEM_UTIL=0.86 SERVE_PORT=8192 \
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/direct.sh start \
  --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
```

**要点**：

- **不要用 `scripts/start-api-server-mtp.bat`**。它启用 `--spec-method mtp`，而 final build 已删除全部 `mtp.*` 张量，该路线已失效。
- **不要设 `VLLM_SPEC_METHOD`**。原因见阶段 5 的警告。
- 首次启动可能触发 FlashInfer SM120 内核 JIT 编译（耗时数分钟）；项目脚本已设 `MAX_JOBS=1` 避免并行 nvcc 撑爆内存。

**验证（另开一个 PowerShell 或 WSL 终端）**：

```bash
curl -s http://127.0.0.1:8192/v1/models
curl -s -X POST http://127.0.0.1:8192/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-NVFP4-RTX5090","messages":[{"role":"user","content":"只回答：服务正常"}],"max_tokens":32,"temperature":0}'
```

**记录**：解码 tok/s（用固定长度的长输出任务测）、`nvidia-smi` 显存占用、冷启动 TTFT。

**Gate 4**：`/v1/models` 返回 200、聊天请求返回 200 且内容正常、无 OOM、已记录基线吞吐。停掉服务后再进入阶段 5。

---

### 阶段 5 · 启用 DSpark（vLLM 路线 B）

**目的**：在本机已有的 vLLM 0.27.1 上启用 DSpark，无需安装 Docker。

> ⚠️ **必须避开的陷阱**：项目脚本 `scripts/lib/serve-lib.sh` 在检测到 `VLLM_SPEC_METHOD` 时会**硬编码追加 `--spec-tokens 3`**。而 vLLM 源码要求 DSpark 的 `num_speculative_tokens >= dspark_block_size`（本模型 = 7），小于该值**会输出乱码而不只是接受率下降**。因此**不要**用 `VLLM_SPEC_METHOD=dspark`，必须改用 `VLLM_EXTRA_ARGS` 显式传全套参数。

**推荐方式：复用项目脚本 + 透传参数**

```bash
# 先 dry-run，确认 argv 里出现 --spec-method dspark --spec-tokens 7
# 注意：--dry-run 仍会先执行 preflight（会按索引核对分片是否齐全），因此它也是
#       对阶段 2 升级结果的一次额外验证；只有 VRAM 门限会被跳过。
VLLM_EXTRA_ARGS="--spec-method dspark --spec-model /home/kami/models/Qwen3.8-27B-DSpark-NVFP4 --spec-tokens 7" \
FULL_MAX_MODEL_LEN=122880 FULL_MAX_NUM_SEQS=1 FULL_GPU_MEM_UTIL=0.86 SERVE_PORT=8192 \
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/direct.sh start --dry-run \
  --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
# 期望输出：preflight READY，随后一行 [DRY-RUN] … 其中含 --spec-method dspark --spec-tokens 7

# 确认无误后去掉 --dry-run 正式启动
VLLM_EXTRA_ARGS="--spec-method dspark --spec-model /home/kami/models/Qwen3.8-27B-DSpark-NVFP4 --spec-tokens 7" \
FULL_MAX_MODEL_LEN=122880 FULL_MAX_NUM_SEQS=1 FULL_GPU_MEM_UTIL=0.86 SERVE_PORT=8192 \
bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/direct.sh start \
  --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
```

**隔离排障方式：绕过脚本直接起 vLLM**（上面失败时用，可排除脚本因素）

```bash
/home/kami/vllm/venv/bin/vllm serve /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090 \
  --served-model-name Qwen3.8-27B-NVFP4-RTX5090 \
  --quantization modelopt --kv-cache-dtype fp8 \
  --spec-method dspark \
  --spec-model /home/kami/models/Qwen3.8-27B-DSpark-NVFP4 \
  --spec-tokens 7 \
  --max-model-len 122880 --max-num-seqs 1 \
  --gpu-memory-utilization 0.86 \
  --enable-prefix-caching --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --host 127.0.0.1 --port 8192
```

**参数含义**：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--spec-method` | `dspark` | 本机 vLLM 0.27.1 的 `SpeculativeMethod` 已含该取值 |
| `--spec-model` | 草稿目录 | 独立草稿检查点路径（不由目标模型自带） |
| `--spec-tokens` | `7` | **不得小于草稿 `block_size` = 7** |
| `--max-model-len` | `122880` | 开投机后 32 GB 装不下 262K，取保守值 |
| `--max-num-seqs` | `1` | GDN 状态槽约束（每请求 4 槽） |
| `--gpu-memory-utilization` | `0.86` | 留约 3.5 GB 余量 |

**Gate 5**：日志明确显示 DSpark 草稿已加载、**未回退**到无投机；服务正常返回；`nvidia-smi` 无 OOM；解码吞吐相对阶段 4 基线有提升。

**失败处理**：
- 加载报错 / 输出乱码 / 吞吐不升 → 记录完整日志，进入**阶段 7（SGLang 认证路线）**。
- 若提示草稿量化识别问题，改用 `--speculative-config` JSON 显式给出量化与 block 相关字段。
- 首次请求 OOM → 把 `--gpu-memory-utilization` 降到 0.82 再试。

---

### 阶段 6 · 验收

沿用研究报告第五节的验收清单，逐条核对：

- [ ] `/v1/models` 返回 HTTP 200，模型 ID 正确
- [ ] 不带认证头（本机 vLLM 未设 API key）的聊天请求返回 200
- [ ] 日志显示 DSpark 已加载，未回退到 MTP 或无投机
- [ ] 连续 ≥ 3 个短请求全部成功
- [ ] `nvidia-smi` 无 OOM，显存余量符合预期
- [ ] 长提示词测试：输入 + 输出总 token 不超过 122,880
- [ ] **输出一致性**：同一 prompt 在"无投机（阶段 4）"与"有 DSpark（阶段 5）"下贪婪解码结果一致（DSpark 构造上无损；不一致说明配置有误，例如 `--spec-tokens` 小于 7）
- [ ] 吞吐提升已量化：`（阶段 5 解码 tok/s）÷（阶段 4 基线）`，预期约 1.9–2.0×，若持平或更慢则需分析接受率与验证成本

---

### 阶段 7 · 兜底：SGLang 认证路线（仅当阶段 5 失败）

**目的**：走模型作者认证的组合。本机缺 Docker，需先补装。

```bash
# 1) 安装 Docker Engine（Ubuntu 26.04 官方源）
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"   # 之后需重进 WSL 生效

# 2) 安装 NVIDIA Container Toolkit
sudo apt install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3) 验收容器能看到 GPU
docker pull lmsysorg/sglang:qwen38-27b        # 约 17.9 GB
docker run --rm --gpus all --entrypoint nvidia-smi lmsysorg/sglang:qwen38-27b
# 期望输出中出现 NVIDIA GeForce RTX 5090
```

启动（容器直接挂载已落位的本地目录，无需在容器内重新下载）：

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

**32 GB 上不可删的参数**：`--mamba-ssm-dtype bfloat16`、`--max-mamba-cache-size 8`、`--speculative-draft-model-quantization modelopt_fp4`、`--mem-fraction-static 0.86`、`--max-running-requests 1`。

**注意**：与 vLLM 不能同时运行（端口与显存都会冲突）；本机 Windows 10 的 WSL2 是 NAT，如需局域网访问要另做 portproxy + 防火墙规则。

---

### 阶段 8 · 回滚

任何阶段失败都可完整回滚，**不需要重新下载**（旧分片已在备份目录）：

```bash
cd /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090

# 1) 分片 1 改回原名（内容未变，用 Gate 2 记录的 sha256 复核）
mv model-00001-of-00002.safetensors model-00001-of-00003.safetensors
sha256sum model-00001-of-00003.safetensors     # 期望 cdd37b0e61eccc8a…

# 2) 从备份目录取回旧分片 2、3 与 4 个小文件
cp /mnt/d/WSL/models/backup/Qwen3.8-27B-NVFP4-RTX5090-pre-final/* .

# 3) 移除新版的第 2 分片（避免与旧索引混淆）
rm -f model-00002-of-00002.safetensors

# 4) 校验回到 pre-final
grep -c . crc32.txt                            # 期望 3（旧 crc32.txt 有 3 行）
python3 -c "import json;from collections import Counter;wm=json.load(open('model.safetensors.index.json'))['weight_map'];print(len(wm), Counter(wm.values()))"
# 期望：2402 Counter({'model-00001-of-00003.safetensors': 1312, 'model-00002-of-00003.safetensors': 1077, 'model-00003-of-00003.safetensors': 13})
```

DSpark 草稿目录可保留（不影响回滚后的运行）；若要一并清理，删除 `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4` 即可。

---

## 五、决策点与风险

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| **vLLM + DSpark 未获作者认证**（最大不确定性） | 阶段 5 可能加载失败或退化 | 阶段 4 先建立基线；失败即转阶段 7 的认证路线 |
| `--spec-tokens` < 7 | **输出乱码**（非仅变慢） | 禁用 `VLLM_SPEC_METHOD`，用 `VLLM_EXTRA_ARGS` 显式传 7；阶段 6 做输出一致性比对 |
| 上下文与解码头寸二选一 | 32 GB 装不下"262K + DSpark" | 开投机用 122,880；需要满上下文就关掉 DSpark 单独跑 |
| 并发受限 | 开投机后只能并发 1 | 高并发场景改用无投机的 vLLM（其聚合吞吐更强） |
| 备份跨 9p 拷贝较慢 | 阶段 1 耗时拉长 | 预留 10–30 分钟，勿中断；`mv` 成功前不会删源文件 |
| 升级后 MTP 路线失效 | 既有 `start-api-server-mtp.bat` 不可用 | 明确改用 DSpark；不要在该脚本上排障 |
| 长上下文接受率回退 | 超长文档场景收益下降 | 草稿卡自述长上下文域接受率 −6.1%（训练语料上限 2,048 token） |
| 显存被 Windows 占用 2.6 GB | 可用约 29.5 GB | 保持 `gpu-memory-utilization 0.86`；OOM 时降至 0.82 |
| 端口冲突 | vLLM 8192 / SGLang 30000 | 两条路线不要同时启动 |

**关键决策点**：

1. **Gate 4 之后**：若基线本身异常（final build 加载失败），停止推进，先按阶段 8 回滚并单独排查，不要叠加 DSpark 变量。
2. **Gate 5 之后**：若 DSpark 无收益或输出不一致，直接转阶段 7，不要在 vLLM 参数上反复试错。

---

## 六、执行检查清单

**阶段 0**
- [ ] 端口 8000/8192/30000 空闲
- [ ] 无 vLLM / SGLang 进程
- [ ] GPU 空闲 ≥ 29 GB，根分区可用 ≥ 20 GB，`D:` 可用 ≥ 10 GB
- [ ] 已记录 pre-final 的 4 个小文件 sha256

**阶段 1**
- [ ] 备份目录已建：`D:\WSL\models\backup\Qwen3.8-27B-NVFP4-RTX5090-pre-final\`
- [ ] 旧分片 2、3 已搬入备份目录（6 个文件齐全）
- [ ] 备份 6 个文件 sha256 全部匹配

**阶段 2**
- [ ] 分片 1 已改名为 `model-00001-of-00002.safetensors`
- [ ] 5 个新文件已放入，目录内恰好 2 个分片
- [ ] 索引 2387 张量（1312 / 1075）
- [ ] `mtp_num_hidden_layers = 0`

**阶段 3**
- [ ] `/home/kami/models/Qwen3.8-27B-DSpark-NVFP4` 已建，3 个文件哈希匹配
- [ ] `block_size = 7`

**阶段 4 / 5**
- [ ] `direct.sh --dry-run` 的 argv 中出现 `--spec-method dspark` 与 `--spec-tokens 7`
- [ ] 未使用 `VLLM_SPEC_METHOD`、未使用 `start-api-server-mtp.bat`
- [ ] 日志确认 DSpark 已加载、未回退

**阶段 6**
- [ ] 研究文档第五节 10 条验收项全部通过
- [ ] 有/无投机的贪婪输出一致
- [ ] 吞吐提升已量化记录

**回滚就绪确认**
- [ ] 备份目录内容完整（未在本流程任何一步中删除）