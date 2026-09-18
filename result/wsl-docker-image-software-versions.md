# WSL Docker 镜像软件版本清单

采集时间：2026-09-18 14:07（UTC+8）。数据来源：在 WSL2 Ubuntu 内以 root 运行 `docker run -i --rm --entrypoint bash lmsysorg/sglang:qwen38-27b -s`，读取容器内的包管理器与解释器输出。

## 镜像与容器

| 项 | 值 |
| --- | --- |
| WSL 发行版 | Ubuntu（WSL2，采集时处于 Stopped，由本次命令拉起） |
| Docker Engine | Client 与 Server 均 29.8.1（API 1.56，Go 1.26.8，linux/amd64） |
| Docker 存储 / 默认 runtime | `overlayfs` / `runc` |
| 镜像 | `lmsysorg/sglang:qwen38-27b`（ID `febfb971c735`，磁盘占用 41.9 GB） |
| 关联容器 | `qwen38-sglang`（ID `3a65c31ef1d1`，Entrypoint `/opt/nvidia/nvidia_…`，状态 Exited (0)） |
| 宿主机 GPU | NVIDIA GeForce RTX 5090，驱动 610.88，显存 32607 MiB |

WSL 内核：`6.18.33.2-microsoft-standard-WSL2`。

镜像声明的 CUDA 驱动约束为 `driver>=535`（`NVIDIA_REQUIRE_CUDA`，按品牌分段列至 575.x），并自带前向兼容驱动包 `cuda-compat-13-0` 580.126.20。宿主机 610.88 高于该要求。

本次探测未挂载 GPU 设备，因此容器内 `nvidia-smi` 不可用。默认 runtime 为 `runc`，启动时需显式指定 `--gpus`。

## 基础环境

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| Ubuntu | 24.04.4 LTS | `/etc/os-release` |
| Python | 3.12.3（`/usr/bin/python3`） | 解释器 |
| pip | 26.2.1（`/usr/local/lib/python3.12/dist-packages`） | pip |
| setuptools / wheel | 84.0.0 / 0.48.0 | pip |
| uv | 0.12.4 | CLI |
| GCC / G++ | 13.3.0（`build-essential` 12.10ubuntu1） | CLI |
| CMake | 3.31.1（PATH 优先），apt 包 `cmake` 为 3.28.3 | CLI + `dpkg` |
| Ninja | 1.13.0（pip `ninja`），apt 包 `ninja-build` 为 1.11.1 | CLI + `dpkg` |
| Git / git-lfs | 2.43.0 / 3.4.1 | CLI |
| OpenSSL | 3.0.13（`libssl3t64` 3.0.13-0ubuntu3.12） | `dpkg` |
| OpenSSH | 9.6p1-3ubuntu13.18 | `dpkg` |
| FFmpeg | 6.1.1-3ubuntu5（`libav*` 同版本） | `dpkg` |
| RDMA | `rdma-core` 50.0-2ubuntu0.2，`libibverbs1` 50.0，`librdmacm1t64` 50.0 | `dpkg` |
| 其他 | ccache 4.9.1、tmux 3.4、Vim 9.1、numactl 2.0.18、wget 1.21.4、curl 8.5.0 | CLI + `dpkg` |

apt 包共 952 个（947 个状态 `ii`，5 个被 `apt-mark hold` 的 NVIDIA 包，状态 `hi`）。完整清单见 `qwen38-sglang-image-apt-list.txt`。

## CUDA 与 NVIDIA 原生库

| 项 | 版本 | 说明 |
| --- | --- | --- |
| CUDA Toolkit（`nvcc`） | 13.0，V13.0.88；`CUDA_VERSION=13.0.3` | `/usr/local/cuda-13.0`，metapkg `cuda-toolkit-13-0` 系列 13.0.3-1；pip 侧另有 `nvidia-cuda-nvcc` 13.3.73 |
| 前向兼容驱动 | `cuda-compat-13-0` 580.126.20-1ubuntu1 | 容器自带的 compat 库；WSL2 下 GPU 驱动由 Windows 宿主提供，该包是否实际启用取决于启动时的库加载路径，未验证 |
| CUDA Runtime（cudart） | 13.0.96 | apt 与 pip `nvidia-cuda-runtime` 一致 |
| cuDNN | apt 9.14.0.64（`libcudnn9-cuda-13` hold，dev/headers 同版本）；pip `nvidia-cudnn-cu13` 9.20.0.48 | 系统目录存在 `libcudnn.so.9.14.0`，dist-packages 存在 9.20 一套；`torch.backends.cudnn.version()` 报 92000 |
| NCCL | apt 2.28.3-1+cuda13.0（`libnccl2`/`libnccl-dev` 均 hold）；pip `nvidia-nccl-cu13` 2.30.7；torch 编译期 2.29.7 | `/opt/nccl-2.30.7/lib` 是指向 `dist-packages/nvidia/nccl/lib` 的符号链接，供启动时 `LD_PRELOAD` 使用，即运行期有意采用 pip 的 2.30.7 |
| cuBLAS | 13.1.1.3（apt hold 包 `libcublas-13-0` / `-dev`） | pip `nvidia-cublas` 同版本 |
| cuSOLVER / cuSPARSE / cuFFT / cuRAND | 12.0.4.66 / 12.6.3.3 / 12.0.0.61 / 10.4.0.35 | |
| NPP / nvtx / CUPTI / nvRTC | 13.0.1.2 / 13.0.85 / 13.0.85 / 13.0.88 | |
| NVVM / nvcc(pip) / CCCL / crt | 13.3.73 / 13.3.73 / 13.3.3.4.1 / 13.3.73 | pip 侧独立发布，版本号高于 apt toolkit |
| Nsight Compute / Nsight Systems CLI | 2025.3.1.4 / 2026.4.1.191 | metapkg `cuda-nsight-compute-13-0` 13.0.3-1 |
| nvshmem | 3.4.5（`nvidia-nvshmem-cu13`） | |
| MathDx / cuFile | 25.6.0 / 1.15.1.6 | |
| Python CUDA 绑定 | `cuda-python` 13.3.1、`cuda-bindings` 13.3.1、`cuda-core` 1.0.1、`cuda-pathfinder` 1.6.0、`cuda-toolkit` 13.0.3.0 | |
| 其他 | `nvidia-ml-py` 13.610.43、`nvidia-cudnn-frontend` 1.27.0、`nvidia-cutlass-dsl` 4.7.0、`cubloaty` 0.1.0b3 | |
| TensorRT / DCGM | 未安装 | `dpkg -l` 与 `pip list` 中均无匹配项 |

## PyTorch 栈

| 包 | 版本 |
| --- | --- |
| torch | 2.13.0+cu130（`torch.version.cuda` 13.0，git `cf30153c4c13`） |
| torchvision / torchaudio / torchcodec | 0.28.0+cu130 / 2.11.0+cu130 / 0.15.0+cu130 |
| triton | 3.7.1 |
| torchao | 0.17.0+cu130 |
| torch_memory_saver | 0.0.9.post1 |
| torch_c_dlpack_ext | 0.1.5 |
| numpy / scipy / pandas / pyarrow | 2.3.5 / 1.18.0 / 3.0.5 / 25.0.1 |
| numba / llvmlite | 0.65.1 / 0.47.0 |

## 推理引擎与自研内核

| 包 | 版本 | 备注 |
| --- | --- | --- |
| sglang | 0.0.0.dev0+qwen38.27b.g561c8f3 | editable 安装，指向 `/sgl-workspace/sglang/python`；源码树无 `.git`，版本串由 `vcs-versioning` 生成，基线提交 `561c8f3`，分支标记 `qwen38.27b` |
| sglang-kernel | 0.4.6.post1 | |
| sglang-router | 0.3.2 | |
| sgl-deep-gemm | 0.1.5.post2 | |
| flashinfer-python | 0.6.18 | 另含 `flashinfer-cubin` 0.6.18.dev20260807、`flashinfer-jit-cache` 0.6.18.dev20260807+cu130 |
| flash-attn-4 | 4.0.0b19 | |
| tilelang | 0.1.11 | |
| deep_ep | 2.1.0+local | 源码在 `/sgl-workspace/DeepEP`，提交 `01dc3aa`（v1.2.1-38-g01dc3aa，2026-08-04），存在本地改动 `csrc/kernels/legacy/compiled.cuh` |
| st_attn / vsa | 0.0.7 / 0.0.4 | |
| quack-kernels / humming-kernels | 0.6.4 / 0.1.10 | |
| tokenspeed-mla / tokenspeed-triton | 0.1.8 / 3.8.10.post20260721 | |
| hpc-ops | 0.0.1.dev0+gab1a402 | |
| cache_dit | 1.3.0 | |
| mscclpp | 0.9.1 | |
| mooncake-transfer-engine-cuda13 | 0.3.12.post1 | |
| nixl / nixl-cu13 | 1.4.0 | |
| nccl4py | 0.4.1 | |
| cuda-tile | 1.6.0rc5 | |
| cupy-cuda13x | 14.1.1 | |
| apache-tvm-ffi | 0.1.11 | |
| kernels / kernels-data | 0.14.1 / 0.16.0 | |

## 模型加载与量化

与本项目（NVFP4）直接相关的组件：

| 包 | 版本 |
| --- | --- |
| nvidia-modelopt | 0.45.0 |
| compressed-tensors | 0.18.0 |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| huggingface_hub | 1.27.0（CLI 为 `hf`；`huggingface-cli` 已废弃） |
| gguf | 0.19.0 |
| sentencepiece / tiktoken | 0.2.2 / 0.13.0 |
| mistral_common | 1.11.7 |
| xgrammar / outlines / outlines_core / llguidance | 0.2.1 / 0.1.11 / 0.1.26 / 1.8.0 |
| datasets / fsspec | 5.0.1 / 2026.6.0 |
| modelscope / modelscope-hub | 1.39.1 / 0.2.0 |
| runai-model-streamer（含 s3/gcs/azure） | 0.16.1 |
| av / imageio / moviepy / opencv-python-headless / pillow / soundfile | 16.1.0 / 2.36.0 / 2.2.1 / 4.10.0.84 / 12.3.0 / 0.13.1 |

## 服务端与可观测

| 包 | 版本 |
| --- | --- |
| fastapi / starlette | 0.141.1 / 1.6.0 |
| uvicorn / granian | 0.52.3 / 2.8.1 |
| pydantic / pydantic_core | 2.13.4 / 2.46.4 |
| grpcio / grpcio-health-checking / grpcio-reflection | 1.83.0 / 1.81.1 / 1.81.1 |
| smg-grpc-servicer / smg-grpc-proto | 0.8.0 / 0.4.14 |
| openai / anthropic | 2.6.1 / 0.122.0 |
| httpx / httpcore / requests / websockets | 0.28.1 / 1.0.9 / 2.34.2 / 17.0.1 |
| prometheus_client | 0.26.0 |
| opentelemetry-sdk / -api / semantic-conventions | 1.44.0 / 1.44.0 / 0.65b0 |
| msgspec / orjson / uvloop / pyzmq / msgpack | 0.21.1 / 3.11.9 / 0.22.1 / 27.1.0 / 1.2.1 |
| ipython / pytest / pre_commit / black / isort / mypy_extensions | 9.16.1 / 9.1.1 / 4.6.2 / 26.5.1 / 8.0.1 / 1.1.0 |

## 镜像内 sglang 的实际身份

`sglang 0.0.0.dev0+qwen38.27b.g561c8f3` 不是 PyPI/上游发布版，而是本仓库自己构建的 fork，通过 editable 方式装入镜像。

| 组成 | 含义 | 证据 |
| --- | --- | --- |
| `0.0.0.dev0` | 无实义占位。`python/pyproject.toml` 声明 `dynamic = ["version"]`，构建时由 `SETUPTOOLS_SCM_PRETEND_VERSION` 直接给定版本串（build context 内无 git 元数据） | `docker/qwen38/qwen38_cu13.Dockerfile:220` |
| `+qwen38.27b` | PEP 440 local 段，标识目标模型变体（Qwen3.8 27B） | `python/sglang/_version.py`，`__version_tuple__` 第 5 段 |
| `g561c8f3` | commit 短哈希（`g` 为 git 前缀）；`_version.py` 中 `commit_id = None`，且源码树无 `.git`，容器内无法核对基线 | 同上 |
| editable | `__editable__.sglang-*.pth` + `*_finder.py`，`MAPPING` 把 `sglang` 映射到 `/sgl-workspace/sglang/python/sglang` | `pip install -e python --no-deps` |

`--no-deps` 是刻意的：基础镜像自带 CUDA 标注的 wheel（`sglang-kernel 0.4.6.post1`、`sgl-deep-gemm 0.1.5.post2`），常规安装会用 PyPI 无标注构建覆盖，导入时报未定义符号（`docker/qwen38/cuda_pins.sh` 注释记录）。

代码含本仓库自研部分，不在上游 sglang 中：

- **DSPARK 投机解码**：`--speculative-algorithm` 枚举含 `DSPARK`（`srt/server_args.py:2033`），配套 `speculative_dspark_block_size`、`_sps_table_path`（由 `sglang.benchmark.dspark_sps_profiler` 离线产出）、`_confidence_sts_path`（`dspark_sts_fit`）、`_align_verify_tokens_to_graph_tier`；实现位于 `kernels/ops/speculative/dspark/`、`srt/speculative/dspark_components/`、`dspark_disaggregation.py`。全树 60 个文件命中该关键字。
- **NVFP4 路径**：`srt/layers/quantization/` 下有 `nvfp4_online.py`、`fp4_kv_cache_quant_method.py`、`fp4_utils.py`、`kvfp4_tensor.py`、`mxfp4_*`、`modelopt_quant.py`；110 个文件命中 `nvfp4`。
- **构建配方随镜像发布**：`docker/qwen38/` 内 `qwen38_cu12.Dockerfile`、`qwen38_cu13.Dockerfile`、`cuda_pins.sh`、`apply_deepep_v2_patch.sh`。cu13 配方目标平台为 GB300 aarch64（sm_90a/100a/103a），本镜像为 x86_64，两者不完全对应，实际构建入口待确认。
- 源码树 89 MB、3173 个 `.py`，文件时间戳统一为 2026-08-14（构建日）。

## 依赖冲突（`python3 -m pip check` 输出）

镜像内 324 个 pip 包存在 5 处版本不满足：

| 声明方 | 要求 | 实际安装 |
| --- | --- | --- |
| sglang 0.0.0.dev0+qwen38.27b.g561c8f3 | `flashinfer_python[cu13]==0.6.15.post1` | flashinfer-python 0.6.18 |
| sglang 同上 | `nvidia-cutlass-dsl[cu13]==4.6.0` | nvidia-cutlass-dsl 4.7.0 |
| quack-kernels 0.6.4 | `nvidia-cutlass-dsl==4.6.2` | nvidia-cutlass-dsl 4.7.0 |
| torch 2.13.0+cu130 | `nvidia-nccl-cu13==2.29.7` | nvidia-nccl-cu13 2.30.7 |
| moviepy 2.2.1 | `pillow<12.0,>=9.2.0` | pillow 12.3.0 |

前四条不是安装事故，而是构建配方有意抬高、`pyproject.toml` 未同步：

| 冲突 | 配方动作 | 声明处 |
| --- | --- | --- |
| flashinfer 0.6.18 ≠ 0.6.15.post1 | `pip install --no-deps /tmp/flashinfer`（PR #4358 指定 commit 构建） | `qwen38_cu13.Dockerfile:139`，`pyproject.toml:34` |
| cutlass-dsl 4.7.0 ≠ 4.6.0 / 4.6.2 | `pip install "nvidia-cutlass-dsl[cu13]>=${CUTLASS_DSL_MIN_VERSION}"`（CuTe DSL kernel 需 `enable_multicast_signaling`） | `:140`，`pyproject.toml:48` |
| nccl 2.30.7 ≠ 2.29.7 | `pip install "nvidia-nccl-cu13==${NCCL_PIN_VERSION}"`，DeepEP v2 wheel 要求；末尾另有 `--no-deps --force-reinstall` | `:70`、`:267` |
| moviepy 与 pillow 12.3.0 | 与上述无关，属基础镜像遗留的真实版本越界 | — |

`python/sglang/srt/environ.py` 中 `SGLANG_FLASHINFER_PR4266_SOURCE` 已标记废弃（该 kernel 已进 flashinfer），但 `qwen38_cu13.Dockerfile:198` 仍 `ENV` 设置了 `/opt/flashinfer-src`，导入时因此打印一条告警。

## 其他需要注意的事实

1. DeepEP 位于 `/sgl-workspace/DeepEP`，提交 `01dc3aa`（v1.2.1-38-g01dc3aa，2026-08-04）。其唯一未提交改动 `csrc/kernels/legacy/compiled.cuh` 由 `apply_deepep_v2_patch.sh` 用 sed 写入：`LEGACY_NUM_CPU_TIMEOUT_SECS` 由 100 改为 1000，为 GB300 多机初始化留超时余量；脚本注释说明该改动在构建期不可见，因此脚本会事后断言。
2. 5 个 NVIDIA 包被 `apt-mark hold`（cuBLAS 系列、cuDNN runtime、NCCL），apt 后续升级不会改动它们。
3. `/sgl-workspace/constraints.txt` 是构建期的 pip 版本锁定，共 257 条，少于当前安装的 324 个包；两者差异可用于判断构建后新增或改写了哪些包。
4. `docker/qwen38/*.Dockerfile` 依赖的 nightly 基础镜像（`lmsysorg/sglang:dev`、`dev-cu12`）每晚移动，配方自身提示需按 digest 固定才能复现。

## 复现命令

```bash
# WSL 内 docker.sock 需 root 访问
wsl.exe -d Ubuntu -u root -- bash -c \
  "tr -d '\r' < /mnt/d/tmp/probe.sh | docker run -i --rm --entrypoint bash lmsysorg/sglang:qwen38-27b -s"
```

导出清单：

```bash
pip3 list --format=freeze                              # 324 个 Python 包
dpkg -l | awk '/^[a-z][a-z] /{print $2, $3}'           # 952 个 apt 包（含 hold）
```

只统计 `^ii` 会得到 947，漏掉 5 个 `hi`（hold）的 NVIDIA 包。

## 附件

- `qwen38-sglang-image-pip-list.txt`：`pip3 list --format=freeze` 全量输出（324 行）
- `qwen38-sglang-image-apt-list.txt`：`dpkg -l` 已安装包全量输出（952 行，仅包名与版本，不含 `ii`/`hi` 状态列）
