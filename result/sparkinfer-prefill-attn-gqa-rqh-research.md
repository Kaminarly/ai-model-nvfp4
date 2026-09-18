# sparkinfer: `SPARKINFER_PREFILL_ATTN_GQA_RQH`, int8 KV, and the >=2048-token prefill defect

Research note. Every claim below is quoted from a primary source (raw.githubusercontent.com file
contents); URLs are given per section. Where a source does **not** mention something, that is stated
explicitly.

Sources fetched:

- `https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/CHANGELOG.md` (96,417 B)
- `https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/server/README.md`
- `https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/server/src/model_engine.cpp`
- `https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/server/src/sparkinfer_server.cpp`
  (239,656 B; raw fetch truncates — only the head was readable)
- `https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/runtime/examples/qwen3_gguf_generate.cpp`
- `https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/runtime/include/sparkinfer/kv_cache.h`
- `https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/kernels/include/sparkinfer/kernels/deterministic.h`
- tag-pinned READMEs: `.../v0.5.6/server/README.md`, `.../v0.5.7/server/README.md`,
  `.../v0.5.9/server/README.md`

---

## 1. `SPARKINFER_PREFILL_ATTN_GQA_RQH` upstream

**Yes — it is documented, in `server/README.md`, in the *Determinism* section** (and again,
abbreviated, in the known-defect table). What the doc says it does, verbatim
([main/server/README.md](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/server/README.md)):

> Cost, same hardware and model: decode throughput is unchanged (decode is untouched); TTFT is
> **+2% at short prompts and +8% at 2k**. Above ~2048 tokens the mode also turns the GQA-fused int8
> MMA prefill attention off (`SPARKINFER_PREFILL_ATTN_GQA_RQH=1`), which costs some long-context
> prefill throughput — see the warning below for why.

and, inside the defect blockquote:

> The cliff is exactly at 2048 and it is not the RQH=3 tier's own `n_tokens >= 2048` gate: RQH=2 is
> chosen both below and above it and is equally wrong above, so the length dependence is inside
> `launch_attn_gqa`. Only `RQH=1`, which drops to the per-q-head fallback, is correct there.

So per the doc: it selects the RQH tier used by GQA prefill attention; `RQH=1` drops to the
per-q-head fallback, i.e. off the fused MMA path. It is **not** in the README's Env table — it
appears only in prose (the Determinism section) and abbreviated in the defect table row
`` | `…GQA_RQH=1` | ~0.0001 | ~0.0001 | ~0.0001 | ~0.0001 | 0.00008 | ``.

**When introduced (as far as the sources show).** The identical two passages are present in the
tag-pinned READMEs at **v0.5.6** (`.../v0.5.6/server/README.md`), **v0.5.7**, **v0.5.9** and
`main`. So it is documented at least as far back as the v0.5.6 tag (2026-09-14). I did not check
tags older than v0.5.6, and no source states an introduction date.

**CHANGELOG: nothing.** Across the whole `CHANGELOG.md` (95,445 characters read), the literal
strings `SPARKINFER_PREFILL_ATTN`, `GQA_RQH` and `RQH` occur **zero** times (regex count over the
full file). There is therefore **no** CHANGELOG entry — in 0.5.9, 0.5.10 or anywhere else — saying
this variable was REMOVED, RENAMED, made default, or replaced. `#976` also occurs **zero** times.
The word `escape` occurs **zero** times.

The nearest CHANGELOG statement about the mechanism (in **0.5.6**, i.e. *outside* the 0.5.7–0.5.10
range asked about), which never names the variable:

> ## [0.5.6] — 2026-09-14
> ### Correctness
> - **Deterministic mode is faster and more accurate.** It still forced the non-fused GQA prefill on
>   the strength of a bug #980 had already fixed (#982).

---

## 2. `SPARKINFER_KV_INT8`

**README table row** ([main/server/README.md](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/server/README.md)):

> | `SPARKINFER_KV_INT8` | model-dependent | Same as `qwen3_gguf_generate` |

That is the *entire* README env-table entry: the purpose column defers to another tool, so the
README itself does not say what it does beyond "model-dependent". The surrounding explanation the
README does give is in the defect blockquote:

> With int8 KV (which the server enables whenever `--ctx` ≥ 4096) and a prompt of 2048 tokens or
> more, the GQA-fused MMA prefill attention disagrees sharply with the token-loop reference, and
> not reproducibly.

**What it actually does (source).** [`runtime/examples/qwen3_gguf_generate.cpp`](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/runtime/examples/qwen3_gguf_generate.cpp):

```cpp
    // int8 KV is the Qwen3-MoE head_dim=128 tensor-core path; Qwen3.6 attention (gated, head_dim=256) writes bf16 KV.
    { const char* e = getenv("SPARKINFER_KV_INT8");   // hybrid: int8 KV when prompt length >= 4k
      kvc.int8_kv = e ? (e[0] != '0') : (cfg.hybrid ? ((argc - 3) >= 4096) : true); }
```

**Default value.** `model-dependent`. Unset (`getenv` returns null): non-hybrid → `true` (int8 KV
always); hybrid → int8 KV iff the *prompt length* (`argc - 3`, the token ids) is ≥ 4096. In the
server the same expression keys off `--ctx`, not the prompt
([`server/src/model_engine.cpp`](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/server/src/model_engine.cpp)):

```cpp
    { const char* e = getenv("SPARKINFER_KV_INT8");
      // Muse Glimmer: int8 KV cache is a confirmed correctness bug, not a precision tradeoff --
      // incoherent output from the very first decode token (#779), root-caused to its per-layer
      // sliding-window/NoPE alternation + sandwich-norm activations not matching what the int8
      // quantize/dequantize kernels were tuned against (Qwen3.6, same cfg.hybrid=true, is
      // unaffected). The CLI tools never caught this because their short eval prompts (<4096
      // tokens) always fell under the bf16 threshold below; the server activates int8 off its
      // configured max_seq (there's no per-request length at KV-pool-init time), and the default
      // max_seq (4096) satisfies ">=4096" unconditionally, so every default-config Muse Glimmer
      // server silently served garbage. Default to bf16 until the kernel bug itself is fixed;
      // SPARKINFER_KV_INT8=1 still force-enables it for anyone debugging that fix.
      kvc.int8_kv = e ? (e[0] != '0')
                      : (impl_->cfg.muse_glimmer ? false
                         : impl_->cfg.hybrid ? (impl_->cfg.max_seq >= 4096) : true); }
```

**Alternative when set to 0: bf16, 2 bytes per KV element.** `SPARKINFER_KV_INT8=0` sets
`int8_kv = false`; the README/source then describe the pool as bf16. Same file:

```cpp
    const int elem_bytes = kv.int8_kv() ? 1 : 2;
```

```cpp
    // pool_bytes is a bf16-DENOMINATED BUDGET, not an allocation: KVCacheManager derives
    // total_blocks = pool_bytes / (n_slots * 2 * bf16_bytes_per_block) and then mallocs at the
    // real element width (int8 just mallocs less). So the `* 2` here is the bf16 element size and
    // is correct as written -- it must stay even when int8_kv is on, or capacity halves.
```

and the log line: `kvc.int8_kv ? 1.0 : 2.0`. [`runtime/include/sparkinfer/kv_cache.h`](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/runtime/include/sparkinfer/kv_cache.h):

```cpp
    bool fp8_kv = false;        // FP8 KV cache compression
    bool int8_kv = false;       // int8 (Q8-style) KV cache; halves the long-context KV read
```

```cpp
    // int8 KV (Q8-style int8 + per-(token,kv_head) fp16 scale). When int8_kv(), k_pool/v_pool hold
    // int8 and k_scale_pool/v_scale_pool hold one __half scale per head vector.
```

So: int8 KV = **1 byte per element plus one fp16 (2-byte) scale per head vector**; the `0`
alternative is **bf16 = 2 bytes per element**. An `fp8_kv` flag exists in `KVCacheConfig` (default
`false`, "FP8 KV cache compression") but **neither call site I read sets it** — in
`qwen3_gguf_generate.cpp` and the server's `model_engine.cpp` only `int8_kv` is assigned. The
CHANGELOG confirms bf16 as the non-int8 path (0.5.6, Muse Glimmer): "Muse Glimmer now honours an
int8 KV cache instead of writing bf16 into it (#1006)… The server was not affected: it keeps Muse
Glimmer on bf16 KV."

**Is int8 KV really enabled automatically at `--ctx >= 4096`, and why?** Yes for the server (hybrid
models): `impl_->cfg.hybrid ? (impl_->cfg.max_seq >= 4096) : true`, and `--ctx` is what sets
`max_seq` (`if (max_seq > 0) impl_->cfg.max_seq = max_seq;`). For Muse Glimmer the default is
forced to bf16 regardless. The README states it plainly: "With int8 KV (which the server enables
whenever `--ctx` ≥ 4096)". The reason stated in the source comment is that the choice is made once,
at KV-pool initialisation, where **"there's no per-request length at KV-pool-init time"**, so the
server uses the configured `max_seq` while the CLI tools use the prompt length; the default
`max_seq` of 4096 therefore satisfies `>= 4096` unconditionally.

---

## 3. Does current `main` still document the known defect? Any workaround named?

**Yes, it still documents it** — verbatim from
[main/server/README.md](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/server/README.md):

> **Known defect, independent of this mode: GQA-fused int8 prefill attention above ~2048 tokens.**
> With int8 KV (which the server enables whenever `--ctx` ≥ 4096) and a prompt of 2048 tokens or
> more, the GQA-fused MMA prefill attention disagrees sharply with the token-loop reference, and
> not reproducibly. `qwen3_gguf_prefill_check` against that reference, Qwen3.6-35B-A3B on an RTX
> 5090, mean KL over 16 teacher-forced positions:
>
> | prefix | 1500 | 2000 | **2100** | 3000 | 4000 |
> |---|---|---|---|---|---|
> | default (fused) | 0.00043 | 0.00022 | **0.18672** | 0.20657 | 0.23978 |
> | `…GQA_RQH=1` | ~0.0001 | ~0.0001 | ~0.0001 | ~0.0001 | 0.00008 |
>
> The cliff is exactly at 2048 and it is not the RQH=3 tier's own `n_tokens >= 2048` gate: RQH=2 is
> chosen both below and above it and is equally wrong above, so the length dependence is inside
> `launch_attn_gqa`. Only `RQH=1`, which drops to the per-q-head fallback, is correct there.
> Reproduce with:
>
> ```bash
> SPARKINFER_KV_INT8=1 ./build/runtime/qwen3_gguf_prefill_check <model.gguf> 4000 16 <4000+ real token ids>
> ```
>
> This is left ON by default rather than silently switched off, because disabling it moves the
> long-context prefill throughput the eval scores against — that call belongs with whoever owns the
> kernel. Deterministic mode simply declines to build on top of it.

(Emphasis in the original is `> **Known defect, …**`; the rest is unmodified.)

**Workaround named:** yes, one, and it is named twice in the README — the full variable name in the
paragraph immediately *above* the blockquote ("...turns the GQA-fused int8 MMA prefill attention off
(`SPARKINFER_PREFILL_ATTN_GQA_RQH=1`)"), and inside the blockquote by tier abbreviation and prose
("`…GQA_RQH=1`", "Only `RQH=1`, which drops to the per-q-head fallback, is correct there"). The
README also states the defect is deliberately **left on by default** — i.e. there is no default
mitigation — and that deterministic mode "declines to build on top of it".

Also still present in the same README is the reproduce command that forces int8 KV on
(`SPARKINFER_KV_INT8=1 … qwen3_gguf_prefill_check`).

---

## 4. CHANGELOG between 0.5.7 and 0.5.10

Source: [main/CHANGELOG.md](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/CHANGELOG.md).
Full-file regex counts: `PREFILL_ATTN` 0, `GQA_RQH` 0, `RQH` 0, `#976` 0, `SPARKINFER_KV_INT8` 0,
`escape` 0, `renamed` 0. "int8 KV" occurs 3 times in the file, both lines outside 0.5.7–0.5.10
(0.5.6 Muse Glimmer / 0.3.8 `#284`).

**Entries in range that mention long-context correctness — yes, two releases:**

> ## [0.5.8] — 2026-09-16
> **Long agent sessions work.** Past 16,384 tokens, decode attended only the attention sink and the
> most recent 4,096 tokens: everything in between was invisible to the tokens being generated, so an
> agent lost sight of its own session within a few turns. That was a default-on decode optimisation
> (#958) whose accuracy gate never used a prompt long enough to reach it. Exact attention is the
> default again; `SPARKINFER_SPARSE_GQA6=1` opts back in.

> ### Long context and agent sessions (#1088)
> - **Exact attention above 16K.** A 44,000-token prompt with one `ERROR` line 20,000 tokens back
>   answered "There is no line containing ERROR"; it now quotes the line and the one before it.
>   Long-context decode is slower for it (95 → 85.7 tok/s at ctx 32768).
> - **A cached prefix followed by a long continuation** is windowed rather than dropped onto the
>   token loop: time to first token for a 44K-token prompt fell from 348 s to 4.7 s.

> ## [0.5.7] — 2026-09-14
> ### Correctness
> - **Prompts past 32,768 tokens lost their Gated-DeltaNet state.** Windowed prefill runs in 16K
>   windows, and every window after the first restarted the Gated-DeltaNet layers (48 of them on
>   Qwen3.8-27B) from zero. Nothing caught it because the accuracy gates stop at 32K. The state now
>   carries into each window, and a pass at position 0 runs the same arithmetic as before (#1071).

**Not found in 0.5.7–0.5.10:** any mention of issue **#976**, of **prefill attention correctness**,
of **int8 KV**, or of a **removal of an escape-hatch env var**. 0.5.9's five sections (Serving,
Chat, Performance — Muse Glimmer, Project) and 0.5.10's single paragraph (accepting `reasoning` /
`refusal` / `annotations` / `audio` / `function_call` back in a request) contain none of these
topics. The only env-var naming change in the range is a *rename that keeps the old name working*
(0.5.8 and 0.5.9):

> - **`SPARKINFER_DRAIN_GRACE_S`** (default 30, `SPARKINFER_SHUTDOWN_GRACE_S` still read); `0` waits
>   for in-flight work as long as it takes.

The nearest int8-KV entry is in **0.5.6** (outside the range):

> ### Muse Glimmer
> Muse Glimmer now honours an int8 KV cache instead of writing bf16 into it (#1006). With int8 KV —
> what the example tools select at 4,096 tokens and up — it produced garbage and fell back to
> token-at-a-time prefill. The server was not affected: it keeps Muse Glimmer on bf16 KV.

---

## 5. Other runtime knobs that disable the fused MMA prefill path / mitigate the defect

Only the following are named by the sources I read:

1. **`SPARKINFER_PREFILL_ATTN_GQA_RQH=1`** — the knob the README names; `RQH=1` "drops to the
   per-q-head fallback" and is the only tier the README calls correct above 2048 tokens. This is the
   direct disable of the fused MMA prefill attention path.
2. **`SPARKINFER_DETERMINISTIC=1`** — not a prefill switch, but the README states it produces the
   same effect above ~2048 tokens: "Above ~2048 tokens the mode also turns the GQA-fused int8 MMA
   prefill attention off (`SPARKINFER_PREFILL_ATTN_GQA_RQH=1`), which costs some long-context
   prefill throughput". It also disables the prefix cache and is described as a bit-reproducibility
   mode, not a correctness workaround for this defect ("**Known defect, independent of this
   mode**").
3. **`SPARKINFER_KV_INT8=0`** — a *source-level* mitigation of the precondition, not one the README
   names as a workaround: the defect paragraph scopes the defect to int8 KV ("With int8 KV … and a
   prompt of 2048 tokens or more"), and `model_engine.cpp` set to `0` forces `kvc.int8_kv = false`
   (bf16, 2 bytes/element). The README's defect blockquote itself names no workaround other than
   `RQH=1`, and states "This is left ON by default rather than silently switched off".

**Explicitly not found:** no other env var, CLI flag or config in the sources I read is documented
as disabling the fused MMA prefill attention path or mitigating this defect. `SPARKINFER_SPARSE_GQA6`
(the 0.5.8 escape hatch) is a *decode*-path sparse-attention knob and is unrelated to prefill
correctness. `SPARKINFER_PREFILL_BATCHED`, `SPARKINFER_PREFILL_CHUNK_TOKENS` and
`SPARKINFER_PREFILL_MIX_MAX` appear in the README's scheduler table but **no source I read presents
them as a mitigation** for this defect, and whether they route around the fused kernel is not stated
anywhere I fetched.

**Unverified / limits of this note:** the `getenv("SPARKINFER_PREFILL_ATTN_GQA_RQH")` call site was
not located. `runtime/src/models/qwen35_prefill.cpp` (367,151 B) is the likely home of
`launch_attn_gqa`, but a raw fetch truncates and grep.app returned HTTP 429 (Vercel checkpoint) on
both attempts; GitHub code search requires authentication. So while the *documentation* names
`SPARKINFER_PREFILL_ATTN_GQA_RQH` in v0.5.6, v0.5.7, v0.5.9 and `main`, I did **not** verify that
current `main`'s C++ still reads that environment variable. Related observation from the caller's
own binary grep: the string is absent from
`ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10`'s `/opt/sparkinfer/bin/sparkinfer_server`; this
note does not establish why.