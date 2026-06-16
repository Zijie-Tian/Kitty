# qlutattn-k1v4 per-token 探索：Channel Reorder 与 SmoothAttention

> 这是一份**研究设计文档**，记录把 qlutattn-k1v4 的 K 量化从 **per-channel** 推向
> **per-token** 的三条探索：per-token 量化路径、QServe 风格 SmoothAttention、以及
> offline channel reorder。全部跑在纯 torch `kitty_sim` fake-quant 路径上，**只是精
> 度代理，不省显存也不提速**（真 Triton kernel 只支持 2/4-bit packing，per-channel
> 轴）。设计动机、实现、已测结果、以及在另一台机器上继续研究的完整复现命令都在这里。
>
> 相关文档：per-channel σ²-binned 混合码本本体见 [`qlutattn_k1v4.md`](qlutattn_k1v4.md)，
> 全量测试结果见 [`qlutattn_k1v4_testing.md`](qlutattn_k1v4_testing.md)。

---

## 0. TL;DR

| 维度 | per-channel（现状，qlutattn-k1v4 本体） | per-token（本文探索） |
| --- | --- | --- |
| K 分组轴 | 沿 **token** 轴分组（每 channel 独立 scale，KIVI K 风格） | 沿 **head_dim** 轴分组（每 token 一个 scale，KIVI V 风格） |
| 混合码本 | 按 channel 的残差 σ² 给不同码本（sign/tern/nf2） | 当前是**单码本**（全 head_dim 同码本） |
| 硬件 | decode 需把新 token 转置进 channel 分组 | decode 时新 token 直接量化，无转置（友好） |
| side info | group=128 → 0.25 bit/value | group=head_dim=64 → 0.5 bit/value |

三条探索的核心结论：

1. **per-token 的精度损失来自码本，不是轴向。** per-token uniform 2-bit 崩到 12.95，
   但 per-token **nf2**（Lloyd 自适应码本）回到 23.50 ≈ per-channel KIVI-2（24.24）。
2. **SmoothAttention 大幅救 uniform（+3.66），但对 nf2 几乎无用（+0.58）。** 因为
   Lloyd 码本已自适应吸收 outlier，smooth 的压平对它是冗余的。
3. **Channel reorder 是一个恒等预处理**（fp16 logits top-1 一致率 99.76%），把同能量
   channel 排到 head_dim 上连续，为「一个 token 内对连续段用不同码本」铺路——这是把
   per-channel σ²-mix 的 1.68-bit 优势搬到 per-token 轴的前提。

研究终态目标：**reorder + per-token 分段混码本 + smooth**，在硬件友好的 per-token
轴上复现 per-channel σ²-mix 的 bit/精度优势。reorder/smooth 两者正交，可叠加。

---

## 1. 背景：为什么要 per-token

qlutattn-k1v4（见 `qlutattn_k1v4.md`）当前是 **per-channel** K 量化：把 K 转置成
`[B, nh, D, T]`，沿 **token 轴**按 `group_size=128` 分组，每个 channel 独立地按其残差
σ² 选码本（低 σ²→`sign`、中→`tern`、高→`nf2`）。这套在精度上很好（1B 全量 24.88，
K≈1.68 bit），但有两个工程上的别扭：

- **decode 不友好**：自回归 decode 每步来一个新 token，per-channel 分组要把这个 token
  的 D 个值分别塞进 D 个 channel 的 token 组里——本质是个转置/scatter，和 KIVI 把
  V 按 per-token 量化的简洁路径相反。
- **混码本绑定在 channel 上**：σ² 是 per-channel 属性，码本选择也按 channel。这跟
  per-token 分组（一个 token 的 D 个值共享）天然冲突。

**per-token**（KIVI 的 V cache 路线）则是把 K 当作 `[B, nh, T, D]`，沿 **head_dim 轴**
分组，每个 token 一个 scale。decode 友好（新 token 直接量化），但一个 scale 要覆盖
整个 head_dim——**channel 间的 outlier 会主导这个共享 scale**，这正是 SmoothAttention
和 reorder 要解决的。

---

## 2. per-token K 量化路径

### 2.1 实现（`src/kitty_sim/kitty_simulate.py`）

`KittyKVCacheConfig` 新增 `k_quant_mode`：

- `"per_channel"`（默认）：原 KIVI-style token 轴分组 + promote/qlut 码本。
- `"per_token"`：K 像 V 一样沿 head_dim 分组，**uniform、无 promote、无 channel
  selection**。校验强制 `promote_ratio=0.0` 且无 per-layer 覆盖（否则 raise）。

核心方法 `_quant_k_pertoken(ks)`：对 `[B, nh, T, D]` 的 K，按 head_dim 当作一个 submean
分组（一个 token 一个 group），走 qlut 码本或 uniform group-wise。prefill 与 decode 两
条 update 分支都改为在 `per_token` 模式下调用它。sink（前 `sink_length=32` token）仍保
持 fp16。

### 2.2 per-token variant（`src/kitty_sim/longbench/runner.py`）

| variant | method slug | tag 后缀 | 配置 |
| --- | --- | --- | --- |
| `qlutattn_pertoken`（别名 `qlut_pertoken`） | `qlutattn-pertoken` | `_kpt` | qlut 单码本：`k_codebook=qlut n_bins=1`，码本由 `QLUT_BIN_CODEBOOKS` 给（默认 `nf2`），V per-token 4-bit |

> uniform-codebook per-token（旧 `kitty_pertoken`，已移除专门 variant）可用
> `--variant custom --k_quant_mode per_token`（`kbits=2 vbits=4 promote_ratio=0`）复现。

> per-token 是单码本：`QLUT_BIN_CODEBOOKS` 只取一个码本名（默认 `nf2`）。要扫 `sign`/
> `tern`/`nf2`，改这个 env 即可。

### 2.3 结果（Llama-3.2-1B，全量 21 数据集，32k）

| K 量化 | 码本 | K bit/value | 均分 | +smooth |
| --- | --- | ---: | ---: | ---: |
| per-token | uniform 2-bit | 2.5 | **12.95** | 16.61 |
| per-token | qlut **nf2**（Lloyd） | 2.5 | **23.50** | 24.08 |
| per-channel（参考） | KIVI-2 | 2.25 | 24.24 | — |
| per-channel（参考） | qlut σ²-mix | 1.68 | 24.88 | — |
| per-channel（参考） | kitty 2-bit+4-bit promote | 2.5 | 26.25 | — |
| fp16 上限 | — | 16 | 27.59 | — |

> **关键发现**：per-token nf2（23.50）几乎追平 per-channel KIVI（24.24），而 per-token
> uniform（12.95）崩溃。**损失是码本造成的，不是 per-token 轴造成的**。nf2 的 Lloyd-Max
> 码本对每个 token 的 head_dim 分布自适应，能在一个共享 scale 下吃下 channel 间的动态范
> 围差异；uniform 量化做不到。
>
> per-token bit/value = codeword 2-bit + side 0.5（μ+scale 两个 fp16 / head_dim=64）= 2.5。
> 比 per-channel 多 0.25 bit side（group 更小：64 vs 128），换 decode 友好。

---

## 3. SmoothAttention（QServe 风格）

### 3.1 动机与原理

post-RoPE K 有显著的 **per-channel outlier**。per-token 量化让一个 scale 覆盖整个
head_dim，于是被最大的 channel 主导，其余 channel 的有效比特被浪费。QServe
（arXiv 2405.04532）的 SmoothAttention 把这个 outlier 从 K «搬» 到 Q：

```
lam[h, i] = lam[h, i + D/2] = max(absmax_K[h, i], absmax_K[h, i + D/2]) ** alpha   (alpha=0.5)
W_q  <-  lam_q * W_q      # 输出通道维，GQA 在 query 头上 broadcast
W_k  <-  W_k / lam_k      # 输出通道维
```

`Q·K^T = (Qλ)·(K/λ)^T`，在精确算术下**完全不变**；但 `K/λ` 的 post-RoPE channel 被压
平向几何均值，正是 per-token 低比特 K 想要的。

### 3.2 RoPE pair 约束（关键）

HF 的 rotate-half 把 channel `i` 与 `i+D/2` 配对共享频率。要让 `diag(λ)` 与旋转**交换**
（从而能折进 pre-RoPE 的投影权重），必须 `λ_i == λ_{i+D/2}`——所以 λ 是 pair 取 max 后
开 alpha 次方，再复制成两半。

### 3.3 限制：q_norm/k_norm 模型

如果模型在投影后有 per-head `q_norm`/`k_norm`（如 **Qwen3**），折进 `W_k` 的 per-channel
scale 会被 norm 重新归一化掉，破坏等价。`fold_scales` 检测到就 raise。**SmoothAttention
只支持 Llama 类（无 per-head QK norm）。**

### 3.4 标定（`scripts/calibrate_smooth_qk.py`）

在 calib 语料上收集 post-RoPE K 的 per-(kv_head, channel) absmax → 算 λ → 折进
`W_q`/`W_k` → `save_pretrained` 出一个 smoothed checkpoint（`<T>_MODEL_PATH` 可直接用）。
附带 `smooth_scales.pt`、`calib_meta.json`（含每层 flatness、k2 NMSE 改善代理、fp16 等价）。

### 3.5 结果

| per-token 码本 | 原始 | +smooth | Δ |
| --- | ---: | ---: | ---: |
| uniform 2-bit | 12.95 | 16.61 | **+3.66** |
| qlut nf2 | 23.50 | 24.08 | +0.58 |

smooth 大幅救 uniform（被 outlier 主导），对 nf2 增益很小（Lloyd 已自适应吸收 outlier，
smooth 的压平对它基本冗余）。

---

## 4. Channel Reorder（route a）

### 4.1 动机

qlutattn-k1v4 的灵魂是**按能量（σ²）给 channel 分配不同码本**。要把这套搬到 per-token
轴，就需要「在一个 token 内部，对不同的 channel 段用不同码本」。但 per-token 是沿
head_dim 连续分组的——**只有当同能量 channel 在 head_dim 上连续，分段混码本才可行**。

route a：**offline 把 Q/K_proj 的输出通道按 σ² 重排**，让同能量 channel 连续。这是一个
恒等变换（重排 channel 顺序不改变 attention，只要 RoPE 同步重排）。

### 4.2 设计（`scripts/preprocess_qlutattn_model.py`）

1. 在 calib 语料上收集 post-RoPE K 的 per-channel σ²。
2. **RoPE pair 约束**：pair `(i, i+D/2)` 必须一起移动（排列与旋转可交换）。每个 pair 按
   `pair_sigma2 = max(σ²[i], σ²[i+D/2])` 打分。
3. **全局单一排列**：HF `rotary_emb` 是 **model-level**（所有层共享一份 `inv_freq`），
   所以 pair 排序对**所有 (层, kv-head)** 取平均后得到**一个全局 pair 排列**（低 σ² 在前），
   `channel_perm = [pair_perm, pair_perm + D/2]`。
4. **折叠**：把排列折进**每一层**的 `W_q`/`W_k` 输出通道（pair 为单位），并把 RoPE
   `inv_freq` 重排成 `inv_freq[pair_perm]` 匹配。
5. 保存 reorder 后的 checkpoint + `reorder_qk.pt {pair_perm, channel_perm, head_dim}`
   + `config.qk_reorder=True` 标记 + `reorder_meta.json`。

### 4.3 加载（`src/kitty_sim/qk_reorder.py`）

HF `LlamaRotaryEmbedding.inv_freq` 是 `persistent=False`——**不随 checkpoint 保存**。所以
`W_q`/`W_k` 的排列折进了权重，但 `inv_freq` 要在加载后由 `apply_qk_reorder(model, path)`
重排（检测到 `reorder_qk.pt` 就 patch `model.model.rotary_emb.inv_freq = inv_freq[pair_perm]`，
非 reorder checkpoint 则 no-op）。已接入 `runner.py:load_model_and_tokenizer`。

> **限制**：model-level `inv_freq` 强制**全局** perm。各层的 σ² 模式不同，全局序是各层
> 的折中——对 per-channel 量化无害（恒等），但会限制后续 per-token 分段的「段纯度」。
> 这是 route a 的固有取舍，留给第 5 节的研究路线。

### 4.4 恒等验证

| 检查 | 结果 |
| --- | ---: |
| fp16 logits top-1 一致率 | **99.76%** |
| mean abs logit diff | 0.0034 |
| max abs logit diff | 0.074 |
| loader `inv_freq == inv_freq[pair_perm]` | ✓ |

残差纯来自 fp16 求和顺序 + bf16→fp16 舍入。

**LongBench 复现**（reorder checkpoint + qlutattn-k1v4 **per-channel**，全量 21 数据集，32k）：

| qlutattn-k1v4 per-channel | 均分 | per-dataset \|Δ\| |
| --- | ---: | ---: |
| 非 reorder | 24.88 | — |
| **reorder** | **24.82** | max 1.28 / mean 0.31 |

均分 Δ = **−0.06**，逐项差异随机有正有负（最大 musique −1.28，是单数据集小样本方差，
非单点崩溃）。**route-a reorder 对 LongBench 恒等。** 残差来自 fp16 求和顺序、bf16→fp16
舍入、以及 qlut Lloyd 码本对输入顺序的轻微数值敏感性，全在采样方差内。

---

## 5. 研究路线图（在另一台机器上继续）

终态目标：**在 per-token 轴上复现 per-channel σ²-mix 的 bit/精度优势**。已铺好的三块拼图：

- **per-token 路径**（§2）：能在 head_dim 轴量化，decode 友好；nf2 已证明 per-token 轴
  本身不掉精度。
- **SmoothAttention**（§3）：压平 K outlier，让任意 per-token 码本受益（uniform 尤甚）。
- **Channel reorder**（§4）：把同能量 channel 排连续，使「段内混码本」可行。

待做（按优先级）：

1. **per-token 分段混码本**：在 reorder 后的模型上，把 head_dim 切成几个连续段（低 σ²
   段用 `sign`/`tern`，高 σ² 段用 `nf2`），复现 per-channel σ²-mix 的 ~1.68 bit。需要
   在 `_quant_k_pertoken` 里加「按段选码本」（段边界来自 `reorder_qk.pt` 的全局 σ² 序）。
2. **smooth + reorder + 分段联合**：两者正交，smooth 先压平再分段量化，预期对 uniform
   段尤其有效。注意 smooth 和 reorder 都改 `W_q`/`W_k`，要确认折叠顺序可交换（都是
   per-channel 对角缩放/置换，应可叠加，但需 fp16 等价复验）。
3. **段纯度 / per-layer**：评估全局 perm 下各层的「段内 σ² 一致性」。若折中太大，考虑
   per-layer perm（但 model-level `inv_freq` 不允许——需要把 RoPE 改成 per-layer，或
   接受按段重新分配码本的损失）。
4. **码本/段边界搜索**：用离线 overlap 代理（见 `qlutattn_k1v4.md` §5b 的
   `build_kq_cache.py` + `eval_qlut_policy.py`）快速搜段数、段边界、每段码本，再上 LongBench。

---

## 6. 使用方法（完整命令）

所有命令默认 GPU0、Llama-3.2-1B、本机 calib 语料 `wikitext-2-raw-v1`。在另一台机器上把
`--model` / `LLAMA32_MODEL_PATH` / `--calib-data` 换成本地路径即可。

### 6.1 生成 reorder 模型（offline，单卡 ~5min）

```bash
cd <repo>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/preprocess_qlutattn_model.py \
  --model /path/to/Llama-3.2-1B-Instruct \
  --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --output /path/to/models/Llama-3.2-1B-Instruct-reorder
# 产出: reorder 后的 checkpoint + reorder_qk.pt + reorder_meta.json (含 fp16 等价检查)
# 不传 --output 时默认写到 repo 的 reorder/<model basename>/（已 gitignore）。
```

### 6.2 跑 reorder + qlutattn-k1v4（per-channel，验证恒等）

把 `LLAMA32_MODEL_PATH` 指向 reorder checkpoint；用 `LLAMA32_MODEL_SLUG` 区分输出目录。
加载时日志会打印 `[qk-reorder] applied RoPE inv_freq permutation`。

```bash
# smoke（2 样本/数据集，长上下文数据集，确保 K 路径被触发）
cd <repo>
LLAMA32_MODEL_PATH=/path/to/models/Llama-3.2-1B-Instruct-reorder \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-reorder \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 0 --variant qlutattn_k1v4 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct-reorder_qlutattn-k1v4/{pred,logs}
```

```bash
# 全量（21 数据集，32k，6 卡×3=18 worker；显式覆盖 GPU1-only 规则）
cd <repo>
LLAMA32_MODEL_PATH=/path/to/models/Llama-3.2-1B-Instruct-reorder \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-reorder \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 --variant qlutattn_k1v4
# -> longbench_out/llama32-1b-instruct-reorder_qlutattn-k1v4/{pred,logs}
```

### 6.3 标定 SmoothAttention（offline，单卡 ~5min；仅 Llama 类）

```bash
cd <repo>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_smooth_qk.py \
  --model /path/to/Llama-3.2-1B-Instruct \
  --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --alpha 0.5 \
  --output /path/to/models/Llama-3.2-1B-Instruct-smooth
# 产出: smoothed checkpoint + smooth_scales.pt + calib_meta.json
# 不传 --output 时默认写到 repo 的 calib/<model basename>/（已 gitignore）。
```

### 6.4 跑 per-token variant（±smooth）

`qlutattn_pertoken`（qlut，`QLUT_BIN_CODEBOOKS` 选码本，默认
nf2）。`+smooth` 就是把 `LLAMA32_MODEL_PATH` 指向 smoothed checkpoint。

```bash
cd <repo>
# per-token qlut-nf2，原始模型（smoke）
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
QLUT_BIN_CODEBOOKS=nf2 MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 0 --variant qlutattn_pertoken --max-samples 2

# per-token qlut-nf2 + smooth（全量）：模型换成 smoothed checkpoint，slug 加 -smooth
LLAMA32_MODEL_PATH=/path/to/models/Llama-3.2-1B-Instruct-smooth \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-smooth \
QLUT_BIN_CODEBOOKS=nf2 MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 --variant qlutattn_pertoken
# uniform-codebook 版：--variant custom --k_quant_mode per_token（无需 QLUT_BIN_CODEBOOKS）。
```

### 6.5 画图

```bash
PYTHONPATH=src python scripts/plot_pertoken_smooth.py
# -> longbench_out/pertoken_smooth_llama32_1b.png（per-token uniform/nf2 ±smooth vs per-channel）
```

---

## 7. 代码清单

| 文件 | 作用 |
| --- | --- |
| `scripts/preprocess_qlutattn_model.py` | offline reorder：收 σ² → 全局 pair 排列 → 折进 W_q/W_k + 重排 inv_freq，存 reorder checkpoint |
| `src/kitty_sim/qk_reorder.py` | loader patch：加载 reorder checkpoint 时重排 `rotary_emb.inv_freq`（已接入 `runner.py`） |
| `scripts/calibrate_smooth_qk.py` | offline SmoothAttention 标定：收 K absmax → λ → 折进 W_q/W_k，存 smoothed checkpoint（仅 Llama 类） |
| `src/kitty_sim/kitty_simulate.py` | `k_quant_mode={per_channel,per_token}` + `_quant_k_pertoken` |
| `src/kitty_sim/longbench/runner.py` | variant `qlutattn_pertoken`；reorder loader 接入 |
| `src/kitty_sim/cli/{utils_cli,eval_longbench}.py` | `--k_quant_mode` CLI + variant choices |
| `scripts/plot_pertoken_smooth.py` | per-token ±smooth vs per-channel 对比图 |
| `tests/test_kitty_pertoken_smooth.py` | per-token / smooth 单元测试 |

> reorder / smooth 产出的大 checkpoint 目录（`reorder/`、`calib/`）已 gitignore；推荐用
> `--output` 存到 `~/models` 等模型目录，和原始模型并列。
