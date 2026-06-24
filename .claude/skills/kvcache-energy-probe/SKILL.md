---
name: kvcache-energy-probe
description: >-
  在真实 LongBench 文本上对 LLM KV-cache 做 post-RoPE 减均值能量分解的分析探针族
  （per-channel μ²/σ² 能量占比、红/蓝通道画像、沿 kv-len 分布、QK 重要性归因、σ² 通道
  离线标定、typed/de-RoPE/Lloyd 低比特 K 码本对决）。当需要诊断「哪些 K/V 通道难量化、
  红/蓝身份是否任务无关、attention-sink 能量集中、选择器跨段/跨任务是否稳定、1-bit sign
  误差从何而来、蓝通道低比特该用什么码本」时使用。也用于把 KV-cache 量化方案（Kitty/KIVI/
  ShadowKV 式）的设计依据用能量证据说清楚。Triggers: KV cache energy analysis, mu/sigma
  decomposition, channel importance, quantization probe, energy heatmap, sigma channel
  calibration, 减均值能量分解, 红蓝通道, KV cache 量化分析.
---

# kvcache-energy-probe

一组围绕**同一个恒等式**的 KV-cache 能量诊断探针。回答「KV-cache 的能量长什么样、哪里
难量化、为什么、换种量化方式值不值」这类问题，用真实 LongBench 长文本上的实测能量证据，
而不是直觉。所有探针共享一个公共库 `scripts/_common.py`，新分析视角应在其上扩展，**不要再
往项目 `scripts/` 散落脚本**。

## 核心概念（先读这一段）

在 **per-channel 取向**（固定 `(head, dim)`、沿 token 轴）、按 `G=128` 连续分组、
**population** 统计下，恒等式精确成立：

```
E[x²] = μ² + σ²        (μ = 组均值, σ² = 组内残差方差; ddof=0 下逐位成立)
```

- **红通道** = `μ²` 主导：能量在组均值里。submean+sign 码本把 μ 用 fp16 精确存，残差小 →
  **天生好量化**。
- **蓝通道** = `σ²` 主导：能量在组内残差里 → sign 误差大 → **难量化**，是低比特 K 的痛点。

实测 Llama-3.2-1B post-RoPE：K 的 μ² 占 ~65%（红偏多）、V 的 μ² 仅 ~15%（蓝偏多）。这条
红/蓝轴是 Kitty 低比特 K 量化、通道选择、de-RoPE 等设计的统一支点。详见
`references/theory-identity.md`。

## 何时用

- 想知道某模型/某任务的 K 或 V cache **能量怎么分布**（per-layer / per-head / per-dim /
  沿 kv-len）。
- 设计或论证 **KV 量化策略**：哪些通道该升精度、magnitude 选择器选错没有、保护 top-k 到
  fp16 的收益上界、红蓝身份能否离线标定一次通用。
- 诊断 **1-bit sign K 为什么塌**、蓝通道 σ² 是 RoPE 伪影还是真重尾、de-RoPE / 更小组 /
  Lloyd-Max 哪个划算。
- 给 attention-sink、massive-norm token、选择器稳定性等现象拿**能量数字**。

## 运行前提

- **conda env**：`kitty`（文档化的栈；`python -m unittest` 跑测试，无 pytest）。
- **GPU1-only 规则**：命令前缀 `CUDA_VISIBLE_DEVICES=1`；代码内固定 `cuda:0`（= 可见的第
  一张卡）。
- **模型路径**：`--model` 支持别名 `llama32 / llama31 / qwen(=qwen3)`，经仓库 `.env` 的
  `KITTY_LLAMA32_1B_PATH / KITTY_LLAMA31_8B_PATH / KITTY_QWEN3_8B_PATH` 解析（注意 `.env`
  默认**没有** `KITTY_LLAMA32_1B_PATH`，跑 llama32 前需补这行，或直接 `--model /abs/path`）。
  也可直接传本地路径或 HF id。
- **LongBench**：`--longbench-dir` 缺省取 `.env` 的 `LONGBENCH_DATA_ROOT/data`。
- `.env` 不被 Python 自动加载 —— 直接跑脚本时，要么 shell 里 `set -a; source .env`，要么用
  别名/显式路径（脚本内部 `os.environ.get` 仅读已导出的变量）。

## 探针速查表

每个探针 = `scripts/<name>.py`，输出图/JSON/PT 到 `--outdir`（默认 `probe_out/`）。逐个
解读见 `references/probe-guide.md`。

| 探针 | 回答的问题 | 关键产物 |
| --- | --- | --- |
| `submean_energy` | K/V per-layer μ²/σ² 占比、恒等式残差、ρ=E|r|/σ | 占比表 + JSON |
| `submean_energy_heatmap` | (layer×head)、(layer×dim) 的 μ² 占比热力图 | 双热力图 + PT |
| `kvlen_energy` | 能量沿 token 位置：attention-sink、μ²/σ² vs 位置、sign-NMSE vs 位置 | 3 子图 |
| `kvlen_channel_energy` | 红/蓝身份是静态还是随 kv-len 漂移；选择器分段稳定性 | 3 子图 |
| `kchannel_sigma_quant` | σ² 集中度、magnitude vs variance 选择器、保护 top-k 收益、pre/post-RoPE σ² | 3 子图 |
| `qk_channel_importance` | 对 attention 区分度(score-variance) 贡献大的通道是红是蓝；3 选择器收益 | 3 子图 |
| `typed_channel_quant` | 红→sign / 蓝→de-RoPE 的 10 种方案 NMSE + 真实 query top-32 overlap | 3 子图 |
| `task_channel_profile` | 红/蓝画像是否任务相关（dim 两两 Pearson + 选择器跨任务重合） | 3 子图 |
| `calibrate_sigma_channels` | 域外(wikitext)标定 σ² 掩码 → 21 任务泛化（energy-capture≥95% 判据） | 3 子图 + 可加载 PT artifact |
| `blue_channel_codebook` | 蓝通道按 rot/tail 子类，6 码本(post vs de-RoPE) NMSE 对决 | 3 子图 |
| `postrope_blue_codebook` | 纯 post-RoPE 蓝通道按 fast/slow，sign/tern/uni/Lloyd 的 bit-NMSE Pareto | 3 子图 |

## 典型命令

以下命令从 **Kitty 仓库根目录**运行（直接给脚本相对路径即可；脚本会自动找到同目录的
`_common.py`，无需 `cd` 或设置 `PYTHONPATH`）。

烟雾测试（小 seq-len，秒级；确认环境/路径通）：

```bash
CUDA_VISIBLE_DEVICES=1 python .claude/skills/kvcache-energy-probe/scripts/submean_energy_heatmap.py \
  --model llama32 --seq-len 2048 --group 128 --tag llama32-1b --outdir /tmp/probe_smoke
```

完整运行（32k 单文档，论文级证据）：

```bash
CUDA_VISIBLE_DEVICES=1 python .claude/skills/kvcache-energy-probe/scripts/submean_energy_heatmap.py \
  --model llama32 --seq-len 32768 --group 128 --tag llama32-1b --outdir probe_out
```

离线标定（需 wikitext parquet 作域外语料）：

```bash
CUDA_VISIBLE_DEVICES=1 python .claude/skills/kvcache-energy-probe/scripts/calibrate_sigma_channels.py \
  --model llama32 --calib-parquet /path/to/wikitext/train-00000-of-00001.parquet \
  --calib-tokens 32768 --tag llama32-1b --outdir probe_out
```

> 跑别的模型：把 `--model` 换成 `llama31 / qwen` 别名，或直接传本地路径 / HF id。所有探针
> 架构无关（GQA 自动按 `num_key_value_heads` 处理；q_norm 自动适配 Qwen3）。

## 公共库与扩展

`scripts/_common.py` 提供 11 组复用原语（取样 / 模型与分段 prefill / DynamicCache 新旧布局
兼容 / 能量分解 `group_decompose→GroupStats(xg,mu,ex2,var)` / RoPE 表与 q·k hook / 量化重建
`submean_sign·submean_codebook·lloyd_codebook` / 选择器 / 路径解析 / 画图）。

**加新探针**：复制任一薄脚本作模板，`import _common as c`，用 `c.resolve_model_path` /
`c.resolve_longbench_dir` / `c.load_model_and_tok` / `c.pick_single_doc` / `c.prefill` /
`c.group_decompose` 拼骨架，只写你这一视角特有的统计与图。

**数值等价红线**（改 `_common.py` 时必须守，否则破坏与历史结果的可比性）：
1. `group_decompose` permute 后**立刻 `.float()`** 再统计；
2. 需要一次性 `mean(dim=(-1,-2))` 形态的（如 `calibrate` 的 `prefill_sigma2`）直接对
   `GroupStats.xg` 本地算，别走链式 `mean(-1).mean(-1)`（fp 顺序会差 ~1e-7）；
3. `GroupStats.var` **不 clamp**，`clamp_min(0)` 由调用方按原脚本逐处复刻；
4. flat-C + sign 重建（`kvlen_energy` / `kchannel_sigma`）保留**本地** inline，不复用
   `submean_sign`（它在 [H,D,T] 上算且回填末尾，语义不同）；
5. 原语不擅自搬设备，`.cpu()` 时机由脚本复刻。

**测试**：`CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -v`（纯 CPU，免模型）
对每个原语与「原脚本内联旧实现」逐元素比对；改库后必须保持全绿。

## references

- `references/theory-identity.md` —— 恒等式推导、ρ 的几何含义、softmax 平移不变性下的
  A/B/N 三类贡献、RoPE 旋转等距 ⇒ de-RoPE 误差守恒、pair-coupled 身份定义。
- `references/probe-guide.md` —— 11 个探针逐个：输入粒度、输出图与标量、怎么读红蓝、别名 CLI。
- `references/findings.md` —— 已有实测结论表（μ² 占比、跨任务 r、energy-capture 判据、
  rot/tail/fast/slow 子类等），跑前的预期与回归对照。
