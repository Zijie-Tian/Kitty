# 探针解读指南（11 个）

每节：**问** = 回答什么；**入** = 输入粒度；**出** = 图与关键标量；**读** = 怎么看红蓝/判定；
**CLI** = 特有参数（公共参数 `--model/--longbench-dir/--seq-len/--group/--chunk/--tag/--outdir`
见 SKILL.md）。所有 `--model` 示例用别名 `llama32`。

## A. 基础分解

### submean_energy
- **问** K/V 每层 μ²/σ² 能量占比、恒等式是否成立、残差形状 ρ。
- **入** 多文档拼接到 `seq_len`（`LB_CONCAT_FILES` 顺序）。
- **出** per-layer 表（`share_μ / share_σ² / identity_resid / ρ`）+ overall + JSON(`--out`)。
- **读** `share_μ` 高 = 红 = 好量化；`identity_resid ~1e-7` 确认口径正确；ρ 偏离 0.798 越多
  sign 越次优。
- **CLI** `--out`（JSON 路径；此探针无 `--tag/--outdir`）。

### submean_energy_heatmap
- **问** 能量在哪些 `layer×head`、`layer×dim` 上偏红/蓝。
- **入** 单条 `≥seq_len` 文档（`LB_SINGLE_FILES`）。
- **出** `layer×head` + `layer×dim` 双热力图（红 μ² 主导 / 蓝 σ² 主导）+ PT。
- **读** `layer×dim` 图能看出 RoPE 频率结构（低 index 快频 pair 偏蓝）；逐格数字仅小模型标。

## B. 维度结构

### kvlen_energy
- **问** 能量沿 token 位置：attention-sink？σ²/μ² 占比随位置漂移？sign-NMSE 随位置？
- **出** per-token 能量(log) + μ²/σ² share vs 位置 + `σ² share per layer×position` 热力图。
- **读** token0 能量尖峰 = sink；σ² share 在 sink 后近平稳、`corr(σ² share, pos)≈0` = 残差
  占比不随 kv-len 集中。
- **注** sign-NMSE 在 flat-C `[C,ng,G]` 上 inline 算（与 `_common.submean_sign` 语义不同）。

### kvlen_channel_energy
- **问** 红/蓝身份是静态属性还是随 kv-len 漂移？σ² 选择器分段稳定吗？
- **出** `dim×position` μ² 热力图 + 「减各自均值」漂移图 + 代表通道曲线 + 选择器
  segment-vs-global / 相邻段重合。
- **读** 漂移图近零 = 身份静态 → 选择可离线化；segment-vs-global 重合高 = 静态选择即可。
- **CLI** `--segments`(默认 8)、`--promote-frac`(0.125)。

### kchannel_sigma_quant
- **问** σ² 在通道间多集中？magnitude(E|K|) vs variance(σ²) 选择器？保护 top-k 到 fp16 的
  NMSE 收益？pre vs post-RoPE σ²？
- **出** σ² 累积集中度曲线 + per-dim σ²(post，可叠 pre) + 提 fp16 后 cache NMSE(variance/
  magnitude)。
- **读** variance 选择器消误差远快于 magnitude（magnitude 在 submean 世界选错对象）；
  post/pre σ² 比值大的 dim = 旋转伪影。
- **CLI** `--capture-pre-rope`（hook k_proj 取 pre-RoPE K，Llama 系无 k_norm）。

## C. 重要性与量化方案

### qk_channel_importance
- **问** 对 attention 区分度(score-variance) 贡献大的通道是红是蓝？哪个选择器最对？
- **入** 单文档 prefill + 最后 `NQ` 个 query（q_proj hook → 补 q_norm(Qwen3) → RoPE → GQA
  按 kv-head 聚合）。
- **出** identity 轴上 energy/A/B 质量分布直方 + B vs identity 散点 + 3 选择器(q²σ²/σ²/E|K|)
  payoff 曲线。
- **读** 区分度 B 的质量落在蓝侧 → 难量化通道决定 attention；`q²σ²`/`σ²` 选择器 >> `E|K|`；
  `B_μ/B` 小 = 区分度主要走残差通路（量化承压）。
- **CLI** `--num-queries`(256)、`--promote-frac`(0.125)。

### typed_channel_quant
- **问** 红→sign / 蓝→de-RoPE 等 10 种方案的 K NMSE（分蓝/红类）+ 真实 query top-32 overlap。
- **入** 量化区 `[sink, T−recent)`；sink/recent 保 fp16；q hook 同 qk。
- **出** per-class NMSE 柱 + 整体 NMSE(标 bits) + top-32 attended-token overlap。
- **读** `typed`(1.25b 等比特) 把蓝类 NMSE 压到接近 `derope_all`；overlap 越接近 1 attention
  越保真；`ptm`(phase-tracked-mu) 保留 int/LUT 点积结构。
- **CLI** `--sink`(32)、`--recent`(128)、`--topk`(32)、`--num-queries`(256)。

## D. 泛化性

### task_channel_profile
- **问** 红/蓝画像是否任务相关（决定选择能否离线一次校准、任务通用）。
- **入** 每个 subtask 一条样本（`DEFAULT_TASKS`，10 个类型差异最大的）。
- **出** `task×dim` 画像热力图 + 跨任务选择器重合矩阵 + 各任务 overall share。
- **读** dim 画像两两 Pearson r≈1 + 重合远高于 chance(12.5%) → 身份是模型属性、任务无关。
- **CLI** `--tasks`、`--min-tokens`(2048)、`--max-tokens`(8192)、`--promote-frac`（此探针
  无 `--seq-len`）。

### calibrate_sigma_channels
- **问** 域外(wikitext)标定的 σ² 掩码能否泛化到 LongBench 21 任务。
- **出** 标定 vs 任务 dim 画像 + energy-capture(95% gate) + NMSE(calib vs oracle vs sign) +
  **可加载 PT artifact**（`masks` / `pair_masks` / `sigma2_profile`）。
- **读** energy-capture ≥ 95% = 静态掩码 ≈ 每任务自适应 → 可部署；artifact 可直接喂量化路径。
- **CLI** `--calib-parquet`(**required**)、`--calib-tokens`(32768)、`--val-tokens`(8192)、
  `--frac`(0.125)、`--frac-grid`。

### blue_channel_codebook
- **问** 蓝通道按 rot/tail 子类，post 空间 vs de-RoPE 的 6 码本(sign/tern/mm2 ×2) 谁赢。
- **出** rot/tail/red 三类 × 6 码本 NMSE 柱 + verdict 行。
- **读** rot-blue: `de-RoPE-sign(1.25b) < post-mm2(2.25b)` = 结构胜位宽；tail-blue: de-RoPE
  收益小、需 bit。
- **CLI** `--rot-ratio`(2.0, post/pre σ² 阈值)、`--sink`、`--recent`。

### postrope_blue_codebook
- **问** 纯 post-RoPE（不 de-RoPE）蓝通道，预算花在更小组 G 还是更多电平。
- **入** 蓝子类靠 RoPE 频率离线判定：fast = 周期 `2π/θ < G`（arcsine 双峰），slow = 周期 ≥ G。
- **出** fast/slow/red × {sign,tern (G128/64/32), uni2, uni3, lloyd2 (G128/64)} 的 bit-NMSE
  散点 + Pareto 前沿。
- **读** fast-blue 上 Lloyd-Max 4 电平 / 更小 G 在 Pareto 前沿，sign/tern 吃亏（双峰打在其假设
  反面）。
- **CLI** `--rope-base`(默认 `cfg.rope_theta`)、`--sink`、`--recent`。

## 两处历史口径差异（已在 `_common` 处理）

- `LB_CONCAT_FILES`（submean_energy 拼接版，第 3/4 = qmsum, musique）与 `LB_SINGLE_FILES`
  （其余单文档版，第 3/4 = musique, qmsum）顺序不同——**两份都保留**，混用会改取样、破坏
  与历史结果可比。
- `pick_single_doc` 的兜底（无 `≥seq_len` 单条时取最长）统一为返回**截断**版；样本够长的
  正常路径与各原脚本逐位一致（smoke 已验证）。
