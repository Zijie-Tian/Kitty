# 已有实测结论（跑前预期 / 回归对照）

主要来自 **Llama-3.2-1B、post-RoPE、G=128、真实 LongBench**。这些是**环境相关的 smoke
证据**（单模型、单/少样本），不是正式 benchmark；换模型 / seq-len / 样本会变。用作跑前的
量级预期与重构回归对照。

## 能量占比（submean_energy / heatmap）

| 量 | 值 | 来源 |
| --- | --- | --- |
| K 的 μ² 占比 | ≈ 64.8%（红偏多）；跨层 65–82% | 本族 smoke 64.68% / kitty_sign 笔记 |
| V 的 μ² 占比 | ≈ 15%（蓝偏多）；跨层 16–21% | 本族 smoke 15.39% / 笔记 |
| 恒等式残差 `|Σμ²+Σσ²−Σex²|/Σex²` | ~1e-7（口径自检） | population 精确 |
| 残差形状 ρ = E|r|/σ（pooled） | ≈ 0.738（偏重尾，介于高斯 0.798 与拉普拉斯 0.707） | submean_energy |

含义：K 适合 submean（红多、残差小）；V 残差主导、量化更难（故 Kitty V 走 per-token）。

## 沿 kv-len（kvlen_energy）

- token0 能量远高于 median（attention-sink）；前 32 token 持有不成比例的 K 能量。
- σ² 占比在 sink 之后**基本平稳**，`corr(σ² share, kv position) ≈ 0` → 残差占比不随位置集中。
- 含义：sink 区特殊（Kitty 保 `sink=32 + recent=128` fp16），其余沿 kv-len 同质，可统一量化。

## 红/蓝身份的静态性（kvlen_channel / task / calibrate）

- 沿 kv-len 漂移小：真趋势性漂移（`|corr(pos)|>0.5 且 range>0.10`）的通道数很少 → 身份近静态。
- 选择器分段稳定：segment-vs-global、相邻段 σ² top-k 重合高。
- 跨任务一致：dim 画像两两 **Pearson r ≥ 0.99**；选择器跨任务重合 **~72–85%**（chance 12.5%）。
- 域外泛化：wikitext 标定掩码在 21 个 LongBench 任务上 **energy-capture ≥ 95%**（部署判据）。
- 含义：σ² 通道选择可**离线标定一次、任务通用**，不必每样本在线重选。

## 选择器对错（kchannel / qk）

- magnitude(E|K|) 与 variance(σ²) 的 top-k 重合**低** → magnitude 在 submean 世界**选错对象**
  （挑了残差小的红通道）。
- variance / `q²σ²` 选择器消 cache-NMSE / score-噪声**远快于** E|K|。
- attention 区分度 B 主要走**残差通路**（`B_μ/B` 小）→「难量化的蓝通道恰决定 attention 选择」。

## RoPE 与蓝通道码本（kchannel / typed / blue / postrope）

- 蓝通道（低 index 快频 RoPE 对）post/pre σ² 比值 **6–8.5×** → σ² 多为旋转伪影。
- de-RoPE 把蓝类 NMSE 降约 **6–8×**；`typed`（红→sign、蓝→de-RoPE sign）在**等比特 1.25b**
  下蓝类接近 `derope_all`、整体显著优于全 post-sign。
- **rot-blue**：`de-RoPE-sign(1.25b) < post-mm2(2.25b)`（结构胜位宽）。
- **tail-blue**：de-RoPE 收益小、需靠 bit（tern/mm2）——真重尾的"第三类"通道。
- **fast-blue**（周期 < G，arcsine 双峰）：Lloyd-Max 4 电平 / 更小组 G 在 Pareto 前沿；
  sign/tern 吃亏（双峰打在其单峰假设反面）。

## 复跑回归对照

- 相同 `--model --seq-len --group` 下，重构脚本与历史 `scripts/` 旧探针输出应一致：
  - `submean_energy` overall `share_μ/share_σ²` 逐位一致；
  - `submean_energy_heatmap` per-element μ² 差 = 0；
  - `typed` / `kchannel` 全数值行逐位一致（仅一个装饰性空行差异）。
- 公共库改动后跑 `python -m unittest discover -s tests` 须全绿（18 项原语等价）。
- 更大模型（3B / 8B）对低比特 K 更鲁棒，红/蓝结论方向不变、阈值更宽（见项目 CLAUDE.md
  低比特 K 研究）。
