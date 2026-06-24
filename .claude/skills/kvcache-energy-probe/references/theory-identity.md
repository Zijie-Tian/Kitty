# 理论内核：减均值能量分解恒等式及其推论

本族探针全部围绕一个恒等式与它的几条推论。这里给出公式，供实现/审阅对照；具体数字见
`findings.md`，每个探针怎么用见 `probe-guide.md`。

## 1. 减均值能量分解 `E[x²] = μ² + σ²`

固定一个 channel `(head, dim)`，沿 token 轴取一组（`G=128` 个连续 token）。记组内元素
`x_t`，组均值 `μ = E_t[x]`，残差 `r_t = x_t − μ`（按定义 `E_t[r] = 0`）：

```
E[x²] = E[(μ + r)²] = μ² + 2μ·E[r] + E[r²] = μ² + σ²
```

其中 `σ² = E[r²]` 是 **population**（ddof=0）方差。因为 `E[r] = 0` 在 population 口径下
**精确**成立，恒等式逐位成立（实测残差 `|Σμ²+Σσ²−Σex²|/Σex² ~ 1e-7`，纯浮点舍入）。

- 用 **样本方差**（ddof=1）会破坏恒等式：`E[r]` 不再恰为 0。本族一律 population。
- 量化含义：submean+sign/tern 码本把 `μ` 用 fp16 **精确存**，量化误差只落在残差 `r` 上、
  正比于 `σ²`。所以一个 channel 好不好量化，由它的 **σ² 占比** `σ²/E[x²]` 决定：
  - `μ²` 占比大 = **红通道** → 残差小 → sign 误差小（好量化）。
  - `σ²` 占比大 = **蓝通道** → 残差大 → sign 误差大（难量化）。

`share_μ² + share_σ² = 1`，互补，故一张以 0.5 为中心的发散热力图（红=μ²主导, 蓝=σ²主导）
即可同时表达两者。

## 2. 残差形状 `ρ = E|r| / √(E[r²]) = E|r|/σ`

`σ²` 只说残差有多大，`ρ` 说残差长什么形状（决定 1-bit sign 是否最优）：

| 分布 | ρ |
| --- | --- |
| 均匀 | √3/2 ≈ 0.866 |
| 高斯 | √(2/π) ≈ 0.798 |
| 实测 K post-RoPE（pooled） | ≈ 0.738 |
| 拉普拉斯 | 1/√2 ≈ 0.707 |

实测 `ρ≈0.738` 介于高斯与拉普拉斯之间、偏重尾。sign 的电平取 `E|r|`（1-bit Lloyd–Max 在
对称单峰假设下的最优电平）；ρ 偏离高斯越多，sign 的次优空间越大（tern / Lloyd 才有收益）。

## 3. QK 三类贡献：softmax 平移不变性下「红通道的能量不决定 attention」

`score_t = Σ_d q_d·k_{t,d}`，把 `k_{t,d} = μ_d + r_{t,d}` 代入。softmax 对 score **加常数
不变**，而 `q_d·μ_d` 在一个组内对所有 t 是同一常数 → **不改变 softmax 分布**。于是把通道对
attention 的作用拆成三类（`qk_channel_importance` 实测）：

- **幅度** `A_d = E|q_d|·E|k_d|`：把 `|score|` 顶多高。红通道（μ 大）天然赢，但只影响绝对
  尺度，不影响 softmax 选谁。
- **区分度** `B_d = rowsum(C ∘ M)_d`，`C = Cov_t(k)`（可拆 `C = C_μ`(组均值漂移) `+ C_res`
  (组内残差)），`M = E_q[q qᵀ]`。`Σ_d B_d = E_q[Var_t(score)]`，是「谁决定 attention 选哪个
  token」的**精确**协方差归因。
- **sign 噪声注入** `N_d = E[q_d²]·σ²_{within,d}`：sign 误差近 per-channel 独立，score 噪声
  方差 `= Σ_d q_d²·MSE_d ∝ N_d`。按 `N_d` 排序即 score-噪声意义下的 oracle 选择器。

**关键张力**：区分度 `B` 主要来自残差通路（`C_res`，量化承压），而非组均值漂移（`C_μ`，
fp16 无损）。即「难量化的蓝通道恰恰是决定 attention 选择的通道」——这是低比特 K 的根本难点，
也是「按 σ²/N 选通道升精度」比「按 E|K| 选」更对的理由。

## 4. RoPE 旋转等距 ⇒ de-RoPE 误差守恒

RoPE 对每个 pair `(i, i+D/2)` 施加 2×2 旋转 `R(θ_t)`（θ 由 token 位置唯一决定），是**正交
等距**：`‖R x‖ = ‖x‖`。于是 de-RoPE 量化：

```
k_pre  = R(−θ_t) k_post           # 逆旋转到 pre-RoPE 空间
rec_pre = quantize(k_pre)          # 在 pre 空间量化
rec_post = R(θ_t) rec_pre          # 旋回
‖rec_post − k_post‖ = ‖R(θ_t)(rec_pre − k_pre)‖ = ‖rec_pre − k_pre‖
```

post 空间误差 **恒等于** pre 空间误差，且不花额外 bit（θ 解析可重放）。蓝通道里的「快频
RoPE 对」其 σ² 大部分是旋转伪影：pre 空间 σ² 小（ρ_pre 大）→ pre sign 误差小 → de-RoPE 后
post 误差也小。**结构（已知旋转）胜位宽**：de-RoPE-sign(1.25b) 可优于 post-mm2(2.25b)。

### pair-coupled 身份

de-RoPE 耦合 pair 两维，必须**同治**。pair 身份 = 两维合并的 μ² 占比：

```
identity_pair = (μ²_i + μ²_{i+D/2}) / (ex2_i + ex2_{i+D/2})，  蓝 pair = identity < 0.5
```

蓝通道按 σ² 来源再分子类（见 `findings.md`）：
- **rot / fast**：post/pre σ² 比值大（或 RoPE 周期 `2π/θ < G`，组内扫过 ≥1 周期 → arcsine
  双峰）—— σ² 是旋转伪影，de-RoPE / 更小组能救。
- **tail / slow**：比值≈1（或周期 ≥ G）—— pre 空间也方差大，是真实重尾内容，只能靠 bit
  （tern / mm2 / Lloyd）。

## 5. 统一统计口径（所有探针必须一致，否则数字不可比）

- per-channel 取向：`K[0].permute(0,2,1)` → `[H, D, T]`，固定 `(h,d)` 沿 token 轴。
- 分组：每 `G=128` 个连续 token 一组，**截掉**末尾不足一组的 token。
- 精度：permute 后**立刻 `.float()`**（fp16 上先 reduce 会偏 σ²），再算 mean/var。
- 口径：population（`var = E[x²] − μ²`，不用样本无偏）。
- post-RoPE：默认统计的是 cache 里的 post-RoPE K / V；pre-RoPE 仅 `kchannel_sigma`/`typed`/
  `blue` 经 k_proj hook 或解析 de-RoPE 取得。
