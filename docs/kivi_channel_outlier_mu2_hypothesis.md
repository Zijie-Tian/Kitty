# KIVI channel-outlier 的病根是 μ² 吗? — 研究计划 + smoke 锚定

> **状态:DRAFT** — 计划已由 smoke 验证成立(本机 Llama-3.2-1B,2048-token);full 32k
> 尚未铺开。本文是供过目的本地草稿,不是对外发布稿。

## 0. 一句话

验证 **KIVI 必须用 per-channel 量化保护的 K "channel outlier",其高能量主要来自
per-channel 的 μ²(恒定 DC 偏置),而非 σ²(残差方差)**;推论是:减全局 per-channel
均值(0 bit,`q·μ` 在 softmax 抵消)就能消掉 KIVI per-channel 设计要解决的那个问题,这
正是 canonical `qlutattn`(per-token sign/nf2)成立的能量依据。

## 1. 假设

- **H1(本研究假设)**:outlier channel 是 **μ² 主导**;减全局 per-channel 均值(免费)
  → channel-间能量异质性塌缩 → per-token 量化从"崩溃"恢复到 ≈ per-channel。
- **H0(零假设)**:outlier 是 **σ² 驱动** → 减均值无用 → per-token 减均值后仍崩。

`q·μ` 为什么免费:`q·(K−μ) = q·K − q·μ`,而 μ 是 per-channel 常数(与 key/token 无关),
`q·μ` 对同一 query 的所有 key 是同一标量 → 在 softmax / top-k 里抵消。**注意这只对
*全局* per-channel 均值成立**(见 §3 口径)。

## 2. 这不是 autoresearch 优化循环(诚实标注)

autoresearch 的内核是 **modify→verify→keep/discard→repeat**(优化)。本目标是**测量 /
假设验证**:模型权重固定,没有"可改来推高指标的代码"。因此执行形态 = **一组有限的、
可证伪的鉴别性实验**,跑完出图/表/JSON,而**不是开环迭代**。唯一能套优化循环的子问题是
"搜索最便宜的减均值+per-token 码本方案",但那是 qlutattn 调参,不是本问题。

## 3. 🔑 决定成败的口径:减的是哪个均值

`kvcache-energy-probe` 默认 `--group 128`(per-128-token **组均值**)。但 qlutattn 减的是
**全局 per-channel 均值**(over 整段 prompt)。两者本质不同:

| 均值 | 与 token 的关系 | `q·μ` 抵消? | 是否免费 |
| --- | --- | --- | --- |
| 全局 per-channel(`--group = seq_len`) | 无关(每通道一个常数) | 是 | **免费** |
| per-128-token 组均值(`--group 128`) | 随位置变 | 否 | **不免费** |

→ **所有实验必须 `--group = seq_len`**,否则会把"局部慢变"那部分(不可免费移除)算进
红利,高估结论。这就是上一轮钉过的 45% vs 64.8% 口径问题。

## 4. Smoke 锚定结果(本机,Llama-3.2-1B,post-RoPE,2048-token LongBench,全 16 层 pooled)

| 量 | 口径 A `--group 2048`(全局 per-channel,= qlutattn 免费移除) | 口径 B `--group 128`(组均值) |
| --- | ---: | ---: |
| **K** μ² 占比 | **52.81%**(层间 47–58%) | **65.01%** |
| **K** σ² 占比 | 47.19% | 34.99% |
| **V** μ² 占比 | **12.45%** | 15.84% |
| **V** σ² 占比 | 87.55% | 84.16% |
| ρ = E\|r\|/σ | 0.73–0.78 | 0.71–0.77 |
| 恒等式残差 | ~1e-9 ✓ | ~1e-8 ✓ |

**结论(smoke 已钉死的 3 件 + 1 个方法学要点):**

1. **口径差是真的**:全局 per-channel(52.8%)vs 组均值(65.0%)差 ~12pp;这 12pp 是
   `q·μ_g` 不抵消的"局部慢变",**不免费**。用探针默认 G=128 会高估免费红利 12pp。
2. **K/V 不对称解释了为什么免费减均值只用在 K**:K μ²=52.8%(DC 重,可免费移除),
   V μ²=12.5%(σ² 重,减均值拿不到能量)→ qlutattn 减均值用在 K(V 走 2-bit tile16c64)。
3. **ρ=0.73–0.78 < 高斯 0.798**:尾部偏重(与另一机器转录的 0.738 一致)→ 高 σ² 通道需
   自适应码本(nf2),固定高斯码本不够。
4. **⚠️ μ² 占比随上下文长度变**:同一 layer8,口径 A 在 2048-tok=51.2%,而 32512-tok
   转录值=44.8%(越长→σ² 累积越多→μ² 占比越低)。**对外数字必须在 full 32k 出**;且
   pooled ~50% 是**错误粒度**(把 ①DC主导 / ②③σ²主导 混平均),真正的 M1 三类比例要
   用 per-(layer,dim) 做 per-channel 分类。

## 5. 完整实验阶梯(每个出一个机械数 + 可证伪预测)

| # | 鉴别问题 | 机械度量 | H1 / H0 预测 | 工具 |
| --- | --- | --- | --- | --- |
| **E1** 相关性 | KIVI-outlier(absmax 最大那批)是不是 μ² 主导? | **M1** = top-k% outlier 中 μ²/E>0.5 的比例;outlier vs 非 outlier 的 μ²/E | M1≫基率 / M1≈基率 | `submean_energy_heatmap`(per-(layer,dim) PT)+ 小聚合 |
| **E2** 反事实 | channel 间能量 spread 有多少是 DC? | **M2** = 塌缩比 =(减均值前 E_max/E_median)/(减均值后 σ²_max/σ²_median) | M2≫1(layer8 见 29×→5.9×) / M2≈1 | `kchannel_sigma_quant` |
| **E3 ⭐** 因果判决 | 单独减全局均值,能否让 per-token 量化恢复到 per-channel? | **M3** = gap-closure =(q[pt,减均值]−q[pt,原始])/(q[pc,原始]−q[pt,原始]),q=真实 query attn top-32 overlap | M3≈1 / M3≈0 | **新薄探针**(见 §6) |
| **E4** 闭环 | 减均值后仍难量化的,是不是高 σ²(给 nf2 那批)且 μ²≈0? | **M4** = 减均值后量化误差 与 σ²_d 相关(应高)vs 与 μ²_d 相关(应≈0) | 误差由 σ² 解释 / μ² 仍有解释力 | `postrope_blue_codebook` + `qk_channel_importance` |
| **E5** 普适 | E1–E2 在全层×head、跨任务稳定吗? | **M5** = 全层 M1/M2 分布;红蓝身份跨任务 Pearson r | 稳定 / 漂移 | `submean_energy_heatmap` + `task_channel_profile` + `calibrate_sigma_channels`(需 wikitext) |

**E3 是判决性的**:E1 只证"outlier *是* μ² 主导"(相关);E3 证"*移除* μ² 修好了 KIVI
per-channel 要解决的那个具体崩溃"(因果)。下游旁证已有:CLAUDE.md 记录修正 submean 维度
把 full LongBench 10.99→21.68(per-token sign on 减均值 K);E3 用 matched 码本做干净的机
理版。

## 6. E3 因果判决探针设计(待实现的新薄探针)

基于 `.claude/skills/kvcache-energy-probe/scripts/_common.py`(`prefill` / `group_decompose`
/ q·k hook / 真实 query top-k overlap 原语,`typed_channel_quant.py` 已有 overlap 计算可
复用):

1. prefill 32k LongBench 单文档,收集 post-RoPE K 与真实 query q;
2. 在**同一 bit 预算 b**(如 b=2 uniform)下重建 4 个 cell:
   - ① per-channel / 原始 K(KIVI 正常区,高参考)
   - ② per-token / 原始 K(KIVI 警告的崩溃区,低)
   - ③ per-channel / 减均值 K(应 ≈ ①)
   - ④ **per-token / 减均值 K**(判决 cell):`k̂ = μ_d + quant_pt_b(k − μ_d)`,μ_d 为
     全局 per-channel 均值,精确存(一通道一个值、且 `q·μ` 抵消 → 免费)
3. 保真度 q[i] = 真实 query 的 attention top-32 overlap vs fp16(逐 query/head/layer 平均);
4. **M3 = (q[④]−q[②])/(q[①]−q[②])**,逐层 + pooled 报告。
5. 预期:③≈①(减均值不伤 per-channel);判决在 ② vs ④。H1→④≈①→M3≈1;H0→④≈②→M3≈0。

输出建议在末行打印 `gap_closure <value>` 便于机械提取。

## 7. 现成探针的 full 命令(E1/E2/E5,无需新代码)

环境:`kitty` env、GPU1-only、LongBench 在 `.env` 的 `LONGBENCH_DATA_ROOT`。

```bash
cd /path/to/Kitty
export LONGBENCH_DATA_ROOT=$HOME/data/LongBench
PY=$HOME/anaconda3/envs/kitty/bin/python
M=$HOME/models/Llama-3.2-1B-Instruct
SK=.claude/skills/kvcache-energy-probe/scripts

# E1 + E5(普适性):per-(layer,dim/head) μ² 热力图,--group=seq_len 取全局 per-channel 口径
CUDA_VISIBLE_DEVICES=1 $PY $SK/submean_energy_heatmap.py --model $M --seq-len 32768 --group 32768 --tag llama32-1b --outdir probe_out
# E2:σ² 集中度 / magnitude-vs-variance 选择器 / pre-post-RoPE σ²
CUDA_VISIBLE_DEVICES=1 $PY $SK/kchannel_sigma_quant.py    --model $M --seq-len 32768 --tag llama32-1b --outdir probe_out
# E4:蓝通道 bit-NMSE Pareto + QK 重要性归因
CUDA_VISIBLE_DEVICES=1 $PY $SK/postrope_blue_codebook.py  --model $M --seq-len 32768 --tag llama32-1b --outdir probe_out
CUDA_VISIBLE_DEVICES=1 $PY $SK/qk_channel_importance.py   --model $M --seq-len 32768 --tag llama32-1b --outdir probe_out
# E5(跨任务):红蓝身份任务无关性 + 域外标定→21 任务泛化(后者需 wikitext parquet)
CUDA_VISIBLE_DEVICES=1 $PY $SK/task_channel_profile.py    --model $M --seq-len 32768 --tag llama32-1b --outdir probe_out
# CUDA_VISIBLE_DEVICES=1 $PY $SK/calibrate_sigma_channels.py --model $M --calib-parquet /path/to/wikitext/train-00000-of-00001.parquet --calib-tokens 32768 --tag llama32-1b --outdir probe_out
```

> `submean_energy_heatmap` / `kchannel_sigma_quant` 等是否都接 `--group` 待跑前确认;`submean_energy`
> 确认接(已用)。M1 三类比例从 heatmap 存的 per-(layer,dim) PT 后处理:按 μ²/E 阈值分
> ①(>0.5)、再按 σ² 分位把 σ²-主导分 ②(低)/③(高)。

## 8. 开放依赖 / 下一步

- **E3 需我写新探针**(§6),是 headline / 最强单一证据。
- **E5 的 `calibrate_sigma_channels` 需 wikitext parquet**(本机是否有待确认;另一机器在
  `$HOME/data/wikitext/...`)。
- 对外发布(Notion 研究笔记)前:所有数字在 **full 32k**、**全层**、**per-channel 三类
  粒度**出齐;按 CLAUDE.md 路由到 `📚 研究笔记 | Research Notes`,且写 Notion 属外发,需
  先过目。

## 附:smoke 复现命令

```bash
cd /path/to/Kitty
export LONGBENCH_DATA_ROOT=$HOME/data/LongBench
PY=$HOME/anaconda3/envs/kitty/bin/python
S=.claude/skills/kvcache-energy-probe/scripts/submean_energy.py
M=$HOME/models/Llama-3.2-1B-Instruct
CUDA_VISIBLE_DEVICES=1 $PY $S --model $M --seq-len 2048 --group 2048 --out /tmp/se_A.json  # 口径 A 52.81%
CUDA_VISIBLE_DEVICES=1 $PY $S --model $M --seq-len 2048 --group 128  --out /tmp/se_B.json  # 口径 B 65.01%
```
