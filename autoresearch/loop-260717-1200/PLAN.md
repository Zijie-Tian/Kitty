# research_mixed loop — K-cache sign/nf2 混合精度 Pareto
约束: 只 sign/nf2 两码本; per-token K 机制不动; V (tile16c64 2bit) 不动。
Metric: mean(qasper, multifieldqa_en) 全量; 最终全量 21 集 LongBench。
参考: fp16 = (23.3+43.6)/2 = 33.45 | 现有 canonical f50 全量运行值 (16.6+33.5)/2 = 25.05

## Batch 1+2 (in flight, 每配置1卡×2worker)
GPU0 f50base  1.75b canonical σ² 50/50 (research=0, 基线+管线验证)
GPU1 q-probe → q75: σ²×E|q| @1.5b (MixKVQ 信号 A/B vs f75)
GPU2 f00     2.25b 纯 nf2 上锚
GPU3 f25     2.00b 均匀
GPU4 f75     1.50b 均匀 (低 bit 端基线)
GPU5 f75s    1.50b 层级配额 (L10/L14 sign-frac=0.5, 其余 0.786)

## 待跑队列 (batch 3+, 按结果决定)
- q50: σ²×E|q| @1.75b A/B vs f50base
- per-head 均衡 f75 (每头各 75% sign) vs 层内全局排序
- RoPE 对偶通道绑定 (i, i+D/2 同码本)
- token 级: 观察窗注意力分档 (需改代码, 视通道轴结果决定)
- 最终: 胜者跑全量 21 集 → Pareto 图

## 最终状态 (2026-07-18)
定版: q65 = sign0.65 + σ²×E|q| (α=1) + band(--nf2-exclude-top 0.05, Qwen 必需/其他无害待验) = 1.60b
全量21: 1B canonical 24.23@1.75 | q65 24.53 | ttm150 24.52@1.50 | f65 24.28 | f75 23.78 | q75 24.27
rollout: MiniCPM q65 18.23(+0.17) ttm150 18.10@1.49 | 3B q65 34.30(+0.06) ttm150 34.18@1.50 | Qwen no-band 12.18(崩溃对照) band版在跑
关键发现: 内部最优~35%nf2 / 乘积信号缺一不可 / 结构消融全阴性 / 全量稀释12× / tier-hi 必须锚定通道最优 / Qwen absmax中毒机制+band修复
