---
name: kv-cache-viz
description: >-
  Kitty 仓库专用的 KV-cache 探测与可视化 skill。只要用户在 Kitty 项目里提到
  "画图 / 可视化 / probe / 分析 KV cache / 看分布 / Pareto / LongBench 结果图"
  等，就应该使用本 skill，而不是手动拼 `python scripts/dump_*.py` 或
  `python scripts/plot_*.py` 命令。本 skill 把 26 个随手脚本统一成
  `kv-cache-viz probe/viz/dump-layer` 命令，提取了公共的 LongBench doc 选择、
  model prefill、cache 提取、画图保存逻辑，并补全了离线画图脚本缺失的
  `layer{L}_{K,V}_fp16.pt` dump 步骤。
---

# KV-Cache 可视化 (`kv-cache-viz`)

把 Kitty 研究里常用的 KV-cache **探测**（从模型 + LongBench 数据 dump 统计量）
和 **离线画图**（读 `.pt`/JSON 生成论文图）整理成一个统一入口。

## 何时触发

在 Kitty 仓库内，用户出现以下意图时务必调用本 skill：

- "画一下 KV cache 分布" / "看一下 K/V 的 channel distribution"
- "probe 一下 sign scale" / "跑一下 mu2sigma2"
- "画 Pareto" / "画 heatmap" / "画 e2e latency/memory 柱状图"
- "分析 KV cache 的能量 / DC share / sigma2"
- "dump layer 8 的 K/V" 用于离线可视化

如果用户只是要求"跑 LongBench 精度对比"，那应该使用 `lutdecoding-acc-bench`
skill 而不是本 skill。

## 统一入口

```bash
bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh <cmd>
```

或用 slash 命令：`/kv-cache-viz <cmd>`

## 子命令

### `probe` — 需要 GPU + LongBench 数据

| 命令 | 原脚本 | 作用 | 输出 |
| --- | --- | --- | --- |
| `probe channel-energy` | `dump_channel_energy_csv.py` | 每层 K channel 能量统计 CSV | `.csv` |
| `probe mu2sigma2` | `dump_kv_mu2_sigma2_dist.py` | K/V 每通道 μ²/σ² 分布 | `.pt`, `.json`, `.png` |
| `probe nf2-pertoken-maxlevel` | `dump_nf2_pertoken_maxlevel.py` | per-token nf2 max level 分布 | `.pt`, `.png` |
| `probe sigma2-block-concentration` | `dump_sigma2_block_concentration.py` | σ² 在 128-token block 内/跨 block 集中度 | `.json`, `.png` |
| `probe signpt-dequant` | `dump_signpt_dequant_dist.py` | sign-pt 反量化前后 channel 分布对比 | `.png` |
| `probe sign-scale` | `dump_sign_scale_dist.py` | sign-group scale 分布 + 16-token 共享实验 | `.pt`, `.json`, `.png` |

### `dump-layer` — 为离线画图准备 layer K/V

```bash
bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh dump-layer \
  --layer 8 --model /path/to/model --longbench-dir /path/to/LongBench/data \
  --tag llama32-1b --outdir probe_out
```

生成 `probe_out/quant_kvcache_analysis/layer8_{K,V}_fp16.pt`，离线 viz 默认会读它。

### `viz` — 离线画图（纯 CPU / 可选 GPU）

| 命令 | 原脚本 | 作用 |
| --- | --- | --- |
| `viz channel-dist` | `plot_kv_channel_dist.py` | layer/head 8 个 channel 的 K/V 分布 |
| `viz channel-dist-multi` | `plot_kcache_channel_dist_multi.py` | 每 channel 单独一张 PNG |
| `viz dcshare-heatmap` | `plot_kcache_dcshare_heatmap.py` | DC-energy share 2D heatmap |
| `viz decomp` | `plot_k_decomp_steps.py` | K = μ + residual + sign-pt 重建的 3D 图 |
| `viz kv-3d-submean` | `plot_kv_3d_submean.py` | K = μ + residual 3D 图 |
| `viz kv-seg-mu2sigma2-3d` | `plot_kv_seg_mu2sigma2_3d.py` | 分段 μ²/σ² 3D 图 |
| `viz codebook-mu2sigma2` | `plot_codebook_mu2sigma2.py` | sign/nf2 codebook 在 μ²/σ² 通道上的示意 |
| `viz kcache-reorder-2d` | `plot_kcache_reorder_2d.py` | σ² 重排后的 2D 空间图 |
| `viz reorder-mixed-codebook` | `plot_reorder_mixed_codebook.py` | 重排 + 混合码本示意 |
| `viz why-sign-beats-minmax` | `plot_why_sign_beats_minmax.py` | sign 打败 minmax 的理论+实证图 |
| `viz kcache-pareto` | `plot_kcache_pareto.py` | 方法散点 Pareto 图 |
| `viz kcache-pareto-combined` | `plot_kcache_pareto_combined.py` | 含 per-token sweep 的合并 Pareto 图 |
| `viz kitty-kv-heatmap` | `plot_kitty_kv_heatmap.py` | Kitty K×V bit sweep heatmap |
| `viz kivistar-heatmap` | `plot_kivistar_heatmap.py` | KIVI* K×V bit sweep heatmap |
| `viz kv-channel-dist-layers` | `plot_kv_channel_dist_layers.py` | 跨层 K/V 分布汇总 |
| `viz pertoken-smooth` | `plot_pertoken_smooth.py` | per-token + SmoothAttention 对比 |
| `viz e2e-perf` | `plot_e2e_perf_bars.py` | end-to-end 性能柱状图 |
| `viz kv-memory-bars` | `plot_kv_memory_bars.py` | KV memory 柱状图 |
| `viz attn-op-latency` | `plot_attn_op_latency_bars.py` | attention op latency 柱状图 |
| `viz qlutattn-energy-quest1024` | `plot_qlutattn_energy_quest1024.py` | Quest 预算 1024 下的能量对比 |

## 公共参数

### probe / dump-layer 公共参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 模型路径 |
| `--longbench-dir` | 必填 | LongBench `data/` 目录 |
| `--seq-len` | 32768 | prefill 长度 |
| `--sink` | 32 | sink token 数 |
| `--recent` | 128 | recent token 数 |
| `--chunk` | 4096 | chunked prefill 步长 |
| `--device` | `cuda:0` | 设备 |
| `--tag` | `model` | 输出文件名标签 |
| `--outdir` | `probe_out/sign_scale` | 输出目录 |

### viz 公共参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--layer` | 8 | 目标层 |
| `--head` | 0 | 目标 head |
| `--pt` | 自动 | K/V dump 路径；默认 `outdir/quant_kvcache_analysis/layer{L}_{K,V}_fp16.pt` |
| `--outdir` | `probe_out` | 输出目录前缀 |
| `--tag` | `llama32-1b` | 输出文件名标签 |
| `--device` | `cuda:0` | 需要 GPU 的 viz 命令使用 |

## 快速开始

```bash
cd /home/zijie/Code/Kitty

# 1. dump 统计量（smoke 用 1024 token 即可）
bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh probe mu2sigma2 \
  --model /home/zijie/models/Llama-3.2-1B-Instruct \
  --longbench-dir /home/zijie/data/LongBench/data \
  --seq-len 1024 --tag llama32-1b --outdir probe_out/sign_scale

# 2. dump layer 8 的 K/V 用于离线画图
bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh dump-layer \
  --layer 8 --model /home/zijie/models/Llama-3.2-1B-Instruct \
  --longbench-dir /home/zijie/data/LongBench/data \
  --seq-len 1024 --tag llama32-1b --outdir probe_out

# 3. 离线画图
bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh viz channel-dist \
  --layer 8 --head 0 --tag llama32-1b --outdir probe_out
```

## 注意事项

- 所有 `probe` 和 `dump-layer` 命令都会加载模型到 GPU，请确认 GPU 可用。
- `viz` 命令默认读 `probe_out/quant_kvcache_analysis/layer{L}_{K,V}_fp16.pt`；
  如果路径不同，用 `--pt` 指定。
- 输出目录若不存在会自动创建。
- 本 skill 是研究辅助工具，不替代 `lutdecoding-acc-bench` 的精度 benchmark。
