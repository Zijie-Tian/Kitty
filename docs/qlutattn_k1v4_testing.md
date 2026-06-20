# QLUT-Attn k1v4 — LongBench 测试文档

`qlutattn-k1v4`(原 `typed`)是按通道残差 σ² 分箱的**混合码本 K-cache 量化**:同层内不同 K
通道用不同码本(低 σ² → `sign` 1.25b,中 σ² → `tern` 1.83b,高 σ²/RoPE 双峰 → `nf2`
2.25b),winner 策略平均 K≈**1.68 bit**,V 固定 per-token 4-bit。它在更低位宽下反超等精度
均匀 tern 基线。方法设计见 `docs/qlutattn_k1v4.md`;本文件是**复现/测试手册**。

> 纯 PyTorch `kitty_sim` fake-quant:**精度代理**,不省真实显存、无 kernel。`nf2` 走
> per-group Lloyd-Max,是运行时瓶颈(`qlutattn-k1v4` 比 `qlutattn_k184v4` 慢约 2x)。

---

## 1. 变体（`--variant`）

| variant | method slug（输出目录后缀） | K 位宽 | 说明 |
| --- | --- | ---: | --- |
| `qlutattn_k1v4` | `qlutattn-k1v4` | ≈1.68 | 主方法,winner policy `["sign","sign","sign","tern","nf2","nf2"]` |
| `qlutattn_k184v4` | `qlutattn-k184v4` | ≈1.83 | 等精度 iso-tern 基线(全 tern K) |
| `fp16` | `fp16` | 16 | 天花板(dense fp16 KV) |

三者 V 都是 per-token 4-bit(`fp16` 除外)。`qlutattn_k1v4` / `qlutattn_k184v4` 内部均走
`k_codebook="qlut"` 的 σ²-分箱码本路径,差别只在 policy。winner policy 可用环境变量
`QLUT_BIN_CODEBOOKS=sign,sign,sign,tern,nf2,nf2` 覆盖(逗号分隔,长度 = `n_bins`)。

---

## 2. 前置

- conda env `kitty`(Transformers 4.57.6 等,见 CLAUDE.md)。
- 模型:`Llama-3.2-1B-Instruct` / `Llama-3.2-3B-Instruct`(本机 `/home/zijie/models/`,
  `run_exp.sh llama32` 默认命中 1B;3B 需显式覆盖见下)。
- LongBench 数据:`$HOME/data/LongBench/data/*.jsonl`(21 数据集,`run_exp.sh` 默认命中)。
- 入口**只有** `scripts/run_exp.sh`。`--max-samples N`(N>0)= smoke(写 `longbench_out/smoke/`),
  省略 = full。完成后自动 score 到 `pred/result.json`。

`--gpus` 多 worker:列表里**重复列同一张卡 = 该卡多 worker**(如 `--gpus 0,0,0,1,1,1`
= GPU0/1 各 3 worker)。一个数据集一个 worker,先完成的卡抢下一个待跑数据集。

---

## 3. Smoke(快速自检,2 样本/数据集)

```bash
cd /home/zijie/Code/Kitty
# 1B,GPU0 单卡 2 worker,2 个长上下文数据集(确保 K 路径被走到)
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpus 0,0 --variant qlutattn_k1v4 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn-k1v4/{pred,logs}
```

预期日志:`variant=qlutattn_k1v4_nb6_v4_cb<hash>`、`status: ok`、自动 score。基线同法:
`--variant qlutattn_k184v4`(慢约 1/2)。

---

## 4. 全量测试（21 数据集,32k 上下文）

### 4a. Llama-3.2-1B（每卡 3 worker = 18 worker,~5GB/worker）

```bash
cd /home/zijie/Code/Kitty
GPUS=0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5
for V in qlutattn_k1v4 qlutattn_k184v4 fp16; do
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpus "$GPUS" --variant "$V"
done
# -> longbench_out/llama32-1b-instruct_{qlutattn-k1v4,qlutattn-k184v4,fp16}/
```

### 4b. Llama-3.2-3B（**每卡 1 worker = 6 worker** — 2/卡@32k 会 OOM,见 §7）

```bash
cd /home/zijie/Code/Kitty
GPUS=0,1,2,3,4,5
for V in qlutattn_k1v4 qlutattn_k184v4 fp16; do
  LLAMA32_MODEL_ID=meta-llama/Llama-3.2-3B-Instruct \
  LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-3B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-3b-instruct \
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/run_exp.sh llama32 --gpus "$GPUS" --variant "$V"
done
# -> longbench_out/llama32-3b-instruct_{qlutattn-k1v4,qlutattn-k184v4,fp16}/
```

> `LLAMA32_MODEL_SLUG=llama32-3b-instruct` **必须设**,否则 3B 结果会写进 1B 目录。
> full 模式可断点续跑(按数据集 jsonl 行数 skip 已完成的)。`nf2` 慢:1B 全量 typed≈2.5h
> (18 worker),3B 全量 qlutattn-k1v4≈6.6h(6 worker),qlutattn_k184v4 约其一半。

---

## 5. 实测结果（本机 6×RTX3090,32k,MAX_GEN=256,2026-06-15）

全量 21 数据集均分:

| 模型 | fp16 | qlutattn_k184v4 (1.83b) | **qlutattn-k1v4 (1.68b)** | 留存 fp16 | k1v4 − tern |
| --- | ---: | ---: | ---: | ---: | ---: |
| **1B** | 27.59 | 22.96 (83.2%) | **24.88 (90.2%)** | 90.2% | **+1.91** |
| **3B** | 36.33 | 30.72 (84.6%) | **34.28 (94.4%)** | 94.4% | **+3.56** |

**结论:** `qlutattn-k1v4` 用**更低 bit**(1.68 < 1.83)在两个规模都 **Pareto 占优** 均匀
tern,且优势随模型增大而扩大(+1.91 → +3.56),模型越大越接近 fp16(留存 90.2% → 94.4%)。
增益集中在 K 保真敏感的检索类(3B:multifieldqa_en +13.1、qasper +10.0、hotpotqa +7.6、
2wikimqa +6.1、multifieldqa_zh +5.6),3B 上几乎全面优于 tern(21 项仅 passage_count −0.03)。

3B per-dataset 亮点(fp16 / tern / k1v4):

| 数据集 | fp16 | tern | k1v4 | k1v4−tern |
| --- | ---: | ---: | ---: | ---: |
| multifieldqa_en | 50.16 | 35.07 | 48.15 | +13.08 |
| qasper | 40.16 | 29.53 | 39.57 | +10.04 |
| hotpotqa | 28.08 | 23.41 | 31.00 | +7.59 |
| 2wikimqa | 26.17 | 23.73 | 29.83 | +6.10 |
| triviaqa | 88.89 | 84.60 | 89.00 | +4.40 |

---

## 6. 评分与产物

- `pred/<dataset>.jsonl`(逐样本预测)、`pred/<dataset>.manifest.json`、`pred/result.json`
  (各数据集分数,run 结束自动生成);`logs/report_<dataset>.json`。
- 部分/抢先评分(只统计已完成数据集):

```bash
PYTHONPATH=src python -m kitty_sim.cli.score_longbench \
  --model longbench_out/llama32-3b-instruct_qlutattn-k1v4/pred --no-strict-complete
```

---

## 7. 注意事项与坑

1. **3B 必须 1 worker/卡。** 单个 3B@32k worker 峰值 ~12–16GB,2 个叠加超单卡 24GB →
   长上下文样本几乎全 `CUDA out of memory`,被记 error 空预测、污染结果。`expandable_segments`
   救不了。1B@32k ~5GB/worker,可 3/卡。
2. **不要靠 driver log 的进度条估速**(多 worker `\r` 进度交错混在一行);看 `pred/*.jsonl`
   行数增长才准。
3. **杀多 worker run** 要先杀调度器 `run_exp.sh`(否则它检测 worker 死亡会复活新 worker),
   再杀 `kitty_sim.cli.eval_longbench`;worker 卡 CUDA 时 SIGKILL 会先变 `<defunct>` 僵尸,
   等 init 回收后显存才释放。`pgrep`/`pkill` 模式用 `[.]`(如 `run_exp[.]sh`)避免匹配到
   自己的命令行而误伤当前 shell。
4. **共享 GPU**:启动前 `nvidia-smi` 确认目标卡空闲;别抢别人的训练任务(会双方 OOM)。
5. `qlutattn-k1v4` 与 `qlutattn_k184v4` 对比时,两臂都不传 `QLUT_BIN_CODEBOOKS` 即用各自内置
   policy;若要扫别的 policy,用 `LLAMA32_MODEL_SLUG` 把 policy 编进输出目录名,避免不同
   policy 混入同一目录(resume 只看行数)。

---

## 8. 离线策略搜索 proxy（不跑 LongBench,秒级）

搜 policy 时先用 top-32 attention-overlap proxy 过滤,再上 LongBench:

```bash
# 1) 一次性缓存 post-RoPE K + 真实 query
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/build_kq_cache.py \
  --model /home/zijie/models/Llama-3.2-1B-Instruct \
  --longbench-dir $HOME/data/LongBench/data --tag llama32-1b
# 2) 评估一个 policy(打印 eff_bits + overlap,写 probe_out/qlut_policy_eval.json)
printf '{"group_size":128,"bin_codebooks":["sign","sign","sign","tern","nf2","nf2"]}' > /tmp/pol.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/eval_qlut_policy.py \
  --cache probe_out/kq_cache_llama32-1b.pt --policy /tmp/pol.json
```

metric = `eff_bits`(越低越好),guard = `overlap ≥ 0.774`(iso-tern)。winner
`[sign,sign,sign,tern,nf2,nf2]` = 1.680 bit / overlap 0.777(Pareto 占优 all-tern 1.835/0.774)。
