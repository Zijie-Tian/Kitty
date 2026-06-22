---
name: lutdecoding-acc-bench
description: >-
  对单个模型运行一套标准化的 KV-cache 量化精度对比实验(Kitty 仓库, LongBench 全量
  21 数据集, 32k): fp16 FULL / ShadowKV / Kitty / KIVI*-2 / KIVI-2 / QLUTATTN
  (per-token sign/nf2 离线 σ²-mix, snf-pt, 50% 混合 ≈1.875bit)。跑完这一组即为一次
  完整精度实验。当用户要"对某模型做精度实验 / 精度评测 / 精度对比""跑 LongBench
  量化对比""测一个新模型的 KV 压缩精度""benchmark 这几种 KV-cache 方法""复现这套
  对比"时,务必使用本 skill——它封装了 QLUTATTN 的离线标定流程、6 个方法的精确命令、
  统一 driver 与结果汇总。即使用户只说"测一下 XX 模型"但上下文是 Kitty 的 KV-cache
  量化研究,也应触发本 skill,而不要自行拼命令。
---

# LUT-Decoding 精度基准 (lutdecoding-acc-bench)

对**一个模型**,在 LongBench 上跑一套固定的 KV-cache 量化精度对比。一次完整实验 =
下面 6 个方法各跑一遍全量 21 数据集,再汇总成一张「方法 × bit × 平均分 × 留存率」表。

除 `fp16`/`kivi*` 外都走 Kitty 的**纯 torch sim fake-quant 路径**:这是**精度代理**
(accuracy proxy),不省真实 KV 显存/不证明加速。这套实验比较的就是各方法在同一精度
代理下的 LongBench 得分。

## 这套实验测什么(6 个方法,固定顺序)

| # | 方法 | run_exp.sh variant | 关键参数 | 输出 method-slug | K bit/value |
| --- | --- | --- | --- | --- | ---: |
| 1 | **F16 FULL** | `fp16` | — | `fp16` | 16(上界) |
| 2 | **ShadowKV** | `shadowkv` | — | `shadowkv` | 稀疏代理 |
| 3 | **Kitty** | `kitty` | 默认 k2/b4/v2/pr0.125 | `kitty-k2b4v2-pr0p125` | ~2.5 |
| 4 | **KIVI\*-2** | `kivi_star` | `KBITS=2 VBITS=2` | `kivi-star-k2v2` | 2.25 |
| 5 | **KIVI-2** | `kivi` | `KBITS=2 VBITS=2` | `kivi-k2v2` | 2.25 |
| 6 | **QLUTATTN** | `qlutattn_k188v4_pt` | **需先离线标定** + `QLUT_CB_MASK=<mask>` | `qlutattn-k188v4-pt` | ~1.875 |

> QUEST + Kitty 曾是第 7 个方法,但已在 commit `8adb49b`(2026-06-17 重构)从仓库
> 删除,当前**不在本基准内**。若日后恢复 `quest_kitty_page16_*` variant,在 driver
> 的方法表里加一行即可。

**QLUTATTN = snf-pt**(per-token sign/nf2 离线 σ²-mix)。每个 post-RoPE K 通道按其
wikitext 残差 σ² 离线固定为 sign(低 σ²,1.25b)或 nf2(高 σ²,2.5b);sign 通道占比
`f` 决定实际 K bit(`bit = f·1.25 + (1−f)·2.5`),`f=0.5` → **1.875bit**。per-channel
均值在 prefill 自标定(对 attention 免费),V 走 per-token 4-bit。详见
仓库 `CLAUDE.md` 的「默认推荐优化算法」段。

## 前置条件

- **在 Kitty 仓库根目录运行**,且 `conda activate kitty`(driver 默认 `REPO=$PWD`)。
- `scripts/run_exp.sh` 是 LongBench 唯一入口;本 skill 的 driver 只是编排它 6 次。
- **GPU 规则(重要)**:本 skill 会启动 GPU 作业。按项目规则,**开跑前必须先与用户
  确认 GPU 卡号与规模**,driver 不会替你问。
- **模型/数据路径**:用本机实际路径(或 `.env` 的 `KITTY_*_PATH`)。本机示例:
  - 模型 `/home/zijie/models/Llama-3.2-1B-Instruct`
  - 标定数据 `/home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet`
- **全量必须** `MAX_MODEL_LEN=32768`,生成长度用 per-target 值(llama32=256, qwen=2048…)。

## 快速开始:一条命令跑全 6 方法(driver)

driver `scripts/run_lutdecoding_bench.sh` 会:先做 QLUTATTN 离线标定 → 依次跑 6 个
方法 → 自动汇总。它把通用 `MODEL_PATH/MODEL_SLUG/MAX_GEN` 映射到 `run_exp.sh` 对应
target 的 per-target env,所以你只设一份。

```bash
# ---- smoke(2 样本/数据集,检索子集,验证命令链正确,~分钟级)----
cd /home/zijie/Code/Kitty && conda activate kitty
TARGET=llama32 \
MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MODEL_SLUG=llama32-1b-instruct \
MAX_GEN=256 GPUS=0,1,2 \
CALIB_DATA=/home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
bash .claude/skills/lutdecoding-acc-bench/scripts/run_lutdecoding_bench.sh smoke
# -> longbench_out/smoke/llama32-1b-instruct_{fp16,shadowkv,kitty-...,kivi-star-k2v2,kivi-k2v2,qlutattn-k188v4-pt}/
```

```bash
# ---- full(全部 21 数据集, 32k;1B 每卡 3 worker -> 6 卡 18 个)----
cd /home/zijie/Code/Kitty && conda activate kitty
TARGET=llama32 \
MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MODEL_SLUG=llama32-1b-instruct \
MAX_GEN=256 GPUS=0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 \
CALIB_DATA=/home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
bash .claude/skills/lutdecoding-acc-bench/scripts/run_lutdecoding_bench.sh full
# -> longbench_out/llama32-1b-instruct_<method-slug>/{pred,logs};末尾打印 + 写汇总 tsv
```

只重跑某几个方法:把方法名接在 mode 后(名字 = 上表第 1 列小写,KIVI* 用 `kivi_star`),
例如只补 QLUTATTN:`... run_lutdecoding_bench.sh full qlutattn`。

**自动跳过已完成的方法(测前先查,缺了才补)**:full 模式下 driver 在跑每个方法**之前**
先用 `scripts/check_method_complete.py` 检查该方法的 21 个数据集是否都已完整——判据是每个
`<dataset>.manifest.json` 的 `status==ok`(`written_samples>=expected_samples` 且无失败,
与 `runner.py` 自身判据一致)。**完整 → 整方法跳过**(连模型都不加载);**缺失/不完整 →
才调 `run_exp.sh`**,而它再按数据集级 resume 只补没跑完的那几个数据集。所以中断后重跑
同一条命令会从断点继续、已完成的方法秒跳。强制全部重跑:`FORCE_RERUN=1`。期望数据集
列表默认取仓库权威的 `LONG_BENCH_DATASETS`(21);设 `DATASETS_CSV=` 子集时按子集判定。
**smoke 模式不跳过**(`run_exp.sh` 每次清空 smoke 目录)。

手动单查某方法是否完整:
```bash
PYTHONPATH=src python .claude/skills/lutdecoding-acc-bench/scripts/check_method_complete.py \
  --pred-dir longbench_out/llama32-1b-instruct_qlutattn-k188v4-pt/pred
# 退出码 0=完整(可跳过) / 2=不完整(需补测);并打印缺哪些数据集
```

## ★ QLUTATTN 离线标定详解(50% 混合度)

这是本基准里唯一需要**离线标定**的方法,务必先做(driver 会自动做;手动跑见下)。

**为什么要标定**:snf-pt 给每个 post-RoPE K 通道二选一码本——低 σ² 通道用便宜的
`sign`(1.25b),高 σ² 通道用富表达的 `nf2`(2.5b)。"哪些通道用哪个"是模型固有属性
(由权重决定),所以离线在 wikitext 上测每通道残差 σ²、排序、按比例切分一次,存成一个
per-channel 掩码;运行时直接加载、不再在线计算。**per-channel 均值不在这里标定**——
它在每个 prompt 的 prefill 阶段自标定(`q·μ` 在 softmax 抵消,对 attention 免费)。

**50% 混合度怎么来**:`--sign-frac 0.5` = σ² 最低的 50% 通道给 sign、其余 50% 给 nf2。
实际 K bit = `0.5·1.25 + 0.5·2.5 = 1.875`。方法名里的 "188" 只是这个 50% 默认点的命名,
**没有意义,实际 bit 永远按比率算**。要别的 bit 就改 `--sign-frac`(driver 用 `SIGN_FRAC`)。

```bash
# 标定(~1min/卡;每个模型只需一次,掩码缓存到模型旁复用)
cd /home/zijie/Code/Kitty && conda activate kitty
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_k168v4_pt.py \
  --model      /home/zijie/models/Llama-3.2-1B-Instruct \
  --calib-data /home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --codebooks  sign,nf2 \
  --sign-frac  0.5 \
  --output     /home/zijie/models/Llama-3.2-1B-Instruct.lutbench_snf_f50.pt
# 打印实际 sign 占比 + nominal eff bit(应 ~1.875);掩码形状 [n_layers, n_kv, head_dim] uint8 (0=sign,1=nf2)
```

参数要点:
- `--codebooks sign,nf2`:低 σ²→sign、高 σ²→nf2(这就是 snf-pt;`sign,tern` 则是 st-pt)。
- `--sign-frac 0.5`:50% 混合度(本基准固定 0.5)。也可用 `--target-bits 1.875` 反解比例。
- `--output`:掩码路径,运行时用 `QLUT_CB_MASK` 指向它。driver 默认存
  `<MODEL_PATH>.lutbench_snf_f50.pt` 并自动复用。
- **换模型必须重新标定**(σ² 是模型固有的);换 `--sign-frac` 也要重标(并改输出名)。

运行时(driver 已包,手动则):
```bash
QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.lutbench_snf_f50.pt \
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct LLAMA32_MAX_GEN=256 \
MAX_MODEL_LEN=32768 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2 --variant qlutattn_k188v4_pt
```
没设 `QLUT_CB_MASK` 会直接报 `FileNotFoundError`——这是有意的保险。

## 分步手动命令(不想用 driver 时,full 全量)

每条都是完整可跑块(smoke 版:加 `--max-samples 2` 且
`DATASETS_CSV=multifieldqa_en,hotpotqa`,输出落到 `longbench_out/smoke/`)。以
Llama-3.2-1B 为例,`LLAMA32_MODEL_PATH` 指向模型,`LLAMA32_MODEL_SLUG` 定输出 model 段。

```bash
cd /home/zijie/Code/Kitty && conda activate kitty
COMMON="LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
LLAMA32_MODEL_SLUG=llama32-1b-instruct LLAMA32_MAX_GEN=256 MAX_MODEL_LEN=32768"
G=0,0,0,1,1,1,2,2,2

# 1) F16 FULL
env $COMMON bash scripts/run_exp.sh llama32 --gpus $G --variant fp16
# 2) ShadowKV
env $COMMON bash scripts/run_exp.sh llama32 --gpus $G --variant shadowkv
# 3) Kitty (paper k2/b4/v2/pr0.125)
env $COMMON bash scripts/run_exp.sh llama32 --gpus $G --variant kitty
# 4) KIVI*-2 (sink=32)
env $COMMON KBITS=2 VBITS=2 bash scripts/run_exp.sh llama32 --gpus $G --variant kivi_star
# 5) KIVI-2 (no sink)
env $COMMON KBITS=2 VBITS=2 bash scripts/run_exp.sh llama32 --gpus $G --variant kivi
# 6) QLUTATTN (先跑上面的标定生成掩码)
env $COMMON QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.lutbench_snf_f50.pt \
  bash scripts/run_exp.sh llama32 --gpus $G --variant qlutattn_k188v4_pt
```

## 汇总结果

每个方法跑完,`run_exp.sh` 会自动 score 出 `…/pred/result.json`(21 集逐项分)。汇总:

```bash
python .claude/skills/lutdecoding-acc-bench/scripts/collect_lutdecoding_results.py \
  --base longbench_out --layout full --model-slug llama32-1b-instruct \
  --mask /home/zijie/models/Llama-3.2-1B-Instruct.lutbench_snf_f50.pt
```
输出一张表(并写 `longbench_out/<model_slug>_lutdecoding_bench.tsv`):

```
method      bit   avg21  retain%   n  slug
F16 FULL     16   27.59   100.0   21  fp16
ShadowKV  sparse   ...     ...    21  shadowkv
Kitty       ~2.5    ...     ...    21  kitty-k2b4v2-pr0p125
KIVI*-2     2.25    ...     ...    21  kivi-star-k2v2
KIVI-2      2.25    ...     ...    21  kivi-k2v2
QLUTATTN   1.875    ...     ...    21  qlutattn-k188v4-pt
```
`MISSING` 行 = 该方法还没跑完(无 result.json)。smoke 用 `--layout smoke`。

## 换模型 / 换 GPU

- **换模型**:改 `MODEL_PATH` + `MODEL_SLUG`(务必改 slug,否则不同模型结果会写进同一
  目录)+ 对应 `MAX_GEN`;**重新跑 QLUTATTN 标定**。非 llama 模型改 `TARGET`
  (`llama`/`qwen`/`glm`/`deepseek`),driver 自动用对应 per-target env 前缀。
  ShadowKV/QLUTATTN 标定仅在 Llama/Qwen 家族验证过;新架构先 smoke。
- **换 GPU/规模**:改 `GPUS`(把一张卡列 N 次 = N 个 worker)。显存参考:1B@32k ~5GB
  → 3 worker/24GB 卡;3B ~12GB → 2/卡;8B ~22GB → 1/卡。**改 GPU 前与用户确认。**

## 注意事项

- 全部是精度代理(fp16/kivi* 用 dense fp16 KV;其余 sim fake-quant),不要当成显存/速度证明。
- snf-pt 与 KIVI*-2 一样带 `sink_length=32`,另外还保最近 `buffer_length=128` token 为
  fp16(KIVI-V 风格近窗);KIVI-2 无 sink。
- 标定掩码缓存在模型旁(`*.lutbench_snf_f50.pt`,被 `.gitignore` 的 `*.pt` 忽略),换
  `--sign-frac` 记得改输出名以免串用。
- full 可断点续跑(已完成的数据集保留);smoke 每次重跑会清掉上次 smoke 目录。
