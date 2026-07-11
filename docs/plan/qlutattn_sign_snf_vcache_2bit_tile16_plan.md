# QLUT-Attn sign / SNF 的 V-cache 2-bit 与 rescued tile16cC 实现计划

> **文档状态：仅计划（Plan Only）**
>
> **日期：2026-07-11**
>
> **本轮明确不做：**不修改量化实现、不修改 runner、不运行单元测试、不启动 GPU、不启动 LongBench。

> **后续状态（2026-07-12）：**计划中的研究版实现与独立测试现已落地；
> 本文件保留原始设计与验收门槛，当前 GPU1 复现命令以仓库根目录
> `AGENTS.md` 的 “Sign / SNF rescued V2 `tile16c64` test method” 为准。
> **目标：**在保持现有 sign 与 SNF K-cache 路径不变的前提下，引入两种 2-bit V-cache 路径：
> 1. per-token asymmetric min-max 2-bit；
> 2. token block 固定为 16、channel block 为可配置 `C` 的 rescued `tile16cC` 2-bit。

---

## 0. 执行摘要

### 0.1 需求解释

现有两条 K-cache 父路径为：

- **sign**：`qlutattn_k125v4_pt`，K 为 per-token sign + per-channel submean；
- **SNF**：`qlutattn_k188v4_pt`，K 为离线 per-channel sign/nf2 mask + per-channel submean。

本计划把两种新的 V-cache 路径分别与这两条 K 路径组合，因此最终形成 **2 × 2，共四个新逻辑变体**：

| K-cache 父方法 | V-cache per-token 2-bit | V-cache rescued tile16cC 2-bit |
| --- | --- | --- |
| sign | `qlutattn_k125v2_pt` | `qlutattn_k125v2_pt_vtile16` |
| SNF | `qlutattn_k188v2_pt` | `qlutattn_k188v2_pt_vtile16` |

本计划的首要不变量是：

> 每个新变体的 K-cache 配置、mask、submean、旋转和 K block 行为都必须与其父变体一致；新旧变体之间只允许 V-cache 配置和 V-cache 调度发生变化。

### 0.2 tile 的语义锁定

`tile16cC` 在本计划中**不是**简单地把 `[16, C]` 的数值放到同一个 min-max 组中。此前误差实验已表明，plain tile 随 `C` 增大明显劣化；真正有希望的方案是已经在 probe 中验证过的 rescued pipeline：

```text
fixed RHT
+ prompt per-channel mean/RMS affine normalization
+ 2-bit tile16cC affine min-max initialization
+ exactly one MSE/least-squares affine refit
+ per-tile fallback to initial min-max
+ inverse affine / inverse RHT
+ prompt per-channel reconstruction-bias correction
```

因此，本计划冻结以下定义：

- 正式公开的 `vtile16` 变体代表上述 **rescued tile**；
- plain min-max tile 只作为内部 oracle/消融，不作为正式 LongBench variant；
- 若后续决定只实现 plain tile，必须先修改本计划的算法定义、测试 oracle、bit accounting 和 LongBench 接受标准，不能继续引用 rescued tile 的历史误差结论。

### 0.3 唯一公开旋钮

tile 的 token 维固定为 `16`，不提供 token-block 环境变量。唯一公开旋钮是 channel block：

```bash
V_TILE_CHANNELS=16   # 也可为 32、64，或任何能够整除 head_dim 的正整数
```

必须在三层做校验：

1. shell/CLI：存在、为正整数；
2. 模型加载后：根据模型 config 得到 `head_dim`，提前检查；
3. 量化核：对真实 tensor 做用户要求的权威断言：

```python
assert value_slice.shape[-1] % v_tile_channels == 0, (
    f"head_dim={value_slice.shape[-1]} must be divisible by "
    f"v_tile_channels={v_tile_channels}"
)
```

同时保留显式 `ValueError` 配置校验，因为 `python -O` 会关闭 `assert`。若启用 FWHT/RHT，还必须独立检查 `head_dim` 是 2 的幂。

---

## 1. 当前研究证据与设计动机

### 1.1 当前 V-cache 基线

现有 qlutattn sign/SNF 的 V-cache 都走同一条路径：

```text
value_slice [B, H, T, D]
→ 沿最后一维 head_dim 做 asymmetric min-max fake quant
→ vbits=4
→ sink=32 与 recent=128 保持 FP16
```

现有 V4 的 scale/min 逻辑来自 `fake_quant_groupwise_lastdim()`。在 Llama-3.2-1B 的 `D=64` 下，每个 token、每个 head 的整个 64-channel 行形成一个量化组。

### 1.2 已有 2-bit 误差观察

此前 probe 已在真实 32k LongBench prompt 上比较 per-token、token-axis block、head-dim block 和二维 tile。关键观察为：

1. **跨相邻 token 的同一 channel 局部更平稳**；
2. **跨 channel 的幅度异质性较强**；
3. plain tile 的 `C` 越大，越容易被跨-channel 动态范围拖累；
4. 2-bit 下这些差异比 4-bit 下更明显；
5. 采用 RHT、per-channel affine、一次 MSE refit 与 bias correction 后，tile16c16/32/64 的误差可明显低于 per-token 2-bit。

现有六文档研究给出的 rescued tile / per-token2 平均误差比，可作为后续生产实现 replay 的 sanity anchor，而不是随机 tensor 单元测试的硬阈值：

| rescued V tile | V reconstruction relMSE / PT2 | attention-output relMSE / PT2 |
| --- | ---: | ---: |
| `tile16c16` | 约 0.654 | 约 0.398 |
| `tile16c32` | 约 0.701 | 约 0.439 |
| `tile16c64` | 约 0.732 | 约 0.470 |

plain tile 的相对误差则随 `C` 增大恶化，因此不能把 plain 实现伪装成 rescued variant。

历史 probe 对不足16-token的尾块会按较小 group 量化，并把该 tail 纳入
`mu/rms/bias`；本生产计划则选择 strict full-block schedule，partial tail 保持
FP16。多数 32k 中段恰好对齐，但例如 QMSum 可能有 remainder。因此上表只能
作为方向性 anchor。进入 Gate F 前要先用“只处理完整16-token块”的 probe 语义
重放，不能声称与旧 aggregate 在所有文档上 bit-exact。

### 1.3 研究图与 probe 的角色

当前研究示意图位于：

```text
docs/figures/vcache_2bit_quantization_strategies.png
```

当前 probe/reference 位于 `.claude/skills/kv-cache-viz/`。它们用于研究和独立 replay，但生产代码必须复制为 `src/kitty_sim/` 下的受测试实现；生产路径不能 import `.claude` 下的未跟踪脚本。

---

## 2. 范围、非目标与兼容性边界

### 2.1 本次实现范围

- 新增四个正式 variant；
- 新增 V per-token 2-bit fake-quant 路径；
- 新增 rescued V tile16cC 2-bit fake-quant 路径；
- token tile 固定为 16；
- `C` 可配置且必须整除 `head_dim`；
- 维持 sink32 + recent128 保护策略；
- 维持 PostQuant 当前 step 读取量化前值的现有语义；
- 添加严格的 prefill/decode block-aligned 调度；
- 添加 CLI、shell、slug、manifest、config hash 和 engagement evidence；
- 添加 0-GPU oracle、调度、接线与回归测试；
- 后续执行真实 dump replay、LongBench smoke 和 full 对比。

### 2.2 明确非目标

- 不实现 packed 2-bit storage；
- 不宣称当前 pure-torch fake-quant 路径真实节省 KV 显存；
- 不实现 Triton/CUDA kernel；
- 不在第一版中把 RHT 永久 fold 到 `W_v/W_o`；
- 不改变 K-cache sign/SNF 算法；
- 不改变现有 V4、Q4_0、KIVI、Kitty 的行为或 slug；
- 不把 token tile 从 16 扩展为可配置值；
- 不把 `PERTOKEN_BLOCK` 复用为 V tile 配置；
- 不把 `group_size` 复用为 V tile channel 配置；
- 不让 `VBITS` 覆盖新命名 variant 固定的 2-bit 语义；
- 第一阶段不承诺 GLM 支持；未完成 GLM parity 前必须显式拒绝，不能静默 fallback。

### 2.3 兼容性目标

实现完成后必须满足：

- 现有 `qlutattn_k125v4_pt` 和 `qlutattn_k188v4_pt` 输出 bit-exact 不变；
- 现有输出目录不改名；
- 新 variant 的 C16/C32/C64 输出目录互不碰撞；
- SNF 的相同 mask 继续由 `QLUT_CB_MASK` 提供；
- 不同 SNF mask/sign fraction 仍需通过不同模型 slug 或显式实验标签隔离；
- 新增配置字段不得导致旧变体意外走新路径；
- 旧 cache API 行为不变。

---

## 3. 正式 variant、alias 与输出 slug 规范

### 3.1 Canonical variant 名称

正式 underscore 名称：

```text
qlutattn_k125v2_pt
qlutattn_k188v2_pt
qlutattn_k125v2_pt_vtile16
qlutattn_k188v2_pt_vtile16
```

同时接受对应 hyphen alias：

```text
qlutattn-k125v2-pt
qlutattn-k188v2-pt
qlutattn-k125v2-pt-vtile16
qlutattn-k188v2-pt-vtile16
```

### 3.2 Method slug

per-token V2：

```text
qlutattn-k125v2-pt
qlutattn-k188v2-pt
```

tile V2 必须编码解析后的 `C` 与 rescued algorithm version `rv1`：

```text
qlutattn-k125v2-pt-vtile16c16-rv1
qlutattn-k125v2-pt-vtile16c32-rv1
qlutattn-k125v2-pt-vtile16c64-rv1

qlutattn-k188v2-pt-vtile16c16-rv1
qlutattn-k188v2-pt-vtile16c32-rv1
qlutattn-k188v2-pt-vtile16c64-rv1
```

`rv1` 对应 `rht-pcaff-mse1-bias-v1`；任何会改变量化结果的 RHT、affine、
MSE、fallback 或 bias 语义都必须 bump 版本并生成新目录。若同时使用 K 侧
`PERTOKEN_BLOCK=N` 或 QUEST，稳定的完整后缀顺序冻结为：

```text
<base>-vtile16c<C>-rv<V>-blk<N>-quest-<mode>
```

例如：

```text
qlutattn-k188v2-pt-vtile16c32-rv1-blk16
```

未启用 block/QUEST 时省略对应后缀，但其相对顺序不能改变。

第一轮公平实验固定 `PERTOKEN_BLOCK=1`，避免 K block 与 V tile 同时变化。

### 3.3 环境污染规则

- `V_TILE_CHANNELS=` 空字符串是显式 unset sentinel；env loader 必须保留这个
  “已设置但为空”的状态，不能再让 `.env` 注入旧值；
- tile variant 的 resolved C 缺失或为空时：fail-fast；不采用静默默认值；
- per-token/non-tile variant：空字符串视为 unset；非空 stale C 则 fail-fast；
- 新命名 variant 若显式传入 `VBITS != 2`：fail-fast；
- sign 新变体允许 `QLUT_BIN_CODEBOOKS` 未设置或恰为 `sign`；其他值 fail-fast；
- sign 变体若不支持 `PERTOKEN_BLOCK>1`：fail-fast，而不是生成错误的 `-blkN` slug；
- Python canonical resolver、direct runner 与 shell preflight 输出必须逐项一致；
  shell 不再维护独立正式 slug 表。

旧 shell 曾对不支持 K block 的 variant 也追加 `-blkN`；这类目录属于历史误
标，不在兼容承诺中。迁移后：合法、确实支持 block 的旧 variant slug 不变；
不支持者遇到 `PERTOKEN_BLOCK>1` 直接报错。

---

## 4. 配置模型设计

### 4.1 `VariantConfig` 新字段

在 `src/kitty_sim/longbench/runner.py::VariantConfig` 中加入能够完整表达 V 算法的字段。推荐保持 `v_codebook` 为调度器，同时加入显式的 tile provenance：

```python
v_codebook: str = "kivi"              # kivi | q4_0 | tile16_rescued
v_tile_tokens: int | None = None       # tile variant 固定为 16
v_tile_channels: int | None = None     # C
v_tile_algo_version: str | None = None # "rht-pcaff-mse1-bias-v1"
v_rht_seed: int | None = None          # 固定 20260711
v_mse_iters: int | None = None         # 固定 1
```

规则：

- V PT2：`vbits=2, v_codebook="kivi"`，所有 tile 字段为 `None`；
- V tile2：

```python
vbits=2
v_codebook="tile16_rescued"
v_tile_tokens=16
v_tile_channels=C
v_tile_algo_version="rht-pcaff-mse1-bias-v1"
v_rht_seed=20260711
v_mse_iters=1
```

RHT seed、算法版本和 MSE 次数第一版不是公开旋钮，但必须进入 config/manifest，防止未来内部算法变化却复用旧结果目录。

固定映射：

```text
slug rv1 <=> v_tile_algo_version="rht-pcaff-mse1-bias-v1"
```

代码应由单一常量表生成该映射；不得在 shell 与 Python 中各自手写不同版本。

### 4.2 `KittyKVCacheConfig` 新字段

在 `src/kitty_sim/kitty_simulate.py::KittyKVCacheConfig` 中加入对应字段，并在 `validate()` 中实现：

1. `vbits == 2` 对新路径成立；
2. `v_codebook` 属于允许集合；
3. `tile16_rescued` 必须有：
   - `v_tile_tokens == 16`；
   - `v_tile_channels` 是正整数；
   - `v_rht_seed` 不为空；
   - `v_mse_iters == 1`；
4. 非 tile 路径不得携带 tile-only 参数；
5. 配置层使用 `ValueError`，tensor 层保留用户要求的 `assert`。

### 4.3 Config hash 和 manifest 兼容策略

新增 dataclass 字段可能改变旧变体的 `asdict(variant)` 结果。当前
`config_hash(manifest)` 包含时间、输出路径和 GPU 等非语义字段，而 shell
resume 只看行数；它不能承担新算法隔离。实现时必须引入两个稳定、用途明确
的 fingerprint，并修复 completed fast-path：

```text
variant_semantic_hash:
  K/V 算法字段 + C + rv version + seed + MSE iters + mask content SHA256

run_config_hash:
  variant_semantic_hash + model identity/config digest + dataset +
  max_model_len + max_gen + sample scope + prompt/template version
```

两种 hash 均使用 canonical sorted JSON，排除：created_at、输出路径、GPU id、
worker id、wall time 和 engagement counters。SNF 不能只 hash mask 路径，必须
记录并 hash mask 文件内容 digest、mask metadata 和 sign fraction。

resume 规则：

1. runner 在判断 dataset complete/skip 前计算 expected `run_config_hash`；
2. 读取现有完整 manifest 并比较 hash；
3. hash 缺失或不一致时拒绝复用，提示新目录/显式 FORCE；
4. shell `prepare_dataset()` 不得再仅凭行数把新 variant 判为可安全 skip；
5. Python completed fast-path 不得把完整 manifest 重写成缺少 variant/hash 的最小 manifest；
6. tile slug 中的 `-rv1` 提供目录级第一道隔离，stable hash 提供第二道校验。

兼容策略如下：

1. 新字段在旧变体上为 `None`；
2. semantic hash 仅纳入对当前算法有意义的字段；
3. 旧变体的 slug、tag 和算法输出保持不变；
4. 新变体的 semantic hash 必须随以下任一项改变：
   - V mode；
   - `C`；
   - algorithm version；
   - RHT seed；
   - MSE iterations；
   - K mask/config；
5. fresh manifest 必须完整记录新 V 字段、两个 hash 和 mask digest；
6. 对旧 full 目录的 resume 不允许仅因新增 `None` 字段而错误覆盖或误判；
7. 旧 manifest 没有 stable hash 时不得自动假设兼容，应要求重新生成或显式迁移。

### 4.4 无 GPU Python preflight：关闭 slug/hash/resume 控制流

当前 shell 在 Python worker 启动前按行数 skip，而 shell 本身无法可靠计算模型、
prompt/template、mask digest 和 per-dataset generation config。第一版明确采用
一次 **无 GPU Python preflight**，不把 skip 决策继续分散在 shell/Python 两套
逻辑中。

建议新增一个只解析元数据、不加载模型权重、不创建 CUDA context 的入口，例如：

```text
python -m kitty_sim.cli.preflight_longbench ... --json
```

它接收与正式 runner 相同的 target/variant/model/config/dataset 参数，输出：

```json
{
  "canonical_variant": "qlutattn_k188v2_pt_vtile16",
  "method_slug": "qlutattn-k188v2-pt-vtile16c32-rv1",
  "resolved_variant": {"...": "..."},
  "variant_semantic_hash": "...",
  "mask_sha256": "...",
  "datasets": {
    "multifieldqa_en": {"run_config_hash": "...", "expected_rows": 200},
    "gov_report": {"run_config_hash": "...", "expected_rows": 200}
  }
}
```

控制流冻结为：

1. `scripts/run_exp.sh` 先解析显式 CLI/env，但不自己拼正式 method slug；
2. shell 调用一次 preflight，取得 canonical slug 和所有 expected hashes；
3. shell 使用 preflight slug 建立唯一输出目录；
4. `prepare_dataset()` 只有在 row count、complete status、expected
   `run_config_hash` 全部一致时才 skip；
5. missing/mismatch manifest 不得按行数直接 skip；
6. shell 将 expected hash 传给每个 Python worker；
7. worker 在生成前独立重算并 assert 相同，完成后写完整 manifest；
8. direct runner 也复用同一 resolver/preflight library；
9. shell 旧 `method_slug()` 要么删除，要么只作为调用 preflight 的薄 wrapper；
10. dry-run/print-slug 测试直接检查这一个 canonical source。

preflight 可以读取本地 `AutoConfig`/portable LongBench JSON 配置和 mask 文件，
但不得加载权重或占用 GPU。若模型 metadata 不可用，应在调度前报错。

### 4.5 旧 manifest 的迁移策略

第一版**不自动 backfill**旧 manifest，因为仅凭旧行数/最小 manifest 无法可信
恢复当时的 mask 内容和完整运行语义。为避免与当前已有 FP16/V4 目录冲突，本
研究的 11 个 full 点全部使用新的实验 model slug：

```text
llama32-1b-instruct-vtile-study-rv1
```

因此：

- FP16 与两个 V4 parent 也重新跑到新 study slug 下；
- 当前 legacy `longbench_out/llama32-1b-instruct_*` 保持只读、不会被覆盖；
- 新 study 目录从第一条结果起就具有 stable hashes；
- 不需要用 `FORCE=1` 绕过 equal-complete legacy 目录；
- 若未来要复用 legacy baseline，必须另写审计式 migration 工具并人工确认，
  不属于第一版实现。

不同 SNF mask/sign fraction 的后续研究仍应使用不同 model slug，同时依赖
`mask_sha256` 做最终 provenance。

---

## 5. V per-token 2-bit 的精确定义

### 5.1 数学定义

对每个 `[B, H, token]` 的整行 `D=head_dim` 数值单独量化。第一版命名
variant 锁定 FP16 研究口径；若模型/cache dtype 不是 FP16，启动时 fail-fast，
避免 BF16/FP32 在未验证的舍入路径上生成同名结果。精确算术必须与当前部署
helper 一致，而不是抽象的实数量化公式：

```text
xh = x.half()
qmin = 0
qmax = 3
min = amin(xh)
max = amax(xh)
scale = clamp(max - min, min=1e-4) / 3
q = clamp(round_to_nearest_even((xh - min) / scale), 0, 3)
x_hat = (q * scale + min).half()
```

其中 `min/max/scale/q/dequant` 都遵循 PyTorch FP16 运算和 `torch.round`
的 round-to-nearest-even 语义。`1e-4` 与 FP16 side-parameter 口径是算法的一
部分；常量组通过 scale clamp 稳定处理，不能产生除零、NaN 或 Inf。

### 5.2 分组语义

新 PT2 variant 的语义应锁定为“每个 token 的整个 head_dim 是一个组”，不要让全局 `group_size=128` 在未来 `D>128` 模型上意外切成多个组。

推荐实现方式：

```python
fake_quant_groupwise_lastdim(value_slice, group_size=value_slice.shape[-1], bit=2)
```

这在当前 Llama `D=64/128` 上与现有行为等价，同时把命名 variant 的 per-token 语义写死。

### 5.3 调度语义

- sink 前 32 token 保持 FP16；
- recent 128 token 保持 FP16；
- prefill 一次量化中段；
- decode 每一步量化刚滑出 recent window 的一个 token；
- 不等待 16-token block；
- PostQuant 下当前 step 返回量化前 V，cache mutation 只影响后续 step。

### 5.4 理论 bit accounting

每个 token/head 有 2-bit code，以及一对 FP16 side information（offset/min 与 scale）：

```text
b_pt2 = 2 + 32 / D
```

因此：

| head_dim | 理论量化区 V bit/value |
| ---: | ---: |
| 64 | 2.500 |
| 128 | 2.250 |

该数字只描述理论 packed representation；当前 fake-quant tensor 仍是 dense FP16。

---

## 6. Rescued V tile16cC 2-bit 的精确定义

### 6.1 输入与分块

输入：

```text
V ∈ FP16[B, H, T, D]
```

固定：

```text
token tile N = 16
channel tile C = V_TILE_CHANNELS
```

必须满足：

```text
C > 0
D % C == 0
D 是 2 的幂（RHT/FWHT 要求）
```

对完整量化区重排为：

```text
[B, H, n_blocks, 16, D/C, C]
```

每个 `[16, C]` tile 展平为一个 `16*C` 数值的量化组，共享一对 affine offset/scale。

### 6.2 Step A：固定 RHT

使用固定 seed `20260711` 生成 Rademacher signs，并在 head_dim 上应用与
probe 完全一致的 normalized randomized Hadamard transform。signs 必须由
CPU generator 生成，不能使用当前 CUDA RNG 状态：

```text
g = torch.Generator(device="cpu").manual_seed(20260711)
s_int8 = torch.randint(0, 2, (D,), generator=g, dtype=torch.int8)
s = (2*s_int8 - 1).to(device=V.device, dtype=V.dtype)

R(V)      = H_norm(V * s)
R_inverse(Y) = H_norm(Y) * s
```

这里 `H_norm` 是除以 `sqrt(D)` 的 normalized FWHT，且 self-inverse。乘
sign 的位置不能交换：forward 是 `H_norm(V*s)`，inverse 是
`H_norm(Y)*s`，不是 `H_norm(V)*s`。

精确 round-trip 顺序冻结为：

```text
forward RHT 后:        V_r = R(V).half().float()
mu/rms:                各自 .half().float() 后存储/使用
tile offset/scale:     .half().float() 后重新编码
inverse RHT 后:        V_hat_0 保持 FP32 用于 bias 计算
bias:                  mean(V_hat_0 - original_FP16_as_FP32).half().float()
最终 cache reconstruction: (V_hat_0 - bias).half()
```

实现时必须冻结：

- CPU generator 与 signs 生成算法；
- signs 的 dtype/device 转换；
- sign multiply 与 FWHT 的顺序；
- normalization 因子；
- 上述每个 FP16 round-trip 的位置。

测试必须证明：

```text
inverse_RHT(RHT(x)) ≈ x
```

第一版 fake-quant cache 会在量化后 inverse RHT 回原始 V basis，以保持 attention 路径不变。未来真实 kernel 可评估把 RHT fold 到 `W_v/W_o`，但不属于本次范围。

### 6.3 Step B：prompt per-channel affine normalization

只对**将被量化的完整 settled prompt blocks**计算统计量，不允许 sink、recent 或 pending tail 污染 calibration：

```text
mu  = mean_token(V_r)
rms = sqrt(mean_token((V_r - mu)^2)).clamp(min=1e-4).half().float()
```

形状：

```text
mu, rms: [B, H, 1, D]
```

`mu` 同样在 FP32 reduction 后执行 `.half().float()`；`rms` 的 epsilon 明确
冻结为 `1e-4`。存储 dtype 为 FP16；归一化：

```text
Z = (V_r - mu) / rms
```

`1e-4`、clamp 的位置和 half-roundtrip 都属于 rv1 数学定义并进入 literal
oracle；不能在生产实现中改用其他 epsilon。

### 6.4 Step C：2-bit min-max 初始化

对每个展平的 `16*C` tile，必须先执行与 probe 相同的
`xh = Z_tile.half()`，并在 FP16 中完成 min/max、scale、code assignment 与
dequant；不能只在最终 reconstruction 上做一次 half：

```text
xh       = Z_tile.half()
offset_0 = min(xh)
scale_0  = clamp(max(xh) - min(xh), min=1e-4) / 3
q_0      = clamp(round_to_nearest_even((xh - offset_0) / scale_0), 0, 3)
Z_0      = (offset_0 + scale_0 * q_0).float()
```

对常量 tile 由 `1e-4` scale clamp 稳定处理。该 baseline 的 offset/scale 是
FP16 side parameter 语义；生产 helper 和 literal oracle 必须锁定相同运算顺序。

### 6.5 Step D：恰好一次 MSE affine refit

候选 refit 的输入先采用 `x = Z_tile.half().float()`，然后固定初始 assignment
并对：

```text
Z ≈ offset + scale * q_0
```

做一次 closed-form least-squares 拟合。若组内元素数为 `n`：

```text
denom   = n * sum(q^2) - sum(q)^2
scale1  = (n * sum(q*Z) - sum(q)*sum(Z)) / denom
offset1 = (sum(Z) - scale1*sum(q)) / n
```

然后：

1. 只有 `denom > 0`、`new_scale > 1e-6` 且 new scale/offset 均 finite 时接受
   LS 参数，否则保留初始参数；
2. 将 `offset1` 执行 `.half().float()`；
3. 将 `scale1` 执行 `.clamp(min=1e-4).half().float()`；
4. 用 round-trip 后参数重新编码 `q_1`；
5. 重建 `Z_1`；
6. 对**传入 group 的原始 normalized `Z_tile`**分别计算
   `SSE1=sum((Z_1-Z_tile)^2)` 与 `SSE0=sum((Z_0-Z_tile)^2)`；
7. 仅当 `SSE1 <= SSE0` 时选择候选；否则回退到 baseline；
8. fallback 必须选择整套 `(q, offset, scale)` 表示，而不只是选择一个 dense
   reconstruction，保证未来 packed representation 语义一致。

禁止：

- 迭代到收敛；
- 默认做 2 次或更多 refit；
- 用全 tensor 总 SSE 代替 per-tile fallback；
- 在未更新 algo version 的情况下改变舍入顺序。

### 6.6 Step E：inverse affine 与 inverse RHT

```text
V_r_hat = Z_hat * rms + mu
V_hat_0 = inverse_RHT(V_r_hat)
```

### 6.7 Step F：prompt per-channel reconstruction-bias correction

在 prompt 的完整量化 blocks 上计算：

```text
bias = mean_token(V_hat_0 - original_V.half().float()).half().float()
```

形状：

```text
bias: [B, H, 1, D]
```

即 bias 在 inverse-RHT 后、最终 output half-roundtrip 前计算。保存为 FP16，
最终输出：

```text
V_hat = (V_hat_0 - bias).half()
```

decode 的新 full tile 必须复用 prompt 初始化的 `mu/rms/bias`，仅重新计算该 tile 的 offset/scale/code。stats 一旦用于第一个量化 block，之后不得重校准，否则旧块和新块将处于不同隐式变换定义中。

### 6.8 理论 bit accounting

量化区内：

- code：2 bit/value；
- 每个 `16*C` tile 有两个 FP16 参数：32 bit；
- 每个 prompt/head/channel 有 `mu/rms/bias` 三个 FP16 参数：48 bit/channel。

当 `T_quantized > 0` 且 stats 已初始化时，量化区近似：

```text
b_tile = 2 + 32/(16*C) + 48/T_quantized
```

当 `T_quantized == 0` 时，不创建或保留 `mu/rms/bias`，metadata bits 为0；
此时整个 cache 保持 FP16，禁止计算 `48/T_quantized`。当前状态机与 crop
规则必须维持：`stats_initialized` 与“cache 中至少保留一个完整量化 tile”
等价。

32k prompt 下，忽略极小 metadata 项前：

| C | tile side info | 量化区约 bit/value |
| ---: | ---: | ---: |
| 16 | 0.12500 | 2.1265 左右 |
| 32 | 0.06250 | 2.0640 左右 |
| 64 | 0.03125 | 2.0327 左右 |

完整 cache bit accounting 必须额外计入：

- sink32 FP16；
- recent128 FP16；
- 最多 15 个 pending FP16 token；
- prompt metadata。

定义 `T_fp16 = T_total - T_quantized`，从而短 prompt 下 sink/recent 重叠不
会被重复计数。每个 head 的完整 tile 公式应实现成纯函数并测试：

```text
total_bits =
    16 * T_fp16 * D
  + 2 * T_quantized * D
  + 32 * (T_quantized * D / (16*C))
  + (48 * D if stats_initialized else 0)

bits_per_value = total_bits / (T_total * D)
```

对应的 per-token2 完整 cache 公式为：

```text
total_bits_pt2 =
    16 * T_fp16 * D
  + 2 * T_quantized * D
  + 32 * T_quantized

bits_per_value_pt2 = total_bits_pt2 / (T_total * D)
```

PT2 没有 prompt-level `mu/rms/bias` metadata；每个已量化 token/head 只有一
对 FP16 min/scale。

---

## 7. Cache 状态与生命周期设计

### 7.1 新增 per-layer 状态

在 `KittyKVCache` 中为 tile V 添加独立于 K 的字典/状态：

```text
v_tile_quant_end[layer_idx]   # 下一个尚未量化的 settled token 绝对位置
v_pc_mean[layer_idx]          # [B,H,1,D], FP16
v_pc_rms[layer_idx]           # [B,H,1,D], FP16
v_error_bias[layer_idx]       # [B,H,1,D], FP16
```

不要复用：

- `k_pt_quant_end`；
- K 的 `k_pc_mean`；
- 全局 `group_size`；
- 其他 layer 的 V stats。

### 7.2 建议的方法拆分

将当前 `_quant_v()` 拆成小而可测的方法：

```text
_quant_v_pertoken(value_slice)
_ensure_v_tile_config(value_slice)
_init_v_tile_prompt_stats(layer_idx, value_slice)
_quant_v_tile_blocks(layer_idx, value_slice)
_quant_v(layer_idx, value_slice)  # dispatcher
```

数学 helper 放在 `src/kitty_sim/utils_quant.py` 或新的小型 `v_tile_quant.py` 中；cache 方法只负责状态、区间和调用，不重复数学逻辑。

### 7.3 Batch 语义

虽然当前 LongBench 常用 batch=1，正式实现不能默默只保存 batch0 的 stats。第一版支持边界冻结为：

```text
batch=1；或等长 dense batch / beam expansion，所有样本共享同一 cache length、
settled frontier 与有效-token mask 语义
```

不同有效 prompt 长度、含不一致 padding frontier 的 batch 暂不支持并应
fail-fast；完整支持它需要 per-sample pointer 与 valid-token mask，不能用单个
scalar `v_tile_quant_end` 假装正确。stats 仍按样本保存：

```text
mu/rms/bias: [B,H,1,D]
```

如某个 cache API 操作改变 batch 顺序或大小，必须同步变换 stats：

- `reorder_cache()`；
- `batch_repeat_interleave()`；
- `batch_select_indices()`。

### 7.4 `reset()`

`reset()` 必须清除：

- key/value tensors；
- `v_tile_quant_end`；
- `v_pc_mean`；
- `v_pc_rms`；
- `v_error_bias`；
- engagement counters。

否则 prompt A 的 stats 会污染 prompt B，尤其会影响 GLM 或复用 cache 的多样本路径。

### 7.5 `crop()` 与 cache 变换

实现前必须给出明确行为并测试：

- 若 `new_len >= v_tile_quant_end`，crop 只裁 recent/pending，保留 pointer/stats；
- 若 `new_len <= sink`，清除 `mu/rms/bias`、清除/重置 pointer，状态回到
  `UNINITIALIZED`；不得对 `new_len-sink` 做 modulo；
- 若 `sink < new_len < v_tile_quant_end` 且 `(new_len-sink) % 16 == 0`，可把 pointer
  设为 `new_len`，因为保留下来的已量化区仍由完整 tile 组成；
- 若 `sink < new_len < v_tile_quant_end` 且切入已量化 tile 内部，第一版必须
  fail-fast；剩余半 tile 已按旧完整 tile 量化，简单回退 pointer 会导致未来
  double-quant；
- stats 初始化后保持冻结；crop 不触发重新 calibration；
- 不采用“无条件回退到最近边界”的模糊行为。

---

## 8. Prefill / decode 严格 16-token 调度

### 8.1 共同符号

```text
sink = 32
recent = buffer_length = 128
tile_tokens = 16
L = 当前 cache 总长度
```

当前可进入量化区的最右边界：

```python
ready_end = max(sink, L - recent)
```

### 8.2 Prefill

只量化完整 tile：

```python
start = sink
aligned_end = start + ((ready_end - start) // 16) * 16
```

处理规则：

1. 若 `aligned_end == start`：不初始化 tile stats，不量化；
2. 若存在完整块：
   - calibration region 为 `[start, aligned_end)`；
   - 初始化 `mu/rms`；
   - 量化所有完整 tile；
   - 从这些重建误差初始化 `bias`；
   - 应用 bias correction；
   - 设置 `v_tile_quant_end[layer]=aligned_end`；
3. `[aligned_end, ready_end)` 不足 16 的 settled tail 保持 FP16；
4. `[ready_end, L)` recent window 保持 FP16；
5. `[0, sink)` 保持 FP16。

### 8.3 Lazy initialization

短 prefill 可能没有完整量化块。此时：

- 不伪造 `mu/rms/bias`；
- 不推进 pointer；
- 随 decode 首次积累出完整 settled 16-token block 时，用该 block 初始化 stats；
- stats 初始化后永久冻结。

状态机显式为：

```text
UNINITIALIZED
  -- first full block [qend,qend+16) -->
CALIBRATED(mu,rms,bias)
```

如果一次 multi-token append 首次释放多个 full blocks，只允许用第一个
`[qend,qend+16)` 执行一个原子操作：

```text
recon, mu, rms, bias = calibrate_and_quantize_first_block(first_block)
write recon exactly once
state = CALIBRATED(mu,rms,bias)
qend += 16 exactly once
```

其中 first block 先以 `mu/rms` 得到无 bias reconstruction，再用该 block 的
重建误差计算 bias，最后对同一 reconstruction 应用 bias 后写回。初始化函数
不能先推进 pointer，也不能让外层再次量化首块。同一次 append 的其余 blocks
立即复用冻结 stats。任何时点都必须满足：pointer 已前进 ⇒ stats 已存在。

该行为可能使极短 prompt 的 stats 仅来自首个16-token block，属于已知风险，必须在短 prompt 单测和 smoke 中记录；不能在后续不断重算 stats。

### 8.4 Decode

每次 append 后：

```python
ready_end = max(sink, L - recent)
qend = v_tile_quant_end.get(layer_idx, sink)

if state is UNINITIALIZED and ready_end - qend >= 16:
    recon, stats = calibrate_and_quantize_first_block(V[..., qend:qend+16, :])
    write recon
    state = CALIBRATED(stats)
    qend += 16

while state is CALIBRATED and ready_end - qend >= 16:
    recon = quantize_with_frozen_stats(V[..., qend:qend+16, :])
    write recon
    qend += 16
```

这样可以：

- 单-token decode 每积累16个 settled token 批量量化一次；
- multi-token append 时一次处理多个 full blocks；
- 永远不量化 partial tile；
- 最多额外保留15个 FP16 pending token；
- 不重复量化旧 block。

### 8.5 PostQuant 语义

现有 `PostQuant=True` 会在 mutation 前 clone 返回值。新路径必须保持：

- 当前 attention step 读到 append 后、量化前的 V；
- cache 内刚刚 settled 的完整 block 被量化；
- 量化结果从下一 attention step 开始影响输出。

### 8.6 多 token update

现有单-token slice 逻辑不能直接假设每次 `T_new=1`。第一版二选一：

1. 推荐：按上面的 `while` 正确支持 `T_new>=1`；
2. 若实现复杂，则在首次非-prefill且 `T_new>1` 时显式 fail-fast。

禁止只量化 multi-token append 中的一个 token 而无报错。

---

## 9. 代码改动地图（逐文件）

### 9.1 `src/kitty_sim/utils_quant.py`

计划新增/调整：

1. 保留现有 `fake_quant_groupwise_lastdim()` 行为不变；
2. 新增 V PT2 薄 wrapper，强制 `group_size=D`；
3. 新增固定 RHT/FWHT helper；
4. 新增 prompt per-channel affine stats helper；
5. 新增 2-bit tile min-max helper；
6. 新增 exactly-one-step LS/MSE refit helper；
7. 新增 per-tile SSE fallback；
8. 新增 rescued tile encode/decode fake-quant helper；
9. 新增理论 bit accounting 纯函数；
10. 所有 helper 必须处理 constant/degenerate/non-finite 风险。

完成门槛：只运行 core 单元测试时无需构造完整 model/cache。

### 9.2 `src/kitty_sim/kitty_simulate.py`

计划新增/调整：

1. 扩展 `KittyKVCacheConfig`；
2. 扩展 `validate()`；
3. 在 `KittyKVCache.__init__` 初始化 V tile state；
4. 将 `_quant_v` 改为接收 `layer_idx`；
5. 实现 per-token2 与 tile dispatcher；
6. 在 prefill 分支接入完整16-token block；
7. 在 decode 分支接入 pointer/while 调度；
8. 保持 V4 和 q4_0 路径不变；
9. 扩展 `reset()`；
10. 扩展 batch/cache transform 方法；
11. 添加 engagement counters；
12. 扩展 `get_kvcache_kitty()` 参数传播。

### 9.3 `src/kitty_sim/longbench/runner.py`

计划新增/调整：

1. 扩展 `VariantConfig` 字段和 tag；
2. 在 `build_variant()` 添加四个分支；
3. sign 新变体硬锁 `bin_codebooks=("sign",)`，禁止 `QLUT_BIN_CODEBOOKS=tern` 造成错误命名；
4. SNF 新变体复用现有 `QLUT_CB_MASK` 加载与存在性检查；
5. V PT2 分支固定 `vbits=2`、`v_codebook="kivi"`；
6. tile 分支解析 `C`，固定完整算法常量；
7. `_cache_factory()` 传播全部新字段；
8. `method_layout_slug()` 添加四个映射并编码 C；
9. 模型加载后根据 config 推导 `head_dim` 做 early validation；
10. 首样本 guard 验证 V path engagement；
11. report/manifest 写入 V mode、C、量化 token/block 计数；
12. 抽出可被 runner 与 preflight 共用的 canonical variant/slug/hash resolver；
13. GLM 未支持前显式拒绝四个新变体。

### 9.4 `src/kitty_sim/cli/eval_longbench.py`

计划新增/调整：

1. `--variant` choices 添加 canonical 名称；
2. 添加 hyphen aliases；
3. 确保 direct Python CLI 可接收 `--v-tile-channels`；
4. parser 测试覆盖合法/非法组合。

同时新增无 GPU preflight CLI（可为独立文件或此入口的
`--resolve-config-only --json` 模式），但其解析逻辑必须调用 runner 共用 resolver，
不能复制一份 variant 表。

### 9.5 CLI 参数落点

第一版只在 `src/kitty_sim/cli/eval_longbench.py` 对应 parser 增加：

```text
--v-tile-channels N
```

默认 `None`。不要放进所有入口共用的 `utils_cli.py`，否则
`eval_kitty.py/custom` 会暴露一个没有 tile strategy 接线的无效参数。只有后续
为这些入口定义完整 tile 语义后，才下沉到公共 CLI helper。解析优先级：

```text
CLI > 显式环境变量 V_TILE_CHANNELS > 缺失并报错
```

### 9.6 `scripts/run_exp.sh`

计划新增/调整：

1. 读取 `V_TILE_CHANNELS`；
2. 可选接受 `--v-tile-channels N`；
3. 在一个函数中按 CLI > env 解析唯一的 `resolved_v_tile_channels`；
4. shell slug 与传给 Python 的 `--v-tile-channels` 都消费这个 resolved 值，
   禁止各自重新解析；
5. 在解析输出目录前验证正整数；
6. tile variant 缺失 C 时 fail-fast；
7. non-tile variant 检测 stale C；
8. 新命名变体检测冲突 `VBITS`；
9. 调用无 GPU Python preflight 取得 canonical `-vtile16c${C}-rv1` slug；
10. 删除 shell 中重复的正式 slug 拼接；如保留函数，只能消费 preflight JSON；
11. 只对真正支持 K block 的变体追加 `-blkN`，且该决定也由 resolver 返回；
12. `prepare_dataset()` 用 preflight hash + manifest + row count 决定 skip；
13. 确保 smoke/full resume 不会跨 C/rv/hash 复用目录；
14. 提供可测试的 dry-run/print-slug 路径，使 canonical resolver 进入自动测试；
15. `bash -n` 必须通过。

### 9.7 `accuracy_simulation/env.sh`

审计并统一扩展显式环境变量 preserve list，至少覆盖本实验使用的：

```text
V_TILE_CHANNELS
PERTOKEN_BLOCK
VBITS
QLUT_CB_MASK
QLUT_BIN_CODEBOOKS
LLAMA32_MODEL_PATH
LLAMA32_MODEL_SLUG
LLAMA32_MAX_GEN
MAX_MODEL_LEN
MAX_SAMPLES
DATASETS_CSV
RUN_MODE
```

确保：

```text
命令行环境变量 > .env
```

preserve 逻辑必须区分“未设置”和“显式设置为空”；后者用于阻止 `.env` 重新
注入 stale C、sample scope 或 dataset scope。第一版不把该实验旋钮写入
`.env.example`；它应在每次实验命令中显式给出。tracked 文件中不得出现本机
模型、数据或 mask 路径。

### 9.8 `src/kitty_sim/glm_kitty_patch.py`

当前 GLM bridge 未完整传播已有 per-token K flags，也不会自动正确传播新 V tile fields。第一版建议：

1. 在 runner 中对 GLM + 四个新变体 fail-fast；
2. 错误消息明确说明 GLM parity 尚未实现；
3. 单测证明不会静默退回普通 KIVI V；
4. 后续独立 phase 再补全 GLM cache config parity 和 GLM smoke。

不要在缺少测试时宣称 GLM 已支持。

### 9.9 Tests

建议拆分：

```text
tests/test_vcache_2bit_core.py
tests/test_vcache_2bit_schedule.py
tests/test_vcache_2bit_wiring.py
```

并扩展：

```text
tests/test_longbench.py
tests/test_no_hardcoded_paths.py（仅运行，不应需要修改）
```

---

## 10. Engagement evidence 与可观测性

仅检查 `KittyKVCache.get_seq_length()>0` 无法证明新 V path 真正执行。必须增加只读计数器：

```text
v_quant_calls
v_quantized_tokens
v_tile_blocks
last_v_quant_mode
last_v_tile_channels
```

要求：

- V PT2 长 prompt：`v_quant_calls>0`、mode=`per_token2`；
- tile 长 prompt：`v_tile_blocks>0`、mode=`tile16_rescued`、C 一致；
- 每个 worker 首样本打印一次明确 evidence；
- report 最好记录计数器；
- manifest 记录请求配置；
- 配置声称 tile、计数器却为0时，长上下文 smoke 必须失败；
- 短 prompt 无 full tile 时允许 block=0，但需要明确 reason，而不能误判为 fallback。


---

## 11. 0-GPU 单元测试计划

### 11.1 Core：V per-token2

#### T-PT2-01：与现有 groupwise helper bit-exact

输入覆盖：

```text
[1,2,7,64]
[2,3,5,128]
random / zero / constant / tiny range / large range
FP16
```

比较：

```python
expected = fake_quant_groupwise_lastdim(x, x.shape[-1], 2)
torch.testing.assert_close(got, expected, atol=0, rtol=0)
```

目的：证明新增 PT2 variant 只是把 V 从4改为2，没有改变现有 per-token 算法。

#### T-PT2-02：D64 whole-head row

确认 D=64 形成一个64值组，不跳过、不拆分。

#### T-PT2-03：D128 whole-head row

确认 D=128 形成一个128值组。

#### T-PT2-04：旧 V4 bit-exact regression

固定 tensor，旧 sign/SNF V4 在改动前后 bit-exact。任何非预期变化均为 merge blocker。

### 11.2 Core：rescued tile oracle

#### T-TILE-01：literal reference

测试中编写独立、循环式 reference；禁止调用生产 helper 或 import probe。参数化：

```text
D64:  C16/C32/C64
D128: C16/C32/C64/C128
B>1, H>1
```

reference 按顺序执行：

1. fixed RHT；
2. prompt `mu/rms`；
3. normalize；
4. reshape 成 `[16,C]`；
5. min-max 初始 code；
6. 一次 closed-form offset/scale LS；
7. offset/scale FP16 round-trip；
8. 重新编码；
9. per-tile SSE fallback；
10. inverse affine；
11. inverse RHT；
12. prompt error-bias correction；
13. 最终 FP16。

若正式实现固定了浮点运算顺序，应做到 bit-exact；若因并行 reduction 无法 bit-exact，应给每个阶段固定严格 tolerance，不能只检查最终“大致接近”。

#### T-TILE-02：exactly one LS step

构造 iter1 与 iter2 输出不同的固定 tensor，断言生产实现等于 iter1，不等于 iter2。
另构造 `denom=0`、`new_scale<=1e-6` 与 scale clamp 生效的 tensor，锁定 valid 条件。

#### T-TILE-03：fallback 单调性

对每个 tile 检查：

```text
SSE(chosen) <= SSE(initial minmax)
```

不能只比较整个 tensor 的总 SSE。

#### T-TILE-04：degenerate groups

覆盖：

- 全零；
- 每 tile 常量但 tile 间不同；
- LS denominator 为0；
- 极小 scale；
- 接近 FP16 上限；
- 正负强偏态；
- 含单个异常值。

要求无 NaN/Inf，常量 tile 可稳定重建。

#### T-TILE-05：RHT determinism

- 固定 seed 每次生成相同 signs；
- CPU 与 CUDA 不改变 signs 序列；
- `FWHT(FWHT(x))` 与输入在固定 tolerance 内一致；
- normalization 和 FP16 round-trip 位置锁定；
- seed 变化会改变 config hash（即使第一版不公开 seed 旋钮）。

#### T-TILE-06：stats shape/dtype

验证：

```text
mu/rms/bias = [B,H,1,D]
storage dtype = FP16
batch 样本分别校准
```

#### T-TILE-07：protected region exclusion

只改变 sink/recent/pending 的值，完整量化中段的 `mu/rms/bias` 与输出不得改变。

#### T-TILE-08：channel tile layout

用每个 channel 都可唯一识别的确定性 tensor，验证 contiguous channel 分块顺序；防止 `permute/reshape` 错把不同 head、token block 或 channel block 混在一起。

#### T-TILE-09：C 等于 D

对 C=D 验证每个16-token块形成一个 `16×D` tile，且不会错误退化成 per-token 分组。

#### T-TILE-10：C 等于1

验证每个 channel 独立在16 token上量化；用于验证最细粒度边界，而不是首轮 LongBench 主点。

### 11.3 参数校验

#### T-C-01：合法 C

```text
D64:  C1/2/4/8/16/32/64
D128: C16/32/64/128
```

首轮 LongBench 只扫16/32/64，但 helper 应支持任何正 divisor。

#### T-C-02：非法 C

```text
C=0
C<0
C>D
C=24 for D64
non-integer env
empty env
missing C on tile variant
stale C on non-tile variant
```

`D=64,C=24` 必须抛出用户要求的 `AssertionError`，错误消息包含：

```text
head_dim=64
v_tile_channels=24
```

#### T-C-03：RHT 独立错误

`D=48,C=16` 虽满足整除，但 D 非2的幂，必须给独立清晰错误。

#### T-C-04：配置层与 tensor 层双重校验

- 构造模型 config 时尽量提前报 `ValueError`；
- 绕过 config 直接调用量化 helper 时仍由真实 tensor `assert` 拦截；
- 在 `python -O` 语义下，显式校验仍然存在。

#### T-C-05：dtype 边界

第一版新命名 variant 对非 FP16 model/cache dtype fail-fast；旧变体 dtype 行为不
受影响。未来如支持 BF16，必须新增独立 variant/algo version 或明确内部 FP16
算术、输出 cast 和全套 oracle，不能静默复用 rv1 结果。

### 11.4 Bit accounting

#### T-BIT-01：PT2

断言：

```text
D64  = 2.5
D128 = 2.25
```

#### T-BIT-02：tile quantized region

断言公式：

```text
2 + 32/(16*C) + 48/Tq
```

并覆盖 C16/C32/C64、不同 Tq。

#### T-BIT-03：full cache

计入 sink/recent/pending/metadata，并覆盖：

- `Tq=0`；
- 只有一个完整 tile；
- 短 prompt；
- 32k prompt；
- pending=0 与 pending=15；
- D64/D128。

`Tq=0` 时断言 metadata bits=0、结果=16 bit/value，且不发生除零。

#### T-BIT-04：明确 theoretical vs actual

报告 helper 输出字段必须命名为 `theoretical_packed_bits`，不得暗示 dense fake-quant tensor 的真实分配已下降。

---

## 12. Cache 调度测试计划

### 12.1 T-SCHED-01：短 prefill

长度覆盖：

```text
T <= sink + recent
T = sink + recent + 1
T = sink + recent + 15
```

期望：没有完整 settled tile 时，V 保持 FP16，pointer 不前进，stats 不伪造。

### 12.2 T-SCHED-02：prefill 只量化完整块

示例：

```text
sink=4
recent=32
T=sink+recent+37
```

期望：

- 32 settled token 量化为两个16块；
- 余5个 settled pending token 保持 FP16；
- sink/recent 原样；
- pointer=`sink+32`；
- C16/C32/C64 分别与 literal oracle 对齐。

### 12.3 T-SCHED-03：decode 等待凑满

沿用5个 pending：

- 再 decode 10步：无新 tile；
- 第11步：恰好量化一个16-token tile；
- pointer 前进16；
- recent/newest 不提前量化。

### 12.4 T-SCHED-04：prefill + decode manual oracle

手工冻结 prompt stats，逐块调用 literal reference，与 cache 最终内容比对。禁止用“最终全序列重新校准后的一次性量化”作为 oracle，因为 decode 必须复用 prefill stats。

### 12.5 T-SCHED-05：PostQuant

断言 `update()` 当前返回值是 mutation 前的 KV；cache 内 settled block 是 mutation 后结果。

### 12.6 T-SCHED-06：PT2 每步滑出一个 token

PT2 继续执行现有 KIVI-style 单-token flush，不等待16。

### 12.7 T-SCHED-07：multi-layer independence

layer0/layer1 使用不同分布：

- stats 分别保存；
- pointer 分别前进；
- counters 可按 layer 或全局明确累计；
- 不得跨层复用。

### 12.8 T-SCHED-08：reset

prompt A → reset → prompt B 的结果必须等于全新 cache 直接处理 B；所有 V state/counters 清空。

### 12.9 T-SCHED-09：no requantization

保存已量化旧块，多次 decode 后旧块逐 bit 不变，防止反复 fake-quant 累积误差。

### 12.10 T-SCHED-10：multi-token append

一次 append 跨过多个16-token边界，必须全部按块处理或明确 fail-fast，不得只处理一个 slice。

### 12.11 T-SCHED-11：batch/beam/crop

分别验证：

- `reorder_cache`；
- `batch_repeat_interleave`；
- `batch_select_indices`；
- aligned crop；
- unaligned crop 的文档化 fail-fast/处理。

### 12.12 T-SCHED-12：lazy calibration

无完整块的 prefill 后，经 decode 首次凑满16 settled token：

- stats 只初始化一次；
- 首个 block 先生成无-bias reconstruction、计算 bias、应用 bias，并且只写回/推进一次；
- 后续 block 复用 stats；
- stats tensor device/dtype 与 cache 一致。

### 12.13 T-SCHED-13：保护窗口不变量

对每次 prefill/decode 检查：

```text
[0,sink) FP16
[ready_end,L) recent FP16
[quant_end,ready_end) pending FP16，长度 < 16
```

---

## 13. Variant / CLI / slug / manifest 测试

### 13.1 T-WIRE-01：父 K 配置完全继承

对 sign parent V4、PT2、tile2 逐字段比较：

```text
k_codebook
bin_codebooks
n_bins
k_quant_mode
pertoken_pc_submean
pertoken_rotate
pertoken_cb_mask
pertoken_block
sink_length
buffer_length
group_size
```

除 V 字段外必须一致。

SNF 用临时合法 mask 比较 parent/new configs；mask 缺失时所有 SNF 新变体必须 fail-fast。

### 13.2 T-WIRE-02：sign 锁定

即使外部残留 `QLUT_BIN_CODEBOOKS=tern`，命名为 k125/sign 的新 variant 也不能静默变成 tern；推荐直接拒绝冲突环境变量。

### 13.3 T-WIRE-03：cache factory

逐项验证：

```text
vbits=2
v_codebook
v_tile_tokens
v_tile_channels
algo version
RHT seed
MSE iters
```

以及所有 K fields。

### 13.4 T-WIRE-04：CLI 与 aliases

canonical 和 hyphen alias 都可解析；未知名称继续拒绝。

### 13.5 T-WIRE-05：slug 唯一且 shell/Python 一致

对 sign/SNF 分别构造：

```text
V4 parent
V PT2
tile C16
tile C32
tile C64
```

所有 slug 唯一；shell 输出目录必须直接采用 Python preflight 返回值，而不是
重新拼接。测试比较 preflight、direct runner 与最终 shell dry-run 的结果完全相等。

参数化覆盖无后缀、`-blkN`、QUEST，以及三者组合；完整顺序必须为：

```text
-vtile16c<C>-rv<V>-blk<N>-quest-<mode>
```

shell 必须有可测试的 dry-run/print-slug 路径；canonical mapping 只有 Python
resolver 一个来源，不靠人工比对两份表。

### 13.6 T-WIRE-06：manifest/hash

fresh manifest 至少包含：

```json
{
  "variant": {
    "vbits": 2,
    "v_codebook": "tile16_rescued",
    "v_tile_tokens": 16,
    "v_tile_channels": 16,
    "v_tile_algo_version": "rht-pcaff-mse1-bias-v1",
    "v_rht_seed": 20260711,
    "v_mse_iters": 1
  },
  "variant_semantic_hash": "...",
  "run_config_hash": "...",
  "mask_sha256": "... for SNF ..."
}
```

断言 PT2/tile、C16/C32/C64、sign/SNF 的 config hash 均正确区分。
另断言时间、GPU、输出路径变化不改变 stable hash；同一路径的 mask 内容变化
必须改变 semantic/run hash；completed fast-path 必须保留完整 manifest。

### 13.7 T-WIRE-07：GLM fail-fast

第一版 GLM 请求新 variant 时必须在运行前报错，不能生成看似合法但实际 fallback 的输出。

### 13.8 T-WIRE-08：路径卫生

新 source、tests、docs 不得包含本机绝对路径；模型、mask、LongBench 路径只来自 `.env` 或显式运行环境。

### 13.9 T-WIRE-09：冲突环境变量

参数化验证：

- named V2 + `VBITS=4`；
- non-tile + `V_TILE_CHANNELS`；
- tile + missing C；
- sign + `PERTOKEN_BLOCK>1`；
- SNF + `PERTOKEN_BLOCK>1`；
- SNF + missing mask；
- sign + conflicting codebook。

每种情况都必须有稳定错误或明确支持行为，不能依赖无声 ignore。

### 13.10 T-WIRE-10：resume/output collision

构造相同模型、不同 C 的 dry run，确认：

- base_dir 不同；
- pred/logs 不同；
- C16 的完成状态不会使 C32 被 skip；
- smoke 清理只清自己的 C 目录；
- full resume 只补当前配置缺失 dataset。

### 13.11 T-WIRE-11：preflight/worker hash handoff

- preflight 不创建 CUDA context、不加载权重；
- 相同语义重复运行输出相同 canonical JSON/hash；
- created_at、GPU、output path 不影响 hash；
- dataset/max-gen/template 改变会改变对应 run hash；
- shell 将篡改后的 expected hash 传给 worker 时，worker fail-fast；
- complete rows + matching hash 才 skip；
- complete rows + missing/mismatched hash 必须拒绝复用；
- completed fast-path 保留完整 manifest。

### 13.12 T-WIRE-12：legacy isolation

dry-run 证明 study slug
`llama32-1b-instruct-vtile-study-rv1` 不会触碰现有
`llama32-1b-instruct_*` 目录；第一版不执行自动 manifest backfill。

---

## 14. 实施阶段与逐步门禁

### Phase A：冻结规范与 baseline

- [ ] 确认四个 variant 名称与 aliases；
- [ ] 确认 tile 正式语义为 rescued pipeline；
- [ ] 确认 token tile 永久固定16；
- [ ] 确认 `V_TILE_CHANNELS` 为必填且无默认；
- [ ] 确认固定 RHT seed=`20260711`；
- [ ] 确认 MSE refit 次数=`1`；
- [ ] 记录旧 sign/SNF V4 固定 tensor 输出用于 regression；
- [ ] 记录 probe 算法常量、FP16 round-trip 顺序；
- [ ] 确认第一版 GLM fail-fast；
- [ ] 不修改任何默认推荐 variant；
- [ ] 不把 probe 脚本变成生产依赖。

**Gate A：**上述语义在 code review 前不得变化；若变化，先更新本计划和 algo version。

### Phase B：量化数学 core

- [ ] 实现 PT2 whole-head wrapper；
- [ ] 实现固定 RHT；
- [ ] 实现 per-channel prompt affine；
- [ ] 实现 tile reshape/grouping；
- [ ] 实现 min-max 2-bit 初始化；
- [ ] 实现一次 LS refit；
- [ ] 实现 FP16 side parameter round-trip；
- [ ] 实现 per-tile fallback；
- [ ] 实现 inverse affine/RHT；
- [ ] 实现 bias correction；
- [ ] 实现 bit accounting；
- [ ] 完成 core/literal oracle tests；
- [ ] 确认没有 cache/runner 依赖。

**Gate B：**core tests 全过；旧 quant helper 未改；constant/degenerate 无 NaN/Inf。

### Phase C：cache config、state 与调度

- [ ] 扩展 `KittyKVCacheConfig`；
- [ ] 扩展 validate；
- [ ] 初始化 per-layer V state；
- [ ] 接入 prefill full-block 调度；
- [ ] 接入 decode pointer/while 调度；
- [ ] 接入 lazy stats；
- [ ] 保持 PostQuant；
- [ ] 支持或拒绝 multi-token append；
- [ ] 扩展 reset；
- [ ] 扩展 batch/beam/crop；
- [ ] 添加 engagement counters；
- [ ] 完成 schedule tests。

**Gate C：**prefill/decode manual oracle 通过；旧 V4/Q4_0 bit-exact；无重复量化。

### Phase D：runner、CLI、shell 与 provenance

- [ ] 添加四个 variant；
- [ ] sign 锁定 sign；
- [ ] SNF 复用 mask 检查；
- [ ] 添加 `--v-tile-channels`；
- [ ] 实现 env/CLI precedence；
- [ ] 实现三层 C validation；
- [ ] 实现 shared canonical resolver；
- [ ] 实现无 GPU Python preflight；
- [ ] shell 只消费 preflight slug/hash；
- [ ] manifest-aware resume；
- [ ] legacy study slug 隔离；
- [ ] tag；
- [ ] manifest/hash；
- [ ] report engagement；
- [ ] GLM fail-fast；
- [ ] `bash -n`；
- [ ] wiring tests。

**Gate D：**不同 C 不碰目录，manifest 与实际 path 完全一致，错误配置启动前失败。

### Phase E：0-GPU 回归

未来实现后按顺序执行；本规划轮不执行：

```bash
cd /path/to/Kitty
PYTHONPATH=src python -m unittest \
  tests.test_vcache_2bit_core \
  tests.test_vcache_2bit_schedule \
  tests.test_vcache_2bit_wiring -v
```

```bash
cd /path/to/Kitty
PYTHONPATH=src python -m unittest \
  tests.test_q4_0_fakequant \
  tests.test_kitty_pertoken_smooth \
  tests.test_longbench \
  tests.test_no_hardcoded_paths -v
bash -n scripts/run_exp.sh
```

```bash
cd /path/to/Kitty
PYTHONPATH=src python -m unittest discover -s tests -v
```

**Gate E：**全量通过、无新 unexpected skip、无 host path、无 NaN/Inf。

### Phase F：真实 dump replay

- [ ] 使用相同真实 V dump；
- [ ] 先让 replay probe 采用生产 strict full-block region，partial tail 保持 FP16；
- [ ] 对齐新的 strict calibration region；
- [ ] 对齐 RHT seed/signs；
- [ ] 对齐 FP16 round-trip；
- [ ] 对齐 tile reshape；
- [ ] 对齐 one-step refit/fallback；
- [ ] 对齐 bias；
- [ ] 比较 C16/C32/C64；
- [ ] 检查生产 helper 与 probe bit-exact 或严格 tolerance；
- [ ] 若不一致，禁止进入 LongBench。

**Gate F：**生产实现与 strict replay 对齐；旧六文档比例只检查方向，不要求
对包含 partial tail 的旧 aggregate bit-exact。

### Phase G：GPU micro integration

对八个新配置各跑一个长 prompt：

```text
sign PT2
sign tile C16/C32/C64
SNF PT2
SNF tile C16/C32/C64
```

检查：

- model/mask load；
- V path engaged；
- prefill 产生 tile；
- decode 至少跨过一次16-token flush；
- reset 后第二样本无状态泄漏；
- cache 无 NaN/Inf；
- 峰值显存；
- quant wall time；
- decode 每16步 latency pulse。

**Gate G：**确认单 worker 显存后再决定并发数，不沿用旧 3 workers/card 假设。

### Phase H：LongBench smoke

所有 LongBench 必须走 `scripts/run_exp.sh`。数据集固定：

```text
multifieldqa_en,gov_report
```

每个新 variant 2 samples，32k，max-gen256。检查 manifest、report、engagement、slug 与输出完整性。

### Phase I：LongBench full

完成 11 点矩阵：

```text
FP16
+ sign: V4 parent / PT2 / tile C16/C32/C64
+ SNF: V4 parent / PT2 / tile C16/C32/C64
```

公平条件：

- Llama-3.2-1B；
- 21 datasets；
- 32k；
- max-gen256；
- SNF 全部使用同一个 f=0.5 mask；
- K `PERTOKEN_BLOCK=1`；
- sink32/recent128；
- 同一代码 commit、模型、数据；
- 只读取当前 `longbench_out/`。

### Phase J：结果分析与文档

- [ ] 汇总 21-task mean；
- [ ] 汇总 per-dataset delta；
- [ ] 分类别汇总；
- [ ] 计算相对各自 V4 parent 的 retention；
- [ ] 比较 tile 相对 PT2；
- [ ] 填写理论 V bits；
- [ ] 记录 wall time/peak memory；
- [ ] 记录失败/重跑/config hash；
- [ ] 在 full 完成前不宣布 winner；
- [ ] 实验记录写入“测试笔记”；
- [ ] 方法分析和最终结论写入“研究笔记”。

---

## 15. 未来 LongBench smoke 命令（实现后执行）

> 以下命令是未来 rollout 规范，本规划轮不执行。tracked 文档使用通用路径；运行前替换 `/path/to/...`。

### 15.1 sign：PT2 + tile C16/C32/C64

```bash
cd /path/to/Kitty

# sign + V per-token 2-bit
V_TILE_CHANNELS= \
  VBITS=2 \
  QLUT_BIN_CODEBOOKS=sign \
  PERTOKEN_BLOCK=1 \
  RUN_MODE=smoke \
  MAX_SAMPLES=2 \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
  LONGBENCH_DATA_ROOT=/path/to/LongBench \
  MAX_MODEL_LEN=32768 \
  LLAMA32_MAX_GEN=256 \
  DATASETS_CSV=multifieldqa_en,gov_report \
  bash scripts/run_exp.sh llama32 --gpu 1 \
    --variant qlutattn_k125v2_pt --max-samples 2

# sign + rescued V tile16cC 2-bit
for c in 16 32 64; do
  V_TILE_CHANNELS="$c" \
  VBITS=2 \
  QLUT_BIN_CODEBOOKS=sign \
  PERTOKEN_BLOCK=1 \
  RUN_MODE=smoke \
  MAX_SAMPLES=2 \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
  LONGBENCH_DATA_ROOT=/path/to/LongBench \
  MAX_MODEL_LEN=32768 \
  LLAMA32_MAX_GEN=256 \
  DATASETS_CSV=multifieldqa_en,gov_report \
  bash scripts/run_exp.sh llama32 --gpu 1 \
    --variant qlutattn_k125v2_pt_vtile16 --max-samples 2
done
```

预期目录：

```text
longbench_out/smoke/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v2-pt/
longbench_out/smoke/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v2-pt-vtile16c16-rv1/
longbench_out/smoke/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v2-pt-vtile16c32-rv1/
longbench_out/smoke/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v2-pt-vtile16c64-rv1/
```

### 15.2 SNF：PT2 + tile C16/C32/C64

```bash
cd /path/to/Kitty

# SNF + V per-token 2-bit；f=0.5 mask 示例
V_TILE_CHANNELS= \
  VBITS=2 \
  QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
  PERTOKEN_BLOCK=1 \
  RUN_MODE=smoke \
  MAX_SAMPLES=2 \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
  LONGBENCH_DATA_ROOT=/path/to/LongBench \
  MAX_MODEL_LEN=32768 \
  LLAMA32_MAX_GEN=256 \
  DATASETS_CSV=multifieldqa_en,gov_report \
  bash scripts/run_exp.sh llama32 --gpu 1 \
    --variant qlutattn_k188v2_pt --max-samples 2

# SNF + rescued V tile16cC 2-bit；同一个 f=0.5 mask
for c in 16 32 64; do
  V_TILE_CHANNELS="$c" \
  VBITS=2 \
  QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
  PERTOKEN_BLOCK=1 \
  RUN_MODE=smoke \
  MAX_SAMPLES=2 \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
  LONGBENCH_DATA_ROOT=/path/to/LongBench \
  MAX_MODEL_LEN=32768 \
  LLAMA32_MAX_GEN=256 \
  DATASETS_CSV=multifieldqa_en,gov_report \
  bash scripts/run_exp.sh llama32 --gpu 1 \
    --variant qlutattn_k188v2_pt_vtile16 --max-samples 2
done
```

预期目录：

```text
longbench_out/smoke/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v2-pt/
longbench_out/smoke/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v2-pt-vtile16c16-rv1/
longbench_out/smoke/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v2-pt-vtile16c32-rv1/
longbench_out/smoke/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v2-pt-vtile16c64-rv1/
```

### 15.3 每个 smoke 后的强制检查

```bash
cd /path/to/Kitty
find longbench_out/smoke -path '*qlutattn-k1*v2*' \
  \( -name 'result.json' -o -name '*.manifest.json' -o -name 'report_*.json' \) \
  -print | sort
```

逐目录确认：

- `status=ok`；
- `failed_sample_ids=[]`；
- manifest 的 V mode/C 正确；
- C16/C32/C64 目录不同；
- engagement counters 非零；
- 日志无 traceback/NaN；
- smoke 重跑只清理自身目录。

每个 smoke 的文件模式为：

```text
longbench_out/smoke/<model>_<method>/pred/<dataset>.jsonl
longbench_out/smoke/<model>_<method>/pred/<dataset>.manifest.json
longbench_out/smoke/<model>_<method>/pred/result.json
longbench_out/smoke/<model>_<method>/logs/report_<dataset>.json
```

---

## 16. 未来 LongBench full 命令（实现且 smoke 通过后执行）

### 16.1 FP16 与父 V4 baseline

```bash
cd /path/to/Kitty

# FP16 ceiling
V_TILE_CHANNELS= \
PERTOKEN_BLOCK=1 \
RUN_MODE=full \
MAX_SAMPLES=-1 \
DATASETS_CSV= \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
LONGBENCH_DATA_ROOT=/path/to/LongBench \
MAX_MODEL_LEN=32768 \
LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant fp16 --max-samples -1

# sign parent: K sign + V per-token 4-bit
V_TILE_CHANNELS= \
VBITS=4 \
QLUT_BIN_CODEBOOKS=sign \
PERTOKEN_BLOCK=1 \
RUN_MODE=full \
MAX_SAMPLES=-1 \
DATASETS_CSV= \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
LONGBENCH_DATA_ROOT=/path/to/LongBench \
MAX_MODEL_LEN=32768 \
LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_k125v4_pt --max-samples -1

# SNF parent: same f=0.5 mask + V per-token 4-bit
V_TILE_CHANNELS= \
VBITS=4 \
QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
PERTOKEN_BLOCK=1 \
RUN_MODE=full \
MAX_SAMPLES=-1 \
DATASETS_CSV= \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
LONGBENCH_DATA_ROOT=/path/to/LongBench \
MAX_MODEL_LEN=32768 \
LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_k188v4_pt --max-samples -1
```

### 16.2 sign full：PT2 + tile C16/C32/C64

```bash
cd /path/to/Kitty

V_TILE_CHANNELS= \
  VBITS=2 \
  QLUT_BIN_CODEBOOKS=sign \
  PERTOKEN_BLOCK=1 \
  RUN_MODE=full \
  MAX_SAMPLES=-1 \
  DATASETS_CSV= \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
  LONGBENCH_DATA_ROOT=/path/to/LongBench \
  MAX_MODEL_LEN=32768 \
  LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 \
    --variant qlutattn_k125v2_pt --max-samples -1

for c in 16 32 64; do
  V_TILE_CHANNELS="$c" \
  VBITS=2 \
  QLUT_BIN_CODEBOOKS=sign \
  PERTOKEN_BLOCK=1 \
  RUN_MODE=full \
  MAX_SAMPLES=-1 \
  DATASETS_CSV= \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
  LONGBENCH_DATA_ROOT=/path/to/LongBench \
  MAX_MODEL_LEN=32768 \
  LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 \
    --variant qlutattn_k125v2_pt_vtile16 --max-samples -1
done
```

### 16.3 SNF full：PT2 + tile C16/C32/C64

```bash
cd /path/to/Kitty

V_TILE_CHANNELS= \
  VBITS=2 \
  QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
  PERTOKEN_BLOCK=1 \
  RUN_MODE=full \
  MAX_SAMPLES=-1 \
  DATASETS_CSV= \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
  LONGBENCH_DATA_ROOT=/path/to/LongBench \
  MAX_MODEL_LEN=32768 \
  LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 \
    --variant qlutattn_k188v2_pt --max-samples -1

for c in 16 32 64; do
  V_TILE_CHANNELS="$c" \
  VBITS=2 \
  QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
  PERTOKEN_BLOCK=1 \
  RUN_MODE=full \
  MAX_SAMPLES=-1 \
  DATASETS_CSV= \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-study-rv1 \
  LONGBENCH_DATA_ROOT=/path/to/LongBench \
  MAX_MODEL_LEN=32768 \
  LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 \
    --variant qlutattn_k188v2_pt_vtile16 --max-samples -1
done
```

### 16.4 Full 完整性检查

预期方法目录包括：

```text
longbench_out/llama32-1b-instruct-vtile-study-rv1_fp16/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v4-pt/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v2-pt/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v2-pt-vtile16c16-rv1/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v2-pt-vtile16c32-rv1/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k125v2-pt-vtile16c64-rv1/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v4-pt/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v2-pt/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v2-pt-vtile16c16-rv1/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v2-pt-vtile16c32-rv1/
longbench_out/llama32-1b-instruct-vtile-study-rv1_qlutattn-k188v2-pt-vtile16c64-rv1/
```

每个 full 目录必须包含：

```text
21 个 dataset jsonl
21 个 complete manifest
21 个 report json
pred/result.json
```

文件模式：

```text
longbench_out/<model>_<method>/pred/<dataset>.jsonl
longbench_out/<model>_<method>/pred/<dataset>.manifest.json
longbench_out/<model>_<method>/logs/report_<dataset>.json
```

只统计当前：

```text
longbench_out/
```

不得把：

```text
archieve/longbench_out/
```

混入当前 scoreboard。

---

## 17. Full 结果表设计

### 17.1 主表

| K family | V mode | C | V 理论 bit/value | LB mean | Δ vs V4 parent | Δ vs PT2 | retention | wall time | peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sign | V4 parent | — | 4.5@D64 |  | 0 |  |  |  |  |
| sign | PT2 | — | 2.5@D64 |  |  | 0 |  |  |  |
| sign | rescued tile | 16 | ~2.1265 quant region |  |  |  |  |  |  |
| sign | rescued tile | 32 | ~2.0640 quant region |  |  |  |  |  |  |
| sign | rescued tile | 64 | ~2.0327 quant region |  |  |  |  |  |  |
| SNF | V4 parent | — | 4.5@D64 |  | 0 |  |  |  |  |
| SNF | PT2 | — | 2.5@D64 |  |  | 0 |  |  |  |
| SNF | rescued tile | 16 | ~2.1265 quant region |  |  |  |  |  |  |
| SNF | rescued tile | 32 | ~2.0640 quant region |  |  |  |  |  |  |
| SNF | rescued tile | 64 | ~2.0327 quant region |  |  |  |  |  |  |

### 17.2 分类别表

至少汇总：

- single-doc QA；
- multi-doc QA；
- summarization；
- few-shot/classification；
- synthetic retrieval/count；
- code completion。

### 17.3 研究判断规则

- 若 tile 在更低理论 bit 下达到或超过同 K family 的 PT2：进入 Pareto 候选；
- 若 tile 分数略低但 bit 显著更低：按 Pareto 保留，不直接判失败；
- 若三个 C 均比 PT2 低超过约 0.5 LongBench 分且没有速度/bit优势：标记 experimental negative result，不升级为推荐；
- 若 full 趋势与六文档误差 proxy 完全相反：先审查 calibration region、RHT、bias reuse、block frontier 和 engagement，不先归因于 benchmark noise；
- 不在完整11点矩阵结束前替换当前默认推荐算法。

---

## 18. 停止、回滚与故障定位标准

出现任一项，立即停止 GPU rollout：

1. 旧 sign/SNF V4 输出发生非预期变化；
2. PT2 不再 bit-exact 匹配现有 whole-head min-max 2-bit；
3. rescued tile 不匹配 literal oracle/probe；
4. invalid C 未在量化前 fail-fast；
5. `head_dim%C` assert 不存在或错误信息不清楚；
6. C 未进入 slug/tag/manifest/hash；
7. shell/Python slug 不一致；
8. reset 后复用上一 prompt 的 `mu/rms/bias`；
9. sink/recent/pending 被提前量化；
10. decode 重复量化旧块；
11. partial tile 被当成 full tile；
12. multi-token update 静默只处理一部分；
13. 产生 NaN/Inf；
14. manifest 声称 tile，但 engagement counters 表明未执行；
15. GLM 静默 fallback；
16. 新 tracked 文件出现 host-local path；
17. smoke C16/C32/C64 落入同一目录；
18. 旧默认 variant/slug 被改变。

故障定位顺序：

```text
数学 oracle
→ FP16 round-trip / RHT / reshape
→ cache interval / pointer
→ state reset / batch transform
→ runner config propagation
→ shell/Python slug
→ engagement guard
→ LongBench task-level behavior
```

这样可以区分“数学实现错误”“调度错误”“配置接线错误”和“方法本身精度不够”。

---

## 19. 建议提交拆分

### Commit A：V quant core + literal oracle

- PT2 wrapper；
- RHT；
- pcaff；
- tile minmax；
- MSE1；
- fallback；
- bias；
- bit accounting；
- core tests。

### Commit B：cache state + strict tile16 schedule

- config；
- state；
- prefill；
- decode；
- reset/batch/crop；
- schedule tests。

### Commit C：四个 variants + CLI + slug + manifest

- runner；
- direct CLI；
- shell；
- env precedence；
- provenance；
- wiring tests。

### Commit D：engagement evidence + GLM explicit policy

- counters；
- first-sample guard；
- report；
- GLM fail-fast 或经过测试的 parity。

### Commit E：probe replay + smoke artifacts

- 只生成 `probe_out/` 与 `longbench_out/smoke/`；
- 不改默认 variant。

### Commit F：full results + research/test notes

- 完成11点矩阵；
- 汇总表；
- Pareto 结论；
- 研究笔记与测试笔记分开记录。

每个 commit 都应能独立 review 和回滚。

---

## 20. 最终 Definition of Done

只有以下全部完成，功能才算完成：

- [ ] 四个新 variant 可由 CLI 和 `scripts/run_exp.sh` 启动；
- [ ] sign/SNF K 配置与父 variant 完全一致；
- [ ] PT2 whole-head min-max 2-bit bit-exact；
- [ ] rescued tile 算法与独立 oracle/probe 对齐；
- [ ] token tile 固定16且不可被环境变量改写；
- [ ] `C` 可配置；
- [ ] `assert head_dim % C == 0` 在真实 tensor 路径存在；
- [ ] C16/C32/C64 slug/tag/manifest/hash互不碰撞；
- [ ] prefill/decode 只量化完整 tile；
- [ ] 最多15个 pending FP16 token；
- [ ] reset/batch/beam/crop 语义明确并测试；
- [ ] PostQuant 语义不变；
- [ ] engagement evidence 证明实际 V path 执行；
- [ ] GLM 不会 silent fallback；
- [ ] 所有新旧 0-GPU tests 通过；
- [ ] 真实 dump replay 通过；
- [ ] 8个新配置 smoke 通过；
- [ ] full 11点矩阵完整；
- [ ] 只基于当前 `longbench_out/` 得出结论；
- [ ] 结果报告同时包含精度、理论 bit、耗时和显存；
- [ ] 现有默认推荐和旧变体行为不变；
- [ ] tracked 文件无本机路径；
- [ ] 明确标注 pure-torch fake-quant 只是精度代理。

---

## 21. 本规划轮的实际动作边界

本轮只创建此计划文档。以下动作全部留到用户明确批准实施后：

```text
修改 src/
修改 scripts/
修改 tests/
运行 unittest
运行 probe replay
启动 GPU
启动 LongBench smoke/full
修改默认 variant
提交 commit
```
