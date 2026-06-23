# SimtVF Per-Token Cast to FP8: pto.vmi 映射

本文档对照 [`example_simtvf_per_token_cast_to_fp8.py`](../../tilelang-deepseek/examples/ascend/example_simtvf_per_token_cast_to_fp8.py) 中 `T.SimtVF` 内核，给出 TileLang 源码、`pto.vmi` 逻辑向量 ISA（RFC 推荐方式）和 `pto.mi` 物理微指令三个层次的映射对比。

参考文档：

- [RFC-VPTO-Logical-Vector-ISA.md](RFC-VPTO-Logical-Vector-ISA.md)：`pto.vmi` 的设计动机、op 分类（A/B/C）、layout 推断模型、sub-group 概念（§9.2）
- [PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md](PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md)：Part (EVEN/ODD) / Part_T (P0~P3) / PK4_B32 的物理效果全梳理
- [per_block_cast_rescale_cast_before_after.md](per_block_cast_rescale_cast_before_after.md)：per-block cast 的 widen → rescale → narrow → store 四阶段管线，以及 256B 约束下的 256 bf16 → 256 FP8 循环粒度
- [PTO-micro-Instruction-SPEC.md](PTO-micro-Instruction-SPEC.md)：`pto.mi` 微指令定义

---

## 1. 算法与源码分析

### 1.1 算法语义：per-token (per-row) 量化到 FP8

对输入 `X[M, N]` (f32)，按 `group_size=128` 分组，对每个 token (行) 在其所在分组内求 absmax，然后用 `absmax / 448.0` 作为 scale，把该行该分组的所有元素除以 scale、clamp 到 `[-448, 448]`，最后 cast 到 FP8 E4M3。

```
对每行 i，每个 128-element 分组 g:
  amax[i]  = max_j |X[i, g*128 + j]|           # per-row-per-group absmax
  scale[i] = max(amax[i], 1e-4) / 448.0          # 防 0，归一化到 FP8 范围
  X_fp8[i, g*128 + j] = clamp(X[i, g*128 + j] / scale[i], -448, 448)   cast to FP8
  X_amax[i, g] = scale[i]                         # 输出 scale (即 amax/448)
```

输出三个张量：
- `X_fp8[M, N]` (FP8 E4M3) — 量化结果
- `X_amax[M, ceil(N/128)]` (f32) — 每 token 每分组的 scale (= amax/448)

### 1.2 TileLang 源码结构

源文件 [example_simtvf_per_token_cast_to_fp8.py](../../tilelang-deepseek/examples/ascend/example_simtvf_per_token_cast_to_fp8.py) L22–59：

```python
@T.prim_func
def per_token_cast(X, X_fp8, X_amax):
    with T.Kernel(N_CORES) as core_id:
        for row, row_g_id in T.Persistent([...], N_CORES, core_id, ...):
            y_ub = T.alloc_shared((blk_m, group_size), dtype)          # 32×128 f32 UB
            y_q_ub_fp8 = T.alloc_shared((blk_m, group_size), T.float8_e4m3fn)

            T.copy(X[row*blk_m:(row+1)*blk_m, ...], y_ub)              # GM → UB (f32)
            with T.SimtVF(threads=1024):
                y_local = T.alloc_fragment((blk_m, group_size), dtype)  # 32×128 f32 reg
                y_amax_local = T.alloc_fragment((blk_m,), dtype)        # 32 f32 (per-row amax)
                y_s_local = T.alloc_fragment((blk_m,), dtype)           # 32 f32 (per-row scale)
                y_q_local = T.alloc_fragment((blk_m, group_size), dtype)# 32×128 f32 (量化后)
                y_q_local_fp8 = T.alloc_fragment((blk_m, group_size), T.float8_e4m3fn)

                T.copy(y_ub, y_local)                                   # UB → reg (f32)
                T.reduce_absmax(y_local, y_amax_local, dim=1)           # per-row absmax

                for i in T.Parallel(blk_m):                             # per-row scale
                    y_amax_local[i] = T.max(y_amax_local[i], 1e-4)
                    y_s_local[i] = y_amax_local[i] / fp8_max            # = amax / 448
                for i, j in T.Parallel(blk_m, group_size):              # divide + clamp
                    y_q_local[i, j] = T.clamp(y_local[i, j] / y_s_local[i], fp8_min, fp8_max)
                T.copy(y_q_local, y_q_local_fp8)                       # f32 → FP8 cast
                for i in T.Parallel(blk_m):                            # store scale
                    X_amax[row*blk_m + i, row_g_id] = y_s_local[i]
                T.copy(y_q_local_fp8, y_q_ub_fp8)                      # reg → UB (FP8)
            T.copy(y_q_ub_fp8, X_fp8[...])                              # UB → GM (FP8)
```

### 1.3 数据流与阶段划分

源码在一次 `T.SimtVF` 内串起 6 个阶段。下表把每个阶段映射到参考文档的算子族和 pto.vmi op：

| 阶段 | TileLang 源码 | 逻辑意图 | 算子族 | pto.vmi op | 参考 |
|------|--------------|----------|--------|-----------|------|
| **A** Load f32 | `T.copy(X, y_ub)` + `T.copy(y_ub, y_local)` | GM→UB→reg 加载 32×128 f32 | 连续访存 | `pto.vmi.vlds` | per_block §4.2 Step 1 |
| **B** Absmax reduce | `T.reduce_absmax(y_local, y_amax_local, dim=1)` | 每行 128 个 f32 求 absmax → 32 个标量 | 跨 lane 归约 | `pto.vmi.vreduce_max` (abs) | RFC §5 / R4 |
| **C** Per-row scale | `T.max(amax, 1e-4)` + `amax / 448` | 32 个标量的 elementwise + scalar div | 逐 lane | `pto.vmi.vmaxs` + `pto.vmi.vmuls` | RFC §5 Category A |
| **D** Divide + clamp | `clamp(y / scale, -448, 448)` | 广播 scale 到 128 列，逐元素除 + clamp | 广播 + 逐 lane | `pto.vmi.vdiv` + `pto.vmi.vmaxs/vmins` | per_block §4.2 Step 4 |
| **E** Cast f32→FP8 | `T.copy(y_q_local, y_q_local_fp8)` | 4 倍窄化 cast | 窄化 vcvt | `pto.vmi.vcvt` (P0) | per_block §4.2 Step 5 |
| **F** Store FP8 + scale | `T.copy(y_q_ub_fp8, X_fp8)` + 写 `X_amax` | reg→UB→GM 写回 FP8 + f32 scale | 打包访存 + 连续访存 | `pto.vmi.vsts` (PK4_B32) + `pto.vmi.vsts` | per_block §4.2 Step 6 |

### 1.4 与 per_block_cast 的关键差异

[per_block_cast_rescale_cast_before_after.md](per_block_cast_rescale_cast_before_after.md) 的场景是 **per-block (per-128-channel) scale**：一个 scale 标量广播到 256 个元素。本 kernel 是 **per-token (per-row) scale**：

| 维度 | per_block_cast | per_token_cast (本文件) |
|------|---------------|----------------------|
| scale 粒度 | per 128-channel block (1 scalar / 256 elem) | per row (1 scalar / 128 elem) |
| 是否有 reduce | 否（scale 从外部输入） | **是**（reduce_absmax 产 scale） |
| 数据 tile | 1D `256×bf16` | 2D `32×128×f32` |
| reduce 轴 | 无 | `dim=1`（行内 128 列归约） |
| sub-group 相关性 | 无 reduce，不涉及 | **reduce 涉及 sub-group 对齐**（见 §3） |

per_token_cast 多了一个 **per-row absmax reduce** 阶段，这是 per_block_cast 没有的。本 kernel 的核心难点就在这个 reduce 如何在 pto.vmi 上表达，以及它如何反向影响 load 的 layout 选择（RFC §9.2 的"逆流拉"推断）。

---

## 2. 数据形状与 256B 约束

### 2.1 tile 形状

```
blk_m = 32 行
group_size = 128 列
tile = 32 × 128 f32 = 4096 个 f32 = 16384 B = 16 KB
```

### 2.2 vmi.vreg 的 256B 约束（RFC §3.1）

`!pto.vmi.vreg<L x T>` 要求 `L × bitwidth(T)` 是 256B (2048bit) 的整数倍：

| T | bitwidth | 单 reg 容纳 | L 最小倍数 |
|---|----------|------------|-----------|
| f32 | 32 | 64 | 64 |
| FP8 (ui8) | 8 | 256 | **256** |

### 2.3 两种可选的逻辑组织

**组织 1：2D hierarchical（推荐，匹配 RFC §9.2 / requirements §5.3）**

把 32×128 tile 看成 2D `R × C`，`C` 绑定 lane-level，`R` 绑定 vlane-level：

```
!pto.vmi.vreg<32 × 128 × f32>

物理分解:
  f32: E_v = 8 (每个 32B VLane 容纳 8 个 f32)
  C = 128 → 每个 row 占 128/8 = 16 个 VLane → 但一个 reg 只有 8 VLane
  → 一行 128 f32 跨 2 个物理 reg (每 reg 64 f32)
  → 32 行 × 128 列 = 4096 f32 = 64 个 64-f32 reg → K = 64

  实际上 R=32 行, 每行 128 f32:
  row 内 128 f32 = 2 reg (64+64)
  32 行 = 32 × 2 = 64 reg
  K = 64 — 远超 P3 core profile 的 K≤4
```

**问题**：32×128 的完整 tile 是 64 个物理 reg，远超寄存器预算。必须**分块驻留**——一次只把部分数据放进寄存器。这呼应 RFC §1.2 / requirements P3/P8："programmer owns the schedule, oversized data is tiled by the programmer"。

**组织 2：分块驻留（实际采用的策略）**

按行分块，一次处理 `R_tile` 行 × 128 列。为了让 reduce（行内 128 列 absmax）能命中 R4a（VLane-aligned），需要 `R_tile × 128` 的物理排布让 reduce axis 对齐 VLane。但 128 列的 reduce 跨多个 reg，无法单条 `vcg*` 完成——这是 R4b（full reduce across K regs）的场景。

折中：**一次处理 8 行 × 128 列** = 1024 f32 = 4 KB = 16 个 64-f32 reg（仍超 K≤4），再切到 **2 行 × 128 列** = 256 f32 = 1 KB = 4 个 64-f32 reg（满足 K=4）。

```
分块: 2 行 × 128 列 f32 = 256 f32 = 4 × 64-f32 reg (K=4, 满足 P3 core profile)

  !pto.vmi.vreg<2 × 128 × f32>   或展平  !pto.vmi.vreg<256 × f32>

  物理排布 (2D, f32, E_v=8):
  ┌──── reg 0 (64 f32) ────┐┌──── reg 1 ────┐┌──── reg 2 ────┐┌──── reg 3 ────┐
  │ row0: c0..c63           ││ row0: c64..c127││ row1: c0..c63 ││ row1: c64..c127│
  │ 8 VLane × 8 f32/VLane  ││               ││               ││               │
  └─────────────────────────┘└────────────────┘└────────────────┘└────────────────┘
```

外层 `for` 循环由 programmer 写（遍历 32 行，步长 2），vmi 只表达"2×128 这一片寄存器驻留数据的 layout 操作"——这正是 RFC §9.2 / requirements P8 的边界。

### 2.4 本文档采用的循环粒度

为与 per_block_cast 文档的 256-element 循环粒度对齐，且满足 K≤4，本文档以 **2 行 × 128 列 = 256 f32** 为一次 vmi 迭代：

- 输入：256 f32（4 reg）
- reduce：每行 128 f32 → 2 个 per-row amax（跨 2 reg 的 R4b）
- 输出 FP8：256 FP8（按 per_block_cast 的 PK4_B32 路径）
- 输出 scale：2 个 f32 标量

外层 programmer 循环：`for row_tile in range(0, 32, 2)` 覆盖完整 32 行。

---

## 3. pto.vmi 代码（RFC 推荐方式）

### 3.1 全局 predicate

```mlir
// ===== 全局 predicate =====
// f32 族全 active mask (b32 族, 每 bit 控制 1 个 f32 lane = 4B)
// 本例完整 tile 下所有 lane 都参与, PAT_ALL 仅是 pto 语法必需 operand
// 真正的 lane 过滤场景是尾块 (末尾行不足 128 列), 由 pto.as 自动补尾部 predicate
%m32 = pto.pset_b32 "PAT_ALL" : !pto.vmi.mask<64xb32>     // f32 族
%m8  = pto.pset_b8  "PAT_ALL" : !pto.vmi.mask<256xb8>      // ui8 族 (FP8 store 用 PK4_B32, 实际走 b32 族)
```

### 3.2 Step A: 加载 f32 输入 (256 f32 = 4 reg)

```mlir
// ===== Step A: 加载 2×128 f32 tile (256 f32 = 4 物理 reg) =====
// 逻辑: 连续加载 256 个 f32
// 物理: 4 × pto.mi.vlds {dist="NORM"} → 4 × !pto.mi.vreg<64xf32>
//
// layout 推断起点 (RFC §9.2 逆流拉):
//   下游 Step B 的 reduce_absmax 需要"每行 128 个 f32 连续排布"才能高效 fold。
//   pto.as 从 reduce 的需求反推: load 用 NORM 连续加载即可,
//   不需要 DINTLV (因为输入是 f32, 不是交织存储的 bf16)。
%y = pto.vmi.vlds %inUb[%elemOff] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<256xf32>
//  ↑ layout: axes=[#axis<"chunk", 4, None, 1>], is_contiguous
//  ↑ 逻辑上可重表达为 !pto.vmi.vreg<2 x 128 x f32> (2D), row=2, col=128
```

### 3.3 Step B: per-row absmax reduce (核心难点)

```mlir
// ===== Step B: per-row absmax reduce =====
// TileLang: T.reduce_absmax(y_local, y_amax_local, dim=1)
// 逻辑: 2 行 × 128 列, 每行求 absmax → 2 个标量
//
// 这是 R4b (full reduce across K regs) 而非 R4a:
//   - reduce axis = col (128 个 f32), 跨 2 个物理 reg (每行 128 f32 = 2 reg)
//   - 不在单个 VLane 内闭合 → 不能用单条 vcgmax
//   - 落入 R4b: fold-then-reduce 或 partial-then-combine
//
// absmax = max(|x|): 先取绝对值再 max
%yabs = pto.vmi.vabs %y : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf32>
//  ↑ Category A, layout 透传, 4 路 fan-out: 4 × pto.mi.vabs

// 2D 视角下的 reduce: axis=col, 每 row 独立归约
// !pto.vmi.vreg<2 x 128 x f32> --reduce_max{axis=col}--> !pto.vmi.vreg<2 x 1 x f32>
%amax = pto.vmi.vreduce_max %yabs {axis = col, sub_group = 2}
    : !pto.vmi.vreg<2 x 128 x f32> -> !pto.vmi.vreg<2 x 1 x f32>
//  ↑ sub_group=2: 256 个 lane 切成 2 个 sub-group (每 row 128 lane 一个 sub-group)
//  ↑ 对齐分析 (RFC §9.2):
//      f32 的 E_v = 8, 每个 VLane 容纳 8 f32
//      一个 64-f32 reg = 8 VLane
//      sub_group=2 → 每个 sub-group = 128 lane = 2 reg = 16 VLane
//      reduce axis (col=128) 跨 2 reg, 16 VLane → 不在单 VLane 内闭合 → R4b
//
//  ↑ pto.as lowering (R4b fold-then-reduce, 每 row 独立):
//      // row 0: 128 f32 跨 reg0(64) + reg1(64), fold 成 1 reg 再 vcmax
//      %r0_fold = pto.mi.vmax %yabs_r0_lo, %yabs_r0_hi, %m32   // fold 2 reg → 1 reg
//      %amax0   = pto.mi.vcmax %r0_fold, %m32                   // reduce 1 reg → 标量 (lane 0)
//      // row 1: 同理
//      %r1_fold = pto.mi.vmax %yabs_r1_lo, %yabs_r1_hi, %m32
//      %amax1   = pto.mi.vcmax %r1_fold, %m32
//  ↑ 结果: !pto.vmi.vreg<2 x 1 x f32> (2 个 per-row amax, 各在 lane 0)
```

### 3.4 Step C: per-row scale (防 0 + 归一化)

```mlir
// ===== Step C: per-row scale = max(amax, 1e-4) / 448 =====
// TileLang:
//   y_amax_local[i] = T.max(y_amax_local[i], 1e-4)
//   y_s_local[i] = y_amax_local[i] / fp8_max    // fp8_max = 448
//
// 2 个标量的 elementwise + vec-scalar op
%amax_clamped = pto.vmi.vmaxs %amax, 1e-4 : !pto.vmi.vreg<2 x 1 x f32>, f32 -> !pto.vmi.vreg<2 x 1 x f32>
//  ↑ Category A, scalar operand 隐式广播 (R6)
//  ↑ lowering: 2 × pto.mi.vmaxs (每 row 一个)

// amax / 448 = amax * (1/448)
%scale = pto.vmi.vmuls %amax_clamped, 0.002232142857 : !pto.vmi.vreg<2 x 1 x f32>, f32 -> !pto.vmi.vreg<2 x 1 x f32>
//  ↑ 用 vmuls (乘以 1/448) 替代 vdiv, 避免 vdiv 的长延迟
//  ↑ lowering: 2 × pto.mi.vmuls
//  ↑ 结果: !pto.vmi.vreg<2 x 1 x f32> = 2 个 per-row scale
```

### 3.5 Step D: divide + clamp (广播 scale + 逐元素)

```mlir
// ===== Step D: y_q = clamp(y / scale, -448, 448) =====
// TileLang:
//   for i, j in T.Parallel(blk_m, group_size):
//       y_q_local[i, j] = T.clamp(y_local[i, j] / y_s_local[i], fp8_min, fp8_max)
//
// scale 是 per-row (2 个标量), 需广播到每 row 的 128 列
// RFC §9.2 / R6: broadcast axis, 1-reg backing, replicate-read
%scale_b = pto.vmi.vbr %scale {axis = row}
    : !pto.vmi.vreg<2 x 1 x f32> -> !pto.vmi.vreg<2 x 128 x f32>
//  ↑ R6 fused reduce+broadcast: scale 从 2×1 广播到 2×128
//  ↑ 物理 backing 只 2 个标量 (1 reg), replicate-read 到 256 lane
//  ↑ lowering: 2 × BRC_B32 (每 row 的 scale 广播到该 row 的 128 lane)
//      %s0_b = pto.mi.vlds %scaleUb[%s0_off] {dist="BRC_B32"} : !pto.mi.vreg<64xf32>  // row0 scale 广播
//      %s1_b = pto.mi.vlds %scaleUb[%s1_off] {dist="BRC_B32"} : !pto.mi.vreg<64xf32>  // row1 scale 广播

// y / scale  →  y * (1/scale), 但 scale 是广播的, 用 vdiv 或 vmuls
// 这里保持 vdiv 以匹配源码语义 (clamp 之前)
%y_div = pto.vmi.vdiv %y, %scale_b : !pto.vmi.vreg<2 x 128 x f32> -> !pto.vmi.vreg<2 x 128 x f32>
//  ↑ Category A, chunk fan-out, broadcast operand via R6
//  ↑ lowering (4 路 fan-out, 每 row 跨 2 reg):
//      %d_r0_lo = pto.mi.vdiv %y_r0_lo, %s0_b_lo, %m32   // row0 前半 64 f32
//      %d_r0_hi = pto.mi.vdiv %y_r0_hi, %s0_b_hi, %m32   // row0 后半 64 f32
//      %d_r1_lo = pto.mi.vdiv %y_r1_lo, %s1_b_lo, %m32
//      %d_r1_hi = pto.mi.vdiv %y_r1_hi, %s1_b_hi, %m32
//  ↑ scale_b 的每 row 128 lane = 2 reg, replicate-read 到对应 row 的 2 个数据 reg

// clamp(y_div, -448, 448): max(min(y_div, 448), -448)
%y_clamped_lo = pto.vmi.vmins %y_div, 448.0 : !pto.vmi.vreg<2 x 128 x f32>, f32 -> !pto.vmi.vreg<2 x 128 x f32>
%y_clamped    = pto.vmi.vmaxs %y_clamped_lo, -448.0 : !pto.vmi.vreg<2 x 128 x f32>, f32 -> !pto.vmi.vreg<2 x 128 x f32>
//  ↑ Category A vec-scalar, 4 路 fan-out 各 2 次 (vmins + vmaxs)
```

### 3.6 Step E: f32 → FP8 narrow cast

```mlir
// ===== Step E: f32 → FP8 窄化 (4 倍窄化) =====
// TileLang: T.copy(y_q_local, y_q_local_fp8)  -- 隐式 cast
//
// 参照 per_block_cast §4.2 Step 5:
//   256 f32 → 256 FP8 (ui8)
//   每个 64-lane f32 reg 各做 vcvt {part=P0} → 256-lane ui8 reg (P0 位有值)
//   4 次 vcvt P0, 产出 4 × 64 = 256 个 FP8 值
//
// RFC 核心: 一条 pto.vmi.vcvt, 不写 part=P0~P3
// 逻辑类型必须满足 256B 约束: 256 × 8bit = 256B ✓
%quantized = pto.vmi.vcvt %y_clamped {rnd = "R", sat = "SAT"}
    : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xui8>
//  ↑ pto.as lowering (4 次 vcvt {part=P0}, 每个 64-lane f32 reg 一次):
//      %fp8_r0_lo = pto.mi.vcvt %y_clamped_r0_lo, %m32 {rnd="R", sat="SAT", part="P0"}
//          : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
//      %fp8_r0_hi = pto.mi.vcvt %y_clamped_r0_hi, %m32 {rnd="R", sat="SAT", part="P0"}
//          : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
//      %fp8_r1_lo = pto.mi.vcvt %y_clamped_r1_lo, %m32 {rnd="R", sat="SAT", part="P0"}
//          : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
//      %fp8_r1_hi = pto.mi.vcvt %y_clamped_r1_hi, %m32 {rnd="R", sat="SAT", part="P0"}
//          : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
//  ↑ layout: axes=[#axis<"chunk", 4>, #axis<"sub_part", 4, P0, 4>]
//     sub_part 轴记录 P0 placement (仅填最低 8-bit, 3 个空洞用 PK4_B32 消费)
```

### 3.7 Step F: store FP8 + scale

```mlir
// ===== Step F1: store FP8 (PK4_B32 打包存储) =====
// 参照 per_block_cast §4.2 Step 6:
//   pto.as 自动将 sub_part 轴消费为 PK4_B32 打包存储
//   4 次 vsts PK4_B32, 每次写 64 bytes, 共 256 bytes
pto.vmi.vsts %quantized, %outUb_fp8[%outOff] : !pto.vmi.vreg<256xui8>, !pto.ptr<ui8, ub>
//  ↑ pto.as lowering:
//      pto.mi.vsts %fp8_r0_lo, %outUb_fp8[%outOff + 0],   %m32 {dist="PK4_B32"}
//      pto.mi.vsts %fp8_r0_hi, %outUb_fp8[%outOff + 64],  %m32 {dist="PK4_B32"}
//      pto.mi.vsts %fp8_r1_lo, %outUb_fp8[%outOff + 128], %m32 {dist="PK4_B32"}
//      pto.mi.vsts %fp8_r1_hi, %outUb_fp8[%outOff + 192], %m32 {dist="PK4_B32"}

// ===== Step F2: store per-row scale (f32 连续存储) =====
// TileLang: X_amax[row*blk_m + i, row_g_id] = y_s_local[i]
// scale 是 2 个 f32 标量, 连续写回 UB
pto.vmi.vsts %scale, %outUb_amax[%amaxOff] : !pto.vmi.vreg<2 x 1 x f32>, !pto.ptr<f32, ub>
//  ↑ pto.as lowering: 1PT_B32 (只写 lane 0) 或 NORM_B32 + 尾部 mask
//      pto.mi.vsts %scale, %outUb_amax[%amaxOff], %m32_2pt {dist="1PT_B32"}
```

### 3.8 外层 programmer 循环

```mlir
// 外层循环由 programmer 写 (RFC §1.2 / requirements P3/P8: programmer owns the schedule)
// 一次迭代处理 2 行 × 128 列, 遍历 32 行需 16 次迭代
// (实际 SimtVF 的 1024 threads 会进一步并行化多个 row_tile, 这里展示单 thread 的串行路径)
scf.for %row_tile = %c0 to %c32 step %c2 {
  // ... Step A–F, %elemOff / %outOff / %amaxOff 随 %row_tile 递增 ...
}
```

---

## 4. pto.mi 物理微指令（展示 lowering 结果）

以下是不用 RFC 推荐方式时，`pto.as` 把 §3 的 `pto.vmi` lowering 后的物理形态。对应一次 2×128 f32 迭代（256 f32 → 256 FP8 + 2 scale）：

```mlir
%m32 = pto.mi.pset_b32 "PAT_ALL" : !pto.mi.mask<b32>

// ===== Step A: 加载 256 f32 (4 reg) =====
%y_r0_lo = pto.mi.vlds %inUb[%elemOff + 0]   {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.mi.vreg<64xf32>   // row0 col 0..63
%y_r0_hi = pto.mi.vlds %inUb[%elemOff + 64]  {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.mi.vreg<64xf32>   // row0 col 64..127
%y_r1_lo = pto.mi.vlds %inUb[%elemOff + 128] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.mi.vreg<64xf32>   // row1 col 0..63
%y_r1_hi = pto.mi.vlds %inUb[%elemOff + 192] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.mi.vreg<64xf32>   // row1 col 64..127

// ===== Step B: per-row absmax (R4b fold-then-reduce) =====
// 先取绝对值
%ya_r0_lo = pto.mi.vabs %y_r0_lo, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%ya_r0_hi = pto.mi.vabs %y_r0_hi, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%ya_r1_lo = pto.mi.vabs %y_r1_lo, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%ya_r1_hi = pto.mi.vabs %y_r1_hi, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>

// fold 每 row 的 2 reg → 1 reg (R4b fold)
%fold0 = pto.mi.vmax %ya_r0_lo, %ya_r0_hi, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%fold1 = pto.mi.vmax %ya_r1_lo, %ya_r1_hi, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>

// reduce 1 reg → 标量 (lane 0), vcmax 返回 (max_value, argmax_index)
%amax0_v = pto.mi.vcmax %fold0, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>   // amax0 在 lane 0
%amax1_v = pto.mi.vcmax %fold1, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>   // amax1 在 lane 0

// ===== Step C: per-row scale = max(amax, 1e-4) * (1/448) =====
%aclamp0 = pto.mi.vmaxs %amax0_v, 1e-4, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%aclamp1 = pto.mi.vmaxs %amax1_v, 1e-4, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%scale0_v = pto.mi.vmuls %aclamp0, 0.002232142857, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%scale1_v = pto.mi.vmuls %aclamp1, 0.002232142857, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>

// ===== Step D: divide + clamp =====
// 广播 scale 到 64-lane (BRC_B32, 每 row 的 scale 在该 row 的 2 个数据 reg 上 replicate-read)
// 注意: scale_v 的有效值在 lane 0, 广播需要先 extract 再 BRC, 或用 vdup
%s0_b_lo = pto.mi.vdup %scale0_v, %m32 {position="LOWEST"} : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>   // row0 scale 广播
%s1_b_lo = pto.mi.vdup %scale1_v, %m32 {position="LOWEST"} : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>   // row1 scale 广播

// y / scale (用 vdiv; 实际可用 vmuls 配合预计算 1/scale)
%d_r0_lo = pto.mi.vdiv %y_r0_lo, %s0_b_lo, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%d_r0_hi = pto.mi.vdiv %y_r0_hi, %s0_b_lo, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%d_r1_lo = pto.mi.vdiv %y_r1_lo, %s1_b_lo, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%d_r1_hi = pto.mi.vdiv %y_r1_hi, %s1_b_lo, %m32 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>

// clamp(d, -448, 448) = max(min(d, 448), -448)
%c_lo_r0_lo = pto.mi.vmins %d_r0_lo, 448.0, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%c_r0_lo    = pto.mi.vmaxs %c_lo_r0_lo, -448.0, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%c_lo_r0_hi = pto.mi.vmins %d_r0_hi, 448.0, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%c_r0_hi    = pto.mi.vmaxs %c_lo_r0_hi, -448.0, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%c_lo_r1_lo = pto.mi.vmins %d_r1_lo, 448.0, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%c_r1_lo    = pto.mi.vmaxs %c_lo_r1_lo, -448.0, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%c_lo_r1_hi = pto.mi.vmins %d_r1_hi, 448.0, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%c_r1_hi    = pto.mi.vmaxs %c_lo_r1_hi, -448.0, %m32 : !pto.mi.vreg<64xf32>, f32, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>

// ===== Step E: f32 → FP8 窄化 (vcvt P0) =====
%fp8_r0_lo = pto.mi.vcvt %c_r0_lo, %m32 {rnd = "R", sat = "SAT", part = "P0"}
    : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
%fp8_r0_hi = pto.mi.vcvt %c_r0_hi, %m32 {rnd = "R", sat = "SAT", part = "P0"}
    : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
%fp8_r1_lo = pto.mi.vcvt %c_r1_lo, %m32 {rnd = "R", sat = "SAT", part = "P0"}
    : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
%fp8_r1_hi = pto.mi.vcvt %c_r1_hi, %m32 {rnd = "R", sat = "SAT", part = "P0"}
    : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>

// ===== Step F1: store FP8 (PK4_B32) =====
pto.mi.vsts %fp8_r0_lo, %outUb_fp8[%outOff + 0],   %m32 {dist = "PK4_B32"}
    : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>
pto.mi.vsts %fp8_r0_hi, %outUb_fp8[%outOff + 64],  %m32 {dist = "PK4_B32"}
    : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>
pto.mi.vsts %fp8_r1_lo, %outUb_fp8[%outOff + 128], %m32 {dist = "PK4_B32"}
    : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>
pto.mi.vsts %fp8_r1_hi, %outUb_fp8[%outOff + 192], %m32 {dist = "PK4_B32"}
    : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>

// ===== Step F2: store per-row scale (1PT_B32, 只写 lane 0) =====
pto.mi.vsts %scale0_v, %outUb_amax[%amaxOff + 0], %m32 {dist = "1PT_B32"}
    : !pto.mi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mi.mask<b32>
pto.mi.vsts %scale1_v, %outUb_amax[%amaxOff + 1], %m32 {dist = "1PT_B32"}
    : !pto.mi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mi.mask<b32>
```

---

## 5. 物理流程全景图

```
UB: f32 input [row0: c0..c127, row1: c0..c127]  (1024 B, 4 × 64-f32 reg)
         │ 4 × vlds NORM
         ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│ y_r0_lo  ││ y_r0_hi  ││ y_r1_lo  ││ y_r1_hi  │  4 × 64-f32 reg
│ row0     ││ row0     ││ row1     ││ row1     │     chunk(4) 连续
│ c0..c63  ││ c64..c127││ c0..c63  ││ c64..c127│     is_contiguous
└──────────┘└──────────┘└──────────┘└──────────┘
         │         │          │         │
       vabs       vabs        vabs       vabs             ← Step B 取绝对值 (4 路)
         │         │          │         │
         ▼         ▼          ▼         ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│ ya_r0_lo ││ ya_r0_hi ││ ya_r1_lo ││ ya_r1_hi │
└──────────┘└──────────┘└──────────┘└──────────┘
         │         │          │         │
         └──vmax───┘          └──vmax───┘               ← R4b fold: 每 row 2 reg → 1 reg
              │                    │
              ▼                    ▼
         ┌──────────┐         ┌──────────┐
         │ fold0    │         │ fold1    │   2 × 64-f32 reg (每 row 一个)
         └──────────┘         └──────────┘
              │                    │
            vcmax                vcmax                   ← reduce 1 reg → 标量 (lane 0)
              │                    │
              ▼                    ▼
         amax0 (lane0)        amax1 (lane0)              ← 2 个 per-row amax

         │                    │
       vmaxs(1e-4)         vmaxs(1e-4)                    ← Step C 防 0
       vmuls(1/448)        vmuls(1/448)                  ← 归一化
         │                    │
         ▼                    ▼
       scale0              scale1                        ← 2 个 per-row scale

         │                    │
       vdup/BRC             vdup/BRC                      ← R6 广播 scale 到 64 lane
         │                    │
         ▼                    ▼
       s0_b (64 f32)       s1_b (64 f32)                  ← 每 row scale 广播, replicate-read 到 2 个数据 reg

         │                    │
         └───vdiv(y, s_b)─────┘                            ← Step D 除法 (4 路 fan-out)
              │
              ▼
         ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
         │ d_r0_lo  ││ d_r0_hi  ││ d_r1_lo  ││ d_r1_hi  │  4 × 64-f32 (y/scale)
         └──────────┘└──────────┘└──────────┘└──────────┘
              │         │          │         │
            vmins(448) vmins(448)  vmins(448) vmins(448)  ← clamp 上界
            vmaxs(-448) vmaxs(-448) vmaxs(-448) vmaxs(-448) ← clamp 下界
              │         │          │         │
              ▼         ▼          ▼         ▼
         ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
         │ c_r0_lo  ││ c_r0_hi  ││ c_r1_lo  ││ c_r1_hi  │  4 × 64-f32 (clamped)
         └──────────┘└──────────┘└──────────┘└──────────┘
              │         │          │         │
            vcvt P0   vcvt P0    vcvt P0   vcvt P0        ← Step E f32→FP8 (Part_T P0)
              │         │          │         │
              ▼         ▼          ▼         ▼
         ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
         │fp8_r0_lo ││fp8_r0_hi ││fp8_r1_lo ││fp8_r1_hi │  4 × 256-ui8 reg (P0 位)
         │ 64 FP8   ││ 64 FP8   ││ 64 FP8   ││ 64 FP8   │     每个 32-bit lane 低 8-bit = FP8
         └──────────┘└──────────┘└──────────┘└──────────┘
              │         │          │         │
            PK4_B32   PK4_B32    PK4_B32   PK4_B32        ← Step F1 FP8 打包存储
              │         │          │         │
              ▼         ▼          ▼         ▼
         UB: fp8 output [fp8[0], fp8[1], ..., fp8[255]]  (256 bytes)

         │                    │
       1PT_B32              1PT_B32                        ← Step F2 scale 存 (只写 lane 0)
         │                    │
         ▼                    ▼
       UB: X_amax[scale0, scale1]                          ← 2 个 f32 scale
```

---

## 6. sub-group 与 reduce 策略分析（RFC §9.2 应用）

本 kernel 的 reduce 是 **per-row absmax over 128 columns**，是 RFC §9.2 sub-group 概念的典型应用场景，但落入 R4b 而非 R4a。

### 6.1 sub_group 划分

```
tile = 2 行 × 128 列 = 256 个 f32 lane
sub_group = 2 (每 row 一个 sub-group, 每个 sub-group = 128 lane)

  ┌── sub-group 0 (row0, 128 lane) ──────────────┐┌── sub-group 1 (row1, 128 lane) ──┐
  │ c0 c1 ... c63  │ c64 ... c127                  ││ c0 ... c63  │ c64 ... c127       │
  │ ← reg0 (64) →  │ ← reg1 (64) →                 ││ ← reg2 (64)→│ ← reg3 (64) →      │
  └────────────────────────────────────────────────┘└──────────────────────────────────┘

reduce axis = col (128 lane), 每个 sub-group 内做 max
```

### 6.2 对齐分析

```
f32: E_v = 8 (每个 32B VLane 容纳 8 个 f32)
sub_group=2 → 每个 sub-group = 128 lane = 16 VLane = 2 reg

  sub-group 对齐 VLane 边界?
  128 lane / 8 lane-per-VLane = 16 VLane → 整数倍 ✓
  → reduce axis 对齐 VLane 边界

  但是: reduce 跨 2 个 reg (16 VLane), 不在单个 VLane 内闭合
  → 不能用单条 vcgmax (vcgmax 只在单 VLane 内 reduce)
  → 落入 R4b: 需要先 fold 多 reg 再 reduce
```

### 6.3 R4a vs R4b 的判定

参考 RFC §9.2 的 sub-group 对齐表：

| sub-group 配置 | reduce axis 范围 | 是否单 VLane 闭合 | lowering 策略 |
|---|---|---|---|
| `sub_group=2`, reduce 128 lane (跨 2 reg) | 16 VLane | 否（跨 reg） | **R4b** fold-then-reduce |
| 若 `sub_group=16`, reduce 16 lane (单 reg 内) | 2 VLane | 否（跨 VLane 但单 reg） | R4b（仍跨 VLane）或 vcpadd |
| 若 reduce 恰好 8 lane (单 VLane) | 1 VLane | 是 | **R4a** 单条 vcgmax |

本 kernel 的 reduce 是 128 列，**必然跨 reg**，因此无论 sub-group 怎么取值都落入 R4b。这是 per-row reduce over 较宽 inner dim 的固有特征。

### 6.4 优化方向：缩小 inner dim 让 reduce 命中 R4a

如果能重新组织 tile 使 reduce axis 恰好 = `E_v`（f32 时 = 8），就能用 R4a。但 per-token 量化的语义是"每行 128 列求 absmax"，inner dim 固定为 128，无法缩小。因此：

- **R4b 是本 kernel reduce 的必然路径**，优化空间在 fold-then-reduce vs partial-then-combine 的选择（pto.as 成本模型决定，见 R4b/M1）
- pto.as 在 K=2（每 row 跨 2 reg）时倾向 fold-then-reduce：`1 × vmax` + `1 × vcmax` per row，少 reduce op

### 6.5 逆流拉推断（RFC §9.2 核心）

```
推断起点: vreduce_max {axis=col, sub_group=2}
  → reduce 跨 2 reg, 落入 R4b, 需要 fold
  → fold 要求每 row 的 128 f32 连续排布在 2 个 reg 里 (reg0=row0 前半, reg1=row0 后半)
  → 反推 load: 用 NORM 连续加载即可满足 (f32 输入, 无交织)
  → load distribution = NORM (不是 DINTLV)

对比 MX block-scale (RFC §9.2 示例):
  → reduce 是 32-element block, 对齐 VLane, 落入 R4a, 需要 parity 布局
  → 反推 load: 用 DINTLV_B16 (bf16 交织输入)

差异根源:
  - MX: bf16 交织输入 + 32-element block reduce (对齐 VLane) → DINTLV
  - per_token: f32 连续输入 + 128-element row reduce (跨 reg) → NORM
  layout 推断方向一致 (从 reduce 逆流到 load), 但结论不同 (取决于输入 dtype 和 reduce 粒度)
```

---

## 7. 三层对比总结

| 维度 | TileLang `T.SimtVF` | `pto.vmi` (RFC 推荐) | `pto.mi` (物理微指令) |
|------|---------------------|----------------------|----------------------|
| **循环粒度** | 32×128 (一次 SimtVF, 1024 threads) | 2×128 = 256 f32 (一次 vmi 迭代, K=4) | 2×128 = 256 f32 (4 reg fan-out) |
| **外层调度** | `T.Persistent` 隐式 | programmer 写 `scf.for` (P3/P8) | programmer 写 `scf.for` |
| **Load f32** | `T.copy` 隐式 | `pto.vmi.vlds` (NORM, 由 reduce 逆流推断) | 4 × `vlds {NORM}` |
| **Absmax reduce** | `T.reduce_absmax` 一行 | `pto.vmi.vabs` + `vreduce_max {axis=col, sub_group=2}` | `vabs`×4 + `vmax`×2 (fold) + `vcmax`×2 |
| **reduce 策略** | 编译器隐式 | R4b (pto.as 选 fold-then-reduce) | 显式 fold + vcmax |
| **per-row scale** | `T.max` + 除法 隐式 | `vmaxs` + `vmuls` (vec-scalar) | `vmaxs`×2 + `vmuls`×2 |
| **scale 广播** | 隐式 | `pto.vmi.vbr {axis=row}` (R6) | `vdup`×2 (每 row scale 广播) |
| **divide + clamp** | `T.clamp(y/s, ...)` 隐式 | `vdiv` + `vmins` + `vmaxs` (4 路 fan-out) | `vdiv`×4 + `vmins`×4 + `vmaxs`×4 |
| **f32→FP8 cast** | `T.copy` 隐式 | `pto.vmi.vcvt {rnd, sat}` (无 part) | 4 × `vcvt {part=P0}` |
| **FP8 store** | `T.copy` 隐式 | `pto.vmi.vsts` (PK4_B32 自动) | 4 × `vsts {PK4_B32}` |
| **scale store** | 赋值隐式 | `pto.vmi.vsts` (1PT_B32) | 2 × `vsts {1PT_B32}` |
| **用户需关心的物理细节** | 零（不可控） | 零（layout 由 pto.as 逆流推断） | 全部（fold 顺序 + P0 + PK4_B32 + 广播 + clamp 拆解） |
| **可读性** | 最低（reduce/cast 不可见） | 最高（每步语义清晰） | 可控但心智负担重 |

### 与 per_block_cast 的对比

| 维度 | per_block_cast | per_token_cast (本文件) |
|------|---------------|----------------------|
| 输入 dtype | bf16 | f32 |
| load distribution | DINTLV_B16 (逆流推断自 32-block reduce) | NORM (逆流推断自 128-row reduce) |
| 有无 reduce | 无 | 有 (per-row absmax, R4b) |
| reduce 策略 | N/A | R4b fold-then-reduce |
| scale 来源 | 外部输入 | reduce 产出 |
| scale 广播 | 1 scalar → 256 elem | 2 scalar → 2×128 elem (per-row) |
| narrow cast 路径 | 相同 (vcvt P0 + PK4_B32) | 相同 |
| 256B 约束处理 | 256 bf16 → 256 FP8 | 256 f32 → 256 FP8 |

---

## 8. 遗留问题

### 8.1 reduce_absmax 的 abs + max 融合

源码用 `T.reduce_absmax`（一步求绝对值最大）。本文档拆成 `vabs` + `vreduce_max` 两步。是否存在原生的 absmax reduce 指令（如 `vcabsmax`）能一步完成？若存在，pto.as 应优先选择它以省一次 fan-out。需在 PTO SPEC 中确认。

### 8.2 per-row scale 的广播粒度

scale 是 per-row 标量（lane 0），广播到该 row 的 128 lane（跨 2 reg）。`vdup {position=LOWEST}` 只取 lane 0 广播到全 reg——但每 row 跨 2 reg，需要 2 次 vdup 或 1 次 vdup + replicate-read。pto.as 如何处理"1 个标量广播到跨 reg 的多个 reg"（R6 的跨 reg broadcast）需要明确。

### 8.3 SimtVF 的 1024 threads 与 vmi 的映射

源码 `T.SimtVF(threads=1024)` 把 32×128 = 4096 个元素分给 1024 个 thread（每 thread 4 元素）。本文档的 vmi 映射按"2×128 = 256 元素 / 迭代"组织，未显式表达 thread 划分。SimtVF 的 thread-level 并行如何映射到 vmi 的物理 reg fan-out（是否每 thread 独立持有一组 reg，还是共享）需要进一步明确。这影响 K 的实际取值（是单 thread 的 K=4，还是跨 thread 聚合的更大 K）。

### 8.4 clamp 的 vmins/vmaxs 与原生 vcvt sat 的关系

`vcvt {sat="SAT"}` 本身带饱和（FP8 E4M3 范围 ±448）。源码显式 `clamp(y/scale, -448, 448)` 后再 cast。是否可以省略显式 clamp，依赖 `vcvt {sat="SAT"}` 的饱和？语义上等价（clamp 到 ±448 再 cast = 直接 sat cast），但数值上需确认 SAT 模式的饱和边界是否恰为 ±448。若等价，可省 `vmins`/`vmaxs` 各 4 条，pto.as 应做此简化。

### 8.5 尾块处理

当 N 不是 128 的整数倍，或 M 不是 32 的整数倍时，末尾 tile 不足 128 列或 32 行。scale 的 reduce 在尾块下需部分 active predicate，FP8 store 的 PK4_B32 在尾块下写入量不足 64 bytes。这些问题与 [per_block_cast_rescale_cast_before_after.md §11](per_block_cast_rescale_cast_before_after.md) 的尾块遗留问题同类，需在 RFC 后续版本统一解决。
