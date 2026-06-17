# Per-Block Cast: Rescale + Narrow Cast Before/After 对比

本文档对照 [`per_block_cast_kernel.py`](../tile_kernels/quant/per_block_cast_kernel.py) 中 Rescale（§2.5）和 Narrow Cast（§2.6）两步的实现，给出 TileLang `T.Parallel` 原始写法、`pto.vmi` 逻辑向量 ISA（RFC 推荐方式）和 `pto.mi` 原始微指令三个层次的 before/after 对比。

参考文档：

- [RFC-VPTO-Logical-Vector-ISA.md](../../PTO-Gym/docs/RFC-VPTO-Logical-Vector-ISA.md)：`pto.vmi` 的设计动机、op 分类、layout 推断模型
- [PTO-micro-Instruction-SPEC.md](../../PTO-Gym/docs/PTO-micro-Instruction-SPEC.md)：`pto.mi` 微指令定义
- [PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md](../../PTO-Gym/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md)：Pack / Unpack / Interleave / Part 的物理效果全梳理
- [per_block_quant_gpu_vs_npu.md](../../a5-kernel-standalone/docs/per_block_quant_gpu_vs_npu.md)：GPU vs NPU 算法对照

---

## 1. 问题：一行赋值隐藏了四阶段管线

### 1.1 源码现象

[`per_block_cast_kernel.py`](../tile_kernels/quant/per_block_cast_kernel.py) 中 Rescale（§2.5）和 Narrow Cast + Store（§2.6）这两步算法步骤，在源码中**完全折叠在一行赋值里**。完整 tile 路径和边界 tile 路径各有一份相同的代码：

**完整 tile 路径**（[L130–131](../tile_kernels/quant/per_block_cast_kernel.py#L130-L131)）：

```python
for i, j in T.Parallel(block_m, block_k):
    out[pid_x * block_m + i, pid_y * block_k + j] = x_fragment[i, j] * sf_fragment[i // num_per_tokens, j // num_per_channels]
```

**边界 tile 路径**（[L136–137](../tile_kernels/quant/per_block_cast_kernel.py#L136-L137)）：

```python
for i, j in T.Parallel(block_m, block_k):
    out[pid_x * block_m + i, pid_y * block_k + j] = x_fragment[i, j] * sf_fragment[i // num_per_tokens, j // num_per_channels]
```

这两行赋值看起来只是一句乘法，但实际上 TileLang 编译器将其展开为参考文档 §2.5 + §2.6 的**完整四阶段管线**。根本原因是三个张量的 dtype 不匹配：

| 张量 | dtype | 角色 |
|------|-------|------|
| `x_fragment` | **BF16** | 输入数据 |
| `sf_fragment` | **FP32** | quant multiply factor |
| `out` | **FP8 E4M3** | 量化输出 |

TileLang 的 dtype 推断系统看到 `BF16 × FP32 → FP8`，必须自动插入类型转换才能让这条赋值语义正确。

### 1.2 编译器隐式插入的四阶段

| 阶段 | 参考 § | 逻辑意图 | TileLang 源码可见性 |
|------|---------|----------|---------------------|
| **4a** bf16 → f32 widen | §2.5 前置 | 加宽输入以便与 f32 scale 做乘法 | ❌ 完全不可见 |
| **4b** f32 × f32 rescale | §2.5 核心 | 乘以 quant multiply factor | ❌ 混在赋值里，看似只是一步乘法 |
| **5a** f32 → FP8 narrow cast | §2.6 核心 | 窄化到 FP8 E4M3 with saturation | ❌ 完全不可见 |
| **5b** FP8 store | §2.6 写回 | 写入全局内存 | ❌ 混在赋值里 |

用户看到的只是一行 `out[...] = x_fragment * sf_fragment`，但编译器在背后默默做了四件事。

### 1.3 CUDA dump 证据

以下编译后的 CUDA 代码来自 [per_block_quant_gpu_vs_npu.md](../../a5-kernel-standalone/docs/per_block_quant_gpu_vs_npu.md) §3.4，是对上面那一行 TileLang 赋值的完整展开：

```cuda
// ===== 阶段 4a: BF16 → FP32 widen (§2.5 前置, 隐式) =====
// 因为 x_fragment 是 BF16 而 sf_fragment 是 FP32，编译器必须先加宽
uint2 v__2 = *(uint2*)(x_fragment + ((i_2 * 16) + (vec * 4)));
((float2*)(&__3))[0] = __bfloat1622float2(...);  // BF16 → FP32
((float2*)(&__3))[1] = __bfloat1622float2(...);

// ===== 阶段 4b: Rescale (§2.5 核心, 隐式) =====
// 乘以 quant multiply factor (广播 scale 到所有元素)
float4 v__3 = make_float4(sf_fragment[0], sf_fragment[0], sf_fragment[0], sf_fragment[0]);
*(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(v__3.x)));
*(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(v__3.z)));

// ===== 阶段 5a: FP32 → FP8 narrow cast (§2.6 核心, 隐式) =====
// 窄化到 FP8 E4M3, 带饱和 (SATFINITE 防止溢出)
(reinterpret_cast<__nv_fp8x2_storage_t*>(&__1))[0] = __nv_cvt_float2_to_fp8x2(
    ((float2*)(&__2))[0], __NV_SATFINITE, __NV_E4M3);
(reinterpret_cast<__nv_fp8x2_storage_t*>(&__1))[1] = __nv_cvt_float2_to_fp8x2(
    ((float2*)(&__2))[1], __NV_SATFINITE, __NV_E4M3);

// ===== 阶段 5b: Store (§2.6 写回, 隐式) =====
// 写入全局内存 (128-bit 向量化存储)
*(fp8_e4_16_t*)(out + ...) = *(fp8_e4_16_t*)(out_local_cast + 0);
```

对照表：

| 参考文档步骤 | TileLang 源码 | 生成 CUDA |
|-------------|--------------|-----------|
| §2.5 Rescale — widen | `x_fragment[i, j]`（dtype=BF16）隐式触发 | `__bfloat1622float2` |
| §2.5 Rescale — multiply | `* sf_fragment[...]`（dtype=FP32） | `tl::mul2` |
| §2.6 Narrow cast | 赋值到 `out`（dtype=FP8 E4M3）隐式触发 | `__nv_cvt_float2_to_fp8x2(SATFINITE, E4M3)` |
| §2.6 Store | 写入 `out[...]` | `*(fp8_e4_16_t*)(out + ...)` 全局存储 |

### 1.4 核心矛盾

**一句话总结**：源码只有一行 `out[...] = x_fragment * sf_fragment`，但 TileLang 的 dtype 推断系统看到 `BF16 × FP32 → FP8`，自动插入 widen(BF16→FP32)、multiply、narrow(FP32→FP8 with saturation)、store 四个阶段。§2.5 和 §2.6 这两步在源码层面是**不可见的**——它们完全由编译器根据张量 dtype 隐式生成。

这带来两个问题：

1. **不可读**：算法步骤与源码行数不对齐。读者无法从源码直接确认"这个内核确实做了 Rescale + Narrow Cast"。必须查阅编译后的 CUDA dump 或参考文档才能补全心智模型。
2. **不可控**：编译器选择 widen/narrow 的时机和方式（如是否先 vintlv 重建连续态再运算，还是保持 parity 交织态 fan-out）完全由 TileLang codegen 决定，用户无法干预。

后续 §3–§5 给出三种替代写法的完整对比，§6–§8 解释这些写法在 A5 SIMD 寄存器上的物理效果。

---

## 2. 场景设定

以 `(128, 128)` block、bf16 输入、FP8 E4M3 输出为例。

**关键约束**：`pto.vmi.vreg` 要求 `L * bitwidth(T)` 是 256B (2048bit) 的整数倍（来自 [RFC-VPTO-Logical-Vector-ISA.md](../../PTO-Gym/docs/RFC-VPTO-Logical-Vector-ISA.md) §3.1）。对于 FP8 (ui8, 8-bit)，最小合法 `L` = 256。这意味着一次 vmi 循环迭代需要处理 **256 bf16 → 256 FP8**，而不是 128。循环粒度翻倍，但量化语义不变。

物理容量参考（来自 [Pack-Unpack-Interleave-Part-Reference.md](../../PTO-Gym/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md) §六）：

| dtype | 每物理 reg 容纳 | lane 宽度 | predicate 族 | vmi.vreg 最小合法 L |
|-------|---------------|----------|-------------|-------------------|
| bf16 | 128 | 16 bit | b16 | 128 |
| f32 | 64 | 32 bit | b32 | 64 |
| fp8 (ui8) | 256 | 8 bit | b8（PK4_B32 store 用 b32 predicate） | **256** |

加宽/窄化时 reg 容量变化（256-element 循环）：

- 2×bf16_reg(128) → 4×f32_reg(64)：chunk×parity 交织态（2 chunk × 2 parity）
- 4×f32_reg(64) → 4×ui8_reg(256) via `vcvt {part=P0}`：P0 位有值，P1/P2/P3 位为零
- 4×ui8_reg → 4 × `vsts PK4_B32` 各写 64 FP8 值 = 256 FP8 总输出

---

## 3. BEFORE：TileLang `T.Parallel`（当前代码）

当前代码将 §2.5 Rescale 和 §2.6 Narrow Cast + Store 完全折叠在两行赋值中（完整 tile 路径 L130–131，边界 tile 路径 L136–137，代码相同）：

```python
# per_block_cast_kernel.py L130–131 / L136–137
for i, j in T.Parallel(block_m, block_k):
    out[pid_x * block_m + i, pid_y * block_k + j] = x_fragment[i, j] * sf_fragment[i // num_per_tokens, j // num_per_channels]
```

编译器看到 `BF16 × FP32 → FP8 E4M3` 的 dtype 链，自动展开为 §1.2 所述的四阶段管线。CUDA dump 的完整展开见 §1.3。

**本写法的关键特征**：

- 四阶段管线（widen → multiply → narrow cast → store）完全隐式，源码不可见
- 编译器自动选择 widen/narrow 的时机，用户无法干预（如是否保持 parity 交织态 fan-out vs 先 vintlv 重建连续态）
- 在 GPU 上这是高效的做法（CUDA 有原生 FP8 intrinsics），但在 NPU 上对应的物理操作完全不同——需要 EVEN/ODD、P0、punpack、PK4_B32 等大量手写细节

---

## 4. AFTER（方案 A）：`pto.vmi` 逻辑向量 ISA（RFC 推荐方式）

按照 [RFC-VPTO-Logical-Vector-ISA.md](../../PTO-Gym/docs/RFC-VPTO-Logical-Vector-ISA.md) §7.1 的模式，跨宽度的 `vcvt` 不需要写 `part=EVEN/ODD` 或 `part=P0~P3`，用户只写逻辑语义。物理布局由 `pto.as` 自动推断和传播。

### 4.1 vmi.vreg 的 256B 约束与 FP8 的倍增问题

根据 [RFC-VPTO-Logical-Vector-ISA.md](../../PTO-Gym/docs/RFC-VPTO-Logical-Vector-ISA.md) §3.1，`!pto.vmi.vreg<L x T>` 要求 `L * bitwidth(T)` 必须是 **256B (2048bit)** 的整数倍：

| T | bitwidth(T) | L 必须是…的倍数 | 单 reg 容纳 |
|---|-------------|----------------|-------------|
| f32 / i32 | 32 | 64 | 64 |
| f16 / bf16 / i16 | 16 | 128 | 128 |
| i8 / ui8 (FP8) | 8 | **256** | 256 |

问题出在 FP8 输出端：128 个 FP8 值占 128 × 8bit = 1024bit = 128B，**不是 256B 的整数倍**。`!pto.vmi.vreg<128xui8>` 是非法类型。最小合法 L 对 ui8 是 256，因此逻辑输出类型必须是 `!pto.vmi.vreg<256xui8>`（2 个物理 ui8 reg）。

这意味着一次 vmi 循环迭代需要处理 **256 个 bf16 输入 → 256 个 FP8 输出**，而不是 128。循环粒度翻倍，但逻辑语义不变——只是每次迭代覆盖两行 128-element 数据。

### 4.2 完整 pto.vmi 代码

```mlir
// ===== 全局 predicate =====
// PAT_ALL = 全 active mask, 对完整 tile 不做任何 lane 过滤, 只是 pto 语法的必要 operand
// 这个例子中 predicate 的真正作用在 Step 2 的 lowering 里: bf16→f32 加宽时 predicate 族从 b16 变到 b32,
// punpack 把一个 128-bit b16 mask 拆成两个 64-bit b32 mask (适配字节粒度), 不是"选择哪些 lane active"
%mb16 = pto.pset_b16 "PAT_ALL" : !pto.vmi.mask<128xb16>    // bf16 族全 active
%mb32 = pto.pset_b32 "PAT_ALL" : !pto.vmi.mask<64xb32>    // f32 族全 active

// ===== Step 1: 加载 bf16 输入 tile (256 bf16 = 2 物理 reg) =====
%input_bf16 = pto.vmi.vlds %inUb[%elemOff] : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<256xbf16>
//  ↑ 物理 lowering: 2 × pto.mi.vlds {dist="NORM"} → 2 × !pto.mi.vreg<128xbf16>
//  ↑ layout: axes=[#axis<"chunk", 2, None, 1>], is_contiguous

// ===== Step 2: bf16 → f32 加宽 (256 bf16 → 256 f32 = 4 物理 reg) =====
// RFC 核心: 一条 pto.vmi.vcvt，不写 part=EVEN/ODD
// 逻辑: 256 bf16 → 256 f32（逻辑连续）
// 物理: 2 个 bf16 reg 各拆成 vcvt EVEN + vcvt ODD → 4 个 f32 reg
%input_f32 = pto.vmi.vcvt %input_bf16 : !pto.vmi.vreg<256xbf16> -> !pto.vmi.vreg<256xf32>
//  ↑ pto.as lowering:
//    // predicate 族变换: bf16→f32 时 b16→b32
//    %mb32_lo  = pto.mi.punpack %mb16, "LOWER"  : !pto.mi.mask<b16> -> !pto.mi.mask<b32>
//    %mb32_hi  = pto.mi.punpack %mb16, "HIGHER" : !pto.mi.mask<b16> -> !pto.mi.mask<b32>
//    // 每个 bf16 reg 各做 EVEN/ODD, 共 4 个 f32 reg
//    %in_c0_e = pto.mi.vcvt %bf16_c0, %mb16 {part="EVEN"} : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
//    %in_c0_o = pto.mi.vcvt %bf16_c0, %mb16 {part="ODD"}  : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
//    %in_c1_e = pto.mi.vcvt %bf16_c1, %mb16 {part="EVEN"} : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
//    %in_c1_o = pto.mi.vcvt %bf16_c1, %mb16 {part="ODD"}  : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
//  ↑ layout: axes=[#axis<"chunk", 2, None, 1>, #axis<"parity", 2, PART_EVEN/ODD, 2>]

// ===== Step 3: 加载 scale 并广播 =====
// scale 是 1 个 f32 值, 通过 BRC_B32 广播到 64-lane f32 vreg
%scale_f32 = pto.vmi.vlds %scaleUb[%scaleOff] {dist = "BRC_B32"}
    : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ 物理 lowering: pto.mi.vlds {dist="BRC_B32"} → !pto.mi.vreg<64xf32> (每个 lane 相同值)
//  ↑ layout: 广播, 单物理 reg

// ===== Step 4: Rescale (§2.5) — f32 × scale =====
// Category A op: chunk + parity 交织态下 4 路 fan-out
// scale_f32 是 64-lane 广播 vreg, pto.as 自动在每个 f32 半区上各应用一次
%rescaled = pto.vmi.vmul %input_f32, %scale_f32 : !pto.vmi.vreg<256xf32>
//  ↑ pto.as lowering (4 路 fan-out, chunk × parity):
//    %re_c0_e = pto.mi.vmul %in_c0_e, %scale_f32, %mb32_lo : !pto.mi.vreg<64xf32>...
//    %re_c0_o = pto.mi.vmul %in_c0_o, %scale_f32, %mb32_hi : !pto.mi.vreg<64xf32>...
//    %re_c1_e = pto.mi.vmul %in_c1_e, %scale_f32, %mb32_lo : !pto.mi.vreg<64xf32>...
//    %re_c1_o = pto.mi.vmul %in_c1_o, %scale_f32, %mb32_hi : !pto.mi.vreg<64xf32>...
//  ↑ layout 透传: axes=[#axis<"chunk", 2>, #axis<"parity", 2>]

// ===== Step 5: f32 → FP8 窄化 (§2.6 Narrow Cast) =====
// RFC 核心: 一条 pto.vmi.vcvt，不写 part=P0~P3
// 逻辑: 256 f32 → 256 FP8 (ui8)（逻辑连续）
// 物理: 每个 64-lane f32 reg 各做 vcvt {part=P0} → !pto.mi.vreg<256xui8>
//       每个 vcvt P0 产出 64 个有效 FP8 值（P0 位有值, P1/P2/P3 位为零）
//       共 4 次 vcvt P0, 产出 4 × 64 = 256 个 FP8 值
%quantized = pto.vmi.vcvt %rescaled {rnd = "R", sat = "SAT"}
    : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xui8>
//  ↑ pto.as lowering:
//    // 4 次 vcvt {part=P0}, 每次输入 64-lane f32, 输出 256-lane ui8 (P0 位)
//    %fp8_c0_e = pto.mi.vcvt %re_c0_e, %mb32_lo {rnd="R", sat="SAT", part="P0"}
//        : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
//    %fp8_c0_o = pto.mi.vcvt %re_c0_o, %mb32_hi {rnd="R", sat="SAT", part="P0"}
//        : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
//    %fp8_c1_e = pto.mi.vcvt %re_c1_e, %mb32_lo {rnd="R", sat="SAT", part="P0"}
//        : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
//    %fp8_c1_o = pto.mi.vcvt %re_c1_o, %mb32_hi {rnd="R", sat="SAT", part="P0"}
//        : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
//  ↑ layout: axes=[#axis<"chunk", 2>, #axis<"parity", 2>, #axis<"sub_part", 4, P0, 4>]
//     其中 sub_part 轴记录 P0 placement (仅填最低 8-bit, 3 个空洞用 PK4_B32 消费)

// ===== Step 6: Store FP8 output (§2.6 Store) =====
// 明确的 pto.vmi.vsts 将 FP8 结果写回 UB
// pto.as 自动将 sub_part 轴消费为 PK4_B32 打包存储
// 4 次 vsts PK4_B32, 每次写 64 bytes, 共 256 bytes
pto.vmi.vsts %quantized, %outUb[%outOff] : !pto.vmi.vreg<256xui8>, !pto.ptr<ui8, ub>
//  ↑ pto.as lowering:
//    // PK4_B32: 取每个 256-lane ui8 vreg 的最低 8-bit, 连续写入 64 bytes
//    pto.mi.vsts %fp8_c0_e, %outUb[%outOff + 0],   %mb32_lo {dist="PK4_B32"}
//    pto.mi.vsts %fp8_c0_o, %outUb[%outOff + 64],  %mb32_hi {dist="PK4_B32"}
//    pto.mi.vsts %fp8_c1_e, %outUb[%outOff + 128], %mb32_lo {dist="PK4_B32"}
//    pto.mi.vsts %fp8_c1_o, %outUb[%outOff + 192], %mb32_hi {dist="PK4_B32"}
```

### 4.3 pto.vmi 写法的关键特征

1. **无 `part` 属性**：`pto.vmi.vcvt` 不写 `EVEN/ODD` 或 `P0~P3`，物理布局由 `pto.as` 根据 layout 推断自动插入
2. **无 `vintlv` / `vdintlv`**：parity 交织态下直接用 Category A op fan-out 运算，不需要重建连续态
3. **无 `punpack` 手写**：predicate 族变换由 `pto.as` 在 vcvt lowering 时自动插入
4. **无 `PK4_B32` 手写**：FP8 打包存储由 `pto.vmi.vsts` lowering 自动生成——sub_part 轴消费为 PK4_B32
5. **Step 6 是明确的 `pto.vmi.vsts`**：计算（vcvt）和存储（vsts）在 pto.vmi 层分离，用户显式控制何时写入 UB。`pto.as` 负责将 sub_part 轴的空洞消费为 PK4_B32 打包，用户不需要知道
6. **逻辑类型满足 256B 约束**：FP8 输出使用 `!pto.vmi.vreg<256xui8>`（256 × 8bit = 256B），使得每次迭代处理 256 bf16 → 256 FP8。循环粒度翻倍，但语义不变

---

## 5. AFTER（方案 B）：`pto.mi` 原始微指令手写（展示 lowering 结果）

这是**不**用 RFC 推荐方式时，用户被迫手写的代码——即 §4.2 的 `pto.vmi` 被 `pto.as` lowering 后的物理形态。对应 256 bf16 → 256 FP8 的循环迭代：

```mlir
%m16 = pto.mi.pset_b16 "PAT_ALL" : !pto.mi.mask<b16>
%m32 = pto.mi.pset_b32 "PAT_ALL" : !pto.mi.mask<b32>

// ===== Predicate 族变换: bf16→f32 加宽时 b16→b32 =====
// bf16 用 b16 族 predicate (128-bit, 每 lane 1 bit)
// f32 需要 b32 族 predicate (64-bit, 每 lane 1 bit)
// 加宽后每个 bf16 reg → 2 × 64 f32, 每个半区需要自己的 b32 predicate
%mb32_lo = pto.mi.punpack %m16, "LOWER"  : !pto.mi.mask<b16> -> !pto.mi.mask<b32>
%mb32_hi = pto.mi.punpack %m16, "HIGHER" : !pto.mi.mask<b16> -> !pto.mi.mask<b32>

// ===== Step 1: 加载 bf16 输入 tile (2 个 bf16 reg = 256 bf16) =====
%bf16_c0 = pto.mi.vlds %inUb[%elemOff + 0]   {dist = "NORM"} : !pto.ptr<bf16, ub> -> !pto.mi.vreg<128xbf16>
%bf16_c1 = pto.mi.vlds %inUb[%elemOff + 128] {dist = "NORM"} : !pto.ptr<bf16, ub> -> !pto.mi.vreg<128xbf16>

// ===== Step 2: bf16 → f32 加宽: 必须手写 EVEN/ODD (每个 bf16 reg 各拆一次) =====
// 每个 128-lane bf16 reg 加宽后拆成两个 64-lane f32 reg
// EVEN 取偶数索引; ODD 取奇数索引
%in_c0_e = pto.mi.vcvt %bf16_c0, %m16 {part = "EVEN"}
    : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
%in_c0_o = pto.mi.vcvt %bf16_c0, %m16 {part = "ODD"}
    : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
%in_c1_e = pto.mi.vcvt %bf16_c1, %m16 {part = "EVEN"}
    : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
%in_c1_o = pto.mi.vcvt %bf16_c1, %m16 {part = "ODD"}
    : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>

// ===== Step 3: scale 广播 =====
%scale = pto.mi.vlds %scaleUb[%scaleOff] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.mi.vreg<64xf32>

// ===== Step 4: Rescale: 4 路 vmul (chunk × parity) =====
// Category A op fan-out: 4 个 f32 reg 各做一次 vmul
// scale 是 64-lane 广播 vreg, 与每个 parity 半区的 64-lane f32 匹配
%re_c0_e = pto.mi.vmul %in_c0_e, %scale, %mb32_lo
    : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%re_c0_o = pto.mi.vmul %in_c0_o, %scale, %mb32_hi
    : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%re_c1_e = pto.mi.vmul %in_c1_e, %scale, %mb32_lo
    : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
%re_c1_o = pto.mi.vmul %in_c1_o, %scale, %mb32_hi
    : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>

// ===== Step 5: f32 → FP8 窄化: 每个 f32 reg 各做 vcvt {part=P0} =====
// P0 是 4-way sub-part placement 的第 0 区: 结果写到每个 32-bit lane group 的最低 8-bit
// 不是 P0/P1 分别填两个半区——所有 4 个 f32 reg 各做 P0, 产出各自的 64 FP8 值
// 输出类型: !pto.mi.vreg<256xui8> (256 × 8bit = 256B, 满足物理 reg 约束)
// 其中只有 P0 位有值 (64 个 FP8), P1/P2/P3 位为零
%fp8_c0_e = pto.mi.vcvt %re_c0_e, %mb32_lo {rnd = "R", sat = "SAT", part = "P0"}
    : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
%fp8_c0_o = pto.mi.vcvt %re_c0_o, %mb32_hi {rnd = "R", sat = "SAT", part = "P0"}
    : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
%fp8_c1_e = pto.mi.vcvt %re_c1_e, %mb32_lo {rnd = "R", sat = "SAT", part = "P0"}
    : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>
%fp8_c1_o = pto.mi.vcvt %re_c1_o, %mb32_hi {rnd = "R", sat = "SAT", part = "P0"}
    : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xui8>

// ===== Step 6: FP8 packed store: 4 条 vsts PK4_B32 =====
// PK4_B32: 取每个 256-lane ui8 vreg 的最低 8-bit (P0 位), 连续写入 64 bytes
// 4 × 64 bytes = 256 bytes = 256 个 FP8 值
pto.mi.vsts %fp8_c0_e, %outUb[%outOff + 0],   %mb32_lo {dist = "PK4_B32"}
    : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>
pto.mi.vsts %fp8_c0_o, %outUb[%outOff + 64],  %mb32_hi {dist = "PK4_B32"}
    : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>
pto.mi.vsts %fp8_c1_e, %outUb[%outOff + 128], %mb32_lo {dist = "PK4_B32"}
    : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>
pto.mi.vsts %fp8_c1_o, %outUb[%outOff + 192], %mb32_hi {dist = "PK4_B32"}
    : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>
```

---

## 6. 物理流程全景图

以下图示展示 256 bf16 → 256 FP8 完整管线在 A5 SIMD 寄存器上的物理效果。每次 vmi 循环迭代处理 256 bf16 元素（2 个 bf16 reg），以满足 vmi.vreg 的 256B 约束。

```
UB: bf16 input [h0, ..., h127, h128, ..., h255]  (512 bytes, 2 × 128 bf16 reg)
         │ 2 × vlds NORM
         ▼
┌─────────────────────────────────────────┐
│ bf16_c0: [h0, ..., h127]               │  128 bf16 lanes, 256B
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ bf16_c1: [h128, ..., h255]             │  128 bf16 lanes, 256B
└─────────────────────────────────────────┘
         │                    │
    ┌────┴────┐          ┌────┴────┐
 vcvt EVEN  vcvt ODD   vcvt EVEN  vcvt ODD    ← Part (parity 轴产出, 每个 bf16 reg 各拆一次)
    │         │          │         │
    ▼         ▼          ▼         ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ in_c0_e  │ │ in_c0_o  │ │ in_c1_e  │ │ in_c1_o  │  ← 4 × 64-lane f32
│ 64 f32   │ │ 64 f32   │ │ 64 f32   │ │ 64 f32   │     chunk×parity 交织态
│ h0,h2,...│ │ h1,h3,...│ │ h128,h130│ │ h129,h131│
│ ...,h126 │ │ ...,h127 │ │ ...,h254 │ │ ...,h255 │
└──────────┘ └──────────┘ └──────────┘ └──────────┘

Predicate 族变换 (b16 → 2 × b32):
┌──────────────────────────────────┐
│ preg_b16: ■■■■■■■■│■■■■■■■■■■■■│  ← 128-bit b16 mask
└──────────────────────────────────┘
         │
    ┌────┴────┐
 punpack     punpack
 LOWER       HIGHER
    │         │
    ▼         ▼
┌──────────┐ ┌──────────┐
│ mb32_lo  │ │ mb32_hi  │  ← 各 64-bit b32 predicate
│ 64-bit   │ │ 64-bit   │     分别用于 EVEN/ODD 半区
│ b32 族   │ │ b32 族   │
└──────────┘ └──────────┘

         │         │          │         │
 4 × vmul × scale (BRC_B32 broadcast, 64-lane f32)
         │         │          │         │
    ┌────┴────┐          ┌────┴────┐
    ▼         ▼          ▼         ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ re_c0_e  │ │ re_c0_o  │ │ re_c1_e  │ │ re_c1_o  │  ← rescaled f32
│ 64 f32   │ │ 64 f32   │ │ 64 f32   │ │ 64 f32   │     仍为 chunk×parity 交织态
└──────────┘ └──────────┘ └──────────┘ └──────────┘
         │         │          │         │
 vcvt P0  vcvt P0   vcvt P0  vcvt P0               ← Part_T (FP8 窄化)
    │         │          │         │                   每个 64-lane f32 reg → 64 FP8 值
    ▼         ▼          ▼         ▼                   P0 位有值, P1/P2/P3 位为零
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ fp8_c0_e │ │ fp8_c0_o │ │ fp8_c1_e │ │ fp8_c1_o │  ← 256-lane ui8 vreg (P0 位)
│ 64 FP8   │ │ 64 FP8   │ │ 64 FP8   │ │ 64 FP8   │     每个 32-bit lane 的低 8-bit = FP8
│ 256B reg │ │ 256B reg │ │ 256B reg │ │ 256B reg │     满足 256B 物理 reg 约束
└──────────┘ └──────────┘ └──────────┘ └──────────┘

FP8 写回 (4 条 vsts PK4_B32):
  vsts(fp8_c0_e, addr + 0,   PK4_B32, mb32_lo)  ← 第 1 组 64 FP8 (64 bytes)
  vsts(fp8_c0_o, addr + 64,  PK4_B32, mb32_hi)  ← 第 2 组 64 FP8 (64 bytes)
  vsts(fp8_c1_e, addr + 128, PK4_B32, mb32_lo)  ← 第 3 组 64 FP8 (64 bytes)
  vsts(fp8_c1_o, addr + 192, PK4_B32, mb32_hi)  ← 第 4 组 64 FP8 (64 bytes)
         │         │          │         │
         ▼         ▼          ▼         ▼
UB: fp8 output [fp8[0], fp8[1], ..., fp8[255]]  (256 bytes)
```

---

## 7. Part (EVEN/ODD) vs Part_T (P0~P3) 的关键区别

这是之前代码示例出错的核心原因。根据 [Pack-Unpack-Interleave-Part-Reference.md](../../PTO-Gym/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md) §2.1–2.2 的梳理：

### Part (EVEN/ODD) — 2-way，跨 reg 级别

```
1 个 bf16 vreg (128 lanes, 256B)
         │
    ┌────┴────┐
 vcvt EVEN  vcvt ODD
    │         │
    ▼         ▼
1 × f32 vreg (64 lanes)   1 × f32 vreg (64 lanes)
 = 偶数位 bf16 加宽        = 奇数位 bf16 加宽
 h0,h2,...,h126 → f32      h1,h3,...,h127 → f32

产出: 2 个 f32 reg, parity 交织态
```

- **EVEN 取偶数索引元素**（h0, h2, h4, ..., h126），加宽到 f32
- **ODD 取奇数索引元素**（h1, h3, h5, ..., h127），加宽到 f32
- 两者在物理上是**交织的半区**，不是逻辑连续的前半/后半
- 后续 Category A op（如 vadd/vmul）可以保持 parity 交织态直接运算（fan-out），不需要 vintlv 重建连续态

### Part_T (P0~P3) — 4-way，reg 内 lane 级别

```
1 个 f32 vreg (64 lanes, 256B)
         │
 vcvt {part=P0}
         │
         ▼
1 个 ui8 vreg (128 lanes, 256B)
 每个 32-bit lane group (4 bytes):
   ┌──────────┬─────┬─────┬─────┐
   │ fp8 (P0) │ P1  │ P2  │ P3  │
   │  低8bit  │零   │零   │零   │
   └──────────┴─────┴─────┴─────┘

P0: 结果写到每个 32-bit lane group 的最低 8-bit (1 byte)
P1: 结果写到第 2 个 8-bit
P2: 结果写到第 3 个 8-bit
P3: 结果写到第 4 个 8-bit

存储时用 PK4_B32: 取每个 lane 的最低 8-bit → 连续 FP8 数组
```

- **P0 是同一 reg 内的 sub-part placement**，不是跨 reg 的半区选择
- 在 block_quant 的场景中：4 个 f32 reg（chunk × parity）各做 **P0**（不是 P0/P1/P2/P3 分别填 4 个半区）
- 每个 f32 reg `vcvt {part=P0}` 产出 64 个 FP8 值，写到一个 256-lane ui8 vreg (`!pto.mi.vreg<256xui8>`) 的 P0 位
- 最终 4 条 `vsts PK4_B32` 各写 64 bytes，共 256 FP8 值

### 常见误区

| 错误理解 | 正确理解 |
|----------|----------|
| "P0 填 EVEN 半区，P1 填 ODD 半区" | P0 是 **reg 内** 4-way placement 的第 0 区。两个 parity 半区各做 P0，产出各自的 FP8 结果 |
| "128 f32 → 128 FP8 需要做 P0/P1/P2/P3 四次 vcvt" | 128 f32 在 parity 交织态下是 2 × 64 f32。每个 64-lane reg 做 P0 即可产出 64 FP8 值。4 次 vcvt 只出现在单 reg 产出 256 个 FP8 值的场景 |
| "PK4_B32 是把 4 个 FP8 打包进 1 个 f32 lane" | PK4_B32 是 **存储端打包**：取每个 256-lane ui8 vreg 的 P0 位（每个 32-bit lane group 的最低 8-bit），连续写入 UB。每次 vsts PK4_B32 写 64 bytes (64 FP8 值) |
| "`!pto.vmi.vreg<128xui8>` 是合法的 FP8 输出类型" | `128 × 8bit = 128B`，不是 256B 的整数倍，**非法**。FP8 输出的最小合法逻辑长度是 256：`!pto.vmi.vreg<256xui8>` (256 × 8bit = 256B)。这导致 vmi 循环粒度翻倍：256 bf16 → 256 FP8 |

---

## 8. Predicate 族变换：punpack 的必要性

**predicate 在这个例子里有两个作用，但大多数人只注意到第一个（平凡），忽略第二个（关键）**：

1. **`PAT_ALL` mask** — 语法必需但语义平凡。完整 tile 下所有 lane 都参与计算，mask 不做任何过滤。真正需要 lane 过滤的场景是边界 tile（如末尾只有 97 行而非 128 行），此时用 `plt/pge` 生成部分 active mask 让尾部 lane 输出零。

2. **`punpack` 族变换** — 不是过滤 lane，而是**适配 predicate 的字节粒度**。这是本例 predicate 的真正核心作用，下文详细解释。

根据 [Pack-Unpack-Interleave-Part-Reference.md](../../PTO-Gym/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md) §2.5：

**为什么要变换 predicate 族？**

A5 的 predicate register 是 256-bit，但每个 bit 控制的字节数取决于当前数据的 dtype 族：

- **b16 族**：1 bit 控制 2 bytes = 1 个 bf16/f16/i16 lane。128-bit 就能覆盖 128 个 bf16 lanes。
- **b32 族**：1 bit 控制 4 bytes = 1 个 f32/i32 lane。64-bit 就能覆盖 64 个 f32 lanes。

bf16→f32 加宽后，数据从 b16 族（128 lanes × 2B = 256B）变成 b32 族（64 lanes × 4B = 256B）。一个物理 reg 的总字节数不变（256B），但 lane 数减半、每 lane 字节数翻倍。**predicate 必须同步变换**，否则 1 bit 控制的字节数和数据的字节宽度不匹配——运算结果会错位。

```
bf16 用 b16 族 predicate (每 1 bit 控制 1 个 bf16 lane):
┌─────────────────────────────────────────┐
│ preg_b16 │ ■ ■ ■ ■ ■ ■ ■ ■ │ ■ ■ ■ ■ ■ ■ ■ ■ │  ← 128-bit
└─────────────────────────────────────────┘
  ← 前 64 位 →          ← 后 64 位 →

f32 需要 b32 族 predicate (每 1 bit 控制 1 个 f32 lane):
加宽后 128 bf16 → 2 × 64 f32
每个 f32 vreg 需要自己的 64-bit b32 predicate

punpack LOWER: 取前 64-bit 展开, 后 64-bit 清零 → mb32_lo
punpack HIGHER: 取后 64-bit 展开, 前 64-bit 清零 → mb32_hi

使用:
  vcvt %in_bf16, %m16 {part="EVEN"} → 用 %m16 (b16 族, 128-bit)
  vmul %in_e, %scale, %mb32_lo      → 用 %mb32_lo (b32 族, 64-bit)
  vmul %in_o, %scale, %mb32_hi      → 用 %mb32_hi (b32 族, 64-bit)
  vsts %fp8_lo, ..., %mb32_lo       → 用 %mb32_lo (b32 族, PK4_B32)
  vsts %fp8_hi, ..., %mb32_hi       → 用 %mb32_hi (b32 族, PK4_B32)
```

**关键点**：

- `vcvt {part=EVEN/ODD}` 使用 **b16 族 predicate**（因为源是 bf16，每 bit 控制 2B）
- 加宽后的 f32 运算使用 **b32 族 predicate**（因为目标是 f32，每 bit 控制 4B）
- `punpack` 是从 b16 到 b32 的必要桥梁，**不做 lane 过滤**——它只是把 predicate 的字节粒度从"1 bit = 2B"适配到"1 bit = 4B"
- 在 `pto.vmi` 层面，`punpack` 由 `pto.as` 自动插入，用户不需要手写

---

## 9. 三层对比总结

| 维度 | TileLang `T.Parallel` | `pto.vmi` (RFC 推荐) | `pto.mi` (物理微指令) |
|------|----------------------|----------------------|----------------------|
| **循环粒度** | 128 bf16 → 128 FP8 (一次 `T.Parallel`) | 256 bf16 → 256 FP8 (一次 vmi 循环) | 256 bf16 → 256 FP8 (4 × 64 f32 → 4 × vcvt P0) |
| **bf16→f32 widen** | 编译器隐式插入 | `pto.vmi.vcvt`（无 `part`） | 4 × `vcvt {part="EVEN/ODD"}` (每个 bf16 reg 各拆一次) |
| **predicate 族变换** | 不暴露 | `pto.as` 自动插 `punpack` | 手写 `punpack LOWER` + `punpack HIGHER` |
| **scale 广播** | `sf_fragment` 隐式 | `pto.vmi.vlds {dist="BRC_B32"}` | `pto.mi.vlds {dist="BRC_B32"}` |
| **Rescale (§2.5)** | `x * sf` 一行隐式 | `pto.vmi.vmul`（chunk×parity fan-out 自动） | 4 路 `vmul` + 各用 `mb32_lo/hi` |
| **f32→FP8 narrow (§2.6)** | 编译器隐式插入 | `pto.vmi.vcvt {rnd, sat}`（无 `part`） | 每个 64-lane f32 reg 做 `pto.mi.vcvt {part="P0"}` → `!pto.mi.vreg<256xui8>` |
| **FP8 store** | `out[...] = ...` 隐式 | **明确的** `pto.vmi.vsts`（PK4_B32 由 pto.as 自动） | 4 × `vsts {dist="PK4_B32"}` |
| **FP8 输出逻辑类型** | 无类型约束 (GPU 原生 FP8) | `!pto.vmi.vreg<256xui8>` (256B 约束) | `!pto.mi.vreg<256xui8>` (物理 256B) |
| **用户需关心的物理细节** | 零（不可控） | 零（layout 由 `pto.as` 推断） | 全部（EVEN/ODD + P0 + punpack + PK4_B32 + 256B 约束) |
| **可读性** | 最低（Rescale+Cast 不可见） | 最高（每步语义清晰，计算/存储分离） | 可控但心智负担重 |
| **可调试性** | 差（需看 CUDA dump） | 好（逻辑 op 一对一映射算法步骤） | 直接对应硬件，但代码膨胀 |

### RFC 的核心收益

1. **源码只表达算法意图**：Rescale = `vmul`，Narrow Cast = `vcvt`，Store = `vsts`——每步与参考文档 §2.2–2.6 的算法步骤一一对应
2. **物理布局自动推断**：`pto.as` 根据 `pto.vmi.vcvt` 的输入输出类型推断 parity 轴、Part_T 轴，自动插入 EVEN/ODD、P0、punpack、PK4_B32
3. **与 TileLang `T.Parallel` 对齐**：`T.Parallel` 的逻辑迭代空间几乎一对一直译成 `pto.vmi` 的逻辑向量 op，宽度转换不再泄漏回用户或 codegen

### RFC 需要补充的关键点（来自 Pack-Unpack-Interleave-Part-Reference.md §八）

1. **Part_T (P0~P3) 轴**：parity 只有 2-way，但 FP8/fp4 需要 4-way sub-part placement。LayoutDescriptor 的 `axes` 需要增加 `sub_part` 轴（cardinality=4, stride=4）来映射 P0~P3
2. **Predicate 族变换是独立于 vreg layout 的维度**：`punpack/ppack` 不在 vreg layout 轴内，是 mask 的独立变换。`!pto.vmi.mask<L x G>` 需要增加 predicate 族转换规则
3. **PK4_B32 不是简单的 width 轴消费**：它是 f32→FP8 的 4 倍窄化存储，涉及 Part_T(P0) + pack-store 两步，需要特殊 lowering 规则
4. **Channel Split/Merge 是独立于 parity 的交织维度**：4 通道和 2 通道交织与 parity（stride-2）的语义不同，可能需要单独的 `channel` 轴

---

## 10. 参考：pto.mi 指令速查

| 指令 | 语法 | 用途 | 延迟 |
|------|------|------|------|
| `pto.mi.vlds {dist="NORM"}` | `%v = pto.mi.vlds %ub[%off] {dist="NORM"} : !pto.ptr<T, ub> → !pto.mi.vreg<NxT>` | 连续加载 | 9 |
| `pto.mi.vlds {dist="BRC_B32"}` | `%v = pto.mi.vlds %ub[%off] {dist="BRC_B32"} : !pto.ptr<f32, ub> → !pto.mi.vreg<64xf32>` | 广播加载 | 9 |
| `pto.mi.vlds {dist="UNPK_B16"}` | `%v = pto.mi.vlds %ub[%off] {dist="UNPK_B16"} : !pto.ptr<bf16, ub> → !pto.mi.vreg<64xf32>` | 加宽 unpack 加载 | 9 |
| `pto.mi.vcvt {part="EVEN"}` | `%r = pto.mi.vcvt %in, %mask {part="EVEN"} : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> → !pto.mi.vreg<64xf32>` | bf16→f32 加宽 (偶数位) | 7 |
| `pto.mi.vcvt {part="ODD"}` | `%r = pto.mi.vcvt %in, %mask {part="ODD"} : !pto.mi.vreg<128xbf16>, !pto.mi.mask<b16> → !pto.mi.vreg<64xf32>` | bf16→f32 加宽 (奇数位) | 7 |
| `pto.mi.vcvt {part="P0"}` | `%r = pto.mi.vcvt %in, %mask {rnd="R", sat="SAT", part="P0"} : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> → !pto.mi.vreg<256xui8>` | f32→FP8 窄化 (P0 位, 64 FP8 值有效) | — |
| `pto.mi.vmul` | `%r = pto.mi.vmul %lhs, %rhs, %mask : !pto.mi.vreg<NxT>, !pto.mi.vreg<NxT>, !pto.mi.mask<G> → !pto.mi.vreg<NxT>` | 逐 lane 乘法 | 8 |
| `pto.mi.vsts {dist="NORM_B32"}` | `pto.mi.vsts %v, %ub[%off], %mask {dist="NORM_B32"} : !pto.mi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mi.mask<b32>` | 连续 f32 存储 | 9 |
| `pto.mi.vsts {dist="PK4_B32"}` | `pto.mi.vsts %v, %ub[%off], %mask {dist="PK4_B32"} : !pto.mi.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mi.mask<b32>` | FP8 打包存储 (取 P0 位, 写 64 bytes) | 9 |
| `pto.mi.vstsx2 "INTLV_B32"` | `pto.mi.vstsx2 %lo, %hi, %ub[%off], "INTLV_B32", %mask : ...` | 交织 f32 存储 | 12 |
| `pto.mi.punpack "LOWER"` | `%r = pto.mi.punpack %mask, "LOWER" : !pto.mi.mask<b16> → !pto.mi.mask<b32>` | predicate 加宽 (前半) | — |
| `pto.mi.punpack "HIGHER"` | `%r = pto.mi.punpack %mask, "HIGHER" : !pto.mi.mask<b16> → !pto.mi.mask<b32>` | predicate 加宽 (后半) | — |

---

## 11. 遗留问题：predicate 在尾块场景下的写法和设计

本文档 §4–§5 的代码示例全部使用 `PAT_ALL` predicate（全 active，不过滤任何 lane），仅覆盖**完整 tile** 场景。但在实际内核中，矩阵行数/列数往往不是 block 大小的整数倍，最后一个 tile 是**尾块（tail block）**——只有部分元素有效，其余应输出零。这一场景下 predicate 的写法和设计存在以下未解决问题：

### 11.1 TileLang 侧：`if` 条件 vs predicate 的映射断层

TileLang 用 Python `if` 语句处理尾块（如 [per_block_cast_lossless_kernel.py](../tile_kernels/quant/per_block_cast_lossless_kernel.py) L114, L131）：

```python
# TileLang: 每个元素逐一判断是否在矩阵范围内
if i * sf_block[0] + pid_token * block_m < num_tokens and j * sf_block[1] + pid_hidden * block_k < hidden:
    x_sf_uint32_fragment[i, j] = ...
```

这个 `if` 的语义是"超出矩阵边界的元素不参与计算"。但在 pto 层面，对应的表达方式是 **predicate mask**：

- `pto.mi` 层面：`plt_b16` / `plt_b32` 逐次生成"前 N 个 lane active"的 predicate
- `pto.vmi` 层面：RFC §3.2 声称 `!pto.vmi.mask<L x G>` 由 `pto.as` 自动根据逻辑长度与矩阵尺寸差值生成尾部 predicate，**用户不需要手写**

**问题**：`T.Parallel` 中的 `if` 是逐元素条件，而 predicate 是逐 lane（8/16/32 字节粒度）的条件。两者**粒度不对齐**：

- `if` 可以精确跳过单个元素（如第 97 个 bf16 值）
- predicate 的最小粒度是 1 lane = 2 bytes (b16) 或 4 bytes (b32)，无法跳过单个 byte

当矩阵边界不落在 lane 边界上时（如 num_tokens=97，不满足 128 的整数倍也不满足 64 的整数倍），predicate 无法精确表达"前 97 个 bf16 active、后 31 个 inactive"——因为 97 不是 b16 lane 数（128）的子集。实际做法是取最接近的 `PAT_VL64` 或 `PAT_VL32` 模式，让前 64 或 32 个 lane active，其余清零。这会在尾块末尾引入少量**零填充元素**，下游 GEMM 需要能容忍这些零值。

### 11.2 `pto.vmi` 层面：mask 类型缺少尾块生成机制

RFC §3.2 定义了 `!pto.vmi.mask<L x G, #layout>`，但只描述了"全 active"和"由 pto.as 自动补齐尾部"两种情况。缺少以下关键机制：

1. **尾块 predicate 的生成方式**：用户在 vmi 层面如何表达"这个 vreg 只有前 N 个元素有效"？是写 `!pto.vmi.mask<97 x b16>`（逻辑长度 97，由 pto.as 自动拆成 `PAT_VL64` + 零填充），还是写 `!pto.vmi.mask<128 x b16>` 配合一个显式的 mask 值？
2. **跨族 predicate 的尾块传递**：bf16→f32 加宽时，`punpack` 把一个 b16 mask 拆成两个 b32 mask。如果原始 b16 mask 是尾块 mask（如 `PAT_VL64`：前 64 个 bf16 lane active），`punpack` 后每个 b32 mask 应该是 `PAT_ALL`（前 64 个 bf16 加宽成 64 个 f32，刚好填满一个 f32 reg）。但如果原始是 `PAT_VL48`（前 48 个 bf16 lane active），`punpack LOWER` 得到的 b32 mask 应该是 `PAT_VL24`——pto.as 如何自动推导这个？
3. **FP8 (ui8) 的尾块 predicate**：ui8 的 lane 数是 256，一个 vreg 容纳 256 个 FP8 值。但 256B 约束使得 vmi 循环粒度翻倍（256 bf16 → 256 FP8），尾块的零填充粒度更粗。当实际有效元素数为 97 时，FP8 输出的尾块 predicate 应该让前 97 个 FP8 lane active、后 159 个清零——但 `!pto.vmi.mask<97 x b8>` 中的 b8 族 predicate 粒度是 1 byte，97 不是 256 的倍数。如何处理？

### 11.3 `pto.mi` 层面：`plt` 的循环模式和跨族传递

在 `pto.mi` 层面，尾块通过 `plt_b16` / `plt_b32` 在循环中逐次消耗 `remaining` 计数器生成 predicate：

```mlir
%mask, %remaining = pto.mi.plt_b16 %remaining : i32 -> !pto.mi.mask<b16>, i32
```

但存在以下问题：

1. **`plt` 与 `punpack` 的组合顺序**：应该先 `punpack` 再 `plt`（把 b16 mask 拆成 b32，再对每个 b32 mask 做 `plt`），还是先 `plt` 再 `punpack`（先对 b16 mask 做 `plt`，再 `punpack` 拆成 b32）？两种顺序产生不同的尾块行为，目前没有明确规则。
2. **`plt` 在 vcvt P0 后的应用**：`vcvt {part=P0}` 把 64-lane f32 reg 映射到 256-lane ui8 reg。原始 f32 的尾块 predicate（如 `PAT_VL32`：前 32 个 f32 lane active）对应 FP8 输出的多少个 lane active？P0 placement 使得每个 f32 lane group 的最低 8-bit 才是有效 FP8，predicate 粒度从 b32 变到 b8，`plt` 需要重新计算 remaining。
3. **`plt` 与 PK4_B32 store 的交互**：PK4_B32 取每个 256-lane ui8 vreg 的最低 8-bit 写入 64 bytes。如果 predicate 是 `PAT_VL32`（前 32 个 f32 lane active），PK4_B32 实际写入多少 bytes？是 32 bytes（32 个 FP8 值），还是 64 bytes（predicate 被 reinterpret 为 b32 族，每个 32-bit lane group 的最低 8-bit 都被提取，包括 inactive lanes 的零值）？

### 11.4 需要进一步讨论的设计方向

| 问题 | 可能的方向 | 需要确认 |
|------|-----------|---------|
| vmi 层面如何表达尾块 | `!pto.vmi.mask<L x G>` 中 `L` 直接写实际有效元素数，pto.as 自动拆成 `PAT_VL*` + 零填充 | L 是否必须满足物理 reg 的 lane 数倍数约束，还是可以写任意值 |
| 跨族 predicate 的尾块传递 | pto.as 在 `punpack` lowering 时自动根据 b16 mask 的 pattern 推导出对应的 b32 mask | 需要明确 `punpack` 对 `PAT_VL*` 模式的推导规则 |
| FP8 尾块的粒度问题 | 接受零填充（尾块 FP8 输出的有效元素数对齐到 b8 lane 数），下游 GEMM 在累加时忽略零值 | 下游 GEMM 是否能容忍零填充？零值在 FP8 E4M3 下是否会被误读为真实值 |
| `plt` + `punpack` 的顺序 | 先 `plt` 再 `punpack`（在原始族上做尾块裁剪，再变换族），与 TileLang `if` 的逻辑对齐更自然 | 需要在实际内核中验证两种顺序的数值结果是否一致 |
| `plt` 在 vcvt P0 后的重算 | vcvt P0 是 B 类 op（消费 parity 轴），predicate 族从 b32 变到 b8，`plt` 的 remaining 需要乘 4（每个 f32 lane → 4 个 ui8 lane） | pto.as 是否自动处理这种 remaining 的倍数变换，还是需要用户显式传递 |

这些问题直接影响 per-block quant 内核在尾块场景下的正确性和性能，需要在 RFC 和 PTO SPEC 的后续版本中解决。
