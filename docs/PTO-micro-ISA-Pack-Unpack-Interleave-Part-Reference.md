# PTO micro ISA: Pack / Unpack / Interleave / Part(EVEN/ODD/P0~P3) 全梳理

> 本文档对 `PTO-micro-Instruction-SPEC.md` 中所有涉及"数据物理摆放而非逻辑语义"的指令和分类做系统梳理，配合图示说明每种机制在物理寄存器上的实际效果。

---

## 一、核心概念：为什么需要这些机制？

A5 SIMD 向量寄存器固定为 **256B / 2048bit**。当逻辑数据出现以下情况时，硬件无法用一条直通指令完成：

| 场景 | 问题 | 硬件应对 |
|------|------|----------|
| **2 倍宽度转换** | 128 个 f16 (256B) → 128 个 f32 (需 512B = 2 个物理 reg) | 只能一半一半地写：EVEN 半区 / ODD 半区 |
| **4 倍宽度转换** | 64 个 f32 (256B) → 256 个 FP8 (256B，但 reg 内 4-way packed) | 需要分 4 个 sub-part (P0/P1/P2/P3) |
| **AoS↔SoA 转换** | 内存中 `[X0,Y0,X1,Y1,...]` → 寄存器 `[X0,X1,...]` + `[Y0,Y1,...]` | 需要 interleave/deinterleave |
| **窄化存储** | f32 reg → f16 内存（物理减半） | 需要 pack-store |
| **FP8 存储** | f32 reg → FP8 内存（只取低 8 位） | 需要 PK4_B32 pack-store |
| **Predicate 跨宽度族** | b16 mask → b32 mask（加宽时 predicate 族变了） | 需要 punpack 拆分 |

**一句话：** 这些机制都是"逻辑连续数据 ↔ 物理交织 reg"的映射手段，它们**不表达计算意图，只表达物理摆放**。

---

## 二、六大分类体系

### 2.1 Part (EVEN / ODD / P0~P3) —— vcvt 的 lane 位置选择器

Part 不是一条独立指令，而是 `vcvt` 的属性，控制**宽度转换时结果写在目标 reg 的哪个 lane 组**：

| Part 族 | 值 | 适用场景 | 语义 |
|---------|-----|----------|------|
| **Part** (2-way) | `EVEN` | 加宽 (narrow→wide)：如 f16→f32, i16→i32 | 结果写到偶数位 lane |
| | `ODD` | 加宽 | 结果写到奇数位 lane |
| | `EVEN` | 窄化 (wide→narrow)：如 f32→f16 | 窄化结果写到目标 reg 的偶数位半区 |
| | `ODD` | 窄化 | 窄化结果写到目标 reg 的奇数位半区 |
| **Part_T** (4-way) | `P0` | 4 倍窄化：如 f32→i8, f32→FP8 | 结果写到 4 个 sub-part 的第 0 区 |
| | `P1` | | 第 1 区 |
| | `P2` | | 第 2 区 |
| | `P3` | | 第 3 区 |

#### 2.1.1 f16→f32 加宽（128 个 f16 → 128 个 f32，跨 2 个物理 reg）

```
逻辑视角:  128 个连续 f16 值
           ┌───────────────────────────────────────────────┐
           │ h0  h1  h2  h3  h4  h5 ... h62  h63 │ h64  h65 ... h126 h127 │
           └───────────────────────────────────────────────┘

物理: 1 个 f16 vreg (256B, 128 lanes)
           ┌─────────────────────────────────────────┐
  vreg_f16 │ h0  h1  h2  h3  h4  h5 ... h62  h63  │h64  h65 ... h126 h127│
           └─────────────────────────────────────────┘
                    ↑ 偶数位 ↑       ↑ 奇数位 ↑        ↑ 偶数位 ↑       ↑ 奇数位 ↑

vcvt PART_EVEN → 1 个 f32 vreg (256B, 64 lanes)
           ┌───────────────────────────────────┐
  vreg_e   │ f32(h0)  f32(h2)  f32(h4) ... f32(h126) │
           │  ← 原始偶数位元素加宽为 f32 →           │
           └───────────────────────────────────┘

vcvt PART_ODD → 1 个 f32 vreg (256B, 64 lanes)
           ┌───────────────────────────────────┐
  vreg_o   │ f32(h1)  f32(h3)  f32(h5) ... f32(h127) │
           │  ← 原始奇数位元素加宽为 f32 →           │
           └───────────────────────────────────┘
```

**关键点：** 一个 128-lane f16 reg 加宽后必须拆成两个 64-lane f32 reg。EVEN 取偶数索引 h0,h2,...,h126；ODD 取奇数索引 h1,h3,...,h127。两者在物理上是**交织的半区**，不是逻辑连续的前半后半。

#### 2.1.2 f16→f32 加宽后保持 parity 交织态（不做 vintlv，直接运算）

```
           ┌───────────────────────────────┐
  vreg_e   │ f32(h0)  f32(h2)  ... f32(h126) │  ← parity 轴: EVEN 半区
           └───────────────────────────────┘
           ┌───────────────────────────────┐
  vreg_o   │ f32(h1)  f32(h3)  ... f32(h127) │  ← parity 轴: ODD 半区
           └───────────────────────────────┘

Category A op (如 vadd) 直接 fan-out:
  vadd(vreg_e_result, vreg_e, vreg_bias_e, mask_b32)  ← 在 EVEN 半区做
  vadd(vreg_o_result, vreg_o, vreg_bias_o, mask_b32)  ← 在 ODD 半区做

最终写回内存（parity→连续）:
  vstsx2(vreg_e_result, vreg_o_result, addr, "INTLV_B32", mask)
  → 内存中: [f32(h0), f32(h1), f32(h2), f32(h3), ...] ← 逻辑连续！
```

#### 2.1.3 f32→f16 窄化（2×64→128 元素，EVEN/ODD 各写半区再 OR 合并）

```
两个 f32 输入:
           ┌───────────────────┐
  vreg_in0 │ f0  f1  f2 ... f63 │   ← 第 1 组 64 个 f32
           └───────────────────┘
           ┌───────────────────┐
  vreg_in1 │ f64 f65 f66 ...f127│  ← 第 2 组 64 个 f32
           └───────────────────┘

vcvt {part=EVEN}: 结果写到 f16 vreg 的偶数位 lane
           ┌─────────────────────────────────────────┐
  vreg_even│ h0=f16(f0)  ___  h2=f16(f2)  ___ ... h126=f16(f63) ___ │
           │  ↑偶数位填值      ↑奇数位空(___=0)                     │
           └─────────────────────────────────────────┘

vcvt {part=ODD}: 结果写到 f16 vreg 的奇数位 lane
           ┌─────────────────────────────────────────┐
  vreg_odd │ ___  h1=f16(f0)  ___  h3=f16(f2) ... ___ h127=f16(f63)│
           │     ↑偶数位空    ↑奇数位填值                            │
           └─────────────────────────────────────────┘

vor 合并:
           ┌─────────────────────────────────────────┐
  result   │ h0  h1  h2  h3  h4  h5 ... h62  h63 │h64  h65 ... h126 h127 │
           └─────────────────────────────────────────┘
           = f16(f0) f16(f1) f16(f2) f16(f3) ...  ← 128 个连续 f16！
```

### 2.2 Part_T (P0~P3) —— 4 倍窄化时的 sub-part placement

#### f32→FP8 窄化（64 个 f32 → 64 个 FP8，单 reg 内 4-way packed）

```
f32 vreg (64 lanes, 256B):
           ┌───────────────────────────────────────┐
  vreg_f32 │ f32[0]  f32[1]  f32[2]  f32[3] ... f32[63] │
           └───────────────────────────────────────┘
           每个 f32 占 4 bytes = 32 bits

vcvt {part=P0}: 结果写到 4 个 sub-part 的第 0 区
           ┌───────────────────────────────────────┐
  vreg_fp8 │ fp8[0]  ___  ___  ___ │ fp8[1] ___ ___ ___ │ ... │ fp8[63] ___ ___ ___ │
           │  P0区    P1  P2  P3  │  P0区   P1  P2  P3 │ ... │  P0区    P1  P2  P3  │
           └───────────────────────────────────────┘
           每个 "lane group" 原占 4 bytes (f32位宽)
           → 窄化后只填最低 1 byte (P0位置), 其余 3 bytes 为零

同理 vcvt {part=P1}: 填第 1 byte
      vcvt {part=P2}: 填第 2 byte
      vcvt {part=P3}: 填第 3 byte

存储时用 PK4_B32: 取每个 lane 的最低 8 bit → pack 成连续的 FP8 数组
           ┌───────────────────┐
  内存     │ fp8[0] fp8[1] fp8[2] ... fp8[63] │ ← 连续 64 bytes
           └───────────────────┘
```

**与 Part (EVEN/ODD) 的区别图示：**

```
Part (EVEN/ODD) — 2-way, 跨 reg 级别:
  ┌──────────┐    vcvt EVEN → ┌──────────┐  (前半区)
  │  f16 reg │                 │  f32 reg │  ← 64 个 f32
  │ 128 lane │    vcvt ODD  → ┌──────────┐  (后半区)
  └──────────┘                 │  f32 reg │  ← 64 个 f32

Part_T (P0~P3) — 4-way, reg 内 lane 级别:
  ┌──────────┐    vcvt P0 → ┌──────────┐
  │  f32 reg │               │  目标reg │  每个 32-bit lane group 里只填 8-bit P0 位置
  │  64 lane │    vcvt P1 → ┌──────────┐  每个 32-bit lane group 里只填 8-bit P1 位置
  └──────────┘
```

### 2.3 Interleave / Deinterleave —— 数据交织重组

分为**向量数据**和 **predicate 数据**两个域。

#### 2.3.1 向量 Interleave / Deinterleave（§12）

| 指令 | 输入 | 输出 | 语义 | A5 支持 |
|------|------|------|------|---------|
| `pto.vintlv` | 2 个 vreg | 2 个 vreg (low, high) | 交织合并 | ✅ |
| `pto.vdintlv` | 2 个 vreg | 2 个 vreg (low, high) | 解交织 | ✅ |
| `pto.vintlvv2` | 2 个 vreg + PART | 1 个 vreg | V2 版：只输出 interleave 的一个半区 | ❌ |
| `pto.vdintlvv2` | 2 个 vreg + PART | 1 个 vreg | V2 版：只输出 deinterleave 的一个半区 | ❌ |

#### vintlv：从 parity 交织态恢复成逻辑连续态

```
输入: vcvt 产出的两个 parity 半区
           ┌───────────────────────────┐
  src0     │ f32(h0)  f32(h2)  f32(h4) ... f32(h62) │  ← EVEN 半区 (偶数位原始值)
  (EVEN)   └───────────────────────────┘
           ┌───────────────────────────┐
  src1     │ f32(h1)  f32(h3)  f32(h5) ... f32(h63) │  ← ODD 半区 (奇数位原始值)
  (ODD)    └───────────────────────────┘

vintlv(src0, src1) → dst0, dst1:
           ┌───────────────────────────────────────────┐
  dst0     │ f32(h0)  f32(h1)  f32(h2)  f32(h3) ... f32(h31) f32(h32) │
  (low)    │  ← src0[0], src1[0], src0[1], src1[1], ...          →   │
           └───────────────────────────────────────────┘
           = 逻辑前 64 个原始值的 f32 连续表示 (h0..h63)

           ┌───────────────────────────────────────────┐
  dst1     │ f32(h64) f32(h65) f32(h66) f32(h67) ... f32(h95) f32(h96)│
  (high)   │  ← src0[32], src1[32], src0[33], src1[33], ...         →  │
           └───────────────────────────────────────────┘
           = 逻辑后 64 个原始值的 f32 连续表示 (h64..h127)

两个 f32 reg 现在是逻辑连续的！
  dst0 = 原始元素 h0..h63 的 f32 值
  dst1 = 原始元素 h64..h127 的 f32 值
```

**元素级简明图示（8 个原始元素为例）：**

```
            src0 (EVEN半区)          src1 (ODD半区)
            ┌───┬───┬───┬───┐       ┌───┬───┬───┬───┐
            │h0 │h2 │h4 │h6 │       │h1 │h3 │h5 │h7 │
            └───┴───┴───┴───┘       └───┴───┴───┴───┘

vintlv 交织:

  dst0 (low):  ┌───┬───┬───┬───┬───┬───┬───┬───┐
               │h0 │h1 │h2 │h3 │h4 │h5 │h6 │h7 │   ← 逻辑连续！前半
               └───┴───┴───┴───┴───┴───┴───┴───┘

  dst1 (high): ┌───┬───┬───┬───┬───┬───┬───┬───┐
               │h8 │h9 │h10│h11│h12│h13│h14│h15│  ← 逻辑连续！后半
               └───┴───┴───┴───┴───┴───┴───┴───┘
```

#### vdintlv：从连续态拆成 parity 交织态（反向操作）

```
输入: 两个连续的 vreg（代表一段交织流）
           ┌───────────────────────────────┐
  src0     │ h0  h1  h2  h3 ... h31  h32 │  ← 交织流的前半
           └───────────────────────────────┘
           ┌───────────────────────────────┐
  src1     │ h33 h34 h35 h36 ... h63  h64 │  ← 交织流的后半
           └───────────────────────────────┘

vdintlv(src0, src1) → low, high:

  low (even):  ┌───┬───┬───┬───┐
               │h0 │h2 │h4 │h6 │   ← src0/src1 的偶数索引元素
               └───┴───┴───┴───┘

  high (odd):  ┌───┬───┬───┬───┐
               │h1 │h3 │h5 │h7 │   ← src0/src1 的奇数索引元素
               └───┴───┴───┴───┘
```

#### 2.3.2 Predicate Interleave / Deinterleave（§5）

| 指令 | 输入 | 输出 | 语义 |
|------|------|------|------|
| `pto.pintlv_b8` / `pto.pintlv_b16` / `pto.pintlv_b32` | 2 个同族 mask | 2 个同族 mask (low, high) | 交织两个 predicate |
| `pto.pdintlv_b8` / `pto.pdintlv_b16` / `pto.pdintlv_b32` | 2 个同族 mask | 2 个同族 mask (low, high) | 解交织两个 predicate |

### 2.4 Pack / Unpack —— 向量数据宽度压缩/展开（§12）

| 指令 | 方向 | 语义 | 约束 |
|------|------|------|------|
| `pto.vpack` | **窄化** (wide→narrow) | `vreg<NxT_wide> → vreg<2NxT_narrow>`，选定 LOWER/HIGHER 半区，另一半填零 | i32/ui32→ui16, i16/ui16→ui8 |
| `pto.vsunpack` | **加宽** (narrow→wide, 符号扩展) | `vreg<NxT_narrow> → vreg<N/2xT_wide>`，选定哪个半区做符号扩展展开 | — |
| `pto.vzunpack` | **加宽** (narrow→wide, 零扩展) | `vreg<NxT_narrow> → vreg<N/2xT_wide>`，选定哪个半区做零扩展展开 | — |

#### vpack（宽→窄，半区放置）

```
输入: f32 vreg (64 lanes, 256B)
           ┌───────────────────────────────────────┐
  vreg_wide│ f32[0]  f32[1]  f32[2]  f32[3] ... f32[63] │
           └───────────────────────────────────────┘

vpack "LOWER": 窄化后放到 ui16 vreg 的前半区 (lane 0..63), 后半区清零
           ┌─────────────────────────────────────────┐
  result   │ ui16[0]  ui16[1] ... ui16[63] │ 0  0 ... 0  0 │
           │ ← 窄化值在前半 →              │ ← 后半填零 →   │
           └─────────────────────────────────────────┘
           = vreg<128xui16>

vpack "HIGHER": 窄化后放到 ui16 vreg 的后半区 (lane 64..127), 前半区清零
           ┌─────────────────────────────────────────┐
  result   │ 0  0 ... 0  0 │ ui16[0]  ui16[1] ... ui16[63] │
           │ ← 前半填零 →  │ ← 窄化值在后半 →              │
           └─────────────────────────────────────────┘
```

#### vsunpack / vzunpack（窄→宽，半区提取+扩展）

```
输入: i16 vreg (128 lanes, 256B)
           ┌─────────────────────────────────────────┐
  vreg_nar │ i16[0] ... i16[63] │ i16[64] ... i16[127] │
           │ ← LOWER 半区 →    │ ← HIGHER 半区 →      │
           └─────────────────────────────────────────┘

vsunpack %src, %part=0 (LOWER): 符号扩展前半区 → 64 个 i32
           ┌───────────────────────────────────────┐
  result   │ i32_signext(i16[0]) ... i32_signext(i16[63]) │
           └───────────────────────────────────────┘
           = vreg<64xi32>

vsunpack %src, %part=64 (HIGHER): 符号扩展后半区 → 64 个 i32
           ┌───────────────────────────────────────┐
  result   │ i32_signext(i16[64]) ... i32_signext(i16[127]) │
           └───────────────────────────────────────┘
           = vreg<64xi32>

vzunpack: 同结构，但用零扩展而非符号扩展
```

**与 vcvt part 的区别：**
- `vcvt {part=EVEN/ODD}`：**类型转换 + lane 位置选择**（有语义）
- `vpack/vsunpack/vzunpack`：**纯物理压缩/展开 + 半区选择**（无语义，只是搬数据）
- RFC 将两者合并为一条 `pto.vmi.vcvt` 是合理的——窄↔宽是真语义，半区摆放是 layout

### 2.5 Predicate Pack / Unpack —— predicate 跨宽度族拆分/合并（§5）

| 指令 | 方向 | Part token | 语义 |
|------|------|------------|------|
| `pto.ppack` | **窄化** | `LOWER` / `HIGHER` | 每 2-bit 组保留 1-bit，打包到指定半区，另一半填零 |
| `pto.punpack` | **加宽** | `LOWER` / `HIGHER` | 从指定半区读取，每 1-bit 扩展成 2-bit（原位 + 零位） |

#### punpack（predicate 加宽展开，b16→b32）

```
这是 block_quant 的关键场景:

f16 用 b16 族 predicate (128-bit, 每 lane 1 bit):
           ┌─────────────────────────────────────────┐
  preg_b16 │ ■ ■ ■ ■ ■ ■ ■ ■ │ ■ ■ ■ ■ ■ ■ ■ ■ │   ← 128 个 b16 predicate bit
           │ ← 前 64 位 →      │ ← 后 64 位 →      │
           └─────────────────────────────────────────┘

f32 需要两个 b32 族 predicate (各 64-bit):
  加宽后 128 个 f16 变成 2 × 64 个 f32
  每个 f32 vreg 需要自己的 64-bit b32 predicate

punpack LOWER:
           ┌───────────────────────────────────────┐
  preg_lo  │ ■ ■ ■ ■ ■ ■ ■ ■ │ 0 0 0 0 0 0 0 0 │   ← 前 64 bit 展开, 后 64 bit 清零
  (b32)    └───────────────────────────────────────┘

punpack HIGHER:
           ┌───────────────────────────────────────┐
  preg_hi  │ 0 0 0 0 0 0 0 0 │ ■ ■ ■ ■ ■ ■ ■ ■ │   ← 前 64 bit 清零, 后 64 bit 展开
  (b32)    └───────────────────────────────────────┘

使用:
  vsts(vreg_fp8_lo, addr,       PK4_B32, preg_lo)   ← 用 LOWER predicate
  vsts(vreg_fp8_hi, addr + 64,  PK4_B32, preg_hi)   ← 用 HIGHER predicate
```

#### ppack（predicate 窄化合并，b32→b16）

```
两个 b32 predicate:
           ┌───────────────────────────────┐
  preg0    │ ■ ■ ■ ■ ■ ■ ■ ■ │   ← 64 个 b32 predicate bit
           └───────────────────────────────┘
           ┌───────────────────────────────┐
  preg1    │ ■ ■ ■ ■ ■ ■ ■ ■ │
           └───────────────────────────────┘

ppack LOWER: 每 2-bit 组保留 1-bit, 打包到 b16 低半区
           ┌─────────────────────────────────────────┐
  result   │ ■_ ■_ ■_ ■_ ■_ ■_ ■_ ■_ │ 0  0  0  0  0  0  0  0 │
           │ ← preg0 交替压缩 →        │ ← 高半区清零 →           │
           └─────────────────────────────────────────┘

ppack HIGHER: 打包到 b16 高半区
           ┌─────────────────────────────────────────┐
  result   │ 0  0  0  0  0  0  0  0 │ ■_ ■_ ■_ ■_ ■_ ■_ ■_ ■_ │
           └─────────────────────────────────────────┘
```

### 2.6 Memory-side Interleave / Pack / Unpack —— load/store 的 dist 模式

#### Load side（`pto.vlds` dist 家族）

| dist 家族 | 语义 | 延迟 |
|-----------|------|------|
| `UNPK_B8` / `UNPK_B16` / `UNPK_B32` | 从内存读窄化 packed 数据，展开成宽 lanes | 9 cycles |
| `UNPK4` | 4-way unpack：4 个 b8 packed group → lanes | 9 cycles |
| `SPLT4CHN` | 分离 4 通道交织数据到 1 个通道平面 | 9 cycles |
| `SPLT2CHN_B8` / `SPLT2CHN_B16` | 分离 2 通道交织数据到 1 个通道平面 | 9 cycles |

#### Load side dual（`pto.vldsx2` —— 一次读两条 vreg）

| dist 家族 | 语义 | 延迟 |
|-----------|------|------|
| `BDINTLV` | Block deinterleave：从内存读 block-交织数据，拆成两个 vreg | 9 cycles |
| `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` | Element-width deinterleave：从内存读交替排列的数据，拆成 even/odd 两个 vreg | 9 cycles |

#### DINTLV_B32 加载图示（内存 AoS → 寄存器 SoA）

```
UB 内存 (f32 AoS 排列, 每 2 个 f32 交织为一组):
           ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  UB       │ X[0] │ Y[0] │ X[1] │ Y[1] │ X[2] │ Y[2] │ ...  │ ...  │
           └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
           offset=0           4       8       12      16      20

vldsx2 %ub[%off], "DINTLV_B32" → low, high:

  low (X):  ┌───────────────────────────┐
            │ X[0]  X[1]  X[2] ... X[63] │   ← 偶数位元素 (stride=2)
            └───────────────────────────┘

  high (Y): ┌───────────────────────────┐
            │ Y[0]  Y[1]  Y[2] ... Y[63] │  ← 奇数位元素 (stride=2)
            └───────────────────────────┘
```

#### Store side（`pto.vsts` dist 家族）

| dist 家族 | 语义 | 延迟 |
|-----------|------|------|
| `NORM_B8` / `NORM_B16` / `NORM_B32` | 连续写回 `UB[base+i] = src[i]` | 9 cycles |
| `1PT_B8` / `1PT_B16` / `1PT_B32` | 只写第 0 个元素 | 9 cycles |
| `PK_B16` | 每 lane 取低 16-bit，pack 存储 | 9 cycles |
| `PK_B32` | 每 lane 取低 32-bit，pack 存储 | 9 cycles |
| `PK_B64` | 每 lane 取低 64-bit，pack 存储 | 9 cycles |
| `PK4_B32` | 每 lane 取低 8-bit，4 倍 pack 存储（**FP8 的关键路径**） | 9 cycles |
| `MRG4CHN_B8` | 合并 4 个 8-bit 通道到 32B block | 9 cycles ⚠️ |
| `MRG2CHN_B8` / `MRG2CHN_B16` | 合并 2 个通道到 32B block | 9 cycles ⚠️ |

> ⚠️ `MRG4CHN_B8` / `MRG2CHN_B8` / `MRG2CHN_B16`：PTO surface 保留，但 A5 硬件验证时报告 unsupported。

#### INTLV_B32 存储图示（寄存器 SoA → 内存 AoS）

```
两个 f32 vreg:
           ┌───────────────────────────┐
  vreg_X   │ X[0]  X[1]  X[2] ... X[63] │
           └───────────────────────────┘
           ┌───────────────────────────┐
  vreg_Y   │ Y[0]  Y[1]  Y[2] ... Y[63] │
           └───────────────────────────┘

vstsx2 %vreg_X, %vreg_Y, %ub[%off], "INTLV_B32", %mask → UB:

  UB:      ┌──────┬──────┬──────┬──────┬──────┬──────┐
           │ X[0] │ Y[0] │ X[1] │ Y[1] │ X[2] │ Y[2] │ ...
           └──────┴──────┴──────┴──────┴──────┴──────┘
           ← 交织写回！逻辑上连续覆盖 128 个原始位置 →
```

#### PK4_B32 存储图示（f32 vreg → FP8 内存）

```
f32 vreg (64 lanes, 每个值经 vcvt 已转成 FP8 存在 P0 位置):
           ┌───────────────────────────────────────┐
  vreg     │ fp8[0]_P0  fp8[1]_P0  fp8[2]_P0 ... fp8[63]_P0 │
           │  每个 32-bit lane 的最低 8 bit = FP8             │
           └───────────────────────────────────────┘

vsts PK4_B32: 取每个 lane 最低 8 bit → 4 个 fp8 值 pack 进 1 个 32-bit 地址单元

  UB:      ┌────────┬────────┬────────┬────────┐
           │ fp8[0] │ fp8[1] │ fp8[2] │ fp8[3] │ ...  ← 连续 64 bytes
           │(8 bit) │(8 bit) │(8 bit) │(8 bit) │
           └────────┴────────┴────────┴────────┘

需要 2 次 vsts PK4_B32 (每次 64 个 fp8 值):
  vsts(vreg_fp8_lo, addr,       PK4_B32, preg_lo)  ← 前 64 个 FP8
  vsts(vreg_fp8_hi, addr + 64,  PK4_B32, preg_hi)  ← 后 64 个 FP8
```

#### Store side dual（`pto.vstsx2`）

| dist 家族 | 语义 | 延迟 |
|-----------|------|------|
| `INTLV_B8` / `INTLV_B16` / `INTLV_B32` | 将两个 vreg 交织写回内存：`UB[2*i]=low[i], UB[2*i+1]=high[i]` | **12 cycles** |

> **注意：** INTLV store 比 NORM/PK store 延迟更高（12 vs 9 cycles）。这是 vstsx2 的固定开销。

### 2.7 Channel Split / Merge —— 通道级别的交织重组

| 类别 | 指令/dist | 语义 |
|------|-----------|------|
| **Split (load)** | `SPLT4CHN` | 4 通道交织数据 → 1 通道平面 |
| | `SPLT2CHN_B8` / `SPLT2CHN_B16` | 2 通道交织数据 → 1 通道平面 |
| **Merge (store)** | `MRG4CHN_B8` | 4 个 8-bit 通道合并到 32B block |
| | `MRG2CHN_B8` / `MRG2CHN_B16` | 2 个通道合并到 32B block |

---

## 三、完整的指令 ↔ 场景映射表

| 逻辑意图 | 物理实现（pto.mi 指令组合） | 涉及的 pack/interleave/part 机制 |
|----------|---------------------------|----------------------------------|
| **f16 → f32 加宽** (128→128 元素, 重建连续) | `vcvt PART_EVEN` + `vcvt PART_ODD` + `vintlv` | Part(EVEN/ODD) + vintlv |
| **f16 → f32 加宽后运算** (保持 parity) | `vcvt PART_EVEN/ODD` → 两路 `vadd` → `vstsx2 INTLV_B32` | Part + parity 轴 + INTLV store |
| **f16 → f32 加宽后写回** | 两路运算 → `vstsx2 INTLV_B32` | Part + INTLV store |
| **f32 → f16 窄化** (2×64→128 元素) | `vcvt EVEN` + `vcvt ODD` + `vor` 合并 | Part(EVEN/ODD) + vor |
| **f32 → FP8 窄化** | `vcvt {part=P0}` + `vsts PK4_B32` | Part_T(P0) + PK4_B32 |
| **i32 → i16 窄化存储** | `vpack LOWER` + `vsts PK_B16` | vpack + PK store |
| **i16 → i32 加宽** | `vsunpack/vzunpack` 选半区 | vsunpack/vzunpack |
| **AoS→SoA (XY pairs)** | `vldsx2 DINTLV_B32` | DINTLV load |
| **SoA→AoS (XY pairs)** | `vstsx2 INTLV_B32` | INTLV store |
| **b16 mask → b32 mask** (加宽时) | `punpack LOWER` + `punpack HIGHER` | punpack |
| **Histogram (chistv2)** | `chistv2 Bin_N0` + `Bin_N1` → `vcvt EVEN/ODD` 每个 half → 4 个 f32 reg | half 轴 + parity 轴 |
| **Softmax exp(x-max)** | `vexpdif %input, %max, %mask, "ODD"` | Part(ODD) on vexpdif |

---

## 四、完整场景流程图

### 4.1 block_quant f16→FP8 的物理流程全景图

```
┌──────────────────────────────────────────────────────────────┐
│  UB 内存                                                      │
│  ┌──────────────────────────────────────────────────┐        │
│  │ f16 输入: [h0, h1, h2, ..., h63, h64, ..., h127] │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
         │ vlds NORM → 1 个 f16 vreg (128 lanes)
         ▼
┌──────────────────────────────────────────┐
│ vreg_f16: [h0, h1, h2, ..., h127]       │
│ 128 lanes, 256B                          │
└──────────────────────────────────────────┘
         │
    ┌────┴────┐
    │         │
 vcvt EVEN  vcvt ODD          ← Part (parity 轴产出)
    │         │
    ▼         ▼
┌──────────┐ ┌──────────┐
│ vreg_e   │ │ vreg_o   │
│ 64 f32   │ │ 64 f32   │
│ EVEN半区 │ │ ODD半区  │
└──────────┘ └──────────┘
         │         │
    vintlv(vreg_e, vreg_o)      ← Interleave 重建连续态
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐ ┌──────────┐
│ dst0     │ │ dst1     │
│ 64 f32   │ │ 64 f32   │
│ h0..h63  │ │ h64..h127│
│ ←连续→   │ │ ←连续→   │
└──────────┘ └──────────┘
         │         │
  (运算: abs_max, scale, divide 等)
         │         │
    ┌────┴────┐
    │         │
 vcvt P0   vcvt P0               ← Part_T (FP8 窄化)
    │         │
    ▼         ▼
┌──────────┐ ┌──────────┐
│ fp8_lo   │ │ fp8_hi   │
│ 64 fp8   │ │ 64 fp8   │
│ (P0位)   │ │ (P0位)   │
└──────────┘ └──────────┘

Predicate 拆分 (b16 → b32):
┌──────────────────────────────┐
│ preg_b16: ■■■■■■■■│■■■■■■■■■│  ← 128-bit b16 mask
└──────────────────────────────┘
         │
  ┌──────┴──────┐
  │             │
punpack LOWER  punpack HIGHER
  │             │
  ▼             ▼
┌──────────┐  ┌──────────┐
│ preg_lo  │  │ preg_hi  │
│ 64-bit   │  │ 64-bit   │
│ b32 族   │  │ b32 族   │
└──────────┘  └──────────┘

FP8 写回:
  vsts(fp8_lo, addr,       PK4_B32, preg_lo)  ← 64 个 FP8
  vsts(fp8_hi, addr + 64,  PK4_B32, preg_hi)  ← 64 个 FP8
         │                │
         ▼                ▼
┌──────────────────────────────────────────────────────────┐
│  UB 内存                                                  │
│  ┌──────────────────────────────────────────────┐        │
│  │ FP8 输出: [fp8[0], fp8[1], ..., fp8[127]]    │ 128 B │
│  └──────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### 4.2 block_quant 另一种路径：保持 parity 交织态不做 vintlv

```
┌──────────────────────────────────────┐
│ vreg_f16 (128 lanes)                  │
└──────────────────────────────────────┘
         │
    ┌────┴────┐
    │         │
 vcvt EVEN  vcvt ODD
    │         │
    ▼         ▼
┌──────────┐ ┌──────────┐
│ vreg_e   │ │ vreg_o   │
│ 64 f32   │ │ 64 f32   │
│ parity态 │ │ parity态 │
└──────────┘ └──────────┘
         │         │
    直接运算 (Category A fan-out):
    vadd(vreg_e, bias_e, mask_e)  ← 各自在 parity 半区运算
    vadd(vreg_o, bias_o, mask_o)
         │         │
         ▼         ▼
┌──────────┐ ┌──────────┐
│ result_e │ │ result_o │
│ 64 f32   │ │ 64 f32   │
│ parity态 │ │ parity态 │
└──────────┘ └──────────┘
         │         │
    vstsx2(result_e, result_o, addr, "INTLV_B32", mask)
    → 内存自动交织成连续的 f32 数组

这种方式省掉了 vintlv 指令（但 vstsx2 比 vsts 多 3 cycles: 12 vs 9）
```

### 4.3 Histogram (chistv2) —— half 轴 + parity 轴叠加

```
          ┌───────────────────┐
          │  UB (u8 源向量)    │
          │  [0..255] 256 byte │
          └───────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
  chistv2 Bin_N0         chistv2 Bin_N1
        │                     │
        ▼                     ▼
  ┌───────────────┐   ┌───────────────┐
  │ vreg_h0       │   │ vreg_h1       │
  │ u16 bins 0..127│   │u16 bins128..255│
  │ ← half 轴: N0 → │   │← half 轴: N1 → │
  │ 128 lanes      │   │ 128 lanes      │
  └───────────────┘   └───────────────┘
        │                     │
  ┌─────┴─────┐         ┌─────┴─────┐
  │           │         │           │
vcvt EVEN  vcvt ODD   vcvt EVEN  vcvt ODD
  │           │         │           │
  ▼           ▼         ▼           ▼
┌─────┐  ┌─────┐    ┌─────┐  ┌─────┐
│n0_e │  │n0_o │    │n1_e │  │n1_o │  ← parity 轴叠加在 half 轴上
│64   │  │64   │    │64   │  │64   │
│u32  │  │u32  │    │u32  │  │u32  │
│bins │  │bins │    │bins │    │bins │
│0..63│  │64.. │    │128..│  │192..│
│  ←even │←odd→│    │←even│  │←odd→│
└─────┘  └─────┘    └─────┘  └─────┘

= 4 个 f32 vreg = half(2) × parity(2) 的二维交织

写回内存:
  vstsx2(n0_e, n0_o, addr_bins+0,   "INTLV_B32")  → bins 0..127 连续写回
  vstsx2(n1_e, n1_o, addr_bins+128, "INTLV_B32")  → bins 128..255 连续写回
```

---

## 五、按 RFC 的 axis 模型重新归类

RFC 的 LayoutDescriptor 用 4 种轴来抽象以上所有机制。对照关系：

| RFC 轴 | 对应的 pto.mi 机制 | 涉及指令 |
|--------|-------------------|----------|
| **`parity`** (stride-2 交织) | `vcvt PART_EVEN/ODD`, `vldsx2 DINTLV_B*`, `vstsx2 INTLV_B*`, `vintlv/vdintlv` | vcvt, vldsx2, vstsx2, vintlv, vdintlv |
| **`width`** (窄↔宽半区) | `vsunpack/vzunpack`, `vpack`, load `UNPK_B*`, store `PK_B*` | vpack, vsunpack, vzunpack, UNPK_B* load, PK_B* store |
| **`half`** (Histogram 128-bin 半区) | `chistv2 Bin_N0/N1` | chistv2 (概念映射，不在 micro ISA surface 中) |
| **`chunk`** (纯连续) | `NORM` / `NORM_B*` | vlds NORM, vsts NORM_B* |

### RFC 轴模型 vs 真实机制的对照图

```
RFC LayoutDescriptor 已定义的轴:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  chunk 轴  ──  纯连续  ──  NORM / NORM_B*                      │
│  (cardinality=K, stride=1)                                      │
│                                                                 │
│  parity 轴 ──  stride-2 交织  ──  vcvt EVEN/ODD, INTLV, DINTLV │
│  (cardinality=2, stride=2)                                      │
│                                                                 │
│  width 轴  ──  窄↔宽半区  ──  vpack/vsunpack, PK_B*/UNPK_B*   │
│  (cardinality=2, stride=? )                                     │
│                                                                 │
│  half 轴   ──  Histogram 半区  ──  chistv2 Bin_N0/N1           │
│  (cardinality=2, stride=? )                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

           ⚠️ RFC 当前缺失的维度:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Part_T 轴 ──  4-way sub-part  ──  vcvt P0~P3, PK4_B32        │
│  (cardinality=4, stride=4)                                      │
│  ❌ parity 轴只有 2-way, 无法表达 4-way FP8/fp4 场景            │
│                                                                 │
│  Predicate 族变换 ──  b16↔b32↔b8  ──  punpack/ppack            │
│  ❌ 不在 vreg layout 内, 是 mask 的独立维度                       │
│                                                                 │
│  Channel 轴 ──  多通道交织  ──  SPLT4CHN, MRG4CHN_B8           │
│  (cardinality=4 或 2)                                           │
│  ❌ 与 parity (stride-2 元素级) 语义不同                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、A5 物理寄存器容量全景图

```
                    1 个物理 vreg = 256B = 2048 bit

  dtype    │ 每reg容纳 │ lane宽度 │ b16族每bit覆盖 │ b32族每bit覆盖 │ b8族每bit覆盖
  ─────────┼───────────┼─────────┼───────────────┼───────────────┼───────────────
  f32/i32  │   64      │  32 bit │       —       │    1 lane     │       —
  f16/i16  │  128      │  16 bit │    1 lane     │    2 lanes    │       —
  bf16     │  128      │  16 bit │    1 lane     │    2 lanes    │       —
  i8/fp8   │  256      │   8 bit │    2 lanes    │    4 lanes    │    1 lane

  加宽时 predicate 族必须匹配:
  f16 (b16族) → f32 (b32族) 需要 punpack
  f32 (b32族) → fp8 (b32族 + PK4_B32) 需要 b32 predicate

  窄化时 reg 容量变化:
  1×f32_reg(64) → 需要 2×f16_reg(128)  或  4×fp8_reg(256)
  1×f16_reg(128) → 需要 2×f32_reg(64)   或  1×fp8_reg(256) via PK4_B32
```

### Predicate 族与 dtype 的绑定关系

```
  dtype 族            │ predicate 族 │ 每 predicate bit 覆盖的 lane 数
  ────────────────────┼─────────────┼────────────────────────────────
  f32 / i32 / si64    │ b32         │ 1 lane
  f16 / bf16 / i16    │ b16         │ 1 lane
  i8 / ui8            │ b8          │ 1 lane

  跨族转换时 predicate 必须同步变换:
  ┌──────────┐           ┌──────────┐
  │ b16 mask │ punpack → │ b32 mask │  (f16→f32 加宽时)
  └──────────┘           └──────────┘

  ┌──────────┐           ┌──────────┐
  │ b32 mask │ ppack   → │ b16 mask │  (f32→f16 窄化时)
  └──────────┘           └──────────┘
```

---

## 七、A5 支持性总结

| 指令 | A5 支持 | 延迟 | 备注 |
|------|---------|------|------|
| `pto.vintlv` / `pto.vdintlv` | ✅ | 12 cycles | 核心 interleave 指令 |
| `pto.vintlvv2` / `pto.vdintlvv2` | ❌ | — | V2 版本，A5 硬件不支持 |
| `pto.vpack` | ✅ | — | i32→ui16, i16→ui8 |
| `pto.vsunpack` / `pto.vzunpack` | ✅ | — | 窄→宽展开 |
| `pto.ppack` / `pto.punpack` | ✅ | — | Predicate 拆分/合并 |
| `pto.pintlv_b*` / `pto.pdintlv_b*` | ✅ | — | Predicate 交织 |
| `pto.vldsx2` (DINTLV/BDINTLV) | ✅ | 9 cycles | Dual load 解交织 |
| `pto.vstsx2` (INTLV) | ✅ | **12 cycles** | Dual store 交织 |
| `pto.vlds {dist=UNPK_B*}` | ✅ | 9 cycles | Load-side unpack |
| `pto.vsts {dist=PK_B*}` | ✅ | 9 cycles | Store-side pack（含 PK4_B32） |
| `pto.vsts {dist=MRG4CHN_B8}` | ⚠️ | 9 cycles | Surface 保留，A5 验证 unsupported |
| `pto.vcvt {part=EVEN/ODD}` | ✅ | 7 cycles (f32→f16) | 2-way part |
| `pto.vcvt {part=P0~P3}` | ✅ | — | 4-way Part_T (FP8/fp4) |
| `pto.vexpdif {EVEN/ODD}` | ✅ | — | Softmax 专用 |

---

## 八、对 RFC (VPTO Logical Vector ISA) 的映射启示

将以上梳理映射到 RFC 的 `pto.vmi → pto.mi` lowering：

| pto.vmi op | 应 lowering 到的 pto.mi 指令 | 需要的 layout 轴 | 需要的 predicate 变换 |
|------------|----------------------------|-----------------|---------------------|
| `pto.vmi.vcvt` (2 倍加宽) | `vcvt PART_EVEN` + `vcvt PART_ODD` | parity 轴 (cardinality=2) | `punpack LOWER/HIGHER` (b16→b32) |
| `pto.vmi.vcvt` (2 倍窄化) | `vcvt EVEN` + `vcvt ODD` + `vor` | parity 轴消费 | — |
| `pto.vmi.vcvt` (4 倍窄化/FP8) | `vcvt {part=P0}` + `vsts PK4_B32` | **Part_T 轴（RFC 未定义，cardinality=4）** | `punpack` (b32→b8/b16 族) |
| `pto.vmi.vcvt` (vsunpack/vzunpack 替代) | `vsunpack/vzunpack` | width 轴 | predicate 族对齐 |
| `pto.vmi.vsts` (连续写回 parity 数据) | `vstsx2 INTLV_B*` | parity 轴消费 | predicate 族需匹配 store 宽度 |
| `pto.vmi.vsts` (FP8 写回) | `vsts PK4_B32` | width 轴 + Part_T 轴 | b32 predicate |
| `pto.vmi.vlds` → `pto.vmi.vlds` | `vldsx2 DINTLV_B*` 或 `vlds UNPK_B*` | parity 轴产出 / width 轴产出 | — |

### RFC 需要补充的关键点

1. **Part_T (P0~P3) 轴**：parity 只有 2-way，但 FP8/fp4 需要 4-way sub-part placement。LayoutDescriptor 的 `axes` 需要增加一种 `sub_part` 轴（cardinality=4, stride=4）来映射 P0~P3。

2. **Predicate 族变换是独立于 vreg layout 的维度**：当 vreg 从 f16(b16 族) 变成 f32(b32 族) 时，predicate 必须同步从 b16 punpack 到 b32。这不是 vreg layout 轴能表达的——它是 **mask 的 layout 变换**，RFC §3.2 的 `!pto.vmi.mask<L x G>` 需要增加 predicate 族转换规则。

3. **PK4_B32 不是简单的 width 轴消费**：它是 f32→FP8 的 4 倍窄化存储，涉及 `Part_T(P0)` + pack-store 两步，需要特殊 lowering 规则。

4. **Channel Split/Merge 是独立于 parity 的交织维度**：4 通道和 2 通道交织与 parity（stride-2）的语义不同，可能需要单独的 `channel` 轴或归入 parity 轴的特殊模式。
