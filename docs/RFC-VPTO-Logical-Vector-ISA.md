# RFC: PTO虚拟SIMD ISA —— 用 layout 元数据消除 part/pack/interleave

---

## 1. 动机

`pto` micro ISA 目前与底层 CCE intrinsic **基本一对一等价**——一条 `pto` 微指令直接对应一条 CCE 指令（`vcvt`、`vintlv`、`vldsx2`、各 `dist` 模式等），它刻意贴着硬件，几乎不做抽象。这让 `pto` 适合作为可精确 lower、可对照硬件验证的底座，但也意味着**硬件 SIMD 寄存器的全部物理约束都直接暴露在pto.mi微指令上**。

`pto` micro ISA 的 `!pto.vreg<NxT>` 物理上恰好是 256 字节（2048 bit）。当一个逻辑值大于 256B，或某个原语天然以"交织/半区"形态产出时，用户被迫在源码里手写物理布局的拼装：`vcvt {part=EVEN/ODD}`、`vpack/vunpack`、`vintlv/vdintlv`、`vldsx2/vstsx2 INTLV_B*`、以及各种 `dist` token。

因此，这类pto.mi指令**并不表达计算意图，只表达物理摆放**。如果我们可以将"逻辑数组 ↔ 物理寄存器"的布局关系推导出来，那么**part / pack / interleave / dist 不再需要显式表达。** 用户写simdvf代码的时候，可以认为寄存器的layout永远是简单连续的。

### 1.1 上层动机：对接 TileLang `T.parallel`

`pto.vmi`(PTO virtual micro instrution) 的目标，是为 **TileLang 的 `T.parallel` 编程模型提供干净的映射**。

TileLang 让用户用 `T.parallel` 描述一个逻辑迭代空间上的逐元素计算——用户**只声明"对这片数据做什么"，不声明"数据怎么摊到GPU warp/thread上"**。最能暴露映射simd架构的问题不是相同dtype的element-wise操作，而是 **`T.cast` 跨数据宽度**的场景：

```python
# TileLang 用户视角：只关心"把 i16 升成 i32 再累加"，不关心寄存器 layout
for i in T.parallel(N):
    C[i] = T.cast(A[i], "int32") + B[i]   # A: i16,  B/C: i32
```

用户写的是一句干净的"加宽后相加"。但在 A5 上，`i16 → i32` 的加宽在硬件层不是一条直通指令：一个 256B 的 i16 向量（128 元素）必须被 `vcvt` 拆成 **EVEN / ODD 两个半区**，各自加宽成一个 256B 的 i32 向量（64 元素），后续运算要在这两个交织的半区上分别进行，写回时还得用 `vstsx2 INTLV_B32` 把两半重新交织成连续内存。

也就是说，一句 `T.cast` 在原始 `pto` 上会炸开成 `vcvt PART_EVEN` + `vcvt PART_ODD` + 两路 `vadd` + `vstsx2 INTLV_B32`。对于从GPU来的开发者的心智模型是"一片连续的 i32 逻辑数据"，会懵逼掉：

- 这片数据要拆成几个 256B 物理 vreg（`K = N * bitwidth / 2048`）；
- 加宽 / 窄化时数据落在 EVEN / ODD 哪个半区，后续 op 要在交织形态上各做一次；
- 跨 vreg 的尾部谓词怎么补；
- 写回内存时该用 NORM 还是 INTLV 交织。

如果 `T.parallel` 直接lower到 `pto.mi`（CCE/AscendC会遇到相同问题），上述每一项都会**泄漏回用户或TileLang codegen**，`T.parallel` 的"只关心数据"承诺就破产了——尤其是宽度转换，会让 codegen或算子哥被迫处理 EVEN/ODD 交织的组合。

本文档提出的`pto.vmi`正是用来抽象这一操作的中间层：

```
TileLang  T.parallel(N) { C[i] = cast<i32>(A[i]) + B[i] }   ← 用户：只有逻辑数据
   │  (直译，逐元素语义保持)
   ▼
pto.vmi      %w = pto.vmi.vcvt %a ; %c = pto.vmi.vadd %w, %b           ← 逻辑向量，layout 由编译器持有
   │  (pto-as：layout-assignment + lowering)
   ▼
pto.mi       vcvt EVEN/ODD + 两路 vadd + vstsx2 INTLV_B32        ← 物理：SIMD 寄存器交织细节
```

也就是说：**`T.parallel` 的逻辑迭代空间，几乎可以一对一直译成 `pto.vmi` 的逻辑向量 op**——逐元素计算映射到 Category A op，`T.cast` 映射到一条无 part 的 `pto.vmi.vcvt`，逻辑长度 `N` 映射到 `!pto.vmi.vreg<N x T>`，`T.parallel` 隐含的"全 active"语义映射到自动补齐的尾部谓词。宽度转换引入的 EVEN/ODD 交织、以及 SIMD 寄存器上的 layout 映射这件 `T.parallel` 无需暴露，后续由pto-as解决。

### 1.2 目标

- `pto.vmi` 给程序员暴露的就**只有逻辑连续语义**。
- 支持 vreg 宽度为 256B 的**整数倍**（`K` 个物理 vreg 后备）。
- 物理布局（含交织、半区、宽窄摆放）由编译器推断和传播，对用户不可见。
- 可完整 lower 回 `pto.mi`，`K=1` 时退化为零开销直通。

---

## 2. 设计概览

`pto.vmi` 由三部分组成：

1. **虚拟类型** `!pto.vmi.vreg<L x T, #layout>` / `!pto.vmi.mask<L x G, #layout>`，携带逻辑形状与一份内部 layout 元数据。
2. **简化后的 surface op 集**：约 30 条纯逻辑 op，相对原始 `pto.mi微指令` 的约 100 条大幅收敛。
3. **`pto.as`**：把 `pto.vmi` 程序 lower 为合法 `pto.mi`，在此过程中推断布局、协调相邻 op 的布局、并在必要处自动变换为连续形态。

**对于pto.vmi.vreg新增layout属性，该属性由编译推导并传播**。

---

## 3. 虚拟类型

### 3.1 `!pto.vmi.vreg<L x T, #layout>`

- `L`：逻辑元素数。`T`：逻辑元素类型（沿用 `pto.mi` 的 `i8/i16/i32/f16/bf16/f32` 等）。
- 约束：`L * bitwidth(T)` MUST 是 2048bit/256Byte 的整数倍。。
- `#layout`：surface 源码通常省略，由 pto-as 填充。

每个 dtype 的合法 `L`：

| T | bits | 单物理 reg 容纳 | L 必须是…的倍数 |
|---|------|----------------|------------------|
| f32 / i32 | 32 | 64 | 64 |
| f16 / bf16 / i16 | 16 | 128 | 128 |
| i8 | 8 | 256 | 256 |

`K=1`（`L` 恰为单 reg 容量）时 `pto.vmi.vreg` 与 `pto.mi.vreg` 一一对应。

### 3.2 `!pto.vmi.mask<L x G, #layout>`

- 虚拟谓词，物理后备为 `K` 个 256-bit `!pto.mi.mask<G>`。
- `G`（`b8/b16/b32`）必须与数据族对齐，沿用 `pto.mi` 的 legality 契约：`f32/i32` 族用 `b32`，`f16/bf16/i16` 族用 `b16`，8-bit 族用 `b8`。

---

## 4. LayoutDescriptor

布局被建模为一组数据结构。

```
#pto.vmi.vreg.layout<
  logical_shape = [L],
  phys_dtype    = T,
  phys_lanes    = 64 | 128 | 256,
  axes = [ #axis<name, cardinality, mode, stride_in_logical>, ... ]
>
```

`is_contiguous := ∀ axis. axis.mode ∈ {NORM, None}`（即只剩 `chunk` 轴）。

### 4.1 轴目录（映射到真实 `pto.mi` 模式）

| 轴语义 | producer 模式 | consumer 模式 |
|--------|---------------|---------------|
| `chunk`（纯连续） | NORM / — | NORM_B* |
| `parity`（stride-2 交织） | `vcvt PART_EVEN/ODD`、`vldsx2 DINTLV_B*` | `vstsx2 INTLV_B*` |
| `half`（Histogram 128-bin 半区） | `chistv2 Bin_N0/Bin_N1` | 作为 fan-out 轴，配合 `parity` 轴分两段连续写回 |
| `width`（窄↔宽） | `vsunpack/vzunpack`、`UNPK_B*` | `vpack`、`PK_B*` |

只有当"逻辑连续数组 ↔ 多个交织 vreg"的偏离需要延迟到消费点才解决时，才配作 layout 轴：`parity` / `width` 的产物在物理上是交织的，取连续视图必须做真实搬移（`INTLV` / `pack`），记住轴就是延迟这次搬移。`half` 这类轴记录 producer 天然按逻辑半区产出，后续作为 fan-out 维度参与同一套 layout lowering。


### 4.2 示例

逻辑连续的 `Vb32Range1024`（4×256B，无交织）：
```
#pto.vmi.vreg.layout<logical_shape=[256], phys_dtype=i32, phys_lanes=64,
             axes=[#axis<"chunk", 4, None, 1>]>   // is_contiguous == true
```

`i16→i32` 加宽后的交织形态（2×256B，parity 轴）：
```
#pto.vmi.vreg.layout<logical_shape=[128], phys_dtype=i32, phys_lanes=64,
             axes=[#axis<"parity", 2, PART_EVEN/ODD, 2>]>
```

`chistv2` 产生的 256-bin u16 Histogram 形态（2×256B，half 轴）：
```
#pto.vmi.vreg.layout<logical_shape=[256], phys_dtype=i16, phys_lanes=128,
             axes=[#axis<"half", 2, Bin_N0/N1, 128>]>
```

对上述 Histogram 执行 `pto.vmi.vcvt i16 -> i32` 后的 u32 累积形态（4×256B，half + parity 轴）：
```
#pto.vmi.vreg.layout<logical_shape=[256], phys_dtype=i32, phys_lanes=64,
             axes=[#axis<"half", 2, Bin_N0/N1, 128>,
                   #axis<"parity", 2, PART_EVEN/ODD, 2>]>
```

---

## 5. Op 三分类与 lowering 契约

每个 surface op 在静态 op-table 中归为 A/B/C 一类。pto.as 据此决定 lowering 策略。

| 类别 | 定义 | lowering | 输出 layout |
|------|------|----------|-------------|
| **A. Native-strided** | 逐 lane、dtype 一致、逻辑索引间独立 | 每物理 reg fan-out 一次，谓词逐 reg 配 | 与输入相同（layout 透传） |
| **B. Mode-rewritable** | 有匹配某条轴的原生 `pto.mi` 模式 | 按其他轴 fan-out，对该轴实例化模式 | 消费或产生该轴 |
| **C. Contiguous-required** | 需 stride-1 逻辑访问且无匹配模式 | 自动插 `.contiguous()` 物化后再做连续 op | 扁平 chunk（`is_contiguous`） |

分类索引：

- **A**：PTO-SPEC §5.3 的全部算术、`vcvt`(逻辑)、`vbr vdup vsel vcmp vcmps`、mask 逻辑、`vprelu vexpdif vaxpy`
- **B**：`pto.vmi.vcvt`(宽窄/parity)、`pto.vmi.chistv2`(Histogram half)、`pto.vmi.vlds/vsts`(各 dist)、`pto.vmi.vbr`(从内存)、内部物化路径
- **C**：sec10 跨物理 reg 归约/扫描、`vgather vscatter`(任意逻辑 index)、`vsqz vusqz`(跨 reg)、`vbitsort vmrgsort4`


---

## 6. pto.mi-as：layout 职责

复杂度没有消失，是从用户代码搬进 `pto.as`。`pto.as` 三步：

1. **layout 推断**：给每个 SSA vreg 推一个内部 layout。部分 producer 天生产出交织形态（如逻辑 widen 在硬件上即 EVEN+ODD），pto.as 直接采纳，零转换成本。
2. **layout 协调（coalescing）**：选 layout 使相邻 op 尽量一致，避免来回交织。本质是类似寄存器分配的 layout 分配/合并问题。
3. **自动做连续化处理**：遇到 C 类消费者（按逻辑 index 的 gather、跨 reg 归约、写 AoS），


### 6.1 性能逃生接口：`pto.vmi.prefer_layout`（语义无关提示）

完全自动布局意味着性取决于 pto.as 在各种场景的优化上；早期 pto.as 可能在归约/gather 密集 kernel 上插冗余指令。为兜底，保留**一个**语义无关的提示：

```mlir
%v2 = pto.vmi.prefer_layout %v {hint = "contiguous" | "interleaved"} : !pto.vmi.vreg<...>
```

- 它**不是布局指令**，可被 pto.as 忽略；忽略时语义与不写完全等价。
- 它**不破坏**"surface 无 part/pack/interleave"的目标——hint 是建议，不指定物理摆放细节。
- 它最终会被删除掉。

这是本 RFC 唯一保留的、与布局相关的面向用户的接口。

---

## 7. Before / After 示例

### 7.1 `i16 -> i32` 加宽后连续写回

场景：`i16` 读入 → 加宽到 `i32` → 加偏置 → 连续写回。

**原始 `pto.mi`（用户被迫写 part + interleave）：**
```mlir
%e  = pto.mi.vcvt %a, %m {part="EVEN"} : !pto.mi.vreg<128xi16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xi32>
%o  = pto.mi.vcvt %a, %m {part="ODD"}  : !pto.mi.vreg<128xi16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xi32>
%se = pto.mi.vadd %e, %be, %m32 : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xi32>
%so = pto.mi.vadd %o, %bo, %m32 : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xi32>
pto.mi.vstsx2 %se, %so, %ub[%c0], "INTLV_B32", %m32
    : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.ptr<i32, ub>, index, !pto.mi.mask<b32>
```

**`pto.vmi`（全逻辑，无 part / interleave）：**
```mlir
%w = pto.vmi.vcvt %a      : !pto.vmi.vreg<128xi16> -> !pto.vmi.vreg<128xi32>   // 内部 2 个物理 reg
%s = pto.vmi.vadd %w, %b  : !pto.vmi.vreg<128xi32>
pto.vmi.vsts %s, %ub[%c0] : !pto.vmi.vreg<128xi32>, !pto.mi.ptr<i32, ub>
```

pto.as 内部把 `%w %s` 保持成 parity 交织（跨 2 物理 reg）：`vcvt` 落成 EVEN+ODD，`vadd` 两 reg 各做一次，末尾 `vsts` **融合**成一条 `vstsx2 INTLV_B32`。源码里一个 part/interleave 都没有。

### 7.2 Histogram：256-bin u32 累积直方图

场景：A5 `chistv2` 从一个 `u8` 源向量生成 256 个 cumulative-histogram bin。硬件每次只产出 128 个 `u16` bin：`Bin_N0` 覆盖 bins `0..127`，`Bin_N1` 覆盖 bins `128..255`。为了跨多个 repeat 累加成 `u32`，原始写法必须把每个 half 再用 `vcvt PART_EVEN/ODD` 加宽成四个 `VL_B32` 物理寄存器，最后用两条 `vstsx2 INTLV_B32` 连续写回。

**原始 `pto.mi`（用户被迫写 Bin_N + part + interleave）：**
```mlir
%m16 = pto.mi.pset "PAT_ALL" : !pto.mi.mask<b16>
%m32 = pto.mi.pset "PAT_ALL" : !pto.mi.mask<b32>
%z32 = arith.constant 0 : i32

%acc_n0e_0 = pto.mi.vbr %z32 : i32 -> !pto.mi.vreg<64xi32>
%acc_n0o_0 = pto.mi.vbr %z32 : i32 -> !pto.mi.vreg<64xi32>
%acc_n1e_0 = pto.mi.vbr %z32 : i32 -> !pto.mi.vreg<64xi32>
%acc_n1o_0 = pto.mi.vbr %z32 : i32 -> !pto.mi.vreg<64xi32>

%acc_n0e, %acc_n0o, %acc_n1e, %acc_n1o =
  scf.for %c = %c0 to %repeat step %c1
      iter_args(%a0 = %acc_n0e_0, %a1 = %acc_n0o_0,
                %a2 = %acc_n1e_0, %a3 = %acc_n1o_0)
      -> (!pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>,
          !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>) {
    %src_c = pto.mi.vlds %src_ub[%c] {dist = "NORM"} : !pto.mi.ptr<i8, ub> -> !pto.mi.vreg<256xi8>
    %h0 = pto.mi.chistv2 %src_c, %p8 {bin_part = "Bin_N0"}
        : !pto.mi.vreg<256xi8>, !pto.mi.mask<b8> -> !pto.mi.vreg<128xi16>
    %h1 = pto.mi.chistv2 %src_c, %p8 {bin_part = "Bin_N1"}
        : !pto.mi.vreg<256xi8>, !pto.mi.mask<b8> -> !pto.mi.vreg<128xi16>

    %h0e = pto.mi.vcvt %h0, %m16 {part = "EVEN"} : !pto.mi.vreg<128xi16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xi32>
    %h0o = pto.mi.vcvt %h0, %m16 {part = "ODD"}  : !pto.mi.vreg<128xi16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xi32>
    %h1e = pto.mi.vcvt %h1, %m16 {part = "EVEN"} : !pto.mi.vreg<128xi16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xi32>
    %h1o = pto.mi.vcvt %h1, %m16 {part = "ODD"}  : !pto.mi.vreg<128xi16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xi32>

    %n0e = pto.mi.vadd %a0, %h0e, %m32 : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xi32>
    %n0o = pto.mi.vadd %a1, %h0o, %m32 : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xi32>
    %n1e = pto.mi.vadd %a2, %h1e, %m32 : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xi32>
    %n1o = pto.mi.vadd %a3, %h1o, %m32 : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xi32>
    scf.yield %n0e, %n0o, %n1e, %n1o
        : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>,
          !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>
  }

pto.mi.vstsx2 %acc_n0e, %acc_n0o, %bin_count[%c0], "INTLV_B32", %m32
    : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.ptr<i32, ub>, index, !pto.mi.mask<b32>
pto.mi.vstsx2 %acc_n1e, %acc_n1o, %bin_count[%c128], "INTLV_B32", %m32
    : !pto.mi.vreg<64xi32>, !pto.mi.vreg<64xi32>, !pto.mi.ptr<i32, ub>, index, !pto.mi.mask<b32>
```

**`pto.vmi`（保留显式 `vcvt`，但不再写 Bin_N / part / interleave）：**
```mlir
%z32 = arith.constant 0 : i32
%hist0 = pto.vmi.vbr %z32 : i32 -> !pto.vmi.vreg<256xi32>

%hist = scf.for %c = %c0 to %repeat step %c1
    iter_args(%acc = %hist0) -> !pto.vmi.vreg<256xi32> {
  %src_c = pto.vmi.vlds %src_ub[%c] : !pto.mi.ptr<i8, ub> -> !pto.vmi.vreg<256xi8>
  %h16 = pto.vmi.chistv2 %src_c, %p8 {bins = 256}
      : !pto.vmi.vreg<256xi8>, !pto.vmi.mask<256xb8> -> !pto.vmi.vreg<256xi16>
  %inc = pto.vmi.vcvt %h16 : !pto.vmi.vreg<256xi16> -> !pto.vmi.vreg<256xi32>
  %next = pto.vmi.vadd %acc, %inc : !pto.vmi.vreg<256xi32>
  scf.yield %next : !pto.vmi.vreg<256xi32>
}

pto.vmi.vsts %hist, %bin_count[%c0] : !pto.vmi.vreg<256xi32>, !pto.mi.ptr<i32, ub>
```

pto.as 内部先把 `%h16` 识别成 `half` 布局：`chistv2` lower 成 `Bin_N0/Bin_N1` 两个 producer。显式的 `pto.vmi.vcvt` 再把 `%h16` 加宽为 `%inc`，并在内部产生 `parity` 轴：lower 时对应每个 half 的 `vcvt PART_EVEN/ODD`。`pto.vmi.vadd` 按 4 个物理 `VL_B32` fan-out；末尾 `pto.vmi.vsts` 按 half fan-out 成两条 `vstsx2 INTLV_B32`，分别写 bins `0..127` 和 `128..255`。源码里保留了 `vcvt` 这个真实语义，但不再出现 `Bin_N0/Bin_N1`、`EVEN/ODD` 或 `INTLV_B32`。

---

## 8. 简化后的 surface op 集

PTO SPEC中sec3–13 的指令按"是否承载逻辑语义"分成三种类。

### 8.1 彻底删除

这些 op **不出现在 pto.vmi surface**：

- sec12：`vintlv vdintlv vintlvv2 vdintlvv2`
- sec3：`vldsx2 vstsx2` 的 x2 交织/解交织形态
- 所有 dist token：`INTLV / DINTLV / PK / UNPK / SPLT / MRG`，以及 `vcvt` 上的 `part=EVEN/ODD` 属性

### 8.2 简化为更简单的op

| 原始 `pto.mi` | `pto.vmi` surface | 说明 |
|------------|----------------|------|
| `vcvt {part=EVEN/ODD}` | `pto.vmi.vcvt`（无 part） | part 是 layout，不是语义 |
| `vpack / vsunpack / vzunpack` | `pto.vmi.vcvt` | 窄↔宽是真语义，半区摆放是 layout |
| `vlds {dist=BRC_*}` | `pto.vmi.vbr`（从内存广播） | 逻辑广播 |
| `vlds {dist=US/DS}` | `pto.vmi.vresample`（可选） | 真重采样，不再以 load 模式出现 |
| `vlds/vsts {dist=NORM}` | `pto.vmi.vlds / pto.vmi.vsts` | 逻辑连续访存 |
| `chistv2 {bin_part=Bin_N0/Bin_N1}` | `pto.vmi.chistv2 {bins=256}` | Histogram 的 half 是 layout，不是用户语义 |

### 8.3 原样保留（本就是逻辑语义）

---

## 9. 待讨论问题（Open Issues）

本节列出当前设计尚未解决的2个关键问题，需要在后续迭代中明确决策。每个问题给出问题本质、影响分析、初步倾向，但不做最终结论。

### 9.1 256-lane SIMT 编程模型——是否需要 128-lane / 64-lane 选项？

**问题**：RFC §3.1 规定 `L * bitwidth(T)` 必须是 256B 的整数倍。大部分 simdvf 代码包含 fp8 / i8 dtype；对于 i8 / fp8，最小合法 `L = 256`（256 × 8bit = 256B），即 vreg 为 `256xi8` 或 `256xfp8`。按照 RFC 的逻辑，所有其他 dtype 的 vreg 也必须对齐到 256 lane 的倍数——`256xfp16`、`256xfp32`。

这意味着 **pto.vmi 实际上创造了一个 256-lane 的 SIMT 编程模型**：不存在 128-lane 或 64-lane 的选项。所有 dtype 统一以 256 lane 为最小执行粒度，不同 dtype 只是每 lane 的 bitwidth 不同。

**影响**：

| 方面 | 正面 | 负面 |
|------|------|------|
| 编程模型 | 统一 256-lane，消除"不同 dtype 不同 lane 数"的心智负担 | 循环粒度变大：f32 计算一次迭代处理 256 个元素（4 物理 reg），而非传统 64 个 |
| 与 GPU SIMT 对比 | 固定 lane 数是 SIMT 的核心特征（CUDA warp 固定 32-thread），类比清晰 | GPU warp 的 32-thread 是最小粒度但允许 sub-group 分组；256-lane 是否也需要类似机制？ |
| 物理效率 | 256-lane f32 = 4 物理 reg，每次循环覆盖更多数据，吞吐率可能更高 | 4 物理 reg 的 fan-out 使单次迭代指令数翻倍，是否会引入reg-spill的问题？ |


### 9.2 sub-group 概念——vreg 内的分组执行粒度

**问题**：由于 §9.1 的推论——所有 vreg 都是 256 lane，对于某些 tile 操作来说 256 lane 太大了。例如一个 128 × 128 的 tile，可能需要把 2 行 pack 进同一个 vreg 来充分利用 vreg 空间（2 × 128 bf16 = 256 bf16 = 2 物理 reg），但对于 reduce 等操作，需要在 vreg 内划分更小的执行粒度——"在每 128 个 lane 的 sub-group 内做 reduce"，而非在整个 256 lane 上。

**核心设计决策**：sub-group 信息应该加在 **vreg 类型**上还是 **op 属性**上？

**选项分析**：

| 选项 | 形式 | 优点 | 缺点 |
|------|------|------|------|
| **加在 vreg 上** | `!pto.vmi.vreg<L x T, sub_group=N, #layout>` | 一次声明全局生效，下游 op 不需重复指定 | vreg 类型过于复杂；layout 已经是编译器持有的内部属性，再叠加 sub-group 严重膨胀类型系统；同一 vreg 在不同 op 中可能需要不同分组粒度 |
| **加在 op 上** | `pto.vmi.vreduce %v {sub_group=2}` | vreg 类型保持简洁；sub-group 只影响特定 op 的语义解释；与现有 A/B/C op 分类自然衔接 | 每个需要 sub-group 的 op 都要单独指定，写法略冗余 |

**涉及的 op**：

- **reduce**：`pto.vmi.vreduce %v {sub_group=N}` — 在每 `L/N` 个 lane 的 sub-group 内归约
- **vci**（向量-标量交互）：从 sub-group 内提取标量或将标量广播到 sub-group
- **vld / vst**：已sub-group为单位做strided load/store

**典型场景——MX block-scale exponent reduce**：一个 `!pto.vmi.vreg<256xbf16>` 在逻辑上仍然是连续的 256 个 bf16 元素，但对于 MX 的 32-element block-scale，需要将它切分成 8 个 sub-group（每 32 个 lane 为一组），在每个 sub-group 内做 max-reduce 取 shared exponent。逻辑上用户只想说"对这片数据按 32-element block 取 max exponent"，不关心物理上 DINTLV 解交织、expMask 位提取、VLane 对齐等细节——这些由 pto-as 推导映射到具体的 `pto.mi` 指令组合。

```mlir
// 用户视角：一个连续 vreg，按 sub-group 做 reduce
%x      = pto.vmi.vlds %x_ub              : !pto.mi.ptr<bf16,ub> → !pto.vmi.vreg<256xbf16>
%exp    = pto.vmi.vand %x, %exp_mask       : !pto.vmi.vreg<256xbf16> → !pto.vmi.vreg<256xu16>   // extract exponent field
%maxe   = pto.vmi.vreduce_max %exp {sub_group=8}
           : !pto.vmi.vreg<256xu16> → !pto.vmi.vreg<8xu16>       // 每 32 lanes 取 max → 8 个 shared exponent
%scaleb = pto.vmi.vbr %maxe                : !pto.vmi.vreg<8xu16> → !pto.vmi.vreg<256xu16>       // broadcast 回原 vreg 粒度
```

**pto-as 推导路径**——从下游 op 的 sub-group 需求反向推断最优 layout：

`pto.vmi.vlds` 本身不携带任何关于 UB 数据物理交织的信息——它只声明"我要从 UB 加载一片连续的 `256xbf16`"。pto-as 无法从这条 load 指令推断出数据在 UB 中是交织存储的，也无法推断出应该用 `DINTLV` 还是 `NORM`。**layout 推断的驱动力来自下游消费者**：

1. **推断起点**：`vreduce_max {sub_group=8}`。这个 op 声明了分组粒度——256 个 u16 切成 8 个 32-lane sub-group，每组取 max。对于 bf16/u16，`E_v = 16`，每个 32B VLane 容纳 16 个 u16。32-lane sub-group = 2 VLane × 16 lanes/VLane = **恰好对齐 VLane 边界**。这让 pto-as 确认：reduce 可以用 Category B 的 `vcgmax`，无需转换。

2. **反向推断最优 layout**：既然 reduce 的最优路径是 `vcgmax`（每个 VLane 内独立取 max），pto-as 需要确保数据在到达 reduce 时已经按 VLane 级别组织好——即**每个 VLane 内的 16 个 u16 正好是同一个 32-element block 的 exponent**。要做到这一点，最省指令的方式是让 load 直接产出 parity 布局（even/odd 拆成 2 个 vreg），然后 vand 各自提取 exponent，最后 vmax fold 成 1 个 vreg——此时 fold 后的 vreg 中每个 VLane 的 16 lanes 恰好覆盖一个 32-element block 的 even+odd max exponent，直接喂给 `vcgmax`。如果 load 用 `NORM` 加载再手动拆分，反而多了一步 layout 变换。因此 **pto-as 从下游 reduce 的需求反向推断：load 应该用 `DINTLV_B16`**。

3. **逐指令 lowering**：

   - `%x = pto.vmi.vlds`：pto-as 根据下游 reduce 的 sub-group 需求，选择 `DINTLV_B16` distribution → lower 到 `pto.vldsx2 "DINTLV_B16"`（一次加载拆成 even/odd 两个物理 vreg：`vdExp0` 和 `vdExp1`）。`%x` 的 layout 含 `parity` 轴。

   - `%exp = pto.vmi.vand`：Category A op，layout 透传。lower 到两条 `pto.mi.vand`（even vreg & `expMaskBF16`, odd vreg & `expMaskBF16`），得到 `vdExpExtract0` 和 `vdExpExtract1`。各自的 parity 轴保留。

   - `%maxe = pto.vmi.vreduce_max {sub_group=8}`：这是推断起点。逻辑上是"256 个 u16 切成 8 个 32-lane sub-group，每组取 max"。pto-as 推导出两个 lowering 步骤：
     - parity 布局的 even/odd 两个 vreg 用 `pto.mi.vmax` 折叠为 1 个 vreg（`(K-1)× vmax`，将 2 reg → 1 reg）。fold 后 parity 轴消失，但数据已经按 VLane 级别组织——每个 VLane 的 16 lanes 覆盖一个 32-element block 的 max exponent。
     - `sub_group=8` 对齐 VLane 边界，lower 到 `pto.mi.vcgmax`——每 VLane 内取 max，1 op per physical reg，no cross-reg combine, no materialization。输出 8 个 shared exponent（每 VLane lane 0 一个）。

   - `%scaleb = pto.vmi.vbr`：R6 broadcast reuse——8 个 shared exponent 广播回 256 个 lane，物理 backing 只 1 reg，replicate-read。

4. **关键洞察**：layout 推断不是从 load 向下游"顺流推"，而是从 reduce 的 sub-group 需求向上游"逆流拉"。`pto.vmi.vlds` 只声明了逻辑意图（"加载 256 个连续 bf16"），**具体用什么 distribution token 是 pto-as 为了服务下游 reduce 的最优路径而做出的选择**。这正是 P5 的精神——lowering 是 local instruction-selection search，不是固定 recipe。

**sub-group size 大于 32B**：

| `sub_group` 值 | 与 VLane 关系 | pto-as 推断的最优 load layout | lowering 策略 | 物理指令 |
|---|---|---|---|---|
| `8`（256 / 32 = 8 sub-groups，bf16 时每组 = 2 VLane） | 对齐 VLane 边界 | `DINTLV_B16`（parity layout 服务后续 fold+vcgmax） | fold → `vcgmax` | `vldsx2 DINTLV_B16` + `vand`×2 + `vmax` + `vcgmax` |
| `4`（256 / 64 = 4 sub-groups，每组 = 4 VLane） | 对齐 VLane 边界 | `DINTLV_B16` 或 `NORM`（取决于 fold 链长度 vs materialization 成本） | 跨 VLane 组 `vcgmax`，fold 链更长 | `vldsx2` + `vand`×2 + `vmax`×3 + `vcgmax` |

也就是说，**sub-group 的值决定了 reduce axis 与 VLane 边界的对齐关系，从而反向推断出 load 应用的最优 distribution**。当 sub-group 的 lane 数恰好是 `E_v` 的整数倍时，reduce axis 对齐 VLane，pto-as 选择 parity layout 以最大化 `vcg*` 的命中概率；否则落入 Category C 需要转换，load 退回 `NORM`。


**初步倾向**：建议 **加在 op 上**，原因：

1. vreg 已经有 layout 属性（编译器持有），再加 sub-group 会让类型系统过于复杂
2. sub-group 语义本质上是对**操作的分组方式**描述，而非对数据的固有属性描述——同一个 `!pto.vmi.vreg<256xbf16>` 可以在不同 op 中以不同 sub-group 粒度使用
3. op 属性与现有 A/B/C 分类自然衔接：sub-group 信息可以作为 B 类 op 的模式属性参与 lowering
4. MX block-scale 场景验证了 op 属性的合理性：同一个 `vreg<256xbf16>` 在 `vreduce_max {sub_group=8}` 中以 32-lane 粒度 reduce，在 `vadd` 中以全 256-lane 粒度运算——分组粒度是 op 的语义，不是 vreg 的固有属性