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
   │  (pto.mi-as：layout-assignment + lowering)
   ▼
pto.mi       vcvt EVEN/ODD + 两路 vadd + vstsx2 INTLV_B32        ← 物理：SIMD 寄存器交织细节
```

也就是说：**`T.parallel` 的逻辑迭代空间，几乎可以一对一直译成 `pto.vmi` 的逻辑向量 op**——逐元素计算映射到 Category A op，`T.cast` 映射到一条无 part 的 `pto.vmi.vcvt`，逻辑长度 `N` 映射到 `!pto.vmi.vreg<N x T>`，`T.parallel` 隐含的"全 active"语义映射到自动补齐的尾部谓词。宽度转换引入的 EVEN/ODD 交织、以及 SIMD 寄存器上的 layout 映射这件 `T.parallel` 无需暴露，后续由pto.mi-as解决。

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
- `#layout`：surface 源码通常省略，由 pto.mi-as 填充。

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
- 用户对逻辑长度 `L` 写一个谓词；pto.mi-as 自动拆成 `K-1` 个 `PAT_ALL` + 1 个尾部 `plt/pge`。

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

- Category A 全部：sec6 `vabs vneg vexp vln vsqrt vrelu vnot`；sec7 `vadd vsub vmul vdiv vmax vmin vand vor vxor vshl vshr vaddc vsubc`；sec8 `vadds vmuls vmaxs vmins vshls vshrs vl
