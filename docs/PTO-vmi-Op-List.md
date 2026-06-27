# PTO 虚拟微指令(`pto.vmi`)——指令清单(含操作数与数据类型)

[toc]

---

### 0.1 记号

| 记号 | 含义 |
|---|---|
| `V<L×T>` | `!pto.vmi.vreg<L×T>` 逻辑向量，`L` = 逻辑元素数 |
| `M<L×G>` | `!pto.vmi.mask<L×G>` 虚拟谓词，`G ∈ {b8,b16,b32}` |
| `Ptr<T>` | `!pto.ptr<T,ub>` UB 指针 |
| `s` | 标量 |
| `[pmode]` | 可选治理谓词（governing mask），`{pmode = "merge"\|"zero"}`，默认 `zero` |
| `K` | 一个逻辑 vreg 的物理后备数，`K = L·bitwidth(T) / 2048` |
| `E_v` | 一个物理 vreg 的 lane 数（f32/i32=64, f16/bf16/i16=128, i8/fp8=256） |
| BlockLane | 硬件 32B 原子归约单元，vreg = 8 BlockLane；BlockLane 容纳 `32B / bitwidth(T)` 个 lane |

### 0.2 Category A/B/C —— 精确 lowering 契约

RFC §5 给了三类定义，这里展开成 pto.as 的可执行判据：

| 类别 | 判据 | pto.as 行为 | 输出 layout |
|---|---|---|---|
| **A. Native-strided** | 逐 lane、dtype 一致、lane 间独立，输入输出同 dtype 同 `L` | 每 `K` 个物理 reg 各 fan-out 一次该 `pto.mi` op；治理谓词按物理 mask 族逐 reg 配（必要时 `ppack/punpack`） | 与输入相同（layout 透传，含 parity/half 等轴一并透传） |
| **B. Mode-rewritable** | 存在匹配某条 layout 轴的原生 `pto.mi` 模式 | 沿**其他**轴 fan-out，对匹配轴实例化对应模式（`PART_EVEN/ODD`、`Bin_N0/N1`、`PK/UNPK`、`INTLV/DINTLV`…） | 消费或产生该轴 |
| **C. Contiguous-required** | 需 stride-1 逻辑访问且无匹配模式（跨 reg 归约/scan、任意 index gather/scatter、跨 reg squeeze） | 先自动插 `.contiguous()` 物化（`INTLV`/`pack`/搬移）把交织轴拍平，再做连续 op | 扁平 chunk（`is_contiguous`） |

> A 类的"layout 透传"是 vmi 的核心红利：parity/half 轴上的数据无需解交织即可逐 lane 运算，搬移延迟到真正需要连续视图的 C 类消费者。

### 0.3 谓词传播总则

1. **governing mask 伴随数据轴**：`[pmode]` 在 lowering 时随数据 fan-out 到每个物理 reg；mask 族 `G` 必须与数据族对齐（f32/i32→b32，f16/bf16/i16→b16，i8/fp8→b8）。跨族（如 widen i16→i32）时 mask 族也变，pto.as 用 `ppack/punpack` 或重新 `pset` 生成对应族 mask。
2. **inactive lane 行为**：`pmode="zero"`（默认）inactive lane 写 0；`pmode="merge"` 保留目的原值。reduce 类 op 的 inactive lane 见 §2.3。
3. **A5 load 不可谓词化**：`vlds` 的尾部谓词不能挂在 load 上，必须迁移到消费侧 op 或 store。`vsts` 在 A5 上可谓词化。
4. **tail mask 物化**：`pset "PAT_ALL"` / `pge "PAT_VLn"` / `plt %rem` 三件套覆盖全 active、头尾 active、数据相关尾三种模式（见 §10）。

---

## Part 1 —— 虚拟类型形式化

### 1.1 `!pto.vmi.vreg<L×T>`

- **合法性**：`L · bitwidth(T)` MUST 是 2048bit/256B 的整数倍。等价地 `L` 必须是 `E_v` 的整数倍。
- **物理后备**：`K = L·bitwidth(T) / 2048` 个 `!pto.mi.vreg<E_v×T>`。`K=1` 时 vmi.vreg 与 pto.mi.vreg 一一对应。`K<1`时按照`K=1`处理。
- **`#layout`**：surface 源码省略，由 pto.as 填充。layout 是**编译器内部属性**，不进入用户类型签名（用户只写 `L×T`）。
- **合法 `L` 与 dtype**：

| T | bits | E_v | L 必须是 … 的2的幂次倍数 |
|---|---|---|---|
| f32 / i32 | 32 | 64 | 64 |
| f16 / bf16 / i16 | 16 | 128 | 64 |
| i8 / fp8 | 8 | 256 | 64 |

### 1.2 `!pto.vmi.mask<L×G>`

- 虚拟谓词，物理后备 `K` 个 256-bit `!pto.mi.mask<G>`。
- `G` 与数据族对齐（见 §0.3-1）。
- **合法性**：`L` 与所修饰 vreg 的 `L` 一致；`K` 一致。

### 1.3 compact reduce 结果

reduce/broadcast 涉及"小于 256B 的 compact 标量向量"（如 `V<2×f32>`、`V<8×u16>`）。其第一根轴的维度与`group`相等。

---

## Part 2 —— `group` 语义形式化

> **决策**（对齐 RFC §9.2 倾向 + Op-List 现有命名）：sub-group 信息**加在 op 属性上**，命名为 `group`。同一 vreg 可在不同 op 以不同 `group` 粒度使用——`V<256×bf16>` 在 `vcmax {group=8}` 中切成 8 个 sub-group reduce，在 `vadd` 中按全 256-lane 粒度运算。`group` 是 **op 的语义**，不是 vreg 固有属性。

### 2.1 定义

对 reduce op（`vcadd`/`vcmax`/`vcmin`）与广播 op（`vbrc`），`{group=C}` 中的 `C` 表示 **group 数（组数）**，而非每 sub-group 的 lane 数：

- **reduce**：把 `V<L×T>` 的 `L` 个 lane 切成 **`C` 个 sub-group**，每 sub-group `L/C` lane，各产 1 个 compact scalar；输出 `V<C×T>`（`C` 个 scalar，低 `C` 槽有效）。
- **vbrc**：把 compact 输入 `V<C×T>` 的 `C` 个 scalar 分别广播回各自 `(L/C)`-lane sub-group，输出 `V<L×T>`。

`C` 必须整除 `L`；**每 sub-group 的 lane 数**为 `L/C`，其字节数 `W = (L/C)·bitwidth(T)/8` 决定与 BlockLane 边界的关系，从而决定 Category。

### 2.2 `group` → Category 决策表

记 `W = (L/C) · bitwidth(T) / 8`（一个 sub-group 的字节数，`L/C` 为每 sub-group 的 lane 数），`BlockLane = 32B`。

| `W` 与 BlockLane 关系 | Category | pto.as lowering | 物理指令 |
|---|---|---|---|
| `W == 32`（sub-group 恰为 1 BlockLane） | **B** | 每 BlockLane 独立 reduce，1 op per physical reg，无跨 reg 合并 | `vcgadd`/`vcgmax`/`vcgmin` |
| `W = k·32, k>1`（sub-group 跨 k 个 BlockLane，对齐 BlockLane 边界） | **B**（fold + vcg） | 先 `(k-1)` 次 `vadd/vmax/vmin` 把 k 个 BlockLane fold 成 1 个 BlockLane 宽的中间值，再 `vcg*` | `vmax×(k-1)` + `vcg*` |
| `W` 不是 32 的整数倍（sub-group 不对齐 BlockLane） | **C** | 物化成连续形态后用全向量 `vc*`，或 pto.as 选择 `NORM` load + 重排 | `vcadd`/`vcmax`/`vcmin`（物化后） |


### 2.3 reduce 的 inactive lane 与 argmax

- **inactive lane**：`vcmax/vcmin` 视 inactive 为 `-INF/+INF`（fp）或类型字面 min/max（int）；全 inactive 时 `result[0]` 为该极值。`vcadd` 视 inactive 为 0。与 SPEC §10 一致。

  ```mlir
  %val = pto.vmi.vcmax %x, %m {group = C}
      : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L×G> -> !pto.vmi.vreg<C×T>
  ```

### 2.4 典型场景：MX block-scale exponent reduce（bf16，group=8）

`V<256×bf16>` 取 shared exponent，每 32-element block 一个 max → `256/32 = 8` 个 sub-group，故 `group=8`（每 sub-group `256/8 = 32` lane）。bf16 `bitwidth=16`，`W=32·16/8=64B=2 BlockLane` → 落 §2.2 第二行（`k=2`，对齐 BlockLane，Category B）。

- load：pto.as 从下游 `group=8` 逆推，选 `DINTLV_B16`（parity）让 even/odd 各成一 vreg。
- `vand` 提 exponent：A 类透传，两 parity vreg 各 `vand`。
- fold：`(k-1)=1` 次 `vmax` 把 even/odd 两 vreg fold 成 1 个，parity 轴消失，每 BlockLane 16 lane 覆盖一个 32-element block 的 max exponent。
- `vcgmax`：每 BlockLane 取 max → 8 个 shared exponent（与 `group=8` 一致；8 BlockLane 各产 1 个，一一对应）。
- 输出：compact `V<8×bf16>`（`C=8` 个 shared exponent）。

---

## Part 3 —— Group 1:Load / Store

`vlds`/`vsts` 是逻辑连续访存;`dist` token(`NORM`/`INTLV`/`PK`/`UNPK`/`1PT`…)对用户不可见,
由 pto.as 根据入口 layout 自动选择(详见配套文档 §3)。`vsts` 的 `{mode}` 用于 compact 标量
按 `group` 写出(见配套文档 §3.1)。

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vlds` | A | `Ptr<T>` | `V<L×T>` | i8–i32, f16/bf16, f32 |
| `vsts` | A | `V<L×T>`, `Ptr<T>`, `M`, `[pmode]` | — | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 连续 load:UB → vreg
%v = pto.vmi.vlds %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>

// 连续 store:vreg → UB(带治理谓词,A5 上 store 可谓词化)
pto.vmi.vsts %v, %ub_out[%offset], %mask : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<b32>

// TODO:广播 load:标量/块复制进 vreg
%vb = pto.vmi.vlds %ub[%offset], <dist> : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>

// tail/partial load:A5 上 load 不可谓词化,%mask 迁移到消费侧/store
%vt = pto.vmi.vlds %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>   // 尾谓词在下游 vadd/vsts 生效
```

---

## Part 4 —— Group 2:index-gen

复制与索引物化。产生 `broadcast` 轴或索引向量;
直到 Category-B/C 边缘需要展开形式前,绝不展开成 `K` 份存储副本。

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vci` | A | `s`  | `V<E×i32>` (`{ASC/DESC}`) | i8–i32, f16, f32 |

示例:

```mlir
// vci:生成 [base, base+1, ...] lane 索引(ASC/DESC 由属性给出,%base 为起始标量)
%idx = pto.vmi.vci %base {order = "ASC"} : i32 -> !pto.vmi.vreg<64xi32>
```

---

## Part 5 —— Group 3:Eltwise compute


| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vadd` `vsub` `vmul` `vdiv` `vmax` `vmin` | A | `V<T>`, `V<T>` (或 bcast), `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 (`vdiv` f16/f32) |
| `vand` `vor` `vxor` | A | `V<T>`, `V<T>`, `[pmode]` | `V<T>` | i8–i32 |
| `vnot` | A | `V<T>`, `[pmode]` | `V<T>` | i8–i32 (bit-typed) |
| `vshl` `vshr` | A | `V<T>`, `V<T>` (向量 count), `[pmode]` | `V<T>` | i8–i32 |
| `vadds` `vmuls` `vmaxs` `vmins` `vshls` `vshrs` | A | `V<T>`, `s`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vabs` `vneg` `vrelu` | A | `V<T>`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vexp` `vln` `vsqrt` | A | `V<T>`, `[pmode]` | `V<T>` | f16, f32 |
| `vcmp` | A | `V<T>`, `V<T>`, `M` | `M` | i8–i32, f16/bf16, f32 |
| `vcmps` | A | `V<T>`, `s`, `M` | `M` | i8–i32, f16/bf16, f32 |
| `vsel` | A | `M`, `V<T>`, `V<T>`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vselr` | A | `V<T>`, `V<index>` | `V<T>` (permute) | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 二元算术(逐 lane,带治理谓词)
%s = pto.vmi.vadd %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>
%m = pto.vmi.vmax %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>

// 向量-标量(标量隐式广播)
%scaled = pto.vmi.vmuls %x, %scale, %mask : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>
%shifted = pto.vmi.vshrs %data, %c4, %mask : !pto.vmi.vreg<64xi32>, i16, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xi32>

// 一元算术/激活
%a = pto.vmi.vabs %v, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>
%e = pto.vmi.vexp %v, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>

// 比较 → 谓词(第三个 M 为治理谓词,限定哪些 lane 参与比较;inactive lane 结果置 0)
%lt = pto.vmi.vcmp %a, %b, %m, "lt" : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.mask<b32>

// 标量比较 → 谓词
%ges = pto.vmi.vcmps %a, %c0, %m, "ge" : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<b32> -> !pto.vmi.mask<b32>

// 谓词选择:%mask 为真取 %x,否则取 %y
%out = pto.vmi.vsel %x, %y, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>

// 寄存器 gather/permute
%p = pto.vmi.vselr %x, %idx : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xf32>


---

## Part 6 —— Group 4:Broadcast

`vbrc` 是逻辑标量→向量/已归约→扇出广播(R6)。未分组形式便宜;分组形式(每 BlockLane
partial 扇回自身 lane)是难情形。

| op (form) | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vbrc` (未分组) | A | `s` | `V<L×T>` | i8–i32, f16/bf16, f32 |
| `vbrc` (`{group=C}`) | B | `V<C×T>` | `V<L×T>` | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 未分组广播:标量/已归约值扇出到整条 vreg(寄存器内,无 UB roundtrip)
%bc = pto.vmi.vbrc %maxe : f32 -> !pto.vmi.vreg<64xf32>

// 分组广播:C=8 个 scalar 各扇回自身 (L/C)=8 lane sub-group(无直接物理指令对应,实现由 pto.as决定)
%scaleb = pto.vmi.vbrc %maxe {group = 8} : !pto.vmi.vreg<8xf32> -> !pto.vmi.vreg<64xf32>
```


---

## Part 7 —— Group 5: reduce

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vcadd` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i8–i32, f16, f32 |
| `vcmax` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i16–i32, f16, f32 |
| `vcmin` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i16–i32, f16, f32 |

示例:

```mlir
// 全数组求和归约(到标量)
%sum = pto.vmi.vcadd %x, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<1xf32>

// 全数组 max 归约(到标量)
%mx = pto.vmi.vcmax %x, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<1xf32>

// sub-group 归约:group=8 → 256 lane 切 8 组(每组 32 lane),各取一个 max → 8 个 compact scalar
%maxe = pto.vmi.vcmax %exp {group = 8} : !pto.vmi.vreg<256xu16>, !pto.vmi.mask<b16> -> !pto.vmi.vreg<8xu16>
```


---

## Part 8 —— Group 6:Convert (cvt)

一个逻辑 `vcvt`,其 *目标 dtype 即布局*。`pto.as` 展开为 dtype-specific cast 链 +
part/width staging + 匹配的 store distribution,并拖出谓词伴随。

**属性**:`{to=<dtype>, rnd=<R>, sat=<SAT>}`(`to` 可由返回类型推断,为显式冗余;`rnd`/`sat`
控制窄化时的取整与饱和)。`part`/`PART_*`/`PK`/`UNPK` **不出现在 surface**,由 pto.as 作为
内部 layout 轴(`parity`/`width`/`sub_part`)填充。

| op (form) | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vcvt` | B | `V<L×Tn>`, `[pmode]` | `V<L×Tm>` | (i8–i32, f8-f32)↔(i8–i32, f8-f32)|
| `vinterpret_cast` | A | `V<L×T>` | `V<L×T'>` | 任意bit 重解释,显式,无布局推断|


> **`vinterpret_cast`** —— bit 级重解释(`bitcast`)。**不**是 `vcvt`:不产生 parity/width
> 轴、无布局可推断、无 dtype cast 链,只是把同一组 bit 按新 dtype 读出。因此
> Category 留空、不带 `[pmode]`,刻意保留为显式操作(作者自行保证语义合法)。

示例:

```mlir
// 宽化 16→32(radix-2,parity EVEN/ODD 由 pto.as 展开)
%w = pto.vmi.vcvt %a, %mask : !pto.vmi.vreg<128xf16>, !pto.vmi.mask<b16> -> !pto.vmi.vreg<128xf32>

// 窄化 32→16
%n = pto.vmi.vcvt %a, %mask : !pto.vmi.vreg<128xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<128xf16>

// 量化 f32 → fp8
%q = pto.vmi.vcvt %s, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xfp8>

// bit 重解释(显式,不经 vcvt)
%r = pto.vmi.vinterpret_cast %a : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xi32>
```


---

## Part 9 —— Group 7:SFU

特殊功能/领域加速器操作。混合类别:`chistv2` 产生 `half` 轴(B);sort 与
gather/scatter 是 Category-C tile/permute 操作;融合激活/算术操作是 Category-A
`vreg→vreg`。

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vhist` | B | `V<L×i*>` (bin idx), `[pmode]` | `V` (Bin_N0/N1 counts, half 轴) | i8–i32 (bin index) |
| `vgather` | C | `Ptr<T>`, `%idx`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vgatherb` | C | `Ptr<T>`, `%idx`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vscatter` | C | `V<T>`, `%idx`, `Ptr<T>`, `[pmode]` | — | i8–i32, f16/bf16, f32 |
| `vexpdif` | A | `V<f*>` (x), `V<f32>` (max) †, `[pmode]` | `V<f32>` | f16/f32 → f32 |
| `vaxpy` | A | `V<T>` (x), `V<T>` (y), `s` (α) †, `[pmode]` | `V<T>` | f16, f32 |
| `vlrelu` | A | `V<T>`, `[pmode]` | `V<T>` | f16, f32 |
| `vprelu` | A | `V<T>`, `s`/param, `[pmode]` | `V<T>` | f16, f32 |
| `vmull` | B | `V<i32>`, `V<i32>`, `[pmode]` | `V<i64>` (hi+lo, 2 reg; 产生 `width` 轴) | i32/u32 |
| `vmula` | A | `V<T>` (acc), `V<T>`, `V<T>` †, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 直方图/per-bin 计数
%h = pto.vmi.vhist %bin_idx, %mask : !pto.vmi.vreg<256xi8>, !pto.vmi.mask<256xb8> -> !pto.vmi.vreg<256xi16>

// 索引 gather(B32 / byte)
%g = pto.vmi.vgather %src, %offsets, %mask : !pto.ptr<f32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>
%gb = pto.vmi.vgatherb %src, %offsets, %mask : !pto.ptr<i32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<256xi32>

// 索引 scatter
pto.vmi.vscatter %v, %dest, %offsets, %mask : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<b32>

// 融合 exp(x − max)(softmax)
%e = pto.vmi.vexpdif %x, %max, %mask, "EVEN" : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>

// 融合 α·x + y
%y = pto.vmi.vaxpy %x, %acc, %alpha, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>

// leaky / parametric ReLU
%lr = pto.vmi.vlrelu %x, %slope, %mask : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>
%pr = pto.vmi.vprelu %x, %alpha, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>

// 宽化 32×32→64 乘(产生 width 轴,hi+lo 两 reg)
%res = pto.vmi.vmull %a, %b, %mask : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xi64>

// 融合乘加
%acc = pto.vmi.vmula %acc, %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>
```

---

## Part 10 —— Group 8:Predicate ops

`pset`/`pge`/`plt` 的 mask 族(`b8/b16/b32`)由返回类型 `M<L×G>` 决定,**不**进 op 名
(即 `pset : !pto.vmi.mask<b32>`,而非 `pset_b32`)。族必须与所修饰的数据族对齐。

| op | Mask in | In | Out | Datatypes |
|---|---|---|---|---|
| `pset` | gen | — (命名模式 `PAT_*`) | `M` | b8/b16/b32 |
| `pge` | gen | — (lane-count 模式 `PAT_VLn`) | `M` (tail) | b8/b16/b32 |
| `plt` | gen | `s` (i32, 如 `%rem`) | `M` (tail), `s`(next) | b8/b16/b32 |

TODO: 双输出考虑一下如何设计。
示例:

```mlir
// 物化全 active / tail 模式 mask(gen,无输入)
%all  = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<b32>
%tail = pto.vmi.pge "PAT_VL16" : !pto.vmi.mask<b32>   // 前 16 个 b32 lane active

// 数据相关 tail mask(从标量剩余计数生成)
%mt, %next = pto.vmi.plt %rem : i32 -> !pto.vmi.mask<b32>, i32

```

TODO:增加vinterleave和vdeinterleave
