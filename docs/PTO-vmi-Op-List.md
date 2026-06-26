# PTO 虚拟微指令(`pto.vmi`)——指令清单(含操作数与数据类型)

[toc]

---

## Part 0 —— 阅读说明

每条指令列出 **操作数签名**(输入 → 输出)与 **数据类型**,按 SPEC 的分组组织。
`†` 标记的签名在 SPEC 中未给出显式 MLIR 示例,由功能描述/成本行推断
(多数有 SPEC Part 15 或 requirements Appendix A 的真实示例支撑)。

### 0.1 操作数记号

| 记号 | 含义 |
|---|---|
| `V<T>` | `!pto.vmi.vreg<L×T>` 逻辑向量 |
| `M` | `!pto.vmi.mask<L×b*>` 谓词(`b8/b16/b32` = 1 bit 控制 1/2/4 byte lane) |
| `Ptr<T>` | `!pto.ptr<T,ub>` UB 指针 |
| `s` | 标量 |
| `[pmode]` | 可选治理谓词(支持 `{pmode = "merge"}`,默认 `zero`) |

### 0.2 Category 图例(来自 requirements §4.1)

- **A** —— Element-wise操作，layout透传。
- **B** —— Layout修改。
- **C** —— Layout未知，可能无法准确推导（默认行为是pto-as推导成continous）。

---

## Part 2 —— Group 1:Load / Store

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vload` | A | `Ptr<T>` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vstore` | A | `V<T>`, `Ptr<T>`, `M` | — | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 连续 load:UB → vreg
%v = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>

// 连续 store:vreg → UB(带治理谓词,A5 上 store 可谓词化)
pto.vmi.vstore %v, %ub_out[%offset], %mask : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<b32>

// TODO:广播 load:标量/块复制进 vreg
%vb = pto.vmi.vload %ub[%offset], <dist> : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>

// tail/partial load:A5 上 load 不可谓词化,%mask 迁移到消费侧/store
%vt = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>   // 尾谓词在下游 vadd/vstore 生效
```

---

## Part 3 —— Group 2:index-gen

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

## Part 4 —— Group 3:Eltwise compute


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

## Part 5 —— Group 4:Broadcast

`vbrc` 是逻辑标量→向量/已归约→扇出广播(R6)。未分组形式便宜;分组形式(每 VLane
partial 扇回自身 lane)是难情形。

| op (form) | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vbrc` (`{group=C}`) | B | `V<T>` | `V<T>` | i8–i32, f16/bf16, f32 |

示例:

```mlir
// 未分组广播:标量/已归约值扇出到整条 vreg(寄存器内,无 UB roundtrip)
%bc = pto.vmi.vbrc %maxe : f32 -> !pto.vmi.vreg<64xf32>

// 分组广播:每 VLane partial 扇回自身 C 条 lane(无直接物理指令对应,实现由 pto.as决定)
%scaleb = pto.vmi.vbrc %maxe {group = 8} : !pto.vmi.vreg<8xf32> -> !pto.vmi.vreg<64xf32>
```


---

## Part 6 —— Group 5: reduce

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

// sub-group 归约:每 C=8 lane 取一个 max → 8 个 partial
%maxe = pto.vmi.vcmax %exp {group = 8} : !pto.vmi.vreg<256xu16>, !pto.vmi.mask<b16> -> !pto.vmi.vreg<8xu16>
```


---

## Part 7 —— Group 6:Convert (cvt)

一个逻辑 `vcvt`,其 *目标 dtype 即布局*。`pto.as` 展开为 dtype-specific cast 链 +
part/width staging + 匹配的 store distribution,并拖出谓词伴随。

| op (form) | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vcvt` | B | `V<L×Tn>`, `[pmode]` | `V<L×Tm>` | (i8–i32, f8-f32)↔(i8–i32, f8-f32)|
| `vbitcast` | A | `V<L×T>` | `V<L×T'>` | 任意bit 重解释,显式,无布局推断|


> **`vbitcast`** —— bit 级重解释(`bitcast`)。**不**是 `vcvt`:不产生 parity/width
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
%r = pto.vmi.vbitcast %a : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xi32>
```


---

## Part 8 —— Group 7:SFU

特殊功能/领域加速器操作。混合类别:`chistv2` 产生 `half` 轴(B);sort 与
gather/scatter 是 Category-C tile/permute 操作;融合激活/算术操作是 Category-A
`vreg→vreg`。

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vhist` | B | `V<L×i*>` (bin idx), `[pmode]` | `V` (Bin_N0/N1 counts, half 轴) † | i8–i32 (bin index) |
| `vgather` | C | `Ptr<T>`, `%idx`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vgatherb` | C | `Ptr<T>`, `%idx`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vscatter` | C | `V<T>`, `%idx`, `Ptr<T>`, `[pmode]` | — | i8–i32, f16/bf16, f32 |
| `vexpdif` | A | `V<f*>` (x), `V<f32>` (max) †, `[pmode]` | `V<f32>` | f16/f32 → f32 |
| `vaxpy` | A | `V<T>` (x), `V<T>` (y), `s` (α) †, `[pmode]` | `V<T>` | f16, f32 |
| `vlrelu` | A | `V<T>`, `[pmode]` | `V<T>` | f16, f32 |
| `vprelu` | A | `V<T>`, `s`/param, `[pmode]` | `V<T>` | f16, f32 |
| `vmull` | A | `V<i32>`, `V<i32>`, `[pmode]` | `V<i64>` (hi+lo, 2 reg) | i32/u32 |
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

// 宽化 32×32→64 乘
%res = pto.vmi.vmulh %a, %b, %mask : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<32xi64>

// 融合乘加
%acc = pto.vmi.vmula %acc, %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<b32> -> !pto.vmi.vreg<64xf32>
```

---

## Part 9 —— Group 8:Predicate ops



| op | Mask in | In | Out | Datatypes |
|---|---|---|---|---|
| `pset` | gen | — (命名模式 `PAT_*`) | `M` | b8/b16/b32 |
| `pge` | gen | — (lane-count 模式 `PAT_VLn`) | `M` (tail) | b8/b16/b32 |
| `plt` | gen | `s` (i32, 如 `%rem`) | `M` (tail) | b8/b16/b32 |


示例:

```mlir
// 物化全 active / tail 模式 mask(gen,无输入)
%all  = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<b32>
%tail = pto.vmi.pge "PAT_VL16" : !pto.vmi.mask<b32>   // 前 16 个 b32 lane active

// 数据相关 tail mask(从标量剩余计数生成)
%mt, %next = pto.vmi.plt %rem : i32 -> !pto.vmi.mask<b32>, i32

```
