# PTO Virtual Micro-Instruction (`pto.vmi`) Design

[toc]

---
## Part 0 —— Background

### 0.0 Positioning

`pto.vmi` (PTO virtual micro instruction) is an intermediate layer sitting between the **upper-level programming model** (e.g. TileLang `T.parallel`) and the **low-level `pto.mi` physical ISA**: it exposes only logical-contiguous semantics and per-element computation intent, while the physical SIMD register layout (interleave / half / width-placement / part / pack / dist) is held and propagated by `pto.as` and is invisible to the user. Full motivation is in [RFC-VPTO-Logical-Vector-ISA.md](RFC-VPTO-Logical-Vector-ISA.md).

```
TileLang  T.parallel(N) { C[i] = cast<i32>(A[i]) + B[i] }   ← user: logical data only
   │  (literal translation, per-element semantics preserved)
   ▼
pto.vmi      %w = pto.vmi.vcvt %a ; %c = pto.vmi.vadd %w, %b           ← logical vector, layout held by the compiler
   │  (pto-as: layout-assignment + lowering)
   ▼
pto.mi       vcvt EVEN/ODD + two-way vadd + vstsx2 INTLV_B32        ← physical: SIMD register interleave details
```

- **Upper → vmi**: the logical iteration space of `T.parallel` is translated one-to-one into `pto.vmi` logical-vector ops — per-element computation → Category A op, `T.cast` → a single `pto.vmi.vcvt` with no `part`, logical length `N` → `!pto.vmi.vreg<N×T>`, the "all-active" semantics → an auto-filled tail predicate.
- **vmi → pto.mi**: `pto.as` performs layout inference + coalescing + contiguization, lowering the logical vector into concrete `pto.mi` instructions (with `part/pack/interleave/dist`). When `K=1` it degenerates to a zero-overhead pass-through.

### 0.1 Notation

| Symbol | Meaning |
|---|---|
| `V<L×T>` | `!pto.vmi.vreg<L×T>` logical vector, `L` = number of logical elements |
| `M<L>` | `!pto.vmi.mask<L>` virtual predicate |
| `Ptr<T>` | `!pto.ptr<T,ub>` UB pointer |
| `s` | scalar |
| `[pmode]` | optional governing predicate (governing mask), `{pmode = "merge"\|"zero"}`, default `zero` |
| `[dist-mode]` | optional access shape (vload/vstore only), `{dist-mode = "continuous"\|"unpack"\|"dintlv"\|"brc"}`, default `continuous` |

> **Physical notation (appendix)**: the following symbols are physical quantities internal to `pto.as` and **do not** appear in surface signatures; they are used only in this document's physical views and lowering notes. Surface users only write `L×T`; `K`/`E_v`/`BlockLane` are held by `pto.as`.
>
> - `K`: number of physical backing registers for one logical vreg, `K = L·bitwidth(T) / 2048` (when `K_raw < 1`, `K = 1` is used; see §1.1).
> - `E_v`: number of lanes in one physical vreg (f32/i32 = 64, f16/bf16/i16 = 128, i8/fp8 = 256).
> - `BlockLane`: the hardware 32B atomic reduction unit; each physical vreg = 8 BlockLanes; each BlockLane holds `32B / bitwidth(T)` lanes.

### 0.2 Category A/B/C —— precise lowering contract

RFC §5 gives the three-category definition; here it is expanded into executable criteria for `pto.as`. The essential distinction among the three is **whether the op modifies the register layout, and what assumption it makes about the layout**:

| Category | Relationship to layout | pto.as behavior | Output layout |
|---|---|---|---|
| **A. Layout-passthrough** | **Does not modify** the register layout (pass-through) | Fan out the `pto.mi` op once per `K` physical reg; configure the governing predicate per physical reg by mask family (using `ppack/punpack` when needed) | Same as input (layout pass-through, including parity/half axes passed through) |
| **B. Layout-rewritable** | **Modifies the register layout by rule** | Fan out along **other** axes; instantiate the matching mode (`PART_EVEN/ODD`, `Bin_N0/N1`, `PK/UNPK`, `INTLV/DINTLV`…) on the matched axis | Consumes or produces that axis |
| **C. Contiguous-required** | **Strong assumption on physical layout** (requires a stride-1 contiguous view, with no matching mode to satisfy in place) | **Before** the op, auto-insert a `.contiguous()` materialization (`INTLV`/`pack`/move) to convert the register layout to continuous, then perform the contiguous op | Flat chunk (`is_contiguous`) |

> **C's strong assumption**: a Category-C op cannot execute in place on an arbitrary register layout — it assumes the input is already continuous. Hence `pto.as` explicitly inserts a layout materialization before the Category-C op, flattening upstream non-contiguous axes (parity/half/sub_part) into continuous before feeding the op. This is the only transition point from A/B (can carry layout) to C (strong contiguity assumption).

### 0.3 Predicate propagation rules

1. **Governing mask follows the data axis**: during lowering `[pmode]` fans out with the data to every physical reg; the mask family `G` must align with the data family (f32/i32→b32, f16/bf16/i16→b16, i8/fp8→b8). On cross-family ops (e.g. widen i16→i32) the mask family changes too; `pto.as` uses `ppack/punpack` or re-`pset` to produce the matching family.
2. **Inactive-lane behavior**: with `pmode="zero"` (default) inactive lanes write 0; with `pmode="merge"` the destination's original value is preserved. Inactive-lane behavior for reduce ops is in §2.3.
3. **A5 load is not predicate-capable**: the tail predicate of `vload` cannot be attached to the load; it must be migrated to the consuming op or the store. `vstore` is predicate-capable on A5.
4. **Tail mask materialization**: the `pset "PAT_ALL"` / `pge "PAT_VLn"` / `plt %rem` trio covers all-active, head-tail-active, and data-dependent-tail modes (see §10).

---

## Part 1 —— Virtual type formalization

### 1.1 `!pto.vmi.vreg<L×T>`

- **Legality**: for a full vector, `L · bitwidth(T)` MUST be an integer multiple of 2048 bit / 256 B; compact/partial vregs smaller than 256 B are allowed, and their physical backing is still allocated as one 256 B vreg.
- **Physical backing**: let `K_raw = L·bitwidth(T) / 2048`. A full vector uses `K = K_raw` `!pto.mi.vreg<E_v×T>` registers; when `K_raw < 1`, `K=1` is used, the low `L` logical slots are valid, and the remaining physical slots do not belong to the logical value. When `K=1` and not partial, vmi.vreg corresponds one-to-one with pto.mi.vreg.
- **`#layout`**: omitted in surface source, filled by `pto.as`. Layout is a **compiler-internal property** and does not enter the user type signature (the user writes only `L×T`).
- **Legal `L` and dtype**:

| T | bits | E_v | L must be a power-of-2 multiple of … |
|---|---|---|---|
| f32 / i32 | 32 | 64 | 64 |
| f16 / bf16 / i16 | 16 | 128 | 64 |
| i8 / fp8 | 8 | 256 | 64 |

### 1.1.1 Common vreg logical/physical views

This subsection gives three diagrams for each common vreg: **logical view**, **physical view (contiguous)**, and **physical view (non-contiguous)**. `fp16/fp32` correspond to the type names `f16/f32` in this document.
Each physical vreg is fixed at `256 B = 2048 bit = 8 × 32 B BlockLane`. The numbers in the diagrams are logical lane ids;
`pad/undef` means that physical slot does not belong to the logical value, and the consumer must ignore it via the logical length / predicate.
A non-contiguous layout does not change the logical order of `V<L×T>`; it only changes the mapping from logical lane to physical reg/lane:
`parity(EVEN/ODD)` is a stride-2 even/odd interleave; `sub_part(P0~P3)` is the byte slot within a 4 B group,
used only for fp8/i8 carriers, not a native lane axis of fp16/fp32.

| Logical type | Logical bytes | `K_raw` | Allocated physical vregs | Valid slots per physical vreg |
|---|---:|---:|---:|---|
| `V<256×fp8>` | 256 B | 1 | 1 | all 256 fp8 lanes valid |
| `V<256×fp16>` | 512 B | 2 | 2 | all 128 fp16 lanes valid each |
| `V<256×fp32>` | 1024 B | 4 | 4 | all 64 fp32 lanes valid each |
| `V<64×fp16>` | 128 B | 1/2 | 1 | low 64 fp16 lanes valid, high 64 invalid |
| `V<64×fp8>` | 64 B | 1/4 | 1 | low 64 fp8 lanes valid, high 192 invalid |

#### `V<256×fp8>`: 1 physical reg (K=1)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**Physical view (contiguous)** — 1 physical reg, each BlockLane = 32 B = 32 fp8 lanes

```text
   BL0          BL1                  BL7
┌─────────────┬─────────────┬───┬─────────────┐
│ x0 ... x31  │ x32 ... x63 │...│x224 ... x255 │
└─────────────┴─────────────┴───┴─────────────┘
                   P0 (256B)
```

#### `V<256×fp16>`: 2 physical regs (K=2)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**Physical view (contiguous)** — 2 physical regs, each BlockLane = 32 B = 16 fp16 lanes

```text
   BL0           BL1               BL7
┌──────────┬──────────┬───┬──────────┐
│ x0..x15  │ x16..x31 │...│x112..x127 │
└──────────┴──────────┴───┴──────────┘
                  P0 (256B)

   BL0           BL1               BL7
┌──────────┬──────────┬───┬──────────┐
│x128..x143│x144..x159│...│x240..x255 │
└──────────┴──────────┴───┴──────────┘
                  P1 (256B)
```

**Physical view (non-contiguous, parity EVEN/ODD)** — even lanes go to P0, odd lanes to P1 (e.g. a `DINTLV_B16` load or parity state kept after `vdintlv`; both regs have all 128 lanes valid)

```text
   P0 (EVEN)                                   P1 (ODD)
┌────┬────┬────┬─────┬──────┬──────┐  ┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x2 │ x4 │ ... │ x252 │ x254 │  │ x1 │ x3 │ x5 │ ... │ x253 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘  └────┴────┴────┴─────┴──────┴──────┘
   128 even lanes valid                   128 odd lanes valid
```

> Restore contiguous: `INTLV_B16(P0, P1) → [x0 x1 x2 x3 ... x255]`.

#### `V<256×fp32>`: 4 physical regs (K=4)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**Physical view (contiguous)** — 4 physical regs, each BlockLane = 32 B = 8 fp32 lanes

```text
   BL0       BL1               BL7
┌───────┬───────┬───┬───────┐
│ x0..7 │ x8..15│...│x56..63│
└───────┴───────┴───┴───────┘
            P0 (256B)

   BL0        BL1              BL7
┌────────┬────────┬───┬────────┐
│x64..71 │x72..79 │...│x120..127│
└────────┴────────┴───┴────────┘
            P1 (256B)

   BL0          BL1             BL7
┌──────────┬──────────┬───┬──────────┐
│x128..135 │x136..143 │...│x184..191 │
└──────────┴──────────┴───┴──────────┘
            P2 (256B)

   BL0          BL1             BL7
┌──────────┬──────────┬───┬──────────┐
│x192..199 │x200..207 │...│x248..255 │
└──────────┴──────────┴───┴──────────┘
            P3 (256B)
```

**Physical view (non-contiguous, parity EVEN/ODD)** — even lanes go to P0/P2, odd lanes to P1/P3 (typical origin: `V<256×fp16> → V<256×fp32>` widen retaining parity; all 4 regs have 64 valid lanes)

```text
 P0 (chunk0 EVEN)   P1 (chunk0 ODD)    P2 (chunk1 EVEN)   P3 (chunk1 ODD)
┌────┬────┬─────┐ ┌────┬────┬─────┐ ┌──────┬──────┬─────┐ ┌──────┬──────┬─────┐
│ x0 │ x2 │x126 │ │ x1 │ x3 │x127 │ │ x128 │ x130 │x254 │ │ x129 │ x131 │x255 │
└────┴────┴─────┘ └────┴────┴─────┘ └──────┴──────┴─────┘ └──────┴──────┴─────┘
    64 lane            64 lane            64 lane            64 lane
```

> Restore contiguous: `INTLV_B32(P0, P1) → [x0..x127]`, `INTLV_B32(P2, P3) → [x128..x255]`, then concatenate back in chunk order.

**Physical view (non-contiguous, P0/P1/P2/P3)** — 4-way stride-4 interleave: every 4 logical elements land in one reg each (`x0,x4,...` → P0; `x1,x5,...` → P1; `x2,x6,...` → P2; `x3,x7,...` → P3); all 4 regs have 64 valid lanes (corresponds to the sub_part / part_T 4-way axis)

```text
   P0                   P1                   P2                   P3
┌────┬────┬─────┐  ┌────┬────┬─────┐  ┌────┬────┬─────┐  ┌────┬────┬─────┐
│ x0 │ x4 │x252 │  │ x1 │ x5 │x253 │  │ x2 │ x6 │x254 │  │ x3 │ x7 │x255 │
└────┴────┴─────┘  └────┴────┴─────┘  └────┴────┴─────┘  └────┴────┴─────┘
     64 lane              64 lane              64 lane              64 lane
```

#### `V<64×fp16>`: 1 partial physical reg (K=1, low 64 lanes valid)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x62  │ x63  │
└────┴────┴────┴─────┴──────┴──────┘
                  64 lane
```

**Physical view (contiguous)** — 1 physical reg, low 64 lanes valid, each BlockLane = 16 fp16 lanes

```text
   BL0          BL1         BL2          BL3          BL4   BL5   BL6   BL7
┌──────────┬──────────┬──────────┬──────────┬──────┬──────┬──────┬──────┐
│ x0..x15  │ x16..x31 │ x32..x47 │ x48..x63 │      │      │      │      │
└──────────┴──────────┴──────────┴──────────┴──────┴──────┴──────┴──────┘
<------------- 128B logical payload -------------><---- 128B outside logical value ---->
                          P0 (256B)
```

**Physical view (non-contiguous, part EVEN/ODD)** — single `V<64×fp32> → V<64×fp16>` narrowing carrier: 64 valid fp16 placed on even/odd positions of 128 physical lanes

```text
   EVEN carrier (phys lane 0,2,...,126 valid)
┌────┬───┬────┬───┬─────┬─────┬───┬─────┬───┐
│ x0 │ _ │ x1 │ _ │ ... │ x62 │ _ │ x63 │ _ │
└────┴───┴────┴───┴─────┴─────┴───┴─────┴───┘
```

> Contrast: the fp8/i8 `sub_part(P0~P3)` is a byte slot within a 4 B group (`[P0 P1 P2 P3] [P0 P1 P2 P3] ...`), a different axis from fp16's part EVEN/ODD; see `V<64×fp8>`.

#### `V<64×fp8>`: 1 partial physical reg (K=1, low 64 lanes valid)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x62  │ x63  │
└────┴────┴────┴─────┴──────┴──────┘
                  64 lane
```

**Physical view (contiguous)** — 1 physical reg, low 64 lanes valid, each BlockLane = 32 fp8 lanes

```text
      BL0           BL1          BL2   BL3   BL4   BL5   BL6   BL7
┌─────────────┬─────────────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ x0 ... x31  │ x32 ... x63 │      │      │      │      │      │      │
└─────────────┴─────────────┴──────┴──────┴──────┴──────┴──────┴──────┘
<-- 64B logical payload --><--------------- 192B outside logical value --------------->
                          P0 (256B)
```

**Physical view (non-contiguous, sub_part P0)** — from a `V<64×fp32> → V<64×fp8>` `vcvt`: instead of placing the 64 B contiguously at the low end, the 0th byte of each 4 B group is filled with the valid fp8 (the `PK4_B32` extraction target)

```text
P0: 256B fp8 carrier, viewed as 64 groups × 4B, only the P0 slot valid per group
┌───────────┬───────────┬─────┬────────────┐
│ x0  _  _  _│ x1  _  _  _│ ... │ x63  _  _  _│
│ P0 P1 P2 P3│ P0 P1 P2 P3│     │ P0 P1 P2 P3 │
└───────────┴───────────┴─────┴────────────┘
   grp0          grp1              grp63
```

> This sparse view is a lowering layout and does not change the logical view of `V<64×fp8>`; `vstore` extracts from it when lowered to `PK4_B32`, writing out contiguous 64 B fp8.

### 1.2 `!pto.vmi.mask<L>`

- Virtual predicate, physically backed by `K` 256-bit `!pto.mi.mask<G>`.
- `G` is derived from the annotated data type (aligned with the data family, see §0.3-1) and does not appear in the user type signature.
- **Legality**: `L` matches the `L` of the annotated vreg; `K` matches.

### 1.3 compact reduce results

reduce/broadcast involves "compact scalar vectors smaller than 256 B" (e.g. `V<2×f32>`, `V<8×u16>`). The dimension of their first axis equals `group`.

---

## Part 2 —— `group` semantics formalization

### 2.1 Definition

For reduce ops (`vcadd`/`vcmax`/`vcmin`) and the broadcast op (`vbrc`), `C` in `{group=C}` denotes the **number of groups (group count)**, not the number of lanes per group:

- **reduce**: split the `L` lanes of `V<L×T>` into **`C` groups**, `L/C` lanes each, each producing 1 compact scalar; output `V<C×T>` (`C` scalars, the low `C` slots valid).
- **vbrc**: broadcast each of the `C` scalars of the compact input `V<C×T>` back to its own `(L/C)`-lane group, output `V<L×T>`.

`C` must divide `L`; the **lanes per group** is `L/C`, whose byte count `W = (L/C)·bitwidth(T)/8` determines the relationship to the BlockLane boundary and thereby the Category.

**Legal compact vreg types for reduce results**: the reduce output `V<C×T>` is a "compact scalar vector smaller than 256 B" (see §1.3), and `C` may only take **`1 / 2 / 4 / 8`** — i.e. the only legal reduce-result virtual register types are `V<1×T>`, `V<2×T>`, `V<4×T>`, `V<8×T>`. Hence the `C` of the op attribute `{group=C}` must **strictly match** the `C` of the return type `V<C×T>` (`group=8` ⇄ returns `V<8×T>`, `group=4` ⇄ `V<4×T>`, and so on); a mismatch is illegal IR and is rejected by `pto.as` at the type-checking stage.

### 2.2 `group` → Category decision table

Let `W = (L/C) · bitwidth(T) / 8` (the byte count of one group, `L/C` being the lanes per group), `BlockLane = 32 B`.

| `W` vs BlockLane | Category | pto.as lowering | Physical instruction |
|---|---|---|---|
| `W == 32` (group is exactly 1 BlockLane) | **B** | reduce each BlockLane independently, 1 op per physical reg, no cross-reg combine | `vcgadd`/`vcgmax`/`vcgmin` |
| `W = k·32, k>1` (group spans k BlockLanes, aligned to BlockLane boundary) | **B** (fold + vcg) | first `(k-1)` `vadd/vmax/vmin` to fold k BlockLanes into one BlockLane-wide intermediate, then `vcg*` | `vmax×(k-1)` + `vcg*` |


### 2.3 Inactive lanes and argmax of reduce

- **inactive lane**: `vcmax/vcmin` treat inactive as `-INF/+INF` (fp) or the literal type min/max (int); when all inactive, `result[0]` is that extreme value. `vcadd` treats inactive as 0. Consistent with SPEC §10.

  ```mlir
  %val = pto.vmi.vcmax %x, %m {group = C}
      : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```

### 2.4 Typical scenario: MX block-scale exponent reduce (bf16, group=8)

`V<256×bf16>` takes a shared exponent, one max per 32-element block → `256/32 = 8` groups, so `group=8` (`256/8 = 32` lanes per group). bf16 `bitwidth=16`, `W=32·16/8=64 B = 2 BlockLane` → falls on the second row of §2.2 (`k=2`, BlockLane-aligned, Category B).

- load: `pto.as` reverse-infers from the downstream `group=8` and selects `DINTLV_B16` (parity) so even/odd each become one vreg.
- `vand` extracts the exponent: Category-A pass-through, each parity vreg `vand`-ed.
- fold: `(k-1)=1` `vmax` folds the even/odd two vregs into one; the parity axis disappears, and each BlockLane's 16 lanes cover the max exponent of one 32-element block.
- `vcgmax`: take max per BlockLane → 8 shared exponents (matching `group=8`; the 8 BlockLanes each produce 1, one-to-one).
- output: compact `V<8×bf16>` (`C=8` shared exponents).

#### 32×32 tile → `V<256×bf16>` placement

A 32×32 bf16 tile has 1024 elements in total, requiring **4 `V<256×bf16>`** (each 256 lanes = 8 rows × 32 cols). It is sliced row-first into four 8-row × 32-col row-slabs, one slab per vreg: lane `n·32 + j` = tile row `row_off + n`, col `j` (`n ∈ 0..7`, `j ∈ 0..31`; `row_off ∈ {0,8,16,24}` for V0~V3).

```text
                   32×32 bf16 tile
   col 0                              col 31
┌──────────────────────────────────────────────┐
│ row0   c0  c1  c2  ...  c30  c31              │
│ row1   ...                                   │   ←─ V0 (row 0–7)   8×32=256
│ ...                                          │
│ row7   ...                                   │
├──────────────────────────────────────────────┤
│ row8   ...                                   │
│ ...                                          │   ←─ V1 (row 8–15)  8×32=256
│ row15  ...                                   │
├──────────────────────────────────────────────┤
│ row16  ...                                   │   ←─ V2 (row 16–23) 8×32=256
│ ...                                          │
│ row23  ...                                   │
├──────────────────────────────────────────────┤
│ row24  ...                                   │
│ ...                                          │   ←─ V3 (row 24–31) 8×32=256
│ row31  ...                                   │
└──────────────────────────────────────────────┘
```

> **Correspondence with the `group=8` reduce**: `group=8` slices 256 lanes into 8 groups of 32 lanes each. Under the 8×32 row-first placement, each 32 lanes is exactly **one complete row** (32 elements). So the 8 groups of one vreg are exactly the 8 rows it holds (row 0–7), and `vcgmax` takes one max per row → 8 shared exponents, one-to-one with the 8 rows — this is precisely the MX block-scale "one shared exponent per 32-element block" semantics (each row of the tile is one 32-element block).

---

## Part 3 —— Group 1: Load / Store

`vload`/`vstore` are logical memory accesses. **`[dist-mode]` explicitly declares the access shape**, default `continuous` (contiguous);
options are `unpack` (widening unpack), `dintlv` (de-interleave), `brc` (broadcast). The physical `dist` token
(`NORM_B*` / `UNPK_B*` / `DINTLV_B*` / `BRC_B*`) is invisible to the user; `pto.as` **derives the matching `B*` suffix
from the element type `T` of the UB pointer** (see table below), then instantiates the concrete `pto.mi` instruction.
The original `dist`-token layout inference (reverse-inference from a contiguous-view consumer) is still done internally by `pto.as` (see companion doc §3).

> Relationship between `[dist-mode]` and layout inference: `[dist-mode]` is the user's explicit declaration of the **access shape** (how to read/write UB),
> while layout-axis inference is `pto.as`'s implicit decision about the **register-side placement** (how data is laid out after entering the vreg). The two are orthogonal:
> even with `[dist-mode=continuous]`, `pto.as` may still lower the load to `DINTLV_B*` to serve a downstream reduce
> (producing a `parity` axis) — here the physical dist differs from the surface dist-mode, which is a legal `pto.as` optimization.

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vload` | A | `Ptr<T>`, `[dist-mode]`, `[pmode]` | `V<L×T>` | i8–i32, f16/bf16, f32 |
| `vstore` | A | `V<L×T>`, `Ptr<T>`, `M`, `[dist-mode]`, `[pmode]` | — | i8–i32, f16/bf16, f32 |

### `[dist-mode]` values and pointer type → hardware dist inference

`[dist-mode]` defaults to `continuous`; the `B*` suffix is determined by the element width of `Ptr<T>` (`T` being 8/16/32-bit → `B8`/`B16`/`B32`).

| `[dist-mode]` | Semantics | vload → physical dist | vstore → physical dist |
|---|---|---|---|
| `continuous` (default) | contiguous stride-1 access | `NORM` / `NORM_B*` | `NORM_B*` |
| `unpack` | widening unpack: narrow source expanded to wider lanes by `T` | `UNPK_B*` | — (no unpack for store) |
| `dintlv` | de-interleave/interleave: paired even/odd halves | `DINTLV_B*` (dual load, `vldsx2`) / `BDINTLV` | `INTLV_B*` (dual store, `vstsx2`) |
| `brc` | broadcast: scalar/block copy into vreg | `BRC_B*` / `BRC_BLK` | — |

> **Pointer type → `B*` suffix**: `!pto.ptr<f32,ub>` → `B32`; `!pto.ptr<bf16,ub>` → `B16`;
> `!pto.ptr<i8,ub>` → `B8`. A `continuous` load uses the element-width-agnostic `NORM`, and a store uses `NORM_B*`.
> The `B*` suffix of `dintlv`/`brc`/`unpack` is likewise derived from `Ptr<T>`. `dintlv` is physically a dual form (mapping to `vldsx2`/`vstsx2`),
> but the surface still expresses it as a single `vload`/`vstore` + `[dist-mode=dintlv]` with **single input/single output** — the load produces 1 logical vreg and the store consumes 1 logical vreg;
> the split into / composition from the EVEN/ODD two halves is done by `pto.as` at lowering time (the logical view is always contiguous).

Examples:

```mlir
// Contiguous load (default dist-mode): UB → vreg
%v = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=continuous → pto.mi.vlds {dist="NORM"}

// Contiguous store: vreg → UB (with governing predicate; store is predicate-capable on A5)
pto.vmi.vstore %v, %ub_out[%offset], %mask : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=continuous → pto.mi.vsts {dist="NORM_B32"}

// Broadcast load: scalar/block copy into vreg
%vb = pto.vmi.vload %ub[%offset] {dist-mode = "brc"} : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=brc → pto.mi.vlds {dist="BRC_B32"} (each lane = UB[base])

// Widening unpack load: narrow source expanded to wider lanes
%u = pto.vmi.vload %ub[%offset] {dist-mode = "unpack"} : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ pto.as:Ptr<bf16> → B16,dist-mode=unpack → pto.mi.vlds {dist="UNPK_B16"}

// De-interleave load (single surface output; pto.as splits into EVEN/ODD two physical regs at lowering)
%v = pto.vmi.vload %ub[%offset] {dist-mode = "dintlv"}
    : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=dintlv → pto.vldsx2 "DINTLV_B32"
//    one surface vload yields 1 logical vreg; pto.as splits its physical backing into
//    EVEN/ODD two half regs (parity axis) at lowering; the logical view stays 64 contiguous f32.

// Interleave store (single surface input; pto.as composes the EVEN/ODD two halves into a dual store at lowering)
pto.vmi.vstore %v, %ub_out[%offset], %mask {dist-mode = "dintlv"}
    : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>
//  ↑ pto.as:Ptr<f32> → B32,dist-mode=dintlv → pto.vstsx2 "INTLV_B32"
//    one surface vstore takes 1 logical vreg; its physical backing is the EVEN/ODD two halves,
//    and pto.as composes them into a dual store at lowering to write back contiguous memory.

// tail/partial load: A5 load is not predicate-capable, %mask migrated to consumer/store
%vt = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>   // tail predicate takes effect at downstream vadd/vstore
```

---

## Part 4 —— Group 2: index-gen

Copy and index materialization. Produces a `broadcast` axis or an index vector;
it is never expanded into `K` stored copies until a Category-B/C edge requires the expanded form.

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vci` | A | `s`  | `V<E×i32>` (`{ASC/DESC}`) | i8–i32, f16, f32 |

Examples:

```mlir
// vci: generate [base, base+1, ...] lane indices (ASC/DESC given by attribute, %base is the start scalar)
%idx = pto.vmi.vci %base {order = "ASC"} : i32 -> !pto.vmi.vreg<64xi32>
```

---

## Part 5 —— Group 3: Eltwise compute


| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vadd` `vsub` `vmul` `vdiv` `vmax` `vmin` | A | `V<T>`, `V<T>` (or bcast), `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 (`vdiv` f16/f32) |
| `vand` `vor` `vxor` | A | `V<T>`, `V<T>`, `[pmode]` | `V<T>` | i8–i32 |
| `vnot` | A | `V<T>`, `[pmode]` | `V<T>` | i8–i32 (bit-typed) |
| `vshl` `vshr` | A | `V<T>`, `V<T>` (vector count), `[pmode]` | `V<T>` | i8–i32 |
| `vadds` `vmuls` `vmaxs` `vmins` `vshls` `vshrs` | A | `V<T>`, `s`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vabs` `vneg` `vrelu` | A | `V<T>`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vexp` `vln` `vsqrt` | A | `V<T>`, `[pmode]` | `V<T>` | f16, f32 |
| `vcmp` | A | `V<T>`, `V<T>`, `M` | `M` | i8–i32, f16/bf16, f32 |
| `vcmps` | A | `V<T>`, `s`, `M` | `M` | i8–i32, f16/bf16, f32 |
| `vsel` | A | `M`, `V<T>`, `V<T>`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vselr` | A | `V<T>`, `V<index>` | `V<T>` (permute) | i8–i32, f16/bf16, f32 |

Examples:

```mlir
// Binary arithmetic (per-lane, with governing predicate)
%s = pto.vmi.vadd %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%m = pto.vmi.vmax %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// Vector-scalar (scalar implicitly broadcast)
%scaled = pto.vmi.vmuls %x, %scale, %mask : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%shifted = pto.vmi.vshrs %data, %c4, %mask : !pto.vmi.vreg<64xi32>, i16, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xi32>

// Unary arithmetic / activation
%a = pto.vmi.vabs %v, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%e = pto.vmi.vexp %v, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// Compare → predicate (the third M is the governing predicate, restricting which lanes participate; inactive-lane result is 0)
%lt = pto.vmi.vcmp %a, %b, %m, "lt" : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.mask<64>

// Scalar compare → predicate
%ges = pto.vmi.vcmps %a, %c0, %m, "ge" : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.mask<64>

// Predicate select: take %x where %mask is true, else %y
%out = pto.vmi.vsel %x, %y, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// Register gather/permute
%p = pto.vmi.vselr %x, %idx : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xf32>
```

---

## Part 6 —— Group 4: Broadcast

`vbrc` is logical scalar→vector / reduced→fan-out broadcast (R6). The ungrouped form is cheap; the grouped form (per-BlockLane
partial fan-out back to its own lanes) is the hard case.

| op (form) | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vbrc` (ungrouped) | A | `s` | `V<L×T>` | i8–i32, f16/bf16, f32 |
| `vbrc` (`{group=C}`) | B | `V<C×T>` | `V<L×T>` | i8–i32, f16/bf16, f32 |

Examples:

```mlir
// Ungrouped broadcast: scalar/reduced value fanned out to the whole vreg (in-register, no UB roundtrip)
%bc = pto.vmi.vbrc %maxe : f32 -> !pto.vmi.vreg<64xf32>

// Grouped broadcast: each of the C=8 scalars fanned back to its own (L/C)=8-lane group (no direct physical instruction; implementation decided by pto.as)
%scaleb = pto.vmi.vbrc %maxe {group = 8} : !pto.vmi.vreg<8xf32> -> !pto.vmi.vreg<64xf32>
```


---

## Part 7 —— Group 5: reduce

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vcadd` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i8–i32, f16, f32 |
| `vcmax` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i16–i32, f16, f32 |
| `vcmin` (`{group=C}`) | B | `V<L×T>`, `[pmode]` | `V<C×T>` | i16–i32, f16, f32 |

Examples:

```mlir
// Full-array sum reduce (to scalar)
%sum = pto.vmi.vcadd %x, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1xf32>

// Full-array max reduce (to scalar)
%mx = pto.vmi.vcmax %x, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1xf32>

// Group reduce: group=8 → 256 lanes split into 8 groups (32 lanes each), take one max each → 8 compact scalars
%maxe = pto.vmi.vcmax %exp {group = 8} : !pto.vmi.vreg<256xu16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8xu16>
```


---

## Part 8 —— Group 6: Convert (cvt)

One logical `vcvt`, whose *destination dtype is the layout*. `pto.as` expands it into a dtype-specific cast chain +
part/width staging + a matching store distribution, and drags the predicate along.

**Attributes**: `{to=<dtype>, rnd=<R>, sat=<SAT>}` (`to` can be inferred from the return type and is an explicit redundancy; `rnd`/`sat`
control rounding and saturation on narrowing). `part`/`PART_*`/`PK`/`UNPK` **do not appear in the surface**; they are filled by `pto.as` as
internal layout axes (`parity`/`width`/`sub_part`).

| op (form) | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vcvt` | B | `V<L×Tn>`, `[pmode]` | `V<L×Tm>` | (i8–i32, f8-f32)↔(i8–i32, f8-f32)|
| `vinterpret_cast` | A | `V<L×T>` | `V<L×T'>` | arbitrary bit re-interpretation, explicit, no layout inference|


> **`vinterpret_cast`** —— bit-level re-interpretation (`bitcast`). **Not** `vcvt`: it produces no parity/width
> axis, has no layout to infer, and has no dtype cast chain — it merely reads the same bits under a new dtype. Hence
> its Category is left empty and it carries no `[pmode]`; it is deliberately kept as an explicit op (the author guarantees semantic legality).

Examples:

```mlir
// Widen 16→32 (radix-2, parity EVEN/ODD expanded by pto.as)
%w = pto.vmi.vcvt %a, %mask : !pto.vmi.vreg<128xf16>, !pto.vmi.mask<128> -> !pto.vmi.vreg<128xf32>

// Narrow 32→16
%n = pto.vmi.vcvt %a, %mask : !pto.vmi.vreg<128xf32>, !pto.vmi.mask<128> -> !pto.vmi.vreg<128xf16>

// Quantize f32 → fp8
%q = pto.vmi.vcvt %s, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xfp8>

// bit re-interpretation (explicit, not via vcvt)
%r = pto.vmi.vinterpret_cast %a : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xi32>
```


---

## Part 9 —— Group 7: SFU

Special-function / domain-accelerator operations. Mixed categories: `chistv2` produces a `half` axis (B); sort and
gather/scatter are Category-C tile/permute ops; fused activation/arithmetic ops are Category-A
`vreg→vreg`.

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vhist` | B | `V<L×i*>` (bin idx), `[pmode]` | `V` (Bin_N0/N1 counts, half axis) | i8–i32 (bin index) |
| `vgather` | C | `Ptr<T>`, `%idx`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vgatherb` | C | `Ptr<T>`, `%idx`, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |
| `vscatter` | C | `V<T>`, `%idx`, `Ptr<T>`, `[pmode]` | — | i8–i32, f16/bf16, f32 |
| `vexpdif` | A | `V<f*>` (x), `V<f32>` (max) †, `[pmode]` | `V<f32>` | f16/f32 → f32 |
| `vaxpy` | A | `V<T>` (x), `V<T>` (y), `s` (α) †, `[pmode]` | `V<T>` | f16, f32 |
| `vlrelu` | A | `V<T>`, `[pmode]` | `V<T>` | f16, f32 |
| `vprelu` | A | `V<T>`, `s`/param, `[pmode]` | `V<T>` | f16, f32 |
| `vmull` | B | `V<i32>`, `V<i32>`, `[pmode]` | `V<i64>` (hi+lo, 2 reg; produces a `width` axis) | i32/u32 |
| `vmula` | A | `V<T>` (acc), `V<T>`, `V<T>` †, `[pmode]` | `V<T>` | i8–i32, f16/bf16, f32 |

Examples:

```mlir
// Histogram / per-bin count
%h = pto.vmi.vhist %bin_idx, %mask : !pto.vmi.vreg<256xi8>, !pto.vmi.mask<256> -> !pto.vmi.vreg<256xi16>

// Index gather (B32 / byte)
%g = pto.vmi.vgather %src, %offsets, %mask : !pto.ptr<f32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%gb = pto.vmi.vgatherb %src, %offsets, %mask : !pto.ptr<i32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<256> -> !pto.vmi.vreg<256xi32>

// Index scatter
pto.vmi.vscatter %v, %dest, %offsets, %mask : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64>

// Fused exp(x − max) (softmax)
%e = pto.vmi.vexpdif %x, %max, %mask, "EVEN" : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// Fused α·x + y
%y = pto.vmi.vaxpy %x, %acc, %alpha, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// leaky / parametric ReLU
%lr = pto.vmi.vlrelu %x, %slope, %mask : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
%pr = pto.vmi.vprelu %x, %alpha, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>

// Widening 32×32→64 multiply (produces a width axis, hi+lo two regs)
%res = pto.vmi.vmull %a, %b, %mask : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xi64>

// Fused multiply-add
%acc = pto.vmi.vmula %acc, %a, %b, %mask : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
```

---

## Part 10 —— Group 8: Predicate ops

The mask family (`b8/b16/b32`) of `pset`/`pge`/`plt` is derived from the data type annotated by the return type `M<L>`, and **does not** enter the op name
(i.e. `pset : !pto.vmi.mask<L>`, not `pset_b32`). The family must align with the data family of what it annotates.

| op | Mask in | In | Out | Datatypes |
|---|---|---|---|---|
| `pset` | gen | — (naming pattern `PAT_*`) | `M` | b8/b16/b32 |
| `pge` | gen | — (lane-count pattern `PAT_VLn`) | `M` (tail) | b8/b16/b32 |
| `plt` | gen | `s` (i32, e.g. `%rem`) | `M` (tail), `s`(next) | b8/b16/b32 |

Examples:

```mlir
// Materialize all-active / tail-pattern mask (gen, no input)
%all  = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<16>
%tail = pto.vmi.pge "PAT_VL16" : !pto.vmi.mask<16>   // first 16 lanes active

// Data-dependent tail mask (generated from a scalar remaining count)
%mt, %next = pto.vmi.plt %rem : i32 -> !pto.vmi.mask<16>, i32

```

---

## Part 11 —— Group 9: Data rearrange

In-register data movement and permutation, not accessing UB. `vintlv`/`vdintlv` are **Category-A** ops: per-lane, dtype
consistent, input and output share the same `L` and `T`, and they **do not change the vreg layout**.

Common use cases for continuously stored data within one vector register:
* real + imaginary
* value + index

| op | Cat | In | Out | Datatypes |
|---|---|---|---|---|
| `vintlv` | A | `V<L×T>`, `V<L×T>`, `[pmode]` | `V<L×T>`, `V<L×T>` | i8–i32, f16/bf16, f32 |
| `vdintlv` | A | `V<L×T>`, `V<L×T>`, `[pmode]` | `V<L×T>`, `V<L×T>` | i8–i32, f16/bf16, f32 |

Examples:

```mlir
// Interleave: two sources merged by even/odd into paired results (logical: low/high halves)
// low  = {lhs[0], rhs[0], lhs[1], rhs[1], ...}
// high = {lhs[L/2], rhs[L/2], lhs[L/2+1], rhs[L/2+1], ...}
%lo, %hi = pto.vmi.vintlv %a, %b, %mask
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>

// De-interleave: paired source split by even/odd (AoS → SoA)
// lo = {lhs[0], lhs[2], lhs[4], ...}   // even
// hi = {lhs[1], lhs[3], lhs[5], ...}   // odd
%even, %odd = pto.vmi.vdintlv %x, %y, %mask
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>
```

---

## Part 12 —— End-to-end example: Block MX Quant

The same Block MX Quant quantization logic, unfolded top-down across three layers of abstraction: first how TileLang organizes the whole kernel, then how the traditional `pto.mi` exposes hardware details at the vector layer, and finally how `pto.vmi` removes that burden through semantic-level ops.

### 12.1 TileLang layer: the complete kernel algorithm

TileLang organizes the whole block-mx quant kernel — block partitioning, amax reduction, scale generation — then hands the quantization execution path to the layer below:

```python
# ===========================================================================
# per_block_cast_kernel: the complete kernel for Block MX Quant
#   - Grid: (ceil_div(num_tokens, block_m), ceil_div(hidden, block_k))
#   - Threads: 256
#   - Each block handles block_m × block_k elements
#   - On the block_k dimension, 256 elements are processed at a time
# ===========================================================================
@T.prim_func
def per_block_cast_kernel(
    x: T.Tensor[(num_tokens, hidden), in_config.dtype],          # input: f16 matrix
    out: T.Tensor[(num_tokens, hidden), out_config.dtype],       # output: fp8 matrix
    out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],  # scale factor
):
    with T.Kernel(
        ceil_div(num_tokens, block_m),   # grid dim 0
        ceil_div(hidden, block_k),       # grid dim 1
        threads=num_threads              # 256 threads per block
    ) as (pid_x, pid_y):
    ...
        if:
            for i, j in T.Parallel(block_m, block_k):
                out[...] = x_fragment[i, j] * sf_fragment[i // num_per_tokens, j // num_per_channels]  # 3. quantize and write back
        else:
          ...
```

### 12.2 Traditional pto.mi (CCE) style: the algorithm is drowned by physical details

The same quantization main path — given a reciprocal scale, multiply a block of `f16` data by the scale, convert to `fp8`, and write back — when written directly with low-level MI instructions, every physical detail is exposed in the code.

> This handles one 256-element sub-block along the block_k direction in TileLang.

```mlir
// ===========================================================================
// Block MX Quant quantization execution path — traditional MI style
// Processes a num_per_tokens × 256 tile: load f16 → cvt f32 → mul scale → cvt fp8 → store
// Requires manual handling of all physical details: register splitting, part selection,
// mask granularity, vor merging, etc.
// ===========================================================================
module attributes {pto.backend = "pto", pto.target_arch = "a5"} {
  func.func @ComputeY1ToFP8_fp16_e4m3_MI(
      %arg0: i16, %arg1: i16,
      %arg2: !pto.ptr<f16, ub>,           // xAddr: input f16 data
      %arg3: !pto.ptr<f16, ub>,           // mxScale1ReciprocalAddr: reciprocal scale
      %arg4: !pto.ptr<f8E4M3FN, ub>,      // y1Addr: output fp8 data
      %arg5: i16, %arg6: i16) attributes {pto.kernel} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %vl_half = arith.index_cast %arg6 : i16 to index
    %load_stride = arith.muli %vl_half, %c2 : index

    pto.vecscope {
      // -- load scale: 128 f16, dist mode E2B_B16 --
      %scale_128 = pto.mi.vlds %arg3[%c0] {dist = "E2B_B16"}
        : !pto.ptr<f16, ub> -> !pto.mi.vreg<128xf16>

      %mask_b16 = pto.mi.pset_b16 "PAT_ALL" : !pto.mi.mask<b16>   // for f16 conversion
      %scale_fp32 = pto.mi.vcvt %scale_128, %mask_b16 {part = "EVEN"}
        : !pto.mi.vreg<128xf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
      %mask_b32 = pto.mi.pset_b32 "PAT_ALL" : !pto.mi.mask<b32>   // for f32 ops
      %mask_b8  = pto.mi.pset_b8  "PAT_ALL" : !pto.mi.mask<b8>    // for fp8 write-back

      %block_count = arith.index_cast %arg1 : i16 to index
      scf.for %i = %c0 to %block_count step %c1 {
        %offset = arith.muli %i, %load_stride : index

        // (1) load 256 f16 → DINTLV_B16 interleaves into low(128) + high(128)
        %low, %high = pto.mi.vldsx2 %arg2[%offset], "DINTLV_B16"
          : !pto.ptr<f16, ub>, index -> !pto.mi.vreg<128xf16>, !pto.mi.vreg<128xf16>

        // (2) f16 → f32: low and high each split EVEN/ODD, becoming 4 vreg<64xf32>
        %cvt_low_even  = pto.mi.vcvt %low,  %mask_b16 {part = "EVEN"} : ... -> !pto.mi.vreg<64xf32>
        %cvt_high_even = pto.mi.vcvt %high, %mask_b16 {part = "EVEN"} : ... -> !pto.mi.vreg<64xf32>
        %cvt_low_odd   = pto.mi.vcvt %low,  %mask_b16 {part = "ODD"}  : ... -> !pto.mi.vreg<64xf32>
        %cvt_high_odd  = pto.mi.vcvt %high, %mask_b16 {part = "ODD"}  : ... -> !pto.mi.vreg<64xf32>

        // (3) multiply reciprocal scale: 4 independent vmul, one per f32 lane above
        %mul0 = pto.mi.vmul %cvt_low_even,  %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul1 = pto.mi.vmul %cvt_high_even, %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul2 = pto.mi.vmul %cvt_low_odd,   %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul3 = pto.mi.vmul %cvt_high_odd,  %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>

        // (4) f32 → fp8: 4 vcvt, each packed into the P0/P1/P2/P3 part
        %p0 = pto.mi.vcvt %mul0, %mask_b32 {part = "P0", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p1 = pto.mi.vcvt %mul1, %mask_b32 {part = "P1", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p2 = pto.mi.vcvt %mul2, %mask_b32 {part = "P2", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p3 = pto.mi.vcvt %mul3, %mask_b32 {part = "P3", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>

        // (5) merge: 3 vor splice P0~P3 back into one complete vector
        %merge01 = pto.mi.vor %p0, %p1, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %merge012 = pto.mi.vor %merge01, %p2, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %merged = pto.mi.vor %merge012, %p3, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>

        // (6) write back
        pto.mi.vsts %merged, %arg4[%offset], %mask_b8
          : !pto.mi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, !pto.mi.mask<b8>
      }
    }
    return
  }
}
```

**Pain points of the traditional pto.mi (CCE) style:**

This layer is no longer describing an algorithm. Every "why" is answered not by "the algorithm needs it" but by "the hardware looks like this":

- A logical vector is split into `low` / `high` because the physical register is only 128 wide
- `f16 -> f32` distinguishes `EVEN` / `ODD` because the bitwidth halves during type conversion
- `f32 -> fp8` is distributed across `P0`~`P3`, and each instruction must explicitly carry hardware parameters like `rnd` (rounding mode) and `sat` (saturation overflow); getting them wrong directly causes numerical errors, because fp8 packing must match the hardware part mechanism
- Multiple parts are spliced back with `vor`, because the earlier steps had to split in the first place
- From the moment of loading, data is distributed across multiple registers in an interleaved manner (`DINTLV_B16` → low/high → EVEN/ODD); the index stride of the original elements becomes 4, and the developer must keep this mapping in mind at all times to track each element
- Each operation requires choosing the right mask granularity (`b16` / `b32` / `b8`), because different data types have different bitwidths; a wrong choice is a bug

**The loop body is 17 instructions, of which 12 have nothing to do with the quantization algorithm — they purely describe how the hardware splits and splices data.**

### 12.3 pto.vmi style: write only semantics, not hardware

The same quantization main path — processing a `num_per_tokens × 256` tile — when written with pto.vmi, all physical details are handled automatically by the compiler:

```mlir
// ===========================================================================
// Block MX Quant quantization execution path — pto.vmi surface syntax
// Same num_per_tokens × 256 tile, only 5 semantic actions needed
// ===========================================================================
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {

  func.func @ComputeY1ToFP8_fp16_e4m3_VMI(
      %dataLen: i16,                       // total length of input data
      %block_count: i16,                    // main-loop iteration count = block_k / 256
      %xAddr: !pto.ptr<f16, ub>,           // input f16 data address
      %mxScale1ReciprocalAddr: !pto.ptr<f16, ub>,  // reciprocal scale address
      %y1Addr: !pto.ptr<f8E4M3FN, ub>,     // output fp8 data address
      %ubBlockSize: i16,                   // UB block size
      %vlForHalfNumber: i16)               // vector length of the half number
      attributes {pto.kernel} {

    // -- constants --
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %vl_half = arith.index_cast %vlForHalfNumber : i16 to index
    %load_stride_y8 = arith.muli %vl_half, %c2 : index

    pto.vecscope {
      // =====================================================================
      // Stage 1: load reciprocal scale, broadcast to 256-wide, convert to f32
      // =====================================================================
      %scale_f16 = pto.vmi.vload %mxScale1ReciprocalAddr[%c0]
        : !pto.ptr<f16, ub> -> !pto.vmi.vreg<8xf16>                // load 8 f16 scales
      %scale_f16_vec = pto.vmi.vbrc %scale_f16 {group = 8}
        : !pto.vmi.vreg<8xf16> -> !pto.vmi.vreg<256xf16>          // broadcast to 256-wide
      %scale_fp32 = pto.vmi.vcvt %scale_f16_vec
        : !pto.vmi.vreg<256xf16> -> !pto.vmi.vreg<256xf32>        // f16 -> f32

      // =====================================================================
      // Stage 2: per-tile quantization — load f16 → cvt f32 → mul scale → cvt fp8 → store
      // 256 f16 elements per iteration
      // =====================================================================
      %block_count_idx = arith.index_cast %block_count : i16 to index
      scf.for %i = %c0 to %block_count_idx step %c1 {
        %x_off = arith.muli %i, %load_stride_y8 : index
        %y_off = arith.muli %i, %load_stride_y8 : index

        // (1) load one tile of f16 data (256 elements)
        %x_f16 = pto.vmi.vload %xAddr[%x_off]
          : !pto.ptr<f16, ub> -> !pto.vmi.vreg<256xf16>

        // (2) f16 -> f32
        %x_fp32 = pto.vmi.vcvt %x_f16
          : !pto.vmi.vreg<256xf16> -> !pto.vmi.vreg<256xf32>

        // (3) multiply reciprocal scale
        %res_fp32 = pto.vmi.vmul %x_fp32, %scale_fp32
          : !pto.vmi.vreg<256xf32>, !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf32>

        // (4) f32 -> fp8 (e4m3)
        %res_fp8 = pto.vmi.vcvt %res_fp32
          : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf8E4M3FN>

        // (5) write back fp8 result
        pto.vmi.vstore %res_fp8, %y1Addr[%y_off]
          : !pto.vmi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, !pto.vmi.mask<256>
      }
    }
    return
  }
}
```
