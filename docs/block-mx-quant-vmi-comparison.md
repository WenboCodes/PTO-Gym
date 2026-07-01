# Block MX Quant 场景下 pto.vmi 虚拟指令示例

## 背景：向量量化 kernel 的开发痛点

以 Block MX Quant（per-block FP8 量化）为例，一个典型的量化 kernel 包含以下步骤：

1. 把一块 `f16` 数据加载到向量寄存器
2. 算出每个 block 的 amax，进而得到 scale / reciprocal scale
3. 把数据转成 `f32`，乘上 reciprocal scale
4. 再转成 `fp8`，写回显存

如果直接使用底层向量指令（CCE/pto.mi）编写，开发者需要手工处理大量与算法无关的硬件细节：

- 一个逻辑向量在物理寄存器上被拆成几份（`low` / `high`）
- 类型转换时如何区分 `EVEN` / `ODD` part
- `f32 -> fp8` 时如何分到 `P0` / `P1` / `P2` / `P3`
- 多个 part 最后如何用 `vor` 拼回去
- 不同数据类型对应不同 mask 粒度（`b16` / `b32` / `b8`）

这些细节跟算法没有任何关系——它们纯粹是simd向量寄存器的宽度约束。**VMI 的设计目标正是解决这个问题：让开发者只表达"做什么"，不用管"寄存器怎么拆"。**

下面用同一份 Block MX Quant 量化逻辑，从上到下展开三层抽象：先看 TileLang 如何组织整个 kernel，然后看传统 MI 在向量层暴露了多少硬件细节，最后看 VMI 如何通过语义级算子消除这些负担。

---

## 1. TileLang 层：完整的 kernel 算法

TileLang 负责组织整个 block-mx quant kernel——block 划分、amax 归约、scale 生成，然后把量化执行路径交给下层：

```python
# ===========================================================================
# per_block_cast_kernel：Block MX Quant 的完整 kernel
#   - Grid: (ceil_div(num_tokens, block_m), ceil_div(hidden, block_k))
#   - Threads: 256
#   - 每个 block 处理 block_m × block_k 个元素
#   - block_k 方向上每次处理 256 个元素
# ===========================================================================
@T.prim_func
def per_block_cast_kernel(
    x: T.Tensor[(num_tokens, hidden), in_config.dtype],          # 输入：f16 矩阵
    out: T.Tensor[(num_tokens, hidden), out_config.dtype],       # 输出：fp8 矩阵
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
                out[...] = x_fragment[i, j] * sf_fragment[i // num_per_tokens, j // num_per_channels]  # 3. 量化写回
        else:
          ...
```

---

## 2. 传统 pto.mi(CCE) 写法：算法被物理细节淹没

同样的量化主路径——已有 reciprocal scale，把一块 `f16` 数据乘 scale 后转 `fp8` 写回——用底层 MI 指令直接编写时，所有物理细节都会暴露在代码中。

> 这里处理的是 TileLang 中 block_k 方向上的一次 256 元素的子块。

```mlir
// ===========================================================================
// Block MX Quant 量化执行路径 — 传统 MI 写法
// 处理 num_per_tokens × 256 的分块：load f16 → cvt f32 → mul scale → cvt fp8 → store
// 需要手工处理寄存器拆分、part 选择、mask 粒度、vor 合并等所有物理细节
// ===========================================================================
module attributes {pto.backend = "pto", pto.target_arch = "a5"} {
  func.func @ComputeY1ToFP8_fp16_e4m3_MI(
      %arg0: i16, %arg1: i16,
      %arg2: !pto.ptr<f16, ub>,           // xAddr：输入 f16 数据
      %arg3: !pto.ptr<f16, ub>,           // mxScale1ReciprocalAddr：reciprocal scale
      %arg4: !pto.ptr<f8E4M3FN, ub>,      // y1Addr：输出 fp8 数据
      %arg5: i16, %arg6: i16) attributes {pto.kernel} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %vl_half = arith.index_cast %arg6 : i16 to index
    %load_stride = arith.muli %vl_half, %c2 : index

    pto.vecscope {
      // -- 加载 scale：128 个 f16，dist 模式为 E2B_B16 --
      %scale_128 = pto.mi.vlds %arg3[%c0] {dist = "E2B_B16"}
        : !pto.ptr<f16, ub> -> !pto.mi.vreg<128xf16>

      %mask_b16 = pto.mi.pset_b16 "PAT_ALL" : !pto.mi.mask<b16>   // f16 转换用
      %scale_fp32 = pto.mi.vcvt %scale_128, %mask_b16 {part = "EVEN"}
        : !pto.mi.vreg<128xf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
      %mask_b32 = pto.mi.pset_b32 "PAT_ALL" : !pto.mi.mask<b32>   // f32 运算用
      %mask_b8  = pto.mi.pset_b8  "PAT_ALL" : !pto.mi.mask<b8>    // fp8 写回用

      %block_count = arith.index_cast %arg1 : i16 to index
      scf.for %i = %c0 to %block_count step %c1 {
        %offset = arith.muli %i, %load_stride : index

        // (1) 加载 256 个 f16 → DINTLV_B16 交错拆成 low(128) + high(128)
        %low, %high = pto.mi.vldsx2 %arg2[%offset], "DINTLV_B16"
          : !pto.ptr<f16, ub>, index -> !pto.mi.vreg<128xf16>, !pto.mi.vreg<128xf16>

        // (2) f16 → f32：low 和 high 各自再拆 EVEN/ODD，变成 4 个 vreg<64xf32>
        %cvt_low_even  = pto.mi.vcvt %low,  %mask_b16 {part = "EVEN"} : ... -> !pto.mi.vreg<64xf32>
        %cvt_high_even = pto.mi.vcvt %high, %mask_b16 {part = "EVEN"} : ... -> !pto.mi.vreg<64xf32>
        %cvt_low_odd   = pto.mi.vcvt %low,  %mask_b16 {part = "ODD"}  : ... -> !pto.mi.vreg<64xf32>
        %cvt_high_odd  = pto.mi.vcvt %high, %mask_b16 {part = "ODD"}  : ... -> !pto.mi.vreg<64xf32>

        // (3) 乘 reciprocal scale：4 条独立的 vmul，一一对应上面 4 路 f32
        %mul0 = pto.mi.vmul %cvt_low_even,  %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul1 = pto.mi.vmul %cvt_high_even, %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul2 = pto.mi.vmul %cvt_low_odd,   %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>
        %mul3 = pto.mi.vmul %cvt_high_odd,  %scale_fp32, %mask_b32 : ... -> !pto.mi.vreg<64xf32>

        // (4) f32 → fp8：4 条 vcvt，分别打入 P0/P1/P2/P3 part
        %p0 = pto.mi.vcvt %mul0, %mask_b32 {part = "P0", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p1 = pto.mi.vcvt %mul1, %mask_b32 {part = "P1", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p2 = pto.mi.vcvt %mul2, %mask_b32 {part = "P2", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %p3 = pto.mi.vcvt %mul3, %mask_b32 {part = "P3", rnd = "R", sat = "SAT"} : ... -> !pto.mi.vreg<256xf8E4M3FN>

        // (5) 合并：3 条 vor 把 P0~P3 拼回一个完整向量
        %merge01 = pto.mi.vor %p0, %p1, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %merge012 = pto.mi.vor %merge01, %p2, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>
        %merged = pto.mi.vor %merge012, %p3, %mask_b8 : ... -> !pto.mi.vreg<256xf8E4M3FN>

        // (6) 写回
        pto.mi.vsts %merged, %arg4[%offset], %mask_b8
          : !pto.mi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, !pto.mi.mask<b8>
      }
    }
    return
  }
}
```

**传统 pto.mi(CCE) 写法的痛点：**

这一层已经不是在描述算法了。每一个"为什么"的答案都不是"算法需要"，而是"硬件长这样"：

- 一个逻辑向量被拆成 `low` / `high`，原因是物理寄存器只有 128 宽
- `f16 -> f32` 时区分 `EVEN` / `ODD`，原因是类型转换时位宽减半
- `f32 -> fp8` 时分到 `P0`~`P3`，且每条指令需显式携带 `rnd`（舍入模式）、`sat`（饱和溢出）等硬件参数，用错会直接导致数值错误，原因是 fp8 打包需要匹配硬件 part 机制
- 多个 part 用 `vor` 拼回去，因为前面各步骤不得不拆
- 数据从加载开始就以交错（interleave）方式分布在多个寄存器中（`DINTLV_B16` → low/high → EVEN/ODD），原始元素的索引步长变成 4，开发者需要始终在脑中维持这套映射关系才能追踪每个元素
- 每种操作需要选对 mask 粒度（`b16` / `b32` / `b8`），因为不同数据类型的位宽不同，选错即 bug

**loop body 17 条指令，其中 12 条跟量化算法毫无关系，纯粹在描述硬件如何拆分和拼接数据。**

---

## 3. VMI 写法：只写语义，不写硬件

同样的量化主路径——处理 `num_per_tokens × 256` 的分块——用 VMI 编写时，所有物理细节由编译器自动处理：

```mlir
// ===========================================================================
// Block MX Quant 量化执行路径 — VMI surface 语法
// 同样处理 num_per_tokens × 256 的分块，只需 5 个语义动作
// ===========================================================================
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {

  func.func @ComputeY1ToFP8_fp16_e4m3_VMI(
      %dataLen: i16,                       // 输入数据总长度
      %block_count: i16,                    // 主循环迭代次数 = block_k / 256
      %xAddr: !pto.ptr<f16, ub>,           // 输入 f16 数据地址
      %mxScale1ReciprocalAddr: !pto.ptr<f16, ub>,  // reciprocal scale 地址
      %y1Addr: !pto.ptr<f8E4M3FN, ub>,     // 输出 fp8 数据地址
      %ubBlockSize: i16,                   // UB block 大小
      %vlForHalfNumber: i16)               // half number 的向量长度
      attributes {pto.kernel} {

    // -- 常量 --
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %vl_half = arith.index_cast %vlForHalfNumber : i16 to index
    %load_stride_y8 = arith.muli %vl_half, %c2 : index

    pto.vecscope {
      // =====================================================================
      // 阶段 1：加载 reciprocal scale，广播到 256 宽，转成 f32
      // =====================================================================
      %scale_f16 = pto.vmi.vload %mxScale1ReciprocalAddr[%c0]
        : !pto.ptr<f16, ub> -> !pto.vmi.vreg<8xf16>                // 加载 8 个 f16 scale
      %scale_f16_vec = pto.vmi.vbrc %scale_f16 {group = 8}
        : !pto.vmi.vreg<8xf16> -> !pto.vmi.vreg<256xf16>          // 广播到 256 宽
      %scale_fp32 = pto.vmi.vcvt %scale_f16_vec
        : !pto.vmi.vreg<256xf16> -> !pto.vmi.vreg<256xf32>        // f16 -> f32

      // =====================================================================
      // 阶段 2：逐块量化 — load f16 → cvt f32 → mul scale → cvt fp8 → store
      // 每次处理 256 个 f16 元素
      // =====================================================================
      %block_count_idx = arith.index_cast %block_count : i16 to index
      scf.for %i = %c0 to %block_count_idx step %c1 {
        %x_off = arith.muli %i, %load_stride_y8 : index
        %y_off = arith.muli %i, %load_stride_y8 : index

        // (1) 加载一块 f16 数据（256 个元素）
        %x_f16 = pto.vmi.vload %xAddr[%x_off]
          : !pto.ptr<f16, ub> -> !pto.vmi.vreg<256xf16>

        // (2) f16 -> f32
        %x_fp32 = pto.vmi.vcvt %x_f16
          : !pto.vmi.vreg<256xf16> -> !pto.vmi.vreg<256xf32>

        // (3) 乘 reciprocal scale
        %res_fp32 = pto.vmi.vmul %x_fp32, %scale_fp32
          : !pto.vmi.vreg<256xf32>, !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf32>

        // (4) f32 -> fp8 (e4m3)
        %res_fp8 = pto.vmi.vcvt %res_fp32
          : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf8E4M3FN>

        // (5) 写回 fp8 结果
        pto.vmi.vstore %res_fp8, %y1Addr[%y_off]
          : !pto.vmi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, !pto.vmi.mask<256>
      }
    }
    return
  }
}
```

