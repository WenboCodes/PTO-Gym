
    func.func @ComputeY1ToFP8_fp16_e4m3_vmi(%arg0: i16, %arg1: i16, %arg2: !pto.ptr<f16, ub>, %arg3: !pto.ptr<f16, ub>, %arg4: !pto.ptr<f8E4M3FN, ub>, %arg5: i16, %arg6: i16) attributes {pto.kernel} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %0 = arith.index_cast %arg1 : i16 to index
      %1 = arith.index_cast %arg6 : i16 to index
      %2 = arith.muli %1, %c2 : index
      pto.vecscope {
        %result = pto.mi.vlds %arg3[%c0] {dist = "E2B_B16"} : !pto.ptr<f16, ub> -> !pto.mi.vreg<128xf16>
        %3 = pto.mi.pset_b16 "PAT_ALL" : !pto.mi.mask<b16>
        %4 = pto.mi.vcvt %result, %3 {part = "EVEN"} : !pto.mi.vreg<128xf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
        %5 = pto.mi.pset_b32 "PAT_ALL" : !pto.mi.mask<b32>
        %6 = pto.mi.pset_b8 "PAT_ALL" : !pto.mi.mask<b8>
        scf.for %arg7 = %c0 to %0 step %c1 {
          %7 = arith.muli %arg7, %2 : index
          %low, %high = pto.mi.vldsx2 %arg2[%7], "DINTLV_B16" : !pto.ptr<f16, ub>, index -> !pto.mi.vreg<128xf16>, !pto.mi.vreg<128xf16>
          %8 = pto.mi.vcvt %low, %3 {part = "EVEN"} : !pto.mi.vreg<128xf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
          %9 = pto.mi.vcvt %high, %3 {part = "EVEN"} : !pto.mi.vreg<128xf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
          %10 = pto.mi.vcvt %low, %3 {part = "ODD"} : !pto.mi.vreg<128xf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
          %11 = pto.mi.vcvt %high, %3 {part = "ODD"} : !pto.mi.vreg<128xf16>, !pto.mi.mask<b16> -> !pto.mi.vreg<64xf32>
          %12 = pto.mi.vmul %8, %4, %5 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
          %13 = pto.mi.vmul %9, %4, %5 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
          %14 = pto.mi.vmul %10, %4, %5 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
          %15 = pto.mi.vmul %11, %4, %5 : !pto.mi.vreg<64xf32>, !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<64xf32>
          %16 = pto.mi.vcvt %12, %5 {part = "P0", rnd = "R", sat = "SAT"} : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xf8E4M3FN>
          %17 = pto.mi.vcvt %13, %5 {part = "P1", rnd = "R", sat = "SAT"} : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xf8E4M3FN>
          %18 = pto.mi.vcvt %14, %5 {part = "P2", rnd = "R", sat = "SAT"} : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xf8E4M3FN>
          %19 = pto.mi.vcvt %15, %5 {part = "P3", rnd = "R", sat = "SAT"} : !pto.mi.vreg<64xf32>, !pto.mi.mask<b32> -> !pto.mi.vreg<256xf8E4M3FN>
          %20 = pto.mi.vor %16, %17, %6 : !pto.mi.vreg<256xf8E4M3FN>, !pto.mi.vreg<256xf8E4M3FN>, !pto.mi.mask<b8> -> !pto.mi.vreg<256xf8E4M3FN>
          %21 = pto.mi.vor %20, %18, %6 : !pto.mi.vreg<256xf8E4M3FN>, !pto.mi.vreg<256xf8E4M3FN>, !pto.mi.mask<b8> -> !pto.mi.vreg<256xf8E4M3FN>
          %22 = pto.mi.vor %21, %19, %6 : !pto.mi.vreg<256xf8E4M3FN>, !pto.mi.vreg<256xf8E4M3FN>, !pto.mi.mask<b8> -> !pto.mi.vreg<256xf8E4M3FN>
          pto.mi.vsts %22, %arg4[%7], %6 : !pto.mi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, !pto.mi.mask<b8>
        }
      }
      return
    }

