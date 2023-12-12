module {
  func.func @fig_2_3(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6_i32 = arith.constant 6 : i32
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %1 = arith.index_cast %arg3 : index to i32
        %2 = arith.subi %c6_i32, %1 : i32
        %3 = arith.cmpi sle, %0, %2 : i32
        scf.if %3 {
          %4 = memref.load %arg0[%arg2] : memref<?xi32>
          %5 = memref.load %arg1[%arg3] : memref<?xi32>
          %6 = arith.addi %5, %4 : i32
          memref.store %6, %arg1[%arg3] : memref<?xi32>
        }
      }
    }
    return
  }
}
