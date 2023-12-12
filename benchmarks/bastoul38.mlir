module {
  func.func @fig_3_8(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      memref.store %0, %arg0[%arg2] : memref<?xi32>
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %1 = memref.load %arg1[%arg3] : memref<?xi32>
        %2 = memref.load %arg0[%arg2] : memref<?xi32>
        %3 = arith.addi %1, %2 : i32
        %4 = arith.divsi %3, %c2_i32 : i32
        memref.store %4, %arg1[%arg3] : memref<?xi32>
      }
    }
    return
  }
}
