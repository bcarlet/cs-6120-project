module {
  func.func @pascal(%arg0: memref<?x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg1 = %c0 to %c10 step %c1 {
      %0 = arith.index_cast %arg1 : index to i32
      memref.store %c1_i32, %arg0[%arg1, %c0] : memref<?x10xi32>
      %1 = arith.addi %0, %c1_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.addi %0, %c-1_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      scf.for %arg2 = %c1 to %2 step %c1 {
        %5 = arith.index_cast %arg2 : index to i32
        %6 = arith.addi %5, %c-1_i32 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg0[%4, %7] : memref<?x10xi32>
        %9 = memref.load %arg0[%4, %arg2] : memref<?x10xi32>
        %10 = arith.addi %8, %9 : i32
        memref.store %10, %arg0[%arg1, %arg2] : memref<?x10xi32>
      }
    }
    return
  }
}
