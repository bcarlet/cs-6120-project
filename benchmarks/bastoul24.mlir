module {
  func.func @fig_2_4(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg1 = %c1 to %c4 step %c1 {
      %0 = arith.index_cast %arg1 : index to i32
      %1 = memref.load %arg0[%arg1] : memref<?xi32>
      %2 = arith.addi %1, %c1_i32 : i32
      memref.store %2, %arg0[%arg1] : memref<?xi32>
      %3 = arith.muli %0, %0 : i32
      %4 = arith.addi %3, %c1_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      scf.for %arg2 = %c1 to %5 step %c1 {
        %6 = arith.index_cast %arg2 : index to i32
        %7 = memref.load %arg0[%arg2] : memref<?xi32>
        %8 = arith.addi %7, %c2_i32 : i32
        memref.store %8, %arg0[%arg2] : memref<?xi32>
        %9 = arith.addi %6, %c1_i32 : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = arith.cmpi sge, %6, %c2_i32 : i32
        scf.for %arg3 = %c0 to %10 step %c1 {
          scf.if %11 {
            %14 = memref.load %arg0[%arg3] : memref<?xi32>
            %15 = arith.addi %14, %c3_i32 : i32
            memref.store %15, %arg0[%arg3] : memref<?xi32>
          }
          %12 = memref.load %arg0[%arg3] : memref<?xi32>
          %13 = arith.addi %12, %c4_i32 : i32
          memref.store %13, %arg0[%arg3] : memref<?xi32>
        }
      }
      scf.for %arg2 = %c0 to %c7 step %c1 {
        %6 = memref.load %arg0[%arg2] : memref<?xi32>
        %7 = arith.addi %6, %c5_i32 : i32
        memref.store %7, %arg0[%arg2] : memref<?xi32>
        %8 = memref.load %arg0[%arg2] : memref<?xi32>
        %9 = arith.addi %8, %c6_i32 : i32
        memref.store %9, %arg0[%arg2] : memref<?xi32>
      }
    }
    return
  }
}
