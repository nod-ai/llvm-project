// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s

//      CHECK: func.func @reduce
// CHECK-SAME: %[[in_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[out_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[root_rank:[A-Za-z0-9_]+]]: index,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
//      CHECK: %[[result_buff:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.reduce sum, %[[in_tensor]], %[[out_tensor]], %[[root_rank]], %[[communicator]], %[[chain]]
// CHECK-SAME: (memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>, index, !ccl.communicator, !ccl.chain) -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
//      CHECK: return %[[result_buff]], %[[result_chain]] : memref<?xf32, strided<[?], offset: ?>>, !ccl.chain

func.func @reduce(
    %in_tensor : tensor<?xf32>,
    %out_tensor : tensor<?xf32>,
    %root_rank: index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<?xf32>, !ccl.chain) {
   %res, %chain_out = ccl.reduce sum, %in_tensor, %out_tensor, %root_rank, %communicator, %chain :
       (tensor<?xf32>, tensor<?xf32>, index, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
  return %res, %chain_out : tensor<?xf32>, !ccl.chain
}
