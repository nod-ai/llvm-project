// RUN: mlir-opt %s --pass-pipeline=ccl-unrank-memrefs -split-input-file | FileCheck %s

//      CHECK: func.func @reduce
// CHECK-SAME: %[[in_tensor:[A-Za-z0-9_]+]]: tensor<?xf32>,
// CHECK-SAME: %[[out_ranked:[A-Za-z0-9_]+]]: memref<?xf32>,
// CHECK-SAME: %[[root_rank:[A-Za-z0-9_]+]]: index,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (memref<?xf32>, !ccl.chain)
func.func @reduce(
    %in_tensor : tensor<?xf32>,
    %out_ranked : memref<?xf32>,
    %root_rank: index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (memref<?xf32>, !ccl.chain) {
//      CHECK: %[[out_unranked:[A-Za-z0-9_]+]] = memref.cast %[[out_ranked]] : memref<?xf32> to memref<*xf32>
//      CHECK: %[[result_unranked:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.reduce sum, %[[in_tensor]], %[[out_unranked]], %[[root_rank]], %[[communicator]], %[[chain]]
// CHECK-SAME: (tensor<?xf32>, memref<*xf32>, index, !ccl.communicator, !ccl.chain) -> (memref<*xf32>, !ccl.chain)
   %res_tensor, %chain_out = ccl.reduce sum, %in_tensor, %out_ranked, %root_rank, %communicator, %chain :
       (tensor<?xf32>, memref<?xf32>, index, !ccl.communicator, !ccl.chain) -> (memref<?xf32>, !ccl.chain)
//      CHECK: %[[result_ranked:[A-Za-z0-9_]+]] = memref.cast %[[result_unranked]] : memref<*xf32> to memref<?xf32>
//      CHECK: return %[[result_ranked]], %[[result_chain]] : memref<?xf32>, !ccl.chain
  return %res_tensor, %chain_out : memref<?xf32>, !ccl.chain
}

// -----

//      CHECK: func.func @recv
// CHECK-SAME: %[[out_tensor:[A-Za-z0-9_]+]]: tensor<?xf32>,
// CHECK-SAME: %[[source:[A-Za-z0-9_]+]]: index,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (tensor<?xf32>, !ccl.chain)
func.func @recv(
    %out_tensor : tensor<?xf32>,
    %source: index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<?xf32>, !ccl.chain) {
//      CHECK: %[[result_tensor:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.recv %[[out_tensor]], %[[source]], %[[communicator]], %[[chain]]
// CHECK-SAME: (tensor<?xf32>, index, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
   %res_tensor, %chain_out = ccl.recv %out_tensor, %source, %communicator, %chain :
       (tensor<?xf32>, index, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
//      CHECK: return %[[result_tensor]], %[[result_chain]] : tensor<?xf32>, !ccl.chain
  return %res_tensor, %chain_out : tensor<?xf32>, !ccl.chain
}
