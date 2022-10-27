// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s

//      CHECK: func.func @send
// CHECK-SAME: %[[in_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[destination:[A-Za-z0-9_]+]]: index,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> !ccl.chain
func.func @send(
    %in_tensor : tensor<?xf32>,
    %destination: index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> !ccl.chain {
//      CHECK: %[[result_chain:[A-Za-z0-9_]+]] = ccl.send %[[in_tensor]], %[[destination]], %[[communicator]], %[[chain]]
// CHECK-SAME: (memref<?xf32, strided<[?], offset: ?>>, index, !ccl.communicator, !ccl.chain) -> !ccl.chain
   %chain_out = ccl.send %in_tensor, %destination, %communicator, %chain :
       (tensor<?xf32>, index, !ccl.communicator, !ccl.chain) -> !ccl.chain
//      CHECK: return %[[result_chain]] : !ccl.chain
  return %chain_out : !ccl.chain
}

// -----

//      CHECK: func.func @recv
// CHECK-SAME: %[[out_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[source:[A-Za-z0-9_]+]]: index,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
func.func @recv(
    %out_tensor : tensor<?xf32>,
    %source: index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<?xf32>, !ccl.chain) {
//      CHECK: %[[result_buff:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.recv %[[out_tensor]], %[[destination]], %[[communicator]], %[[chain]]
// CHECK-SAME: (memref<?xf32, strided<[?], offset: ?>>, index, !ccl.communicator, !ccl.chain) -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
   %res_tensor, %chain_out = ccl.recv %out_tensor, %source, %communicator, %chain :
       (tensor<?xf32>, index, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
//      CHECK: return %[[result_buff]], %[[result_chain]] : memref<?xf32, strided<[?], offset: ?>>, !ccl.chain
  return %res_tensor, %chain_out : tensor<?xf32>, !ccl.chain
}

// -----

//      CHECK: func.func @bcast
// CHECK-SAME: %[[io_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[source:[A-Za-z0-9_]+]]: index,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
func.func @bcast(
    %io_tensor : tensor<?xf32>,
    %source: index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<?xf32>, !ccl.chain) {
//      CHECK: %[[result_buff:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.bcast %[[io_tensor]], %[[destination]], %[[communicator]], %[[chain]]
// CHECK-SAME: (memref<?xf32, strided<[?], offset: ?>>, index, !ccl.communicator, !ccl.chain) -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
   %res_tensor, %chain_out = ccl.bcast %io_tensor, %source, %communicator, %chain :
       (tensor<?xf32>, index, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
//      CHECK: return %[[result_buff]], %[[result_chain]] : memref<?xf32, strided<[?], offset: ?>>, !ccl.chain
  return %res_tensor, %chain_out : tensor<?xf32>, !ccl.chain
}

// -----

//      CHECK: func.func @reduce
// CHECK-SAME: %[[in_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[out_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[root_rank:[A-Za-z0-9_]+]]: index,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
func.func @reduce(
    %in_tensor : tensor<?xf32>,
    %out_tensor : tensor<?xf32>,
    %root_rank: index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<?xf32>, !ccl.chain) {
//      CHECK: %[[result_buff:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.reduce sum, %[[in_tensor]], %[[out_tensor]], %[[root_rank]], %[[communicator]], %[[chain]]
// CHECK-SAME: (memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>, index, !ccl.communicator, !ccl.chain) -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
   %res_tensor, %chain_out = ccl.reduce sum, %in_tensor, %out_tensor, %root_rank, %communicator, %chain :
       (tensor<?xf32>, tensor<?xf32>, index, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
//      CHECK: return %[[result_buff]], %[[result_chain]] : memref<?xf32, strided<[?], offset: ?>>, !ccl.chain
  return %res_tensor, %chain_out : tensor<?xf32>, !ccl.chain
}

// -----

//      CHECK: func.func @allreduce
// CHECK-SAME: %[[in_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[out_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
func.func @allreduce(
    %in_tensor : tensor<?xf32>,
    %out_tensor : tensor<?xf32>,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<?xf32>, !ccl.chain) {
//      CHECK: %[[result_buff:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.allreduce sum, %[[in_tensor]], %[[out_tensor]], %[[communicator]], %[[chain]]
// CHECK-SAME: (memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>, !ccl.communicator, !ccl.chain) -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
   %res_tensor, %chain_out = ccl.allreduce sum, %in_tensor, %out_tensor, %communicator, %chain :
       (tensor<?xf32>, tensor<?xf32>, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
//      CHECK: return %[[result_buff]], %[[result_chain]] : memref<?xf32, strided<[?], offset: ?>>, !ccl.chain
  return %res_tensor, %chain_out : tensor<?xf32>, !ccl.chain
}

// -----

//      CHECK: func.func @reduce_scatter
// CHECK-SAME: %[[in_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[out_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
func.func @reduce_scatter(
    %in_tensor : tensor<?xf32>,
    %out_tensor : tensor<?xf32>,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<?xf32>, !ccl.chain) {
//      CHECK: %[[result_buff:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.reduce_scatter sum, %[[in_tensor]], %[[out_tensor]], %[[communicator]], %[[chain]]
// CHECK-SAME: (memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>, !ccl.communicator, !ccl.chain) -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
   %res_tensor, %chain_out = ccl.reduce_scatter sum, %in_tensor, %out_tensor, %communicator, %chain :
       (tensor<?xf32>, tensor<?xf32>, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
//      CHECK: return %[[result_buff]], %[[result_chain]] : memref<?xf32, strided<[?], offset: ?>>, !ccl.chain
  return %res_tensor, %chain_out : tensor<?xf32>, !ccl.chain
}

// -----

//      CHECK: func.func @allgather
// CHECK-SAME: %[[in_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[out_tensor:[A-Za-z0-9_]+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME: %[[communicator:[A-Za-z0-9_]+]]: !ccl.communicator,
// CHECK-SAME: %[[chain:[A-Za-z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
func.func @allgather(
    %in_tensor : tensor<?xf32>,
    %out_tensor : tensor<?xf32>,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<?xf32>, !ccl.chain) {
//      CHECK: %[[result_buff:[A-Za-z0-9_]+]], %[[result_chain:[A-Za-z0-9_]+]] = ccl.allgather %[[in_tensor]], %[[out_tensor]], %[[communicator]], %[[chain]]
// CHECK-SAME: (memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>, !ccl.communicator, !ccl.chain) -> (memref<?xf32, strided<[?], offset: ?>>, !ccl.chain)
   %res_tensor, %chain_out = ccl.allgather %in_tensor, %out_tensor, %communicator, %chain :
       (tensor<?xf32>, tensor<?xf32>, !ccl.communicator, !ccl.chain) -> (tensor<?xf32>, !ccl.chain)
//      CHECK: return %[[result_buff]], %[[result_chain]] : memref<?xf32, strided<[?], offset: ?>>, !ccl.chain
  return %res_tensor, %chain_out : tensor<?xf32>, !ccl.chain
}
