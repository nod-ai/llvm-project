// RUN: mlir-opt %s --pass-pipeline="ccl-unrank-memrefs" -split-input-file
// | FileCheck %s

func.func @reduce(
    %in_tensor : memref<?xf32>,
    %out_tensor : memref<?xf32>,
    %root_rank: index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (memref<?xf32>, !ccl.chain) {

   %res_tensor, %chain_out = ccl.reduce sum, %in_tensor, %out_tensor, %root_rank, %communicator, %chain :
       (memref<?xf32>, memref<?xf32>, index, !ccl.communicator, !ccl.chain) -> (memref<?xf32>, !ccl.chain)
  return %res_tensor, %chain_out : memref<?xf32>, !ccl.chain
}
