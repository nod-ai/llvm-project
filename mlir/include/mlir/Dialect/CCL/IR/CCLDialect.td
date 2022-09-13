//===- CCLDialect.td - CCL dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the CCL (Collective Communications Library) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_IR_CCLDIALECT
#define MLIR_DIALECT_CCL_IR_CCLDIALECT

include "mlir/IR/OpBase.td"

def CCL_Dialect : Dialect {
  let summary = "Collective Communications Library dialect";
  let description = [{
    CCL is a dialect that describes distributed communication very similar to
    MPI (Message Passing Interface) and
    NCCL (NVIDIA Collective Communications Library).
    This dialect differs from them in that it works with tensors instead of continous
    buffers.
    ```mlir
    %communicator = ccl.com_world : communicator
    %rank = ccl.rank %communicator : (i64) -> i64
    %tensor = tensor.from_elements %rank :  tensor<1xi64>
    %in_chain = ccl.create_chain : ccl.chain
    %reduced, %out_chain = ccl.allreduce %tensor, %communicator, %in_chain
      { reduction_op = "sum" } : (tensor<1xi64>, communicator) -> tensor<1xi64>
    ```
  }];

  let name = "ccl";
  let cppNamespace = "::mlir::ccl";

  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_CCL_IR_CCLDIALECT
