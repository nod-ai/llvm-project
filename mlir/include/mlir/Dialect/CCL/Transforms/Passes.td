//===-- Passes.td - CCL pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_TRANSFORMS_PASSES
#define MLIR_DIALECT_CCL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CCLToFuncCalls : InterfacePass<"ccl-to-func-calls", "::mlir::FunctionOpInterface"> {
  let summary = "Convert CCL ops to calls to external functions.";
  let constructor = "::mlir::ccl::createCCLToFuncCallsPass()";
}

#endif // MLIR_DIALECT_CCL_TRANSFORMS_PASSES
