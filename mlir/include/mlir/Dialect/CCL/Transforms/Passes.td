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

def CCLToFuncCalls : Pass<"ccl-to-func-calls", "ModuleOp"> {
  let summary = "Convert CCL ops to calls to external functions.";
  let constructor = "::mlir::ccl::createCCLToFuncCallsPass()";
  let options = [
    Option<"funcNamePrefix", "func-name-prefix", "std::string",
        /*default=*/"\"ccl_mpi_\"", "Prefix to put at each function name.">,
  ];
}

def UnrankMemrefs : Pass<"ccl-unrank-memrefs"> {
  let summary = "Substitute all operand and result ranked memrefs with unranked memrefs for CCL ops.";
  let description = [{Inserts memref.cast to make the transformation.}];
  let constructor = "::mlir::ccl::createUnrankMemrefsPass()";
}

#endif // MLIR_DIALECT_CCL_TRANSFORMS_PASSES
