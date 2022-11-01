//===- Passes.h - CCL passes entry points ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_CCL_TRANSFORMS_PASSES_H

#include "mlir/Dialect/CCL/IR/CCL.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include <memory>

namespace mlir {
namespace ccl {

#define GEN_PASS_DECL_CCLTOFUNCCALLS
#define GEN_PASS_DECL_UNRANKMEMREFS
#include "mlir/Dialect/CCL/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createCCLToFuncCallsPass();
void populateCCLToFuncCallsPatterns(
    RewritePatternSet &patterns, const std::string& funcNamePrefix);

std::unique_ptr<Pass> createUnrankMemrefsPass();
void populateUnrankMemrefsPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/CCL/Transforms/Passes.h.inc"

} // namespace ccl
} // namespace mlir

#endif // MLIR_DIALECT_CCL_TRANSFORMS_PASSES_H
