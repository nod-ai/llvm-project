//===- CCLToFuncCalls.cpp - Pass to convert CCL ops to extern func calls --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CCL/IR/CCL.h"
#include "mlir/Dialect/CCL/Transforms/Passes.h"

namespace mlir {
namespace ccl {

#define GEN_PASS_DEF_CCLTOFUNCCALLS
#include "mlir/Dialect/CCL/Transforms/Passes.h.inc"

struct CCLToFuncCallsPass
    : public impl::CCLToFuncCallsBase<CCLToFuncCallsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<CCLDialect>();
  }
  CCLToFuncCallsPass() = default;

  void runOnOperation() override;
};

void CCLToFuncCallsPass::runOnOperation() {}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCCLToFuncCallsPass() {
  return std::make_unique<CCLToFuncCallsPass>();
}

} // namespace ccl
} // namespace mlir