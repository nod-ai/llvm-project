//===- CCLToFuncCalls.cpp - Pass to convert CCL ops to extern func calls --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CCL/IR/CCL.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CCL/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <string>

namespace mlir {
namespace ccl {

#define GEN_PASS_DEF_CCLTOFUNCCALLS
#define GEN_PASS_DEF_UNRANKMEMREFS
#include "mlir/Dialect/CCL/Transforms/Passes.h.inc"

namespace {
StringAttr funcName(Operation& op, const std::string& prefix) {
  return StringAttr::get(op.getContext(), llvm::Twine(prefix) + op.getName().stripDialect());
}

template <typename OpTy>
llvm::SmallVector<Value, 6> funcOperands(OpTy op, PatternRewriter &rewriter) {
  return op.getOperation();
}

template <>
llvm::SmallVector<Value, 6> funcOperands<ReduceOp>(ReduceOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  OpBuilder builder(op.getContext());
  auto reductionEnumVal = builder.create<arith::ConstantIntOp>(
    loc, static_cast<int>(op.getReductionOp()), builder.getI16Type());
  llvm::SmallVector<Value, 6> res = op.getOperands();
  res.insert(res.begin() + 3, reductionEnumVal.getResult());
  return res;
}

template <typename OpTy>
struct CCLOpToFuncCallConverter : public OpRewritePattern<OpTy> {
  CCLOpToFuncCallConverter(
    const std::string& funcNamePrefix,
    MLIRContext *context, PatternBenefit benefit = 1, ArrayRef<StringRef> generatedNames = {})
    : OpRewritePattern<OpTy>(context, benefit, generatedNames),
    funcNamePrefix(funcNamePrefix) {}
  LogicalResult match(OpTy op) const final {
    for (auto operand: op.getOperands()) {
      if (operand.getType().template isa<TensorType>() || operand.getType().template isa<MemRefType>()) {
        return failure("Only unranked memref operands are allowed.");
      }
    }
    for (auto resultType: op.getResultTypes()) {
      if (resultType.template isa<TensorType>() || resultType.template isa<MemRefType>()) {
        return failure("Only unranked memref results are allowed.");
      }
    }
    return success();
  }
  void rewrite(OpTy op, PatternRewriter &rewriter) const final {
    auto operands = funcOperands(op, rewriter);
    rewriter.replaceOpWithNewOp<func::CallOp>(
      op.getOperation(), funcName(*op.getOperation(), funcNamePrefix),
      op.getResultTypes(), op.getOperands());
  }

  private:
  std::string funcNamePrefix;
};

struct CCLToFuncCallsPass
    : public impl::CCLToFuncCallsBase<CCLToFuncCallsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, func::FuncDialect, CCLDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override;
};

void CCLToFuncCallsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateCCLToFuncCallsPatterns(patterns, funcNamePrefix);

  ConversionTarget target(getContext());
  target.addIllegalDialect<CCLDialect>();

    if (failed(applyFullConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
  }
}

} // namespace

void populateCCLToFuncCallsPatterns(
    RewritePatternSet &patterns, const std::string& funcNamePrefix) {
  // clang-format off
  patterns.add<
    CCLOpToFuncCallConverter<ReduceOp>
   >(funcNamePrefix, patterns.getContext());
  // clang-format on
}

std::unique_ptr<Pass>
createCCLToFuncCallsPass() {
  return std::make_unique<CCLToFuncCallsPass>();
}

llvm::SmallVector<Value, 6> unrankMemrefs(ValueRange values,
    ImplicitLocOpBuilder& builder) {
  llvm::SmallVector<Value, 6> res;
  for (const Value& val: values) {
    if (!val.getType().isa<MemRefType>()) {
      res.push_back(val);
    } else {
      MemRefType rankedMemrefType = val.getType().cast<MemRefType>();
      auto castOp = builder.create<memref::CastOp>(
        UnrankedMemRefType::get(rankedMemrefType.getElementType(),
        rankedMemrefType.getMemorySpace()), val);
      res.push_back(castOp.getResult());
    }
  }
  return res;
}

llvm::SmallVector<Type, 6> unrankMemrefTypes(TypeRange types) {
  llvm::SmallVector<Type, 6> res;
  for (const auto& type: types) {
    if (type.isa<MemRefType>()) {
      MemRefType rankedMemrefType = type.cast<MemRefType>();
      res.push_back(UnrankedMemRefType::get(rankedMemrefType.getElementType(),
        rankedMemrefType.getMemorySpace()));
    } else {
      res.push_back(type);
    }
  }
  return res;
}

llvm::SmallVector<Value, 6> rankMemrefs(ValueRange values,
    TypeRange requiredTypes,
    ImplicitLocOpBuilder& builder) {
  llvm::SmallVector<Value, 6> res;
  assert(values.size() == requiredTypes.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i].getType() == requiredTypes[i]) {
      res.push_back(values[i]);
    } else {
      auto castOp = builder.create<memref::CastOp>(requiredTypes[i], values[i]);
      res.push_back(castOp.getResult());
    }
  }
  return res;
}

struct UnrankMemrefsConverter : public OpInterfaceRewritePattern<CCLOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult match(CCLOp cclOp) const final {
    Operation& op = *cclOp.getOperation();
    bool hasRankedMemref = false;
    for (auto operand: op.getOperands()) {
      if (operand.getType().isa<MemRefType>()) {
        return success();
      }
    }
    for (auto resultType: op.getResultTypes()) {
      if (resultType.isa<MemRefType>()) {
        return success();
      }
    }
    return failure("At least one unranked memref operand or result is required.");
  }

  void rewrite(CCLOp cclOp, PatternRewriter &rewriter) const final {
    Operation& op = *cclOp.getOperation();
    // {
    //   llvm::SmallVector<Value, 6> args = op.getOperands();
    //   args.erase(args.begin() + 2);
    //   AllReduceOp newOp = rewriter.replaceOpWithNewOp<AllReduceOp>(&op, op.getResultTypes(), args, op.getAttrs());
    //   newOp.dump();
    //   newOp.getOperation()->getParentOp()->dump();
    //   return;
    // }

    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    llvm::SmallVector<Value, 6> unrankedOperands = unrankMemrefs(op.getOperands(), builder);
    llvm::SmallVector<Type, 6> unrankedResultTypes = unrankMemrefTypes(op.getResultTypes());
    bool areResultsUnranked = std::equal(unrankedResultTypes.begin(), unrankedResultTypes.end(),
            op.getResultTypes().begin(), op.getResultTypes().end());

    if (std::equal(unrankedOperands.begin(), unrankedOperands.end(),
        op.getOperands().begin(), op.getOperands().end()) && areResultsUnranked) {
      return;
    }

    Operation* newOp = builder.create(builder.getLoc(), op.getName().getIdentifier(),
      unrankedOperands, unrankedResultTypes, op.getAttrs());
    llvm::SmallVector<Value, 6> newResults = areResultsUnranked ?
        newOp->getResults() :
        rankMemrefs(newOp->getResults(), op.getResultTypes(), builder);
    rewriter.replaceOp(&op, newResults);
    newOp->dump();
    newOp->getParentOp()->dump();
  }
};

struct UnrankMemrefsPass
    : public impl::UnrankMemrefsBase<UnrankMemrefsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<CCLDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override;
};

void UnrankMemrefsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateUnrankMemrefsPatterns(patterns);

  // ConversionTarget target(getContext());
  //   if (failed(applyPartialConversion(getOperation(), target,
  //                                     std::move(patterns)))) {
  //     signalPassFailure();
  // }
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void populateUnrankMemrefsPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    UnrankMemrefsConverter
   >(patterns.getContext());
  // clang-format on
}

std::unique_ptr<Pass> createUnrankMemrefsPass() {
  return std::make_unique<UnrankMemrefsPass>();
}

} // namespace ccl
} // namespace mlir
