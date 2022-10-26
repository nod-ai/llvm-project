//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CCL/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/CCL/IR/CCL.h"

using namespace mlir::bufferization;

namespace mlir {
class DialectRegistry;

namespace ccl {

namespace {
struct ReduceOpInterface
    : public BufferizableOpInterface::ExternalModel<ReduceOpInterface,
                                                    ReduceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 1;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    if (opResult.getResultNumber() == 0) {
      return {&op->getOpOperand(1)};
    }
    return {};
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 1) {
      return {op->getResult(0)};
    }
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto reduceOp = cast<ReduceOp>(op);
    if (!reduceOp.getInTensor().getType().cast<TensorType>() &&
        !reduceOp.getOutTensor().getType().cast<TensorType>()) {
      return success();
    }

    Value inTensorBuffer;
    if (reduceOp.getInTensor().getType().cast<TensorType>()) {
      FailureOr<Value> inTensorBufferRes =
          getBuffer(rewriter, reduceOp.getInTensor(), options);
      if (failed(inTensorBufferRes)) {
        return failure();
      }
      inTensorBuffer = *inTensorBufferRes;
    } else {
      inTensorBuffer = reduceOp.getInTensor();
    }

    Value outTensorBuffer;
    if (reduceOp.getInTensor().getType().cast<TensorType>()) {
      FailureOr<Value> outTensorBufferRes =
          getBuffer(rewriter, reduceOp.getOutTensor(), options);
      if (failed(outTensorBufferRes)) {
        return failure();
      }
      outTensorBuffer = *outTensorBufferRes;
    } else {
      outTensorBuffer = reduceOp.getInTensor();
    }

    Type resultTensorMemRefType =
        reduceOp.getResultTensor().getType().isa<BaseMemRefType>()
            ? reduceOp.getResultTensor().getType()
            : getMemRefType(reduceOp.getResultTensor(), options, /*layout=*/{},
                            outTensorBuffer.getType()
                                .cast<BaseMemRefType>()
                                .getMemorySpaceAsInt());

    replaceOpWithNewBufferizedOp<ReduceOp>(
        rewriter, reduceOp, resultTensorMemRefType,
        reduceOp.getResultChain().getType(), inTensorBuffer, outTensorBuffer,
        reduceOp.getRoot(), reduceOp.getCommunicator(), reduceOp.getInChain(),
        reduceOp.getReductionOpAttr());
    return success();
  }
};
} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CCLDialect *dialect) {
    ReduceOp::attachInterface<ReduceOpInterface>(*ctx);
  });
}
} // namespace ccl
} // namespace mlir
