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
#include "mlir/IR/DialectRegistry.h"

using namespace mlir::bufferization;

namespace mlir {
namespace ccl {
namespace {

struct SendOpInterface
    : public BufferizableOpInterface::ExternalModel<SendOpInterface, SendOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *opPtr, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<SendOp>(opPtr);
    if (!op.getInTensor().getType().cast<TensorType>()) {
      return success();
    }

    Value inTensorBuffer;
    if (op.getInTensor().getType().cast<TensorType>()) {
      FailureOr<Value> inTensorBufferRes =
          getBuffer(rewriter, op.getInTensor(), options);
      if (failed(inTensorBufferRes)) {
        return failure();
      }
      inTensorBuffer = *inTensorBufferRes;
    } else {
      inTensorBuffer = op.getInTensor();
    }

    replaceOpWithNewBufferizedOp<SendOp>(
        rewriter, op, op.getResultChain().getType(), inTensorBuffer,
        op.getDestination(), op.getCommunicator(), op.getInChain());
    return success();
  }
};

struct RecvOpInterface
    : public BufferizableOpInterface::ExternalModel<RecvOpInterface, RecvOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    if (opResult.getResultNumber() == 0) {
      return {&op->getOpOperand(0)};
    }
    return {};
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 0) {
      return {op->getResult(0)};
    }
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *opPtr, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<RecvOp>(opPtr);
    if (!op.getOutTensor().getType().cast<TensorType>()) {
      return success();
    }

    Value outTensorBuffer;
    if (op.getOutTensor().getType().cast<TensorType>()) {
      FailureOr<Value> outTensorBufferRes =
          getBuffer(rewriter, op.getOutTensor(), options);
      if (failed(outTensorBufferRes)) {
        return failure();
      }
      outTensorBuffer = *outTensorBufferRes;
    } else {
      outTensorBuffer = op.getOutTensor();
    }

    Type resultTensorMemRefType =
        op.getResultTensor().getType().isa<BaseMemRefType>()
            ? op.getResultTensor().getType()
            : getMemRefType(op.getResultTensor(), options, /*layout=*/{},
                            outTensorBuffer.getType()
                                .cast<BaseMemRefType>()
                                .getMemorySpaceAsInt());

    replaceOpWithNewBufferizedOp<RecvOp>(
        rewriter, op, resultTensorMemRefType, op.getResultChain().getType(),
        outTensorBuffer, op.getSource(), op.getCommunicator(), op.getInChain());
    return success();
  }
};

struct BcastOpInterface
    : public BufferizableOpInterface::ExternalModel<BcastOpInterface, BcastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    if (opResult.getResultNumber() == 0) {
      return {&op->getOpOperand(0)};
    }
    return {};
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 0) {
      return {op->getResult(0)};
    }
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *opPtr, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<BcastOp>(opPtr);
    if (!op.getIoTensor().getType().cast<TensorType>()) {
      return success();
    }

    Value ioTensorBuffer;
    if (op.getIoTensor().getType().cast<TensorType>()) {
      FailureOr<Value> ioTensorBufferRes =
          getBuffer(rewriter, op.getIoTensor(), options);
      if (failed(ioTensorBufferRes)) {
        return failure();
      }
      ioTensorBuffer = *ioTensorBufferRes;
    } else {
      ioTensorBuffer = op.getIoTensor();
    }

    Type resultTensorMemRefType =
        op.getResultTensor().getType().isa<BaseMemRefType>()
            ? op.getResultTensor().getType()
            : getMemRefType(op.getResultTensor(), options, /*layout=*/{},
                            ioTensorBuffer.getType()
                                .cast<BaseMemRefType>()
                                .getMemorySpaceAsInt());

    replaceOpWithNewBufferizedOp<BcastOp>(
        rewriter, op, resultTensorMemRefType, op.getResultChain().getType(),
        ioTensorBuffer, op.getRoot(), op.getCommunicator(), op.getInChain());
    return success();
  }
};

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

  LogicalResult bufferize(Operation *opPtr, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<ReduceOp>(opPtr);
    if (!op.getInTensor().getType().cast<TensorType>() &&
        !op.getOutTensor().getType().cast<TensorType>()) {
      return success();
    }

    Value inTensorBuffer;
    if (op.getInTensor().getType().cast<TensorType>()) {
      FailureOr<Value> inTensorBufferRes =
          getBuffer(rewriter, op.getInTensor(), options);
      if (failed(inTensorBufferRes)) {
        return failure();
      }
      inTensorBuffer = *inTensorBufferRes;
    } else {
      inTensorBuffer = op.getInTensor();
    }

    Value outTensorBuffer;
    if (op.getOutTensor().getType().cast<TensorType>()) {
      FailureOr<Value> outTensorBufferRes =
          getBuffer(rewriter, op.getOutTensor(), options);
      if (failed(outTensorBufferRes)) {
        return failure();
      }
      outTensorBuffer = *outTensorBufferRes;
    } else {
      outTensorBuffer = op.getOutTensor();
    }

    Type resultTensorMemRefType =
        op.getResultTensor().getType().isa<BaseMemRefType>()
            ? op.getResultTensor().getType()
            : getMemRefType(op.getResultTensor(), options, /*layout=*/{},
                            outTensorBuffer.getType()
                                .cast<BaseMemRefType>()
                                .getMemorySpaceAsInt());

    replaceOpWithNewBufferizedOp<ReduceOp>(
        rewriter, op, resultTensorMemRefType, op.getResultChain().getType(),
        inTensorBuffer, outTensorBuffer, op.getRoot(), op.getCommunicator(),
        op.getInChain(), op.getReductionOpAttr());
    return success();
  }
};

struct AllReduceOpInterface
    : public BufferizableOpInterface::ExternalModel<AllReduceOpInterface,
                                                    AllReduceOp> {
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

  LogicalResult bufferize(Operation *opPtr, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<AllReduceOp>(opPtr);
    if (!op.getInTensor().getType().cast<TensorType>() &&
        !op.getOutTensor().getType().cast<TensorType>()) {
      return success();
    }

    Value inTensorBuffer;
    if (op.getInTensor().getType().cast<TensorType>()) {
      FailureOr<Value> inTensorBufferRes =
          getBuffer(rewriter, op.getInTensor(), options);
      if (failed(inTensorBufferRes)) {
        return failure();
      }
      inTensorBuffer = *inTensorBufferRes;
    } else {
      inTensorBuffer = op.getInTensor();
    }

    Value outTensorBuffer;
    if (op.getOutTensor().getType().cast<TensorType>()) {
      FailureOr<Value> outTensorBufferRes =
          getBuffer(rewriter, op.getOutTensor(), options);
      if (failed(outTensorBufferRes)) {
        return failure();
      }
      outTensorBuffer = *outTensorBufferRes;
    } else {
      outTensorBuffer = op.getOutTensor();
    }

    Type resultTensorMemRefType =
        op.getResultTensor().getType().isa<BaseMemRefType>()
            ? op.getResultTensor().getType()
            : getMemRefType(op.getResultTensor(), options, /*layout=*/{},
                            outTensorBuffer.getType()
                                .cast<BaseMemRefType>()
                                .getMemorySpaceAsInt());

    replaceOpWithNewBufferizedOp<AllReduceOp>(
        rewriter, op, resultTensorMemRefType, op.getResultChain().getType(),
        inTensorBuffer, outTensorBuffer, op.getCommunicator(), op.getInChain(),
        op.getReductionOpAttr());
    return success();
  }
};

struct ReduceScatterOpInterface
    : public BufferizableOpInterface::ExternalModel<ReduceScatterOpInterface,
                                                    ReduceScatterOp> {
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

  LogicalResult bufferize(Operation *opPtr, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<ReduceScatterOp>(opPtr);
    if (!op.getInTensor().getType().cast<TensorType>() &&
        !op.getOutTensor().getType().cast<TensorType>()) {
      return success();
    }

    Value inTensorBuffer;
    if (op.getInTensor().getType().cast<TensorType>()) {
      FailureOr<Value> inTensorBufferRes =
          getBuffer(rewriter, op.getInTensor(), options);
      if (failed(inTensorBufferRes)) {
        return failure();
      }
      inTensorBuffer = *inTensorBufferRes;
    } else {
      inTensorBuffer = op.getInTensor();
    }

    Value outTensorBuffer;
    if (op.getOutTensor().getType().cast<TensorType>()) {
      FailureOr<Value> outTensorBufferRes =
          getBuffer(rewriter, op.getOutTensor(), options);
      if (failed(outTensorBufferRes)) {
        return failure();
      }
      outTensorBuffer = *outTensorBufferRes;
    } else {
      outTensorBuffer = op.getOutTensor();
    }

    Type resultTensorMemRefType =
        op.getResultTensor().getType().isa<BaseMemRefType>()
            ? op.getResultTensor().getType()
            : getMemRefType(op.getResultTensor(), options, /*layout=*/{},
                            outTensorBuffer.getType()
                                .cast<BaseMemRefType>()
                                .getMemorySpaceAsInt());

    replaceOpWithNewBufferizedOp<ReduceScatterOp>(
        rewriter, op, resultTensorMemRefType, op.getResultChain().getType(),
        inTensorBuffer, outTensorBuffer, op.getCommunicator(), op.getInChain(),
        op.getReductionOpAttr());
    return success();
  }
};

struct AllGatherOpInterface
    : public BufferizableOpInterface::ExternalModel<AllGatherOpInterface,
                                                    AllGatherOp> {
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

  LogicalResult bufferize(Operation *opPtr, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<AllGatherOp>(opPtr);
    if (!op.getInTensor().getType().cast<TensorType>() &&
        !op.getOutTensor().getType().cast<TensorType>()) {
      return success();
    }

    Value inTensorBuffer;
    if (op.getInTensor().getType().cast<TensorType>()) {
      FailureOr<Value> inTensorBufferRes =
          getBuffer(rewriter, op.getInTensor(), options);
      if (failed(inTensorBufferRes)) {
        return failure();
      }
      inTensorBuffer = *inTensorBufferRes;
    } else {
      inTensorBuffer = op.getInTensor();
    }

    Value outTensorBuffer;
    if (op.getOutTensor().getType().cast<TensorType>()) {
      FailureOr<Value> outTensorBufferRes =
          getBuffer(rewriter, op.getOutTensor(), options);
      if (failed(outTensorBufferRes)) {
        return failure();
      }
      outTensorBuffer = *outTensorBufferRes;
    } else {
      outTensorBuffer = op.getOutTensor();
    }

    Type resultTensorMemRefType =
        op.getResultTensor().getType().isa<BaseMemRefType>()
            ? op.getResultTensor().getType()
            : getMemRefType(op.getResultTensor(), options, /*layout=*/{},
                            outTensorBuffer.getType()
                                .cast<BaseMemRefType>()
                                .getMemorySpaceAsInt());

    replaceOpWithNewBufferizedOp<AllGatherOp>(
        rewriter, op, resultTensorMemRefType, op.getResultChain().getType(),
        inTensorBuffer, outTensorBuffer, op.getCommunicator(), op.getInChain());
    return success();
  }
};

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CCLDialect *dialect) {
    SendOp::attachInterface<SendOpInterface>(*ctx);
    RecvOp::attachInterface<RecvOpInterface>(*ctx);
    BcastOp::attachInterface<BcastOpInterface>(*ctx);
    ReduceOp::attachInterface<ReduceOpInterface>(*ctx);
    AllReduceOp::attachInterface<AllReduceOpInterface>(*ctx);
    ReduceScatterOp::attachInterface<ReduceScatterOpInterface>(*ctx);
    AllGatherOp::attachInterface<AllGatherOpInterface>(*ctx);
  });
}
} // namespace ccl
} // namespace mlir
