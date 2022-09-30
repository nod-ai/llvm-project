//===- CCLOps.td - CCL dialect definition ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the CCL (Collective Communications Library) dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_IR_CCLOPS
#define MLIR_DIALECT_CCL_IR_CCLOPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/Dialect/CCL/IR/CCLInterfaces.td"
include "mlir/Dialect/CCL/IR/CCLTypes.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"

class CCL_Op<string mnemonic, list<Trait> traits = []>
    : Op<CCL_Dialect, mnemonic, traits # [CCL_OpInterface]> {
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def CCL_ComWorldOp : CCL_Op<"com_world", []> {
  let summary = "Get the global/world communicator.";

  let results = (outs CCL_Communicator:$result);
  let assemblyFormat = "attr-dict";
}

def CCL_RankOp : CCL_Op<"rank", []> {
  let summary = "Get the rank of this process.";

  let arguments = (ins CCL_Communicator:$communicator);
  let results = (outs CCL_Rank:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def CCL_CreateChainOp : CCL_Op<"create_chain", []> {
  let results = (outs CCL_Chain:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def CCL_Send : CCL_Op<"send", []> {
  let arguments = (
      ins AnyTensor:$tensor,
      CCL_Rank:$destination,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain);
  let results = (outs CCL_Chain:$out_chain);
}

def CCL_Recv : CCL_Op<"recv", []> {
  let arguments = (
      ins CCL_Rank:$source,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain);
  let results = (outs AnyTensor:$result, CCL_Chain:$out_chain);
}

def CCL_BcastOp : CCL_Op<"bcast",
    [TypesMatchWith<"result type matches type of tensor",
                    "tensor", "result",
                    "$_self.cast<ShapedType>()">]> {
  let arguments = (
      ins AnyTensor:$tensor,
      CCL_Rank:$root,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain);
  let results = (outs AnyTensor:$result, CCL_Chain:$out_chain);
}

def CCL_ReduceOp : CCL_Op<"reduce", []> {
  let description = [{
    Returns a non-empty optional value only when the current rank = `root`.
  }];
  let arguments = (
      ins AnyTensor:$tensor,
      CCL_Rank:$root,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain,
      DefaultValuedAttr<CCL_OpEnum, "OpEnum::SUM">:$reduction_op);
  let results = (outs Optional<AnyTensor>: $result, CCL_Chain:$out_chain);
}

def CCL_AllReduceOp : CCL_Op<"allreduce", [
    TypesMatchWith<"result type matches type of tensor",
                    "tensor", "result",
                    "$_self.cast<ShapedType>()">]> {
  let arguments = (
      ins AnyTensor:$tensor,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain,
      DefaultValuedAttr<CCL_OpEnum, "OpEnum::SUM">:$reduction_op);
  let results = (outs AnyTensor:$result, CCL_Chain:$out_chain);
  let assemblyFormat = "$reduction_op `,` operands attr-dict `:` functional-type(operands, results)";
}

def CCL_ReduceScatterOp : CCL_Op<"reduce_scatter", []> {
  let description = [{
    Stacks along a new dimension `dim`, analogous to `numpy.stack`.
  }];
  let arguments = (
      ins AnyTensor:$tensor,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain,
      DefaultValuedAttr<CCL_OpEnum, "OpEnum::SUM">:$reduction_op);
  let results = (outs AnyTensor:$result, CCL_Chain:$out_chain);
  let assemblyFormat = "$reduction_op `,` operands attr-dict `:` functional-type(operands, results)";
}

def CCL_AllGatherOp : CCL_Op<"allgather", []> {
  let description = [{
    Stacks along a new dimension `dim`, analogous to `numpy.stack`.
  }];
  let arguments = (
      ins AnyTensor:$tensor,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain,
      DefaultValuedAttr<I32Attr, "0">:$dim);
  let results = (outs AnyTensor:$result, CCL_Chain:$out_chain);
}

#endif // MLIR_DIALECT_CCL_IR_CCLOPS
