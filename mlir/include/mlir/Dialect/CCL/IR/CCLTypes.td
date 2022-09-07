//===- CCLDialect.td - CCL dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the CCL (Collective Communications Library) dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_IR_CCLTYPES
#define MLIR_DIALECT_CCL_IR_CCLTYPES

include "mlir/Dialect/CCL/IR/CCLDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

class CCL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<CCL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def CCL_Operation : CCL_Type<"Operation", "operation"> {
  let summary = "CCL handle to an `mlir::Operation *`";
  let description = [{
    This type represents a handle to an instance of an `mlir::Operation *`,
    bound to a value that is usable within a CCL pattern or rewrite.
  }];
}

def CCL_Communicator : CCL_Type<"Communicator", "communicator"> {
  let summary = "Handler to a group of processes that are associated during collective communication.";
}

def CCL_Rank : CCL_Type<"Rank", "rank", [
    // TODO: figure out why adding this predicate fails to compile
    //PredTrait<"Is int", CPred<"$_self.isSignedInteger()">>
    ]> {
  let summary = "Rank within a communicator.";
}

def CCL_OpEnumSum : I16EnumAttrCase<"Sum", 0>;
def CCL_OpEnumProduct : I16EnumAttrCase<"Product", 1>;

def CCL_OpEnum : I16EnumAttr<"OpEnum", "Operation for reductions",
                           [CCL_OpEnumSum, CCL_OpEnumProduct]> {
  let cppNamespace = "::mlir::ccl";
}

#endif // MLIR_DIALECT_CCL_IR_CCLTYPES
