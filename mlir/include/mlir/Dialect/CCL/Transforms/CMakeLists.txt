set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CCL)
add_public_tablegen_target(MLIRCCLTransformsIncGen)

add_mlir_doc(Passes CCLPasses ./ -gen-pass-doc)
