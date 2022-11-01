add_mlir_dialect_library(MLIRCCLTransforms
  BufferizableOpInterfaceImpl.cpp
  CCLToFuncCalls.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/CCL

  DEPENDS
  MLIRCCLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRBufferizationDialect
  MLIRCCLDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
)
