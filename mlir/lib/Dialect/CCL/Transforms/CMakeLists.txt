add_mlir_dialect_library(MLIRCCLTransforms
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/CCL

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRCCLDialect
)
