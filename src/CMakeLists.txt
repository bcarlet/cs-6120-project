add_llvm_executable(polyc
  dep.cpp
  dep.hpp
  is.cpp
  is.hpp
  main.cpp
  scp.cpp
  scp.hpp
  utils.hpp
)

llvm_update_compile_flags(polyc)

target_link_libraries(polyc
  PRIVATE
    ISL
    MLIRDLTIDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRTransforms
)
