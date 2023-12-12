#include <cstddef>
#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include "scp.hpp"

namespace cl = llvm::cl;

namespace
{

cl::opt<std::string> filename(
    cl::Positional,
    cl::desc("<filename>"),
    cl::init("-"),
    cl::value_desc("filename"));

int parse_source(
    llvm::SourceMgr &sm,
    mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module)
{
    auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(filename);

    if (std::error_code error = buffer.getError())
    {
        llvm::errs() << "Failed to open input file: " << error.message() << '\n';
        return EXIT_FAILURE;
    }

    sm.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sm, &context);

    if (!module)
    {
        llvm::errs() << "Failed to load file \"" << filename << "\"\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

}   // namespace

int main(int argc, char *argv[])
{
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "CS 6120 Final Project\n");

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::affine::AffineDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::DLTIDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::SourceMgr sm;

    if (int error = parse_source(sm, context, module))
        return error;

    mlir::PassManager pm(module.get()->getName());

    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
        return EXIT_FAILURE;

    mlir::OpPassManager &opm = pm.nest<mlir::func::FuncOp>();

    opm.addPass(mlir::createCanonicalizerPass());
    opm.addPass(mlir::createSCCPPass());
    opm.addPass(cs6120::create_scp_pass());

    if (mlir::failed(pm.run(*module)))
        return EXIT_FAILURE;

    module->print(llvm::outs());

    return EXIT_SUCCESS;
}
