#include "scp.hpp"

#include <algorithm>
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "is.hpp"
#include "utils.hpp"

namespace arith = mlir::arith;
namespace memref = mlir::memref;
namespace scf = mlir::scf;

namespace
{

class SCoPPass :
    public mlir::PassWrapper<SCoPPass, mlir::OperationPass<mlir::func::FuncOp>>
{
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCoPPass)

    void runOnOperation() override
    {
        llvm::SmallVector<scf::ForOp> loops;
        llvm::SmallVector<scf::IfOp> conditionals;
        llvm::SmallVector<mlir::Value> counters;

        auto func = getOperation();

        func.walk([&](mlir::Operation *op) {
            if (auto loop = llvm::dyn_cast<scf::ForOp>(op))
            {
                loops.push_back(loop);
                counters.push_back(loop.getInductionVar());
            }
            else if (auto conditional = llvm::dyn_cast<scf::IfOp>(op))
            {
                conditionals.push_back(conditional);
            }
        });

        auto scps = form_scps(func, counters);

        cs6120::unique_isl_ctx ctx(isl_ctx_alloc(), isl_ctx_free);

        for (const auto &scp : scps)
        {
            for (auto &op : scp)
            {
                op.walk([&scp, ctx = ctx.get()](mlir::Operation *op) {
                    cs6120::unique_isl_space space(
                        cs6120::build_space(scp, op, ctx),
                        isl_space_free);

                    cs6120::unique_isl_basic_set set(
                        cs6120::build_domain(scp, op, ctx, space.get()),
                        isl_basic_set_free);

                    cs6120::unique_c_str domain(
                        isl_basic_set_to_str(set.get()),
                        std::free);

                    auto attribute = mlir::StringAttr::get(
                        op->getContext(),
                        domain.get());

                    op->setAttr("domain", attribute);
                });
            }
        }
    }

private:
    static llvm::SmallVector<cs6120::SCoP> form_scps(
        mlir::func::FuncOp func,
        llvm::ArrayRef<mlir::Value> counters)
    {
        llvm::SmallVector<cs6120::SCoP> scps;
        std::optional<cs6120::SCoP> current;

        func.getBody().walk<mlir::WalkOrder::PreOrder>(
            [&](mlir::Operation *op)
            {
                if (is_static_control(op, counters))
                {
                    const auto it = op->getIterator();
                    const auto next = ++op->getIterator();

                    if (!current)
                        current = cs6120::SCoP(it, next);
                    else
                        current->set_end(next);

                    if (next == op->getBlock()->end())
                    {
                        scps.push_back(*current);
                        current = std::nullopt;
                    }

                    return mlir::WalkResult::skip();
                }
                else if (current)
                {
                    scps.push_back(*current);
                    current = std::nullopt;
                }

                return mlir::WalkResult::advance();
            }
        );

        return scps;
    }

    static bool is_rich(const cs6120::SCoP &scp)
    {
        for (const auto &op : scp)
        {
            if (llvm::isa<scf::ForOp>(op))
                return true;
        }

        return false;
    }

    static bool is_static_control(
        const mlir::Operation *op,
        llvm::ArrayRef<mlir::Value> counters)
    {
        if (auto loop = llvm::dyn_cast<scf::ForOp>(op))
        {
            auto lower = loop.getLowerBound();
            auto upper = loop.getUpperBound();

            if (loop.getConstantStep() == 1
                && is_affine(lower, counters)
                && is_affine(upper, counters))
            {
                for (const auto &op : loop.getOps())
                {
                    if (!is_static_control(&op, counters))
                        return false;
                }

                return true;
            }

            return false;
        }
        else if (auto conditional = llvm::dyn_cast<scf::IfOp>(op))
        {
            auto condition = conditional.getCondition();

            if (auto cmp = llvm::dyn_cast_or_null<arith::CmpIOp>(condition.getDefiningOp()))
            {
                auto predicate = cmp.getPredicate();
                auto lhs = cmp.getLhs();
                auto rhs = cmp.getRhs();

                if (is_signed_inequality(predicate))
                {
                    if (is_counter(lhs, counters))
                        return is_affine(rhs, counters);

                    if (is_counter(rhs, counters))
                        return is_affine(lhs, counters);
                }
            }

            return false;
        }
        else if (auto load = llvm::dyn_cast<memref::LoadOp>(op))
        {
            for (auto index : load.getIndices())
            {
                if (!is_affine(index, counters))
                    return false;
            }

            return true;
        }
        else if (auto store = llvm::dyn_cast<memref::StoreOp>(op))
        {
            for (auto index : store.getIndices())
            {
                if (!is_affine(index, counters))
                    return false;
            }

            return true;
        }

        return true;
    }

    template<class T>
    static bool is_const(mlir::Value var, T val)
    {
        if (auto constant = llvm::dyn_cast_or_null<arith::ConstantOp>(var.getDefiningOp()))
        {
            if (auto value = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
                return value.getValue() == val;
        }

        return false;
    }

    static bool is_affine(mlir::Value var, llvm::ArrayRef<mlir::Value> counters)
    {
        if (is_counter(var, counters))
            return true;

        auto op = var.getDefiningOp();

        if (!op)
            return false;

        if (llvm::isa<arith::ConstantOp>(op))
            return true;

        if (auto cast = llvm::dyn_cast<arith::IndexCastOp>(op))
            return is_affine(cast.getIn(), counters);

        if (auto sum = llvm::dyn_cast<arith::AddIOp>(op))
        {
            auto lhs = sum.getLhs();
            auto rhs = sum.getRhs();

            return is_affine(lhs, counters) && is_affine(rhs, counters);
        }

        if (auto difference = llvm::dyn_cast<arith::SubIOp>(op))
        {
            auto lhs = difference.getLhs();
            auto rhs = difference.getRhs();

            return is_affine(lhs, counters) && is_affine(rhs, counters);
        }

        if (auto product = llvm::dyn_cast<arith::MulIOp>(op))
        {
            auto lhs = product.getLhs();
            auto rhs = product.getRhs();

            if (llvm::isa_and_nonnull<arith::ConstantOp>(lhs.getDefiningOp()))
                return is_counter(rhs, counters);

            if (llvm::isa_and_nonnull<arith::ConstantOp>(rhs.getDefiningOp()))
                return is_counter(lhs, counters);
        }

        return false;
    }

    static bool is_counter(mlir::Value var, llvm::ArrayRef<mlir::Value> counters)
    {
        if (auto cast = llvm::dyn_cast_or_null<arith::IndexCastOp>(var.getDefiningOp()))
            return is_counter(cast.getIn(), counters);
        else
            return cs6120::contains(counters, var);
    }

    static bool is_signed_inequality(arith::CmpIPredicate predicate)
    {
        switch (predicate)
        {
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::slt:
            return true;
        default:
            return false;
        }
    }
};

}   // namespace

namespace cs6120
{

std::unique_ptr<mlir::Pass> create_scp_pass()
{
    return std::make_unique<SCoPPass>();
}

}   // namespace cs6120
