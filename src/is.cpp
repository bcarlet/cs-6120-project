#include "is.hpp"

#include <cstddef>
#include <string>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/local_space.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/val.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "scp.hpp"
#include "utils.hpp"

namespace arith = mlir::arith;
namespace scf = mlir::scf;

namespace
{

std::string print_value_as_operand(mlir::Value value)
{
    std::string buffer;
    llvm::raw_string_ostream stream(buffer);

    value.printAsOperand(stream, std::nullopt);

    return buffer;
}

__isl_give isl_basic_set *build_set(
    mlir::Operation *condition,
    __isl_keep isl_ctx *ctx,
    __isl_keep isl_space *space)
{
    if (auto cmp = llvm::dyn_cast<arith::CmpIOp>(condition))
    {
        isl_basic_set *(*set_constructor)(isl_aff *, isl_aff *);

        switch (cmp.getPredicate())
        {
        case arith::CmpIPredicate::eq:
            set_constructor = isl_aff_eq_basic_set;
            break;
        case arith::CmpIPredicate::sge:
            set_constructor = isl_aff_ge_basic_set;
            break;
        case arith::CmpIPredicate::sgt:
            set_constructor = isl_aff_gt_basic_set;
            break;
        case arith::CmpIPredicate::sle:
            set_constructor = isl_aff_le_basic_set;
            break;
        case arith::CmpIPredicate::slt:
            set_constructor = isl_aff_lt_basic_set;
            break;
        default:
            return nullptr;
        }

        isl_aff *lhs = cs6120::build_affine(cmp.getLhs(), ctx, space);
        isl_aff *rhs = cs6120::build_affine(cmp.getRhs(), ctx, space);

        return set_constructor(lhs, rhs);
    }

    return nullptr;
}

}   // namespace

namespace cs6120
{

__isl_give isl_space *build_space(
    const SCoP &scp,
    mlir::Operation *op,
    __isl_keep isl_ctx *ctx)
{
    llvm::SmallVector<mlir::Value> dims;
    llvm::SmallVector<mlir::Value> params;

    bool outside =
        contains_if(scp, [op](const auto &root) { return &root == op; });

    while ((op = op->getParentOp()))
    {
        if (auto loop = llvm::dyn_cast<scf::ForOp>(op))
        {
            auto dim = loop.getInductionVar();

            if (outside)
                params.push_back(dim);
            else
                dims.push_back(dim);
        }

        outside |=
            contains_if(scp, [op](const auto &root) { return &root == op; });
    }

    isl_space *space = isl_space_set_alloc(ctx, params.size(), dims.size());

    for (std::size_t i = 0; i < dims.size(); i++)
    {
        std::string name = print_value_as_operand(dims[i]);
        space = isl_space_set_dim_name(space, isl_dim_set, i, name.c_str());
    }

    for (std::size_t i = 0; i < params.size(); i++)
    {
        std::string name = print_value_as_operand(params[i]);
        space = isl_space_set_dim_name(space, isl_dim_param, i, name.c_str());
    }

    return space;
}

__isl_give isl_basic_set *build_domain(
    const SCoP &scp,
    mlir::Operation *op,
    __isl_keep isl_ctx *ctx,
    __isl_keep isl_space *space)
{
    isl_basic_set *set = isl_basic_set_universe(isl_space_copy(space));

    auto is_op = [&op](const mlir::Operation &other)
    {
        return &other == op;
    };

    while (!contains_if(scp, is_op) && (op = op->getParentOp()))
    {
        if (auto loop = llvm::dyn_cast<scf::ForOp>(op))
        {
            isl_aff *dim = build_affine(loop.getInductionVar(), ctx, space);
            isl_aff *lower = build_affine(loop.getLowerBound(), ctx, space);
            isl_aff *upper = build_affine(loop.getUpperBound(), ctx, space);

            isl_basic_set *lb = isl_aff_le_basic_set(lower, isl_aff_copy(dim));
            isl_basic_set *ub = isl_aff_lt_basic_set(dim, upper);

            set = isl_basic_set_intersect(set, lb);
            set = isl_basic_set_intersect(set, ub);
        }
        else if (auto conditional = llvm::dyn_cast<scf::IfOp>(op))
        {
            if (auto condition = conditional.getCondition().getDefiningOp())
            {
                isl_basic_set *cond = build_set(condition, ctx, space);
                set = isl_basic_set_intersect(set, cond);
            }
            else
            {
                return isl_basic_set_free(set);
            }
        }
    }

    return set;
}

__isl_give isl_aff *build_affine(
    mlir::Value expr,
    __isl_keep isl_ctx *ctx,
    __isl_keep isl_space *space)
{
    auto op = expr.getDefiningOp();

    if (!op)
    {
        std::string name = print_value_as_operand(expr);

        auto find = [space, name = name.c_str()](isl_dim_type type)
        {
            return isl_space_find_dim_by_name(space, type, name);
        };

        isl_dim_type type;
        int pos;

        if (pos = find(isl_dim_set); pos >= 0)
            type = isl_dim_set;
        else if (pos = find(isl_dim_param); pos >= 0)
            type = isl_dim_param;
        else
            return nullptr;

        isl_local_space *ls = isl_local_space_from_space(isl_space_copy(space));
        isl_aff *aff = isl_aff_var_on_domain(ls, type, pos);

        return aff;
    }

    if (auto constant = llvm::dyn_cast<arith::ConstantIntOp>(op))
    {
        isl_val *val = isl_val_int_from_si(ctx, constant.value());
        isl_aff *aff = isl_aff_val_on_domain_space(isl_space_copy(space), val);

        return aff;
    }

    if (auto constant = llvm::dyn_cast<arith::ConstantIndexOp>(op))
    {
        isl_val *val = isl_val_int_from_si(ctx, constant.value());
        isl_aff *aff = isl_aff_val_on_domain_space(isl_space_copy(space), val);

        return aff;
    }

    if (auto cast = llvm::dyn_cast<arith::IndexCastOp>(op))
        return build_affine(cast.getIn(), ctx, space);

    if (auto sum = llvm::dyn_cast<arith::AddIOp>(op))
    {
        isl_aff *lhs = build_affine(sum.getLhs(), ctx, space);
        isl_aff *rhs = build_affine(sum.getRhs(), ctx, space);

        return isl_aff_add(lhs, rhs);
    }

    if (auto difference = llvm::dyn_cast<arith::SubIOp>(op))
    {
        isl_aff *lhs = build_affine(difference.getLhs(), ctx, space);
        isl_aff *rhs = build_affine(difference.getRhs(), ctx, space);

        return isl_aff_sub(lhs, rhs);
    }

    if (auto product = llvm::dyn_cast<arith::MulIOp>(op))
    {
        isl_aff *lhs = build_affine(product.getLhs(), ctx, space);
        isl_aff *rhs = build_affine(product.getRhs(), ctx, space);

        return isl_aff_mul(lhs, rhs);
    }

    return nullptr;
}

}   // namespace cs6120
