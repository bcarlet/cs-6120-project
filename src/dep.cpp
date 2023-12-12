#include "dep.hpp"

#include <utility>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/local_space.h>
#include <isl/set.h>
#include <isl/space.h>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "is.hpp"
#include "scp.hpp"

namespace memref = mlir::memref;

namespace
{

mlir::Value get_mem_ref_or_none(mlir::Operation *op)
{
    if (auto load = llvm::dyn_cast<memref::LoadOp>(op))
        return load.getMemRef();

    if (auto store = llvm::dyn_cast<memref::StoreOp>(op))
        return store.getMemRef();

    return mlir::Value();
}

mlir::Operation::operand_range get_indices(mlir::Operation *op)
{
    if (auto load = llvm::dyn_cast<memref::LoadOp>(op))
        return load.getIndices();
    else
        return llvm::cast<memref::StoreOp>(op).getIndices();
}

std::pair<__isl_give isl_aff *, __isl_give isl_aff *> product(
    __isl_take isl_aff *lhs,
    __isl_take isl_aff *rhs)
{
    isl_multi_aff *lm = isl_multi_aff_from_aff(lhs);
    isl_multi_aff *rm = isl_multi_aff_from_aff(rhs);

    isl_multi_aff *product = isl_multi_aff_product(lm, rm);
    product = isl_multi_aff_flatten_domain(product);

    isl_aff *lc = isl_multi_aff_get_at(product, 0);
    isl_aff *rc = isl_multi_aff_get_at(product, 1);

    isl_multi_aff_free(product);

    return std::make_pair(lc, rc);
}

__isl_give isl_basic_set *build_subscript_constraint(
    __isl_take isl_aff *lhs,
    __isl_take isl_aff *rhs)
{
    auto [lc, rc] = product(lhs, rhs);

    return isl_aff_eq_basic_set(lc, rc);
}

__isl_give isl_basic_set *build_subscript_constraint(
    mlir::Operation::operand_range lhs,
    mlir::Operation::operand_range rhs,
    __isl_keep isl_ctx *ctx,
    __isl_keep isl_space *left_space,
    __isl_keep isl_space *right_space)
{
    isl_space *left_aligned = isl_space_align_params(
        isl_space_copy(left_space),
        isl_space_copy(right_space));

    isl_space *right_aligned = isl_space_align_params(
        isl_space_copy(right_space),
        isl_space_copy(left_aligned));

    isl_space *product_space = isl_space_product(left_aligned, right_aligned);
    product_space = isl_space_flatten_range(product_space);

    isl_basic_set *set = isl_basic_set_universe(product_space);

    for (auto l = lhs.begin(), r = rhs.begin(); l != lhs.end() && r != rhs.end(); ++l, ++r)
    {
        isl_aff *la = cs6120::build_affine(*l, ctx, left_space);
        isl_aff *ra = cs6120::build_affine(*r, ctx, right_space);

        isl_basic_set *constraint = build_subscript_constraint(la, ra);

        set = isl_basic_set_intersect(set, constraint);
    }

    return set;
}

}   // namespace

namespace cs6120
{

__isl_give isl_basic_set *build_dependence_polyhedron(
    const SCoP &scp,
    mlir::Operation *s,
    mlir::Operation *r,
    isl_size level,
    __isl_keep isl_ctx *ctx)
{
    auto m1 = get_mem_ref_or_none(s);
    auto m2 = get_mem_ref_or_none(r);

    if (!m1 || m1 != m2)
        return nullptr;

    isl_space *s_space = build_space(scp, s, ctx);
    isl_space *r_space = build_space(scp, r, ctx);

    isl_basic_set *subscripts = build_subscript_constraint(
        get_indices(s), get_indices(r), ctx, s_space, r_space);

    isl_basic_set *s_domain = build_domain(scp, s, ctx, s_space);
    isl_basic_set *r_domain = build_domain(scp, r, ctx, r_space);

    s_domain = isl_basic_set_align_params(s_domain, isl_basic_set_get_space(r_domain));
    r_domain = isl_basic_set_align_params(r_domain, isl_basic_set_get_space(s_domain));

    isl_basic_set *domains = isl_basic_set_flat_product(s_domain, r_domain);

    isl_basic_set *polyhedron = isl_basic_set_intersect(subscripts, domains);

    isl_local_space *s_ls = isl_local_space_from_space(s_space);
    isl_local_space *r_ls = isl_local_space_from_space(r_space);

    isl_size s_dims = isl_local_space_dim(s_ls, isl_dim_set);
    isl_size r_dims = isl_local_space_dim(r_ls, isl_dim_set);

    for (isl_size i = 0; i < level; i++)
    {
        isl_aff *s_aff = isl_aff_var_on_domain(isl_local_space_copy(s_ls), isl_dim_set, s_dims - i - 1);
        isl_aff *r_aff = isl_aff_var_on_domain(isl_local_space_copy(r_ls), isl_dim_set, r_dims - i - 1);

        auto [s_c, r_c] = product(s_aff, r_aff);

        isl_basic_set *precedence = isl_aff_eq_basic_set(s_c, r_c);
        polyhedron = isl_basic_set_intersect(polyhedron, precedence);
    }

    if (s_dims - level > 1 && r_dims - level > 1)
    {
        isl_id *x1 = isl_local_space_get_dim_id(s_ls, isl_dim_set, s_dims - level - 2);
        isl_id *x2 = isl_local_space_get_dim_id(r_ls, isl_dim_set, s_dims - level - 2);

        if (x1 == x2)
        {
            isl_aff *s_aff = isl_aff_var_on_domain(s_ls, isl_dim_set, s_dims - level - 1);
            isl_aff *r_aff = isl_aff_var_on_domain(r_ls, isl_dim_set, r_dims - level - 1);

            auto [s_c, r_c] = product(s_aff, r_aff);

            isl_basic_set *precedence = isl_aff_lt_basic_set(s_c, r_c);
            polyhedron = isl_basic_set_intersect(polyhedron, precedence);

            s_ls = nullptr;
            r_ls = nullptr;
        }

        isl_id_free(x1);
        isl_id_free(x2);
    }

    isl_local_space_free(s_ls);
    isl_local_space_free(r_ls);

    return polyhedron;
}

isl_bool is_dependence(__isl_keep isl_basic_set *polyhedron)
{
    return isl_bool_not(isl_basic_set_is_empty(polyhedron));
}

}   // namespace cs6120
