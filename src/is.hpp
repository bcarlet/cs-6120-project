#ifndef CS6120_IS_HPP_
#define CS6120_IS_HPP_

#include <cstdlib>
#include <memory>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/space.h>

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#define CS6120_DECLARE_UNIQUE(ISL_TYPE) \
    using unique_ ## ISL_TYPE = \
        std::unique_ptr<ISL_TYPE, decltype(&ISL_TYPE ## _free)>

namespace cs6120
{

class SCoP;

CS6120_DECLARE_UNIQUE(isl_aff);
CS6120_DECLARE_UNIQUE(isl_basic_set);
CS6120_DECLARE_UNIQUE(isl_ctx);
CS6120_DECLARE_UNIQUE(isl_space);

using unique_c_str = std::unique_ptr<char, decltype(&std::free)>;

__isl_give isl_space *build_space(
    const SCoP &scp,
    mlir::Operation *op,
    __isl_keep isl_ctx *ctx);

__isl_give isl_basic_set *build_domain(
    const SCoP &scp,
    mlir::Operation *op,
    __isl_keep isl_ctx *ctx,
    __isl_keep isl_space *space);

__isl_give isl_aff *build_affine(
    mlir::Value expr,
    __isl_keep isl_ctx *ctx,
    __isl_keep isl_space *space);

}   // namespace cs6120

#endif
