#ifndef CS6120_DEP_HPP_
#define CS6120_DEP_HPP_

#include <isl/ctx.h>
#include <isl/set.h>

#include "mlir/IR/Operation.h"

namespace cs6120
{

class SCoP;

__isl_give isl_basic_set *build_dependence_polyhedron(
    const SCoP &scp,
    mlir::Operation *s,
    mlir::Operation *r,
    isl_size level,
    __isl_keep isl_ctx *ctx);

isl_bool is_dependence(__isl_keep isl_basic_set *polyhedron);

}   // namespace cs6120

#endif
