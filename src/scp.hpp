#ifndef CS6120_SCP_HPP_
#define CS6120_SCP_HPP_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace cs6120
{

class SCoP
{
public:
    using iterator = mlir::Block::iterator;

    SCoP(iterator begin, iterator end) :
        m_begin(begin),
        m_end(end)
    {
    }

    iterator begin() const { return m_begin; }
    iterator end() const { return m_end; }

    void set_begin(iterator begin) { m_begin = begin; }
    void set_end(iterator end) { m_end = end; }

private:
    iterator m_begin;
    iterator m_end;
};

std::unique_ptr<mlir::Pass> create_scp_pass();

}   // namespace cs6120

#endif
