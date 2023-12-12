#ifndef CS6120_UTILS_HPP_
#define CS6120_UTILS_HPP_

#include <algorithm>

namespace cs6120
{

template<class C, class V>
inline bool contains(const C &collection, const V &value)
{
    auto it = std::find(collection.begin(), collection.end(), value);

    return it != collection.end();
}

template<class C, class P>
inline bool contains_if(const C &collection, P pred)
{
    auto it = std::find_if(collection.begin(), collection.end(), pred);

    return it != collection.end();
}

}   // namespace cs6120

#endif
