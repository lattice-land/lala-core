// Copyright 2021 Pierre Talbot

#ifndef ABSTRACT_DEPS_HPP
#define ABSTRACT_DEPS_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "logic/ast.hpp"

namespace lala {

template <class A>
using abstract_ptr = battery::shared_ptr<A, typename A::allocator_type>;

/** Abstract domains are organized in a directed-acyclic graph (DAG).
 * Therefore, it is not straighforward to copy the whole hierarchy, because the child of a node may be shared by another node, and might have already been copied.
 * This class is useful to copy hierarchy of abstract domains.
 * Moreover, the allocators between the original and the copied hierarchy can be different.
 *
 * The first allocator of the list is used for the internal allocations of this class.
 */
template<class... Allocators>
class AbstractDeps
{
public:
  constexpr static size_t n = battery::tuple_size<battery::tuple<Allocators...>>{};
  static_assert(n > 0, "AbstractDeps must have a non-empty list of allocators.");

  using allocators_type = battery::tuple<Allocators...>;
  using allocator_type = typename battery::tuple_element<0, battery::tuple<Allocators...>>::type;

private:
  struct dep_erasure {
    CUDA virtual ~dep_erasure() {}
  };

  template <class A>
  struct dep_holder : dep_erasure {
    abstract_ptr<A> a;
    CUDA dep_holder(A* ptr): a(ptr, ptr->get_allocator()) {}
    CUDA virtual ~dep_holder() {}
  };

  allocators_type allocators;
  battery::vector<battery::unique_ptr<dep_erasure, allocator_type>, allocator_type> deps;

public:
  CUDA AbstractDeps(const Allocators&... allocators)
  : allocators(allocators...)
  , deps(battery::get<0>(this->allocators))
  {}

  CUDA size_t size() const {
    return deps.size();
  }

  template<class A>
  CUDA abstract_ptr<A> extract(AType aty) {
    assert(aty != UNTYPED);
    assert(deps.size() > aty);
    assert(deps[aty]);
    return static_cast<dep_holder<A>*>(deps[aty].get())->a;
  }

  template<class A2, class A>
  CUDA abstract_ptr<A2> clone(const abstract_ptr<A>& a)
  {
    auto to_alloc = battery::get<typename A2::allocator_type>(allocators);
    if(!a) {
      return abstract_ptr<A2>{to_alloc};
    }
    assert(a->aty() != UNTYPED); // Abstract domain must all have a unique identifier to be copied.
    // If the dependency is not in the list, we copy it and add it.
    if(deps.size() <= a->aty() || !static_cast<bool>(deps[a->aty()])) {
      deps.resize(battery::max((int)deps.size(), a->aty()+1));
      allocator_type internal_alloc = battery::get<0>(allocators);
      A2* a2 = static_cast<A2*>(to_alloc.allocate(sizeof(A2)));
      new(a2) A2(*a, *this);
      dep_holder<A2>* dep_hold = static_cast<dep_holder<A2>*>(internal_alloc.allocate(sizeof(dep_holder<A2>)));
      new(dep_hold) dep_holder<A2>(a2);
      deps[a->aty()] = battery::unique_ptr<dep_erasure, allocator_type>(
        dep_hold, internal_alloc);
      // NOTE: Since we are copying a DAG, `A(*a, *this)` or one of its dependency cannot create `deps[a->aty()]`.
    }
    return extract<A2>(a->aty());
  }

  template <class Alloc>
  CUDA Alloc get_allocator() const {
    return battery::get<Alloc>(allocators);
  }
};

}

#endif
