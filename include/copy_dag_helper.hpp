// Copyright 2021 Pierre Talbot

#ifndef COPY_DAG_HELPER_HPP
#define COPY_DAG_HELPER_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "string.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "logic/ast.hpp"

namespace lala {

/** The dependencies list of the abstract domains DAG when copying abstract domains. */
template<class Alloc = battery::StandardAllocator>
class AbstractDeps
{
  struct dep_erasure {
    CUDA virtual ~dep_erasure() {}
  };

  template <class A>
  struct dep_holder : dep_erasure {
    battery::shared_ptr<A, Alloc> a;
    CUDA dep_holder(A* ptr, const Alloc& alloc): a(ptr, alloc) {}
    CUDA virtual ~dep_holder() {}
  };

  battery::vector<battery::unique_ptr<dep_erasure, Alloc>, Alloc> deps;

public:
  using allocator_type = Alloc;

  CUDA AbstractDeps(const Alloc& alloc = Alloc()): deps(alloc) {}

  CUDA size_t size() const {
    return deps.size();
  }

  template<class A>
  CUDA battery::shared_ptr<A, Alloc> extract(AType uid) {
    assert(uid != UNTYPED);
    assert(deps.size() > uid);
    assert(deps[uid]);
    return static_cast<dep_holder<A>*>(deps[uid].get())->a;
  }

  template<class A, class FromAlloc>
  CUDA battery::shared_ptr<A, Alloc> clone(const battery::shared_ptr<A, FromAlloc>& a)
  {
    if(!a) {
      return nullptr;
    }
    assert(a->uid() != UNTYPED); // Abstract domain must all have a unique identifier to be copied.
    // If the dependency is not in the list, we copy it and add it.
    if(deps.size() <= a->uid() || !static_cast<bool>(deps[a->uid()])) {
      deps.resize(battery::max((int)deps.size(), a->uid()+1));
      Alloc to_alloc = deps.get_allocator();
      deps[a->uid()] = battery::unique_ptr<dep_erasure, Alloc>(
        new(to_alloc) dep_holder<A>(
          new(to_alloc) A(*a, *this),
          to_alloc),
        to_alloc);
      // NOTE: Since we are copying a DAG, `A(*a, *this)` or one of its dependency cannot create `deps[a->uid()]`.
    }
    return extract<A>(a->uid());
  }

  CUDA allocator_type get_allocator() const {
    return deps.get_allocator();
  }
};

}

#endif
