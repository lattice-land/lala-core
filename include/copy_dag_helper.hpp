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
template<class Alloc = battery::StandardAllocator, class FastAlloc = Alloc>
class AbstractDeps
{
public:
  using allocator_type = Alloc;
  using fast_allocator_type = FastAlloc;

private:
  struct dep_erasure {
    CUDA virtual ~dep_erasure() {}
  };

  template <class A>
  struct dep_holder : dep_erasure {
    battery::shared_ptr<A, Alloc> a;
    CUDA dep_holder(A* ptr, const Alloc& alloc): a(ptr, alloc) {}
    CUDA virtual ~dep_holder() {}
  };

  fast_allocator_type falloc;
  battery::vector<battery::unique_ptr<dep_erasure, allocator_type>, allocator_type> deps;

public:
  CUDA AbstractDeps(const allocator_type& alloc = allocator_type(),
    const fast_allocator_type& falloc = fast_allocator_type())
  : deps(alloc), falloc(falloc) {}

  CUDA size_t size() const {
    return deps.size();
  }

  template<class A>
  CUDA battery::shared_ptr<A, Alloc> extract(AType aty) {
    assert(aty != UNTYPED);
    assert(deps.size() > aty);
    assert(deps[aty]);
    return static_cast<dep_holder<A>*>(deps[aty].get())->a;
  }

  template<class A, class FromAlloc>
  CUDA battery::shared_ptr<A, Alloc> clone(const battery::shared_ptr<A, FromAlloc>& a)
  {
    if(!a) {
      return nullptr;
    }
    assert(a->aty() != UNTYPED); // Abstract domain must all have a unique identifier to be copied.
    // If the dependency is not in the list, we copy it and add it.
    if(deps.size() <= a->aty() || !static_cast<bool>(deps[a->aty()])) {
      deps.resize(battery::max((int)deps.size(), a->aty()+1));
      Alloc to_alloc = deps.get_allocator();
      deps[a->aty()] = battery::unique_ptr<dep_erasure, Alloc>(
        new(to_alloc) dep_holder<A>(
          new(to_alloc) A(*a, *this),
          to_alloc),
        to_alloc);
      // NOTE: Since we are copying a DAG, `A(*a, *this)` or one of its dependency cannot create `deps[a->aty()]`.
    }
    return extract<A>(a->aty());
  }

  CUDA allocator_type get_allocator() const {
    return deps.get_allocator();
  }

  CUDA fast_allocator_type get_fast_allocator() const {
    return falloc;
  }
};

}

#endif
