// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_ABSTRACT_DEPS_HPP
#define LALA_CORE_ABSTRACT_DEPS_HPP

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
 * A node of the hierarchy is identified by its address: two `abstract_ptr` pointing to the same object are copied once, and the copies share the same object.
 * Two distinct objects are always copied separately.
 * The addresses are only compared, never dereferenced, hence the original and the copied hierarchy can live in different memory spaces.
 *
 * Preconditions:
 *   - The hierarchy is acyclic.
 *   - The original hierarchy stays alive while it is copied.
 *   - A same object is always copied to the same type `A2` (in `clone<A2>`).
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
  /** `srcs[i]` is the address of the original element that was copied into `holders[i]`. */
  battery::vector<const void*, allocator_type> srcs;
  battery::vector<battery::unique_ptr<dep_erasure, allocator_type>, allocator_type> holders;

  /** If the hierarchy to be copied is the root of the search tree, the children can share some elements with the root.
   * For instance, this is the case of propagators in `PC`.
   * This enables to share data among GPU blocks to avoid duplicating similar data, and to ease the contention on L2 cache. */
  bool shared_copy;

  /** \return The index of `src` in `srcs`, or `-1` if `src` has not been copied yet. */
  CUDA int index_of(const void* src) const {
    for(int i = 0; i < srcs.size(); ++i) {
      if(srcs[i] == src) {
        return i;
      }
    }
    return -1;
  }

  template<class A>
  CUDA abstract_ptr<A> extract(int i) const {
    assert(i >= 0 && i < holders.size());
    assert(holders[i]);
    return static_cast<dep_holder<A>*>(holders[i].get())->a;
  }

public:
  CUDA AbstractDeps(bool shared_copy, const Allocators&... allocators)
  : shared_copy(shared_copy)
  , allocators(allocators...)
  , srcs(battery::get<0>(this->allocators))
  , holders(battery::get<0>(this->allocators))
  {}

  CUDA AbstractDeps(const Allocators&... allocators)
  : AbstractDeps(false, allocators...) {}

  /** \return The number of distinct elements copied so far. */
  CUDA size_t size() const {
    return holders.size();
  }

  CUDA bool is_shared_copy() const {
    return shared_copy;
  }

  template<class A2, class A>
  CUDA NI abstract_ptr<A2> clone(const abstract_ptr<A>& a)
  {
    auto to_alloc = battery::get<typename A2::allocator_type>(allocators);
    if(!a) {
      return abstract_ptr<A2>{to_alloc};
    }
    const void* src = a.get();
    int i = index_of(src);
    if(i != -1) {
      return extract<A2>(i);
    }
    // The element has not been copied yet, so we copy it and add it.
    // NOTE: The constructor `A2(*a, *this)` copies the dependencies of `a` by calling `clone` recursively, which grows `srcs` and `holders`.
    // Therefore, we must not keep an index or a reference into these vectors across this call, and we only register `a` once it is fully copied.
    // Since we are copying a DAG, this recursion never reaches `a` again.
    allocator_type internal_alloc = battery::get<0>(allocators);
    A2* a2 = static_cast<A2*>(to_alloc.allocate(sizeof(A2)));
    new(a2) A2(*a, *this);
    assert(index_of(src) == -1);
    dep_holder<A2>* dep_hold = static_cast<dep_holder<A2>*>(internal_alloc.allocate(sizeof(dep_holder<A2>)));
    new(dep_hold) dep_holder<A2>(a2);
    srcs.push_back(src);
    holders.push_back(battery::unique_ptr<dep_erasure, allocator_type>(dep_hold, internal_alloc));
    return dep_hold->a;
  }

  template <class Alloc>
  CUDA Alloc get_allocator() const {
    return battery::get<Alloc>(allocators);
  }
};

}

#endif
