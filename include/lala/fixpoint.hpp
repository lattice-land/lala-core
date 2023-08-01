// Copyright 2022 Pierre Talbot

#ifndef FIXPOINT_HPP
#define FIXPOINT_HPP

#include "logic/logic.hpp"
#include "universes/primitive_upset.hpp"
#include "battery/memory.hpp"

#ifdef __NVCC__
  #include <cooperative_groups.h>
#endif

namespace lala {

/** A simple form of sequential fixpoint computation based on Kleene fixpoint.
 * At each iteration, the refinement operations \f$ f_1, \ldots, f_n \f$ are simply composed by functional composition \f$ f = f_n \circ \ldots \circ f_1 \f$.
 * This strategy basically corresponds to the Gauss-Seidel iteration method. */
class GaussSeidelIteration {
public:
  CUDA void barrier() {}

  template <class A>
  CUDA void iterate(A& a, local::BInc& has_changed) {
    size_t n = a.num_refinements();
    for(size_t i = 0; i < n; ++i) {
      a.refine(i, has_changed);
    }
  }

  template <class A>
  CUDA void fixpoint(A& a, local::BInc& has_changed) {
    local::BInc changed(true);
    while(changed) {
      changed.dtell_bot();
      iterate(a, changed);
      has_changed.tell(changed);
    }
  }

  template <class A>
  CUDA local::BInc fixpoint(A& a) {
    local::BInc has_changed(false);
    fixpoint(a, has_changed);
    return has_changed;
  }
};

#ifdef __NVCC__

/** A simple form of fixpoint computation based on Kleene fixpoint.
 * At each iteration, the refinement operations \f$ f_1, \ldots, f_n \f$ are composed by parallel composition \f$ f = f_1 \| \ldots \| f_n \f$ meaning they are executed in parallel by different threads.
 * This is called an asynchronous iteration and it is due to (Cousot, Asynchronous iterative methods for solving a fixed point system of monotone equations in a complete lattice, 1977).
 * The underlying lattice on which we iterate must provide two methods:
 * - `a.refine(int, BInc&)`: call the ith refinement functions and set `has_changed` to `true` if `a` has changed. Note that if `a.is_top()` is `true`, then `has_changed` must stay false for all refinement functions.
 * - `a.num_refinements()`: return the number of refinement functions.
 * \tparam Group is a CUDA cooperative group class.
 * \tparam Memory is an atomic memory, that must be compatible with the cooperative group chosen (e.g., don't use atomic_memory_block if the group contains multiple blocks). */
template <class Group, class Memory, class Allocator>
class AsynchronousIterationGPU {
public:
  using memory_type = Memory;
  using allocator_type = Allocator;
  using group_type = Group;
private:
  using atomic_binc = BInc<memory_type>;
  battery::vector<atomic_binc, allocator_type> changed;
  Group group;

  CUDA void assert_cuda_arch() {
    printf("AsynchronousIterationGPU must be used on the GPU device only.\n");
    assert(0);
  }

  CUDA void reset() {
    changed[0].tell_top();
    changed[1].dtell_bot();
    changed[2].dtell_bot();
  }

public:
  CUDA AsynchronousIterationGPU(const Group& group, const allocator_type& alloc = allocator_type()):
    group(group), changed(3, alloc)
  {}

  CUDA void barrier() {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
  #else
    cooperative_groups::sync(group);
  #endif
  }

  template <class A, class M>
  CUDA void iterate(A& a, BInc<M>& has_changed) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
  #else
    size_t n = a.num_refinements();
    for (size_t t = group.thread_rank(); t < n; t += group.num_threads()) {
      a.refine(t, has_changed);
    }
  #endif
  }

  template <class A, class M>
  CUDA size_t fixpoint(A& a, BInc<M>& has_changed) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return 0;
  #else
    reset();
    barrier();
    size_t i;
    for(i = 1; changed[(i-1)%3]; ++i) {
      iterate(a, changed[i%3]);
      changed[(i+1)%3].dtell_bot(); // reinitialize changed for the next iteration.
      barrier();
    }
    // It changes if we performed several iteration, or if the first iteration changed the abstract domain.
    has_changed.tell(changed[1]);
    has_changed.tell(changed[2]);
    return i - 1;
  #endif
  }

  template <class A>
  CUDA local::BInc fixpoint(A& a) {
    local::BInc has_changed(false);
    fixpoint(a, has_changed);
    return has_changed;
  }
};

template <class Allocator>
using BlockAsynchronousIterationGPU = AsynchronousIterationGPU<cooperative_groups::thread_block, battery::atomic_memory_block, Allocator>;

template <class Allocator>
using GridAsynchronousIterationGPU = AsynchronousIterationGPU<cooperative_groups::grid_group, battery::atomic_memory_grid, Allocator>;

#endif

} // namespace lala

#endif
