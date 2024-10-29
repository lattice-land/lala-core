// Copyright 2022 Pierre Talbot

#ifndef FIXPOINT_HPP
#define FIXPOINT_HPP

#include "logic/logic.hpp"
#include "b.hpp"
#include "battery/memory.hpp"
#include "battery/vector.hpp"

#ifdef __CUDACC__
  #include <cooperative_groups.h>
#endif

namespace lala {

/** A simple form of sequential fixpoint computation based on Kleene fixpoint.
 * At each iteration, the deduction operations \f$ f_1, \ldots, f_n \f$ are simply composed by functional composition \f$ f = f_n \circ \ldots \circ f_1 \f$.
 * This strategy basically corresponds to the Gauss-Seidel iteration method. */
class GaussSeidelIteration {
public:
  CUDA void barrier() {}

  template <class A>
  CUDA local::B iterate(A& a) {
    size_t n = a.num_deductions();
    bool has_changed = false;
    for(size_t i = 0; i < n; ++i) {
      has_changed |= a.deduce(i);
    }
    return has_changed;
  }

  template <class A>
  CUDA size_t fixpoint(A& a, local::B& has_changed) {
    size_t iterations = 0;
    local::B changed(true);
    while(changed && !a.is_bot()) {
      changed = iterate(a);
      has_changed.join(changed);
      iterations++;
    }
    return iterations;
  }

  template <class A>
  CUDA local::B fixpoint(A& a) {
    local::B has_changed(false);
    fixpoint(a, has_changed);
    return has_changed;
  }
};

#ifdef __CUDACC__

/** A simple form of fixpoint computation based on Kleene fixpoint.
 * At each iteration, the deduction operations \f$ f_1, \ldots, f_n \f$ are composed by parallel composition \f$ f = f_1 \| \ldots \| f_n \f$ meaning they are executed in parallel by different threads.
 * This is called an asynchronous iteration and it is due to (Cousot, Asynchronous iterative methods for solving a fixed point system of monotone equations in a complete lattice, 1977).
 * The underlying lattice on which we iterate must provide two methods:
 * - `a.deduce(int)`: call the ith deduction functions and returns `true` if `a` has changed.
 * - `a.num_deductions()`: return the number of deduction functions.
 * \tparam Group is a CUDA cooperative group class.
 * \tparam Memory is an atomic memory, that must be compatible with the cooperative group chosen (e.g., don't use atomic_memory_block if the group contains multiple blocks). */
template <class Group, class Memory>
class AsynchronousIterationGPU {
public:
  using memory_type = Memory;
  using group_type = Group;
private:
  using atomic_bool = B<memory_type>;
  atomic_bool changed[3];
  atomic_bool is_bot[3];
  Group group;

  CUDA void assert_cuda_arch() {
    printf("AsynchronousIterationGPU must be used on the GPU device only.\n");
    assert(0);
  }

  CUDA void reset() {
    changed[0].join(true);
    changed[1].meet(false);
    changed[2].meet(false);
    for(int i = 0; i < 3; ++i) {
      is_bot[i].meet(false);
    }
  }

public:
  CUDA AsynchronousIterationGPU(const Group& group):
    group(group)
  {}

  CUDA void barrier() {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
  #else
    cooperative_groups::sync(group);
  #endif
  }

  template <class A>
  CUDA bool iterate(A& a) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    size_t n = a.num_deductions();
    bool has_changed = false;
    for (size_t t = group.thread_rank(); t < n; t += group.num_threads()) {
      has_changed |= a.deduce(t);
      if((t-group.thread_rank()) + group.num_threads() < n) __syncwarp();
    }
    return has_changed;
  #endif
  }

  template <class A, class M>
  CUDA size_t fixpoint(A& a, B<M>& has_changed, volatile bool* stop) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return 0;
  #else
    reset();
    barrier();
    size_t i;
    for(i = 1; changed[(i-1)%3] && !is_bot[(i-1)%3]; ++i) {
      changed[i%3].join(iterate(a));
      changed[(i+1)%3].meet(false); // reinitialize changed for the next iteration.
      is_bot[i%3].join(a.is_bot());
      is_bot[i%3].join(*stop);
      barrier();
    }
    // It changes if we performed several iteration, or if the first iteration changed the abstract domain.
    has_changed.join(changed[1]);
    has_changed.join(changed[2]);
    return i - 1;
  #endif
  }

  template <class A>
  CUDA size_t fixpoint(A& a) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return 0;
  #else
    reset();
    barrier();
    size_t i;
    for(i = 1; changed[(i-1)%3] && !is_bot[(i-1)%3]; ++i) {
      changed[i%3].join(iterate(a));
      changed[(i+1)%3].meet(false); // reinitialize changed for the next iteration.
      is_bot[i%3].join(a.is_bot());
      barrier();
    }
    return i - 1;
  #endif
  }
};

// using BlockAsynchronousIterationGPU = AsynchronousIterationGPU<cooperative_groups::thread_block, battery::atomic_memory_block>;
using GridAsynchronousIterationGPU = AsynchronousIterationGPU<cooperative_groups::grid_group, battery::atomic_memory_grid>;

class BlockAsynchronousIterationGPU {
public:
  using memory_type = battery::atomic_memory_block;
private:
  using atomic_bool = B<memory_type>;
  atomic_bool changed[3];
  atomic_bool is_bot[3];

  CUDA void assert_cuda_arch() {
    printf("BlockAsynchronousIterationGPU must be used on the GPU device only.\n");
    assert(0);
  }

  CUDA void reset() {
    changed[0].join(true);
    changed[1].meet(false);
    changed[2].meet(false);
    for(int i = 0; i < 3; ++i) {
      is_bot[i].meet(false);
    }
  }

public:
  BlockAsynchronousIterationGPU() = default;

  CUDA void barrier() {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
  #else
    __syncthreads();
  #endif
  }

  template <class A>
  CUDA bool iterate(A& a) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    size_t n = a.num_deductions();
    bool has_changed = false;
    for (size_t t = threadIdx.x; t < n; t += blockDim.x) {
      has_changed |= a.deduce(t);
      if((t-threadIdx.x) + blockDim.x < n) __syncwarp();
    }
    return has_changed;
  #endif
  }

  template <class A, class M>
  CUDA size_t fixpoint(A& a, B<M>& has_changed, volatile bool* stop) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return 0;
  #else
    reset();
    barrier();
    size_t i;
    for(i = 1; changed[(i-1)%3] && !is_bot[(i-1)%3]; ++i) {
      changed[i%3].join(iterate(a));
      changed[(i+1)%3].meet(false); // reinitialize changed for the next iteration.
      is_bot[i%3].join(a.is_bot());
      is_bot[i%3].join(*stop);
      barrier();
    }
    // It changes if we performed several iteration, or if the first iteration changed the abstract domain.
    has_changed.join(changed[1]);
    has_changed.join(changed[2]);
    return i - 1;
  #endif
  }

  template <class A>
  CUDA size_t fixpoint(A& a) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return 0;
  #else
    reset();
    barrier();
    size_t i;
    for(i = 1; changed[(i-1)%3] && !is_bot[(i-1)%3]; ++i) {
      changed[i%3].join(iterate(a));
      changed[(i+1)%3].meet(false); // reinitialize changed for the next iteration.
      is_bot[i%3].join(a.is_bot());
      barrier();
    }
    return i - 1;
  #endif
  }

  template <class Alloc, class A>
  CUDA bool iterate(const battery::vector<int, Alloc>& indexes, A& a) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    assert(a.num_deductions() >= indexes.size());
    bool has_changed = false;
    for (size_t t = threadIdx.x; t < indexes.size(); t += blockDim.x) {
      has_changed |= a.deduce(indexes[t]);
      if((t-threadIdx.x) + blockDim.x < indexes.size()) __syncwarp();
    }
    return has_changed;
  #endif
  }

  template <class Alloc, class A>
  CUDA size_t fixpoint(const battery::vector<int, Alloc>& indexes, A& a) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return 0;
  #else
    reset();
    barrier();
    size_t i;
    for(i = 1; changed[(i-1)%3] && !is_bot[(i-1)%3]; ++i) {
      changed[i%3].join(iterate(indexes, a));
      changed[(i+1)%3].meet(false); // reinitialize changed for the next iteration.
      is_bot[i%3].join(a.is_bot());
      barrier();
    }
    return i - 1;
  #endif
  }
};

#endif

} // namespace lala

#endif
