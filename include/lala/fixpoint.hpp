// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_FIXPOINT_HPP
#define LALA_CORE_FIXPOINT_HPP

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

  /** We iterate the function `f` `n` times: \f$ f(0); f(1); \ldots ; f(n); \f$
   * \param `n` the number of call to `f`.
   * \param `bool f(size_t i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA local::B iterate(size_t n, const F& f) const {
    bool has_changed = false;
    for(size_t i = 0; i < n; ++i) {
      has_changed |= f(i);
    }
    return has_changed;
  }

  /** We execute `iterate(n, f)` until we reach a fixpoint or `must_stop()` returns `true`.
   * \param `n` the number of call to `f`.
   * \param `bool f(size_t i)` returns `true` if something has changed for `i`.
   * \param `bool must_stop()` returns `true` if we must stop early the fixpoint computation.
   * \param `has_changed` is set to `true` if we were not yet in a fixpoint.
   * \return The number of iterations required to reach a fixpoint or until `must_stop()` returns `true`.
  */
  template <class F, class StopFun, class M>
  CUDA size_t fixpoint(size_t n, const F& f, const StopFun& must_stop, B<M>& has_changed) {
    size_t iterations = 0;
    local::B changed(true);
    while(changed && !must_stop()) {
      changed = iterate(n, f);
      has_changed.join(changed);
      iterations++;
    }
    return iterations;
  }

  /** Same as `fixpoint` above without `has_changed`. */
  template <class F, class StopFun>
  CUDA size_t fixpoint(size_t n, const F& f, const StopFun& must_stop) {
    local::B has_changed(false);
    return fixpoint(n, f, must_stop, has_changed);
  }

  /** Same as `fixpoint` above with `must_stop` always returning `false`. */
  template <class F, class M>
  CUDA size_t fixpoint(size_t n, const F& f, B<M>& has_changed) {
    return fixpoint(n, f, [](){ return false; }, has_changed);
  }

  /** Same as `fixpoint` above without `has_changed` and with `must_stop` always returning `false`. */
  template <class F>
  CUDA size_t fixpoint(size_t n, const F& f) {
    local::B has_changed(false);
    return fixpoint(n, f, has_changed);
  }
};

#ifdef __CUDACC__

/** This fixpoint engine is parametrized by an iterator engine `I` providing a method `iterate(n,f)`, a barrier `barrier()` and a function `is_thread0()` returning `true` for a single thread.
 *  AsynchronousFixpoint provides a `fixpoint` function using `iterate` of `I`. */
template <class IteratorEngine>
class AsynchronousFixpoint {
  /** We do not use atomic because tearing is seemingly not possible in CUDA (according to information given by Nvidia engineers during a hackathon). */
  local::B changed[3];
  local::B stop[3];

  CUDA void reset() {
    if(is_thread0()) {
      changed[0] = true;
      changed[1] = false;
      changed[2] = false;
      for(int i = 0; i < 3; ++i) {
        stop[i] = false;
      }
    }
  }

public:
  CUDA INLINE bool is_thread0() {
    return static_cast<IteratorEngine*>(this)->is_thread0();
  }

  CUDA INLINE void barrier() {
    static_cast<IteratorEngine*>(this)->barrier();
  }

  template <class F>
  CUDA INLINE local::B iterate(size_t n, const F& f) const {
    return static_cast<const IteratorEngine*>(this)->iterate(n, f);
  }

  /** We execute `I::iterate(n, f)` until we reach a fixpoint or `must_stop()` returns `true`.
   * \param `n` the number of call to `f`.
   * \param `bool f(size_t i)` returns `true` if something has changed for `i`.
   * \param `bool must_stop()` returns `true` if we must stop early the fixpoint computation. This function is called by the first thread only.
   * \param `has_changed` is set to `true` if we were not yet in a fixpoint.
   * \return The number of iterations required to reach a fixpoint or until `must_stop()` returns `true`.
  */
  template <class F, class StopFun, class M>
  CUDA size_t fixpoint(size_t n, const F& f, const StopFun& must_stop, B<M>& has_changed) {
    reset();
    barrier();
    size_t i;
    for(i = 1; changed[(i-1)%3] && !stop[(i-1)%3]; ++i) {
      changed[i%3].join(iterate(n, f));
      if(is_thread0()) {
        changed[(i+1)%3].meet(false); // reinitialize changed for the next iteration.
        stop[i%3].join(must_stop());
      }
      barrier();
    }
    // It changes if we performed several iteration, or if the first iteration changed the abstract domain.
    if(is_thread0()) {
      has_changed.join(changed[1] || i > 2);
    }
    return i - 1;
  }

  /** Same as `fixpoint` above without `has_changed`. */
  template <class F, class StopFun>
  CUDA INLINE size_t fixpoint(size_t n, const F& f, const StopFun& must_stop) {
    local::B has_changed(false);
    return fixpoint(n, f, must_stop, has_changed);
  }

  /** Same as `fixpoint` above with `must_stop` always returning `false`. */
  template <class F, class M>
  CUDA INLINE size_t fixpoint(size_t n, const F& f, B<M>& has_changed) {
    return fixpoint(n, f, [](){ return false; }, has_changed);
  }

  /** Same as `fixpoint` above without `has_changed` and with `must_stop` always returning `false`. */
  template <class F>
  CUDA INLINE size_t fixpoint(size_t n, const F& f) {
    local::B has_changed(false);
    return fixpoint(n, f, has_changed);
  }

  /** Same as `fixpoint` with a new function defined by `g(i) = f(indexes[i])` and `n = indexes.size()`. */
  template <class Alloc, class F, class StopFun, class M>
  CUDA INLINE size_t fixpoint(const battery::vector<int, Alloc>& indexes, const F& f, const StopFun& must_stop, B<M>& has_changed) {
    return fixpoint(indexes.size(), [&](size_t i) { return f(indexes[i]); }, must_stop, has_changed);
  }

  /** Same as `fixpoint` with `g(i) = f(indexes[i])` and `n = indexes.size()`, without `has_changed`. */
  template <class Alloc, class F, class StopFun>
  CUDA INLINE size_t fixpoint(const battery::vector<int, Alloc>& indexes, const F& f, const StopFun& must_stop) {
    local::B has_changed(false);
    return fixpoint(indexes, f, must_stop, has_changed);
  }

  /** Same as `fixpoint` above with `must_stop` always returning `false`. */
  template <class Alloc, class F, class M>
  CUDA INLINE size_t fixpoint(const battery::vector<int, Alloc>& indexes, const F& f, B<M>& has_changed) {
    return fixpoint(indexes, f, [](){ return false; }, has_changed);
  }

  /** Same as `fixpoint` above without `has_changed` and with `must_stop` always returning `false`. */
  template <class Alloc, class F>
  CUDA INLINE size_t fixpoint(const battery::vector<int, Alloc>& indexes, const F& f) {
    local::B has_changed(false);
    return fixpoint(indexes, f, has_changed);
  }
};

/** A simple form of fixpoint computation based on Kleene fixpoint.
 * At each iteration, the functions \f$ f_1, \ldots, f_n \f$ are composed by parallel composition \f$ f = f_1 \| \ldots \| f_n \f$ meaning they are executed in parallel by different threads.
 * This is called an asynchronous iteration and it is due to (Cousot, Asynchronous iterative methods for solving a fixed point system of monotone equations in a complete lattice, 1977).
 * \tparam Group is a CUDA cooperative group class (note that we provide a more efficient implementation for block group below in `BlockAsynchronousIterationGPU`).
*/
template <class Group>
class AsynchronousIterationGPU : public AsynchronousFixpoint<AsynchronousIterationGPU<Group>> {
public:
  using group_type = Group;
private:
  Group group;

  CUDA void assert_cuda_arch() const {
    printf("AsynchronousIterationGPU must be used on the GPU device only.\n");
    assert(0);
  }

public:
  CUDA AsynchronousIterationGPU(const Group& group):
    group(group)
  {}

  CUDA void reset() const {}

  CUDA INLINE bool is_thread0() const {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
      return false;
  #else
    return group.thread_rank() == 0;
  #endif
  }

  /** A barrier used to synchronize the threads within the group between iterations. */
  CUDA INLINE void barrier() {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
  #else
    group.sync();
  #endif
  }

  /** The function `f` is called `n` times in parallel: \f$ f(0) \| f(1) \| \ldots \| f(n) \f$.
   * \param `n` the number of call to `f`.
   * \param `bool f(size_t i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA INLINE bool iterate(size_t n, const F& f) const {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    bool has_changed = false;
    for (size_t i = group.thread_rank(); i < n; i += group.num_threads()) {
      has_changed |= f(i);
    }
    return has_changed;
  #endif
  }
};

using GridAsynchronousFixpointGPU = AsynchronousIterationGPU<cooperative_groups::grid_group>;

/** An optimized version of `AsynchronousIterationGPU` when the fixpoint is computed on a single block.
 * We avoid the use of cooperative groups which take extra memory space.
 */
class BlockAsynchronousFixpointGPU : public AsynchronousFixpoint<BlockAsynchronousFixpointGPU> {
private:
  CUDA void assert_cuda_arch() const {
    printf("BlockAsynchronousFixpointGPU must be used on the GPU device only.\n");
    assert(0);
  }

public:
  BlockAsynchronousFixpointGPU() = default;

  CUDA INLINE bool is_thread0() const {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
      return false;
  #else
    return threadIdx.x == 0;
  #endif
  }

  CUDA INLINE void barrier() {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
  #else
    __syncthreads();
  #endif
  }

  /** The function `f` is called `n` times in parallel: \f$ f(0) \| f(1) \| \ldots \| f(n) \f$.
   * If `n` is greater than the number of threads in the block, we perform a stride loop, without synchronization between two iterations.
   * \param `n` the number of calls to `f`.
   * \param `bool f(size_t i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA INLINE bool iterate(size_t n, const F& f) const {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    bool has_changed = false;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
      has_changed |= f(i);
    }
    return has_changed;
  #endif
  }
};

#endif

} // namespace lala

#endif
