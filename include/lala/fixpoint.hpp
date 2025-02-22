// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_FIXPOINT_HPP
#define LALA_CORE_FIXPOINT_HPP

#include "logic/logic.hpp"
#include "b.hpp"
#include "battery/memory.hpp"
#include "battery/vector.hpp"

#ifdef __CUDACC__
  #include <cooperative_groups.h>
  #include <cub/block/block_scan.cuh>
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
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA local::B iterate(int n, const F& f) const {
    bool has_changed = false;
    for(int i = 0; i < n; ++i) {
      has_changed |= f(i);
    }
    return has_changed;
  }

  /** We execute `iterate(n, f)` until we reach a fixpoint or `must_stop()` returns `true`.
   * \param `n` the number of call to `f`.
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \param `bool must_stop()` returns `true` if we must stop early the fixpoint computation.
   * \param `has_changed` is set to `true` if we were not yet in a fixpoint.
   * \return The number of iterations required to reach a fixpoint or until `must_stop()` returns `true`.
  */
  template <class F, class StopFun, class M>
  CUDA int fixpoint(int n, const F& f, const StopFun& must_stop, B<M>& has_changed) {
    int iterations = 0;
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
  CUDA int fixpoint(int n, const F& f, const StopFun& must_stop) {
    local::B has_changed(false);
    return fixpoint(n, f, must_stop, has_changed);
  }

  /** Same as `fixpoint` above with `must_stop` always returning `false`. */
  template <class F, class M>
  CUDA int fixpoint(int n, const F& f, B<M>& has_changed) {
    return fixpoint(n, f, [](){ return false; }, has_changed);
  }

  /** Same as `fixpoint` above without `has_changed` and with `must_stop` always returning `false`. */
  template <class F>
  CUDA int fixpoint(int n, const F& f) {
    local::B has_changed(false);
    return fixpoint(n, f, has_changed);
  }
};


/** Add the ability to deactive functions in a fixpoint computation.
 * Given a function `g`, we select only the functions \f$ f_{i_1} ; \ldots ; f_{i_k} \f$ for which \f$ g(i_k) \f$ is `true`, and compute subsequent fixpoint without them.
 */
template <class FixpointEngine>
class FixpointSubsetCPU {
private:
  FixpointEngine fp_engine;

  /** The indexes of all functions. */
  battery::vector<int> indexes;

  /** The active subset of the functions is from 0..n-1. */
  int n;

public:
  FixpointSubsetCPU(int n) : n(n), indexes(n) {
    for(int i = 0; i < n; ++i) {
      indexes[i] = i;
    }
  }

  template <class F>
  bool iterate(const F& f) {
    return fp_engine.iterate(n, [&](int i) { return f(indexes[i]); });
  }

  template <class F>
  int fixpoint(const F& f) {
    return fp_engine.fixpoint(n, [&](int i) { return f(indexes[i]); });
  }

  template <class F, class StopFun>
  int fixpoint(const F& f, const StopFun& g) {
    return fp_engine.fixpoint(n, [&](int i) { return f(indexes[i]); }, g);
  }

  template <class F, class StopFun, class M>
  int fixpoint(const F& f, const StopFun& g, B<M>& has_changed) {
    return fp_engine.fixpoint(n, [&](int i) { return f(indexes[i]); }, g);
  }

  /** \return the number of active functions. */
  int num_active() const {
    return n;
  }

  void reset() {
    n = indexes.size();
  }

  /** Compute the subset of the functions that are still active.
   * The subsequent call to `fixpoint` will only consider the function `f_i` for which `g(i)` is `true`. */
  template <class G>
  void select(const G& g) {
    for(int i = 0; i < n; ++i) {
      if(!g(indexes[i])) {
        battery::swap(indexes[i], indexes[--n]);
        i--;
      }
    }
  }

  using snapshot_type = int;

  snapshot_type snapshot() const {
    return snapshot_type(n);
  }

  void restore(const snapshot_type& snap) {
    n = snap;
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
  CUDA INLINE local::B iterate(int n, const F& f) const {
    return static_cast<const IteratorEngine*>(this)->iterate(n, f);
  }

  /** We execute `I::iterate(n, f)` until we reach a fixpoint or `must_stop()` returns `true`.
   * \param `n` the number of call to `f`.
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \param `bool must_stop()` returns `true` if we must stop early the fixpoint computation. This function is called by the first thread only.
   * \param `has_changed` is set to `true` if we were not yet in a fixpoint.
   * \return The number of iterations required to reach a fixpoint or until `must_stop()` returns `true`.
  */
  template <class F, class StopFun, class M>
  CUDA int fixpoint(int n, const F& f, const StopFun& must_stop, B<M>& has_changed) {
    reset();
    barrier();
    int i;
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

  template <class F, class Iter, class StopFun>
  CUDA int fixpoint(int n, const F& f, const Iter& h, const StopFun& must_stop) {
    reset();
    barrier();
    int i;
    for(i = 1; changed[(i-1)%3] && !stop[(i-1)%3]; ++i) {
      changed[i%3].join(iterate(n, f));
      if(is_thread0()) {
        changed[(i+1)%3].meet(false); // reinitialize changed for the next iteration.
        stop[i%3].join(must_stop());
      }
      barrier();
      h();
      barrier();
    }
    return i - 1;
  }

  template <class Alloc, class F, class Iter, class StopFun>
  CUDA INLINE int fixpoint(const battery::vector<int, Alloc>& indexes, const F& f, const Iter& h, const StopFun& must_stop) {
    return fixpoint(indexes.size(), [&](int i) { return f(indexes[i]); }, h, must_stop);
  }

  /** Same as `fixpoint` above without `has_changed`. */
  template <class F, class StopFun>
  CUDA INLINE int fixpoint(int n, const F& f, const StopFun& must_stop) {
    local::B has_changed(false);
    return fixpoint(n, f, must_stop, has_changed);
  }

  /** Same as `fixpoint` above with `must_stop` always returning `false`. */
  template <class F, class M>
  CUDA INLINE int fixpoint(int n, const F& f, B<M>& has_changed) {
    return fixpoint(n, f, [](){ return false; }, has_changed);
  }

  /** Same as `fixpoint` above without `has_changed` and with `must_stop` always returning `false`. */
  template <class F>
  CUDA INLINE int fixpoint(int n, const F& f) {
    local::B has_changed(false);
    return fixpoint(n, f, has_changed);
  }

  /** Same as `fixpoint` with a new function defined by `g(i) = f(indexes[i])` and `n = indexes.size()`. */
  template <class Alloc, class F, class StopFun, class M>
  CUDA INLINE int fixpoint(const battery::vector<int, Alloc>& indexes, const F& f, const StopFun& must_stop, B<M>& has_changed) {
    return fixpoint(indexes.size(), [&](int i) { return f(indexes[i]); }, must_stop, has_changed);
  }

  /** Same as `fixpoint` with `g(i) = f(indexes[i])` and `n = indexes.size()`, without `has_changed`. */
  template <class Alloc, class F, class StopFun>
  CUDA INLINE int fixpoint(const battery::vector<int, Alloc>& indexes, const F& f, const StopFun& must_stop) {
    local::B has_changed(false);
    return fixpoint(indexes, f, must_stop, has_changed);
  }

  /** Same as `fixpoint` above with `must_stop` always returning `false`. */
  template <class Alloc, class F, class M>
  CUDA INLINE int fixpoint(const battery::vector<int, Alloc>& indexes, const F& f, B<M>& has_changed) {
    return fixpoint(indexes, f, [](){ return false; }, has_changed);
  }

  /** Same as `fixpoint` above without `has_changed` and with `must_stop` always returning `false`. */
  template <class Alloc, class F>
  CUDA INLINE int fixpoint(const battery::vector<int, Alloc>& indexes, const F& f) {
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
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA INLINE bool iterate(int n, const F& f) const {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    bool has_changed = false;
    for (int i = group.thread_rank(); i < n; i += group.num_threads()) {
      has_changed |= f(i);
    }
    return has_changed;
  #endif
  }
};

using GridAsynchronousFixpointGPU = AsynchronousIterationGPU<cooperative_groups::grid_group>;

/** An optimized version of `AsynchronousIterationGPU` when the fixpoint is computed on a single block.
 * We avoid the use of cooperative groups which take extra memory space.
 * `syncwarp` is a boolean to tell if `f` in `iterate` is syncing the warp or not, if it does and syncwarp is `true`, `iterate` will always iterate to a multiple of 32 threads by repeating the last index if necessary.
 */
template <bool syncwarp = false>
class BlockAsynchronousFixpointGPU : public AsynchronousFixpoint<BlockAsynchronousFixpointGPU<syncwarp>> {
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

  /** The function `f` is called `n` times in parallel: \f$ f(0) \| f(1) \| \ldots \| f(n-1) \f$.
   * If `n` is greater than the number of threads in the block, we perform a stride loop, without synchronization between two iterations.
   * \param `n` the number of calls to `f`.
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA INLINE bool iterate(int n, const F& f) const {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    bool has_changed = false;
    int n2 = syncwarp && n != 0 ? max(n,n+(32-(n%32))) : n;
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
      has_changed |= f(syncwarp ? (i >= n ? n-1 : i) : i);
    }
    return has_changed;
  #endif
  }
};

#ifdef __CUDACC__

/** This function can be passed to `iterate` of a fixpoint engine in order to perform a local fixpoint per warp.
 * It expects the deduction operation to be split into a `load_deduce` and a `deduce`.
 * TPB: the number of threads per block.
*/
template <int TPB, class A>
__device__ local::B warp_fixpoint(A& a, int i) {
  auto ded = a.load_deduce(i);
  local::B has_changed = false;
  __shared__ bool warp_changed[TPB/32];
  int warp_id = threadIdx.x / 32;
  warp_changed[warp_id] = true;
  while(warp_changed[warp_id]) {
    __syncwarp();
    warp_changed[warp_id] = false;
    __syncwarp();
    if(a.deduce(ded)) {
      has_changed = true;
      /** If something changed, we continue to iterate only if we did not reach bot. */
      if(!a.is_bot()) {
        warp_changed[warp_id] = true;
      }
    }
    __syncwarp();
  }
  return has_changed;
}

#endif

/** Add the ability to deactive functions in a fixpoint computation.
 * Given a function `g`, we select only the functions \f$ f_{i_1} \| \ldots \| f_{i_k} \f$ for which \f$ g(i_k) \f$ is `true`, and compute subsequent fixpoint without them.
 */
template <class FixpointEngine, class Allocator, int TPB>
class FixpointSubsetGPU {
public:
  using allocator_type = Allocator;

private:
  FixpointEngine fp_engine;

  /** The indexes of functions that are active. */
  battery::vector<int, allocator_type> indexes;

  /** A mask to know which functions are still active.
   * We have `mask[i] <=> g(indexes[i])`.
  */
  battery::vector<bool, allocator_type> mask;

  /** A temporary array to compute the prefix sum of `mask`, in order to copy indexes into `indexes2`. */
  battery::vector<int, allocator_type> sum;

  /** A temporary array when copying the new active functions. */
  battery::vector<int, allocator_type> indexes2;

  /** The CUB prefix sum temporary storage. */
  using BlockScan = cub::BlockScan<int, TPB>;
  typename BlockScan::TempStorage cub_prefixsum_tmp;

  // We round n to the next multiple of TPB (the maximum dimension of the block, for now).
  __device__ INLINE int round_multiple_TPB(int n) {
    return n + ((blockDim.x - n % blockDim.x) % blockDim.x);
  }

public:
  FixpointSubsetGPU() = default;

  __device__ void reset(int n) {
    if(threadIdx.x == 0) {
      indexes.resize(n);
      indexes2.resize(n);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < indexes.size(); i += blockDim.x) {
      indexes[i] = i;
    }
  }

  __device__ void init(int n, const allocator_type& allocator = allocator_type()) {
    if(threadIdx.x == 0) {
      indexes = battery::vector<int, allocator_type>(n, allocator);
      indexes2 = battery::vector<int, allocator_type>(n, allocator);
      mask = battery::vector<bool, allocator_type>(round_multiple_TPB(n), false, allocator);
      sum = battery::vector<int, allocator_type>(round_multiple_TPB(n), 0, allocator);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < indexes.size(); i += blockDim.x) {
      indexes[i] = i;
    }
  }

  __device__ void destroy() {
    if(threadIdx.x == 0) {
      indexes = battery::vector<int, allocator_type>();
      indexes2 = battery::vector<int, allocator_type>();
      mask = battery::vector<bool, allocator_type>();
      sum = battery::vector<int, allocator_type>();
    }
    __syncthreads();
  }

  CUDA INLINE bool is_thread0() const {
    return fp_engine.is_thread0();
  }

  CUDA INLINE void barrier() {
    fp_engine.barrier();
  }

  template <class F>
  CUDA INLINE bool iterate(const F& f) {
    return fp_engine.iterate(indexes, f);
  }

  template <class F>
  CUDA INLINE int fixpoint(const F& f) {
    return fp_engine.fixpoint(indexes, f);
  }

  template <class F, class StopFun>
  CUDA INLINE int fixpoint(const F& f, const StopFun& g) {
    return fp_engine.fixpoint(indexes, f, g);
  }

  template <class F, class Iter, class StopFun>
  CUDA INLINE int fixpoint(const F& f, const Iter& h, const StopFun& g) {
    return fp_engine.fixpoint(indexes, f, h, g);
  }

  template <class F, class StopFun, class M>
  CUDA INLINE int fixpoint(const F& f, const StopFun& g, B<M>& has_changed) {
    return fp_engine.fixpoint(indexes, f, g);
  }

  /** \return the number of active functions. */
  CUDA int num_active() const {
    return indexes.size();
  }

  /** Compute the subset of the functions that are still active.
   * The subsequent call to `fixpoint` will only consider the function `f_i` for which `g(i)` is `true`. */
  template <class G>
  __device__ void select(const G& g) {
    assert(TPB == blockDim.x);
    // indexes:   0 1 2 3   (indexes of the propagators)
    // mask:      1 0 0 1   (filtering entailed functions)
    // sum:       1 1 1 2   (inclusive prefix sum)
    // indexes2:      0 3   (new indexes of the propagators)
    if(indexes.size() == 0) {
      return;
    }

    /** I. We perform a parallel map to detect the active functions. */
    for(int i = threadIdx.x; i < indexes.size(); i += blockDim.x) {
      mask[i] = g(indexes[i]);
    }

    /** II. We then compute the prefix sum of the mask in order to compute the new indexes of the active functions. */
    int n = round_multiple_TPB(indexes.size());
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
      BlockScan(cub_prefixsum_tmp).InclusiveSum(mask[i], sum[i]);
      __syncthreads(); // required by BlockScan to reuse the temporary storage.
    }
    for(int i = blockDim.x + threadIdx.x; i < n; i += blockDim.x) {
      sum[i] += sum[i - threadIdx.x - 1];
      __syncthreads();
    }

    /** III. We compute the new indexes of the active functions. */
    if(threadIdx.x == 0) {
      battery::swap(indexes, indexes2);
      indexes.resize(sum[indexes2.size()-1]);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < indexes2.size(); i += blockDim.x) {
      if(mask[i]) {
        indexes[sum[i]-1] = indexes2[i];
      }
    }
  }

  template <class Alloc = allocator_type>
  using snapshot_type = battery::vector<int, Alloc>;

  template <class Alloc = allocator_type>
  CUDA snapshot_type<Alloc> snapshot(const Alloc& alloc = Alloc()) const {
    return snapshot_type<Alloc>(indexes, alloc);
  }

  template <class Alloc>
  __device__ void restore_par(const snapshot_type<Alloc>& snap) {
    for(int i = threadIdx.x; i < snap.size(); i += blockDim.x) {
      indexes[i] = snap[i];
    }
    if(threadIdx.x == 0) {
      assert(snap.size() < indexes.capacity());
      indexes.resize(snap.size());
    }
  }
};

#endif

} // namespace lala

#endif
