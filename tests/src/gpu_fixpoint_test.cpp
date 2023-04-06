// Copyright 2021 Pierre Talbot

#include <random>
#include <limits>
#include <algorithm>
#include "fixpoint.hpp"
#include "allocator.hpp"
#include "memory.hpp"
#include "universes/upset_universe.hpp"

using namespace battery;
using namespace lala;

using zd = ZDec<int, AtomicMemoryBlock<ManagedAllocator>>;

using cpu_gpu_vec = vector<int, ManagedAllocator>;
using cpu_gpu_vec_ptr = shared_ptr<cpu_gpu_vec, ManagedAllocator>;

class Minimum {
  cpu_gpu_vec* data;
  zd result;

public:
  CUDA Minimum(cpu_gpu_vec* data) : data(data), result() {}
  CUDA int num_refinements() { return data->size(); }
  template<class M>
  CUDA void refine(int i, BInc<M>& has_changed) {
    result.tell(zd((*data)[i]), has_changed);
  }
  CUDA int extract() {
    return result;
  }
};

template<class T, class Alloc, class... Args>
__device__ T* make_block_unique(unique_ptr<T, Alloc>& ptr, Args&&... args) {
  __shared__ T* raw_ptr;
  auto block = cooperative_groups::this_thread_block();
  invoke_one(block, [&](){
    ptr = battery::make_unique<T, Alloc>(std::forward<Args>(args)...);
    raw_ptr = ptr.get();
  });
  block.sync();
  return raw_ptr;
}

__device__ void* grid_raw_ptr[2];
__device__ int active_ptr = 0;

template<class T, class Alloc, class... Args>
__device__ T* make_grid_unique(unique_ptr<T, Alloc>& ptr, Args&&... args) {
  auto grid = cooperative_groups::this_grid();
  invoke_one(grid, [&](){
    ptr = battery::make_unique<T, Alloc>(std::forward<Args>(args)...);
    active_ptr = (active_ptr + 1) % 2;
    grid_raw_ptr[active_ptr] = ptr.get();
  });
  grid.sync();
  return static_cast<T*>(grid_raw_ptr[active_ptr]);
}

__global__ void minimum_kernel_on_block(cpu_gpu_vec* g, int* result) {
  using FP_engine = BlockAsynchronousIterationGPU<GlobalAllocator>;
  unique_ptr<FP_engine, GlobalAllocator> fp_engine;
  unique_ptr<Minimum, GlobalAllocator> minimum;
  auto block = cooperative_groups::this_thread_block();
  FP_engine* fp = make_block_unique<FP_engine, GlobalAllocator>(fp_engine, block);
  Minimum* m = make_block_unique<Minimum, GlobalAllocator>(minimum, g);
  fp->fixpoint(*m);
  invoke_one(block, [&](){
    *result = m->extract();
  });
  block.sync();
}

// TODO: even if it finally works, we must use a "grid atomic" for `Minimum.result`.
__global__ void minimum_kernel_on_grid(cpu_gpu_vec* g, int* result) {
  printf("hi\n");
  using FP_engine = GridAsynchronousIterationGPU<GlobalAllocator>;
  unique_ptr<FP_engine, GlobalAllocator> fp_engine;
  unique_ptr<Minimum, GlobalAllocator> minimum;
  printf("hi\n");
  auto grid = cooperative_groups::this_grid();
  printf("%llu entered\n", grid.block_rank());
  FP_engine* fp = make_grid_unique<FP_engine, GlobalAllocator>(fp_engine, grid);
  printf("%llu created fp\n", grid.block_rank());
  Minimum* m = make_grid_unique<Minimum, GlobalAllocator>(minimum, g);
  printf("%llu %p %p\n", grid.block_rank(), fp, m);
  fp->fixpoint(*m);
  invoke_one(grid, [&](){
    *result = m->extract();
  });
  grid.sync();
}

std::vector<int> init_random_vector(size_t size) {
  std::vector<int> v(size);
  std::mt19937 m{std::random_device{}()};
  std::uniform_int_distribution<int> dist{-10000000, 10000000};
  generate(begin(v), end(v), [&dist, &m](){return dist(m);});
  return std::move(v);
}

template <size_t GRID_SIZE, size_t BLOCK_SIZE>
void run_gpu_min(const std::vector<int>& v, cpu_gpu_vec_ptr g) {
  shared_ptr<int, ManagedAllocator> gpu_res =
    make_shared<int, ManagedAllocator>(std::numeric_limits<int>::max());
  if constexpr(GRID_SIZE == 1) {
    minimum_kernel_on_block<<<GRID_SIZE, BLOCK_SIZE>>>(g.get(), gpu_res.get());
  }
  else {
    void* args[] = {g.get(), gpu_res.get()};
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(GRID_SIZE, 1, 1);
    CUDIE(cudaLaunchCooperativeKernel((void*)minimum_kernel_on_grid, dimGrid, dimBlock, args));
  }
  CUDIE(cudaDeviceSynchronize());
  int cpu_res = *(std::min_element(v.begin(), v.end()));
  std::cout << "grid size: " << GRID_SIZE << " | block size: " << BLOCK_SIZE << " | gpu min: " << *gpu_res << " | cpu min: " << cpu_res << std::endl;
  assert(*gpu_res == cpu_res);
}

int main() {
  std::vector<int> v = init_random_vector(10);
  cpu_gpu_vec_ptr g = make_shared<cpu_gpu_vec, ManagedAllocator>(cpu_gpu_vec{v.data(), v.size()});
  run_gpu_min<1, 1>(v, g);
  run_gpu_min<1, 2>(v, g);
  run_gpu_min<1, 100>(v, g);
  run_gpu_min<1, 128>(v, g);
  run_gpu_min<1, 256>(v, g);

  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  if(supportsCoopLaunch == 0) {
    std::cout << "Device does not support cooperative launch, required to synchronize globally on the grid." << std::endl;
    return 0;
  }

  // Does not work yet.
  // run_gpu_min<2, 1>(v, g);
  // run_gpu_min<2, 100>(v, g);
  // run_gpu_min<2, 256>(v, g);
}
