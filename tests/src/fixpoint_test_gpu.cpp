// Copyright 2021 Pierre Talbot

#include <random>
#include <limits>
#include <algorithm>
#include "battery/memory.hpp"
#include "battery/allocator.hpp"
#include "lala/fixpoint.hpp"
#include "lala/universes/arith_bound.hpp"

using namespace battery;
using namespace lala;

using cpu_gpu_vec = vector<int, managed_allocator>;
using cpu_gpu_vec_ptr = shared_ptr<cpu_gpu_vec, managed_allocator>;

template <class AtomicMem>
class Minimum {
  cpu_gpu_vec* data;
  ZUB<int, AtomicMem> result;

public:
  CUDA Minimum(cpu_gpu_vec* data) : data(data), result() {}
  CUDA int num_deductions() { return data->size(); }
  CUDA bool deduce(size_t i) {
    return result.meet(local::ZUB((*data)[i]));
  }
  CUDA int extract() {
    return result;
  }
  CUDA local::B is_bot() const { return false; }
};

__global__ void minimum_kernel_on_block(cpu_gpu_vec* g, int* result) {
  using FP_engine = BlockAsynchronousFixpointGPU<>;
  using Min = Minimum<atomic_memory_block>;
  unique_ptr<FP_engine, global_allocator> fp_engine;
  unique_ptr<Min, global_allocator> minimum;
  auto block = cooperative_groups::this_thread_block();
  FP_engine& fp = battery::make_unique_block<FP_engine, global_allocator>(fp_engine);
  Min& m = battery::make_unique_block<Min, global_allocator>(minimum, g);
  fp.fixpoint(m.num_deductions(), [&](size_t i){ return m.deduce(i); });
  cooperative_groups::invoke_one(block, [&](){
    *result = m.extract();
  });
  block.sync();
}

__global__ void minimum_kernel_on_grid(cpu_gpu_vec* g, int* result) {
  using FP_engine = GridAsynchronousFixpointGPU;
  using Min = Minimum<atomic_memory_grid>;
  unique_ptr<FP_engine, global_allocator> fp_engine;
  unique_ptr<Min, global_allocator> minimum;
  auto grid = cooperative_groups::this_grid();
  FP_engine& fp = battery::make_unique_grid<FP_engine, global_allocator>(fp_engine, grid);
  Min& m = battery::make_unique_grid<Min, global_allocator>(minimum, g);
  fp.fixpoint(m.num_deductions(), [&](size_t i){ return m.deduce(i); });
  cooperative_groups::invoke_one(grid, [&](){
    *result = m.extract();
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
  shared_ptr<int, managed_allocator> gpu_res =
    make_shared<int, managed_allocator>(std::numeric_limits<int>::max());
  if constexpr(GRID_SIZE == 1) {
    minimum_kernel_on_block<<<GRID_SIZE, BLOCK_SIZE>>>(g.get(), gpu_res.get());
  }
  else {
    cpu_gpu_vec* g_ptr = g.get();
    int* res_ptr = gpu_res.get();
    void* args[] = {&g_ptr, &res_ptr};
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(GRID_SIZE, 1, 1);
    CUDAEX(cudaLaunchCooperativeKernel((void*)minimum_kernel_on_grid, dimGrid, dimBlock, args));
  }
  CUDAEX(cudaDeviceSynchronize());
  int cpu_res = *(std::min_element(v.begin(), v.end()));
  std::cout << "grid size: " << GRID_SIZE << " | block size: " << BLOCK_SIZE << " | gpu min: " << *gpu_res << " | cpu min: " << cpu_res << std::endl;
  assert(*gpu_res == cpu_res);
}

int main() {
  std::vector<int> v = init_random_vector(1000000);
  cpu_gpu_vec_ptr g = ::battery::make_shared<cpu_gpu_vec, managed_allocator>(v);
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

  run_gpu_min<2, 1>(v, g);
  run_gpu_min<2, 100>(v, g);
  run_gpu_min<2, 256>(v, g);

  run_gpu_min<16, 1>(v, g);
  run_gpu_min<16, 100>(v, g);
  run_gpu_min<16, 256>(v, g);

  run_gpu_min<48, 1>(v, g);
  run_gpu_min<48, 100>(v, g);
  run_gpu_min<48, 256>(v, g);
}
