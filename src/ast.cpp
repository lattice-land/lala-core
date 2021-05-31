// Copyright 2021 Pierre Talbot

#include "ast.hpp"
#include "cuda_helper.hpp"

namespace lala {

CUDA AVar make_var(int ad_uid, int var_id) {
  assert(ad_uid < 63);
  return (var_id << 6) | ad_uid;
}

}
