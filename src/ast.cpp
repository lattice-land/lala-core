// Copyright 2021 Pierre Talbot

#include "ast.hpp"
#include <cassert>

namespace lala {

CUDA AVar make_var(int ad_uid, int var_id) {
  assert(ad_uid < (1 << 8));
  assert(var_id < (1 << 23));
  return (var_id << 8) | ad_uid;
}

}
