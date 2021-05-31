// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "ast.hpp"

using namespace lala;

TEST(VarTest, MakeVar) {
  int n = 6;
  int ad_uids[n] = {0, 0, 1, 1, 13, 63};
  int var_ids[n] = {0, 1, 0, 1, 124, 10000000};
  for(int i = 0; i < n; ++i) {
    int v = make_var(ad_uids[i], var_ids[i]);
    EXPECT_EQ(AID(v), ad_uids[i]);
    EXPECT_EQ(VID(v), var_ids[i]);
  }
}
