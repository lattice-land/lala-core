// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "ast.hpp"

using namespace lala;

TEST(VarTest, MakeVar) {
  int n = 6;
  int types[n] = {0, 0, 1, 1, 13, (1 << 8) - 1};
  int var_ids[n] = {0, 1, 0, 1, 124, (1 << 23) - 1};
  for(int i = 0; i < n; ++i) {
    int v = make_var(types[i], var_ids[i]);
    EXPECT_EQ(AID(v), types[i]);
    EXPECT_EQ(VID(v), var_ids[i]);
  }
  ASSERT_DEATH(make_var((1 << 8), 0), "");
  ASSERT_DEATH(make_var(0, (1 << 23)), "");
}

TEST(AST, VarEnv) {
  typedef String<StandardAllocator> S;
  AType uid = 3;
  VarEnv<StandardAllocator> env(uid, 3);
  EXPECT_EQ(env.add("x"), make_var(uid, 0));
  EXPECT_EQ(env.add("y"), make_var(uid, 1));
  EXPECT_EQ(env.size(), 2);
  EXPECT_EQ(env.capacity(), 3);
  EXPECT_EQ(env.add("z"), make_var(uid, 2));
  EXPECT_EQ(env.size(), 3);
  EXPECT_EQ(env.capacity(), 3);
  EXPECT_TRUE(env.to_avar("x").has_value());
  EXPECT_EQ(*(env.to_avar("x")), make_var(uid, 0));
  EXPECT_TRUE(env.to_avar("y").has_value());
  EXPECT_EQ(*(env.to_avar("y")), make_var(uid, 1));
  EXPECT_TRUE(env.to_avar("z").has_value());
  EXPECT_EQ(*(env.to_avar("z")), make_var(uid, 2));
  EXPECT_EQ(env.to_avar("w"), thrust::optional<AVar>());
  EXPECT_EQ(env.to_lvar(make_var(uid, 0)), S("x"));
  EXPECT_EQ(env.to_lvar(make_var(uid, 1)), S("y"));
  EXPECT_EQ(env.to_lvar(make_var(uid, 2)), S("z"));
  ASSERT_DEATH(env.to_lvar(make_var(uid, 3)), "");
}
