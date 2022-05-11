// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "ast.hpp"
#include "allocator.hpp"

using namespace lala;
using namespace battery;

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
  using S = String<StandardAllocator>;
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
  EXPECT_EQ(env[0], S("x"));
  EXPECT_EQ(env[1], S("y"));
  EXPECT_EQ(env[2], S("z"));
}

TEST(AST, SFormula) {
  using SF = SFormula<StandardAllocator>;
  SF satisfy(SF::F::make_true(), 2);
  SF maximize(SF::F::make_true(), SF::MAXIMIZE, LVar<StandardAllocator>("x"));
  EXPECT_EQ(satisfy.mode(), SF::SATISFY);
  EXPECT_EQ(satisfy.num_sols(), 2);
  EXPECT_EQ(maximize.mode(), SF::MAXIMIZE);
  EXPECT_EQ(maximize.optimization_lvar(), LVar<StandardAllocator>("x"));
}

TEST(AST, NumVars) {
  using F = TFormula<StandardAllocator>;
  auto var_x = LVar<StandardAllocator>("x");
  auto f1 = make_v_op_z(var_x, LEQ, 1);
  auto f2 = make_v_op_z(var_x, LEQ, 0);
  auto f3 = F::make_binary(f1, AND, f2);
  EXPECT_EQ(num_vars(f1), 1);
  EXPECT_EQ(num_vars(f2), 1);
  EXPECT_EQ(num_vars(f3), 2);
}

TEST(AST, ExtractTy) {
  using F = TFormula<StandardAllocator>;
  auto var_x = LVar<StandardAllocator>("x");
  auto var_y = LVar<StandardAllocator>("y");
  auto f1 = F::make_binary(F::make_lvar(0, var_x), LEQ, F::make_z(10), 0);
  auto f2 = F::make_binary(F::make_lvar(0, var_x), LEQ, F::make_lvar(0, var_y), 1);
  auto f3 = F::make_binary(F::make_lvar(0, var_x), GEQ, F::make_z(0), 0);
  auto f4 = F::make_binary(F::make_lvar(0, var_x), GEQ, F::make_lvar(0, var_y), 1);
  auto f = F::make_binary(f1, AND,
    F::make_binary(f2, AND,
      F::make_binary(f3, AND, f4, 1), 1), 1);
  auto fg = extract_ty(f, 1);
  auto fty1 = battery::get<0>(fg);
  auto fty0 = battery::get<1>(fg);
  EXPECT_EQ(fty1.seq().size(), 2);
  EXPECT_EQ(fty0.seq().size(), 2);
  EXPECT_EQ(fty0.seq(0), f1);
  EXPECT_EQ(fty0.seq(1), f3);
  EXPECT_EQ(fty1.seq(0), f2);
  EXPECT_EQ(fty1.seq(1), f4);
}

TEST(AST, AbstractDeps) {
  AbstractDeps deps;
}