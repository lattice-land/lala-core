// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"

#include <optional>

using namespace lala;
using namespace battery;

TEST(VarTest, MakeVar) {
  constexpr int n = 6;
  int types[n] = {0, 0, 1, 1, 13, (1 << 5) - 1};
  int var_ids[n] = {0, 1, 0, 1, 124, (1 << 26) - 1};
  for(int i = 0; i < n; ++i) {
    AVar v = AVar(types[i], var_ids[i]);
    EXPECT_EQ(v.aty(), types[i]);
    EXPECT_EQ(v.vid(), var_ids[i]);
  }
#ifdef DEBUG
  ASSERT_DEATH(AVar((1 << 5), 0), "");
  ASSERT_DEATH(AVar(0, (1 << 26)), "");
#endif
}

TEST(AST, NumVars) {
  using F = TFormula<standard_allocator>;
  auto var_x = LVar<standard_allocator>("x");
  auto f1 = make_v_op_z(var_x, LEQ, 1);
  auto f2 = make_v_op_z(var_x, LEQ, 0);
  auto f3 = F::make_binary(f1, AND, f2);
  EXPECT_EQ(num_vars(f1), 1);
  EXPECT_EQ(num_vars(f2), 1);
  EXPECT_EQ(num_vars(f3), 2);
}

TEST(AST, ExtractTy) {
  using F = TFormula<standard_allocator>;
  auto var_x = LVar<standard_allocator>("x");
  auto var_y = LVar<standard_allocator>("y");
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
