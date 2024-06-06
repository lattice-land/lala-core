// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/flatzinc_parser.hpp"

#include <optional>

using namespace lala;
using namespace battery;

TEST(VarTest, MakeVar) {
  constexpr int n = 6;
  int types[n] = {0, 0, 1, 1, 13, (1 << 8) - 1};
  int var_ids[n] = {0, 1, 0, 1, 124, (1 << 23) - 1};
  for(int i = 0; i < n; ++i) {
    AVar v = AVar(types[i], var_ids[i]);
    EXPECT_EQ(v.aty(), types[i]);
    EXPECT_EQ(v.vid(), var_ids[i]);
  }
#ifdef DEBUG
  ASSERT_DEATH(AVar((1 << 8), 0), "");
  ASSERT_DEATH(AVar(0, (1 << 23)), "");
#endif
}

template <class Env>
std::optional<AVar> interpret2(Env& env, const char* fzn) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  AVar avar;
  IDiagnostics diagnostics;
  if(env.interpret(*f, avar, diagnostics)) {
    return {avar};
  }
  else {
    return {};
  }
}

void check_env_state1(VarEnv<standard_allocator>& env) {
  EXPECT_EQ(env.num_abstract_doms(), 0);
  EXPECT_EQ(env.num_vars(), 0);
  EXPECT_FALSE(interpret2(env, "var int: x;").has_value()); // Not typed.
  EXPECT_EQ(env.num_abstract_doms(), 0);
  EXPECT_EQ(env.num_vars(), 0);
}

void check_env_state2(VarEnv<standard_allocator>& env) {
  EXPECT_EQ(env.num_abstract_doms(), 1);
  EXPECT_EQ(env.num_vars(), 1);
  EXPECT_EQ(env.num_vars_in(0), 1);
  EXPECT_TRUE(env.contains(AVar(0, 0)));
  EXPECT_TRUE(env.contains("x"));
  EXPECT_TRUE(env.variable_of("x").has_value());
  EXPECT_EQ(*(env.variable_of("x")->get().avar_of(0)), AVar(0, 0));
  EXPECT_FALSE(interpret2(env, "var int: x;").has_value()); // untyped.
  EXPECT_TRUE(interpret2(env, "var int: x :: abstract(0);").has_value());
  EXPECT_FALSE(interpret2(env, "var float: x;").has_value()); // different sort.
}

void check_env_state3(VarEnv<standard_allocator>& env) {
  EXPECT_EQ(env.num_abstract_doms(), 1);
  EXPECT_EQ(env.num_vars(), 2);
  EXPECT_EQ(env.num_vars_in(0), 2);
  EXPECT_TRUE(env.contains(AVar(0, 1)));
  EXPECT_TRUE(env.contains("y"));
  EXPECT_TRUE(env.variable_of("y").has_value());
  EXPECT_EQ(*(env.variable_of("y")->get().avar_of(0)), AVar(0, 1));
}

void check_env_state4(VarEnv<standard_allocator>& env) {
  EXPECT_EQ(env.num_abstract_doms(), 2);
  EXPECT_EQ(env.num_vars_in(1), 1);
  EXPECT_EQ(env.num_vars(), 3);
  EXPECT_TRUE(env.contains(AVar(1, 0)));
  EXPECT_TRUE(env.contains("z"));
  EXPECT_TRUE(env.variable_of("z").has_value());
  EXPECT_FALSE(env.variable_of("z")->get().avar_of(0).has_value());
  EXPECT_EQ(*(env.variable_of("z")->get().avar_of(1)), AVar(1, 0));
}

void check_env_state5(VarEnv<standard_allocator>& env) {
  EXPECT_EQ(env.num_abstract_doms(), 11);
  EXPECT_EQ(env.num_vars(), 4);
  EXPECT_EQ(env.num_vars_in(1), 2);
  EXPECT_EQ(env.num_vars_in(2), 0);
  EXPECT_EQ(env.num_vars_in(9), 0);
  EXPECT_EQ(env.num_vars_in(10), 1);
  EXPECT_TRUE(env.variable_of("w").has_value());
  EXPECT_FALSE(env.variable_of("w")->get().avar_of(4).has_value());
  EXPECT_EQ(*(env.variable_of("w")->get().avar_of(10)), AVar(10, 0));
}

TEST(AST, VarEnv) {
  VarEnv<standard_allocator> env;
  check_env_state1(env);
  auto snap1 = env.snapshot();

  auto x = interpret2(env, "var int: x :: abstract(0);");
  EXPECT_TRUE(x.has_value());
  EXPECT_EQ(x.value(), AVar(0, 0));
  check_env_state2(env);
  auto snap2 = env.snapshot();

  auto y = interpret2(env, "var int: y :: abstract(0);");
  EXPECT_TRUE(y.has_value());
  EXPECT_EQ(y.value(), AVar(0, 1));
  check_env_state3(env);
  auto snap3 = env.snapshot();

  auto z = interpret2(env, "var int: z :: abstract(1);");
  EXPECT_TRUE(z.has_value());
  EXPECT_EQ(z.value(), AVar(1, 0));
  check_env_state4(env);
  auto snap4 = env.snapshot();

  EXPECT_TRUE(interpret2(env, "var int: x :: abstract(1);").has_value());
  auto w = interpret2(env, "var bool: w :: abstract(10);");
  EXPECT_TRUE(w.has_value());
  EXPECT_EQ(w.value(), AVar(10, 0));
  check_env_state5(env);
  auto snap5 = env.snapshot();

  printf("Start restoring snapshots...\n");

  env.restore(snap5);
  check_env_state5(env);
  env.restore(snap4);
  check_env_state4(env);
  env.restore(snap3);
  check_env_state3(env);
  env.restore(snap2);
  check_env_state2(env);
  env.restore(snap1);
  check_env_state1(env);
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
