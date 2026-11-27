// Copyright 2026 Pierre Talbot

#include <gtest/gtest.h>
#include <string>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;
using S = Sort<standard_allocator>;

TEST(SymID, Interning) {
  SymID x("symbol_test_x");
  EXPECT_EQ(x, SymID(std::string("symbol_test_x")));
  EXPECT_EQ(x, SymID(battery::string<standard_allocator>("symbol_test_x")));
  EXPECT_NE(x, SymID("symbol_test_y"));
  EXPECT_STREQ(x.data(), "symbol_test_x");
  EXPECT_EQ(x.str(), "symbol_test_x");
  EXPECT_TRUE(x.is_valid());
  EXPECT_FALSE(SymID().is_valid());
}

TEST(SymID, FindDoesNotIntern) {
  size_t n = SymbolTable::instance().size();
  EXPECT_FALSE(SymID::find("symbol_test_never_interned").has_value());
  EXPECT_EQ(SymbolTable::instance().size(), n);
  SymID z("symbol_test_z");
  auto found = SymID::find("symbol_test_z");
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(*found, z);
}

TEST(SymID, NamesAreStable) {
  // The name returned by `data()` stays valid when the table grows and is rehashed.
  SymID a("symbol_test_a");
  const char* name = a.data();
  for(int i = 0; i < 100000; ++i) {
    SymID("symbol_test_many_" + std::to_string(i));
  }
  EXPECT_EQ(name, a.data());
  EXPECT_STREQ(name, "symbol_test_a");
  EXPECT_STREQ(SymID("symbol_test_many_99999").data(), "symbol_test_many_99999");
}

TEST(SymID, FormulasAndEnvironments) {
  // Formulas built separately share the symbols of their variables.
  auto f1 = F::make_binary(F::make_lvar(UNTYPED, LVar<standard_allocator>("symbol_test_v")), ZLEQ, F::make_z(1));
  auto f2 = F::make_binary(F::make_lvar(UNTYPED, SymID(std::string("symbol_test_v"))), ZLEQ, F::make_z(1));
  EXPECT_EQ(f1, f2);
  EXPECT_NE(f1, F::make_binary(F::make_lvar(UNTYPED, SymID("symbol_test_w")), ZLEQ, F::make_z(1)));

  VarEnv<standard_allocator> env;
  AVar v = env.extends_vars(0, SymID("symbol_test_v"), S(S::Int));
  ASSERT_TRUE(env.variable_of(f1.seq(0).lv()).has_value());
  EXPECT_EQ(env.name_of(v), f2.seq(0).lv());
  EXPECT_TRUE(env.contains("symbol_test_v"));
  EXPECT_FALSE(env.contains("symbol_test_absent"));

  // `restore` removes the variables added after the snapshot from the index.
  auto snap = env.snapshot();
  env.extends_vars(0, SymID("symbol_test_w"), S(S::Int));
  EXPECT_TRUE(env.contains(SymID("symbol_test_w")));
  env.restore(snap);
  EXPECT_FALSE(env.contains(SymID("symbol_test_w")));
  EXPECT_TRUE(env.contains(SymID("symbol_test_v")));

  // A copy of the environment finds the same variables.
  VarEnv<standard_allocator> env2(env);
  ASSERT_TRUE(env2.variable_of(SymID("symbol_test_v")).has_value());
  EXPECT_EQ(env2.variable_of(SymID("symbol_test_v"))->get().avars[0], v);
}
