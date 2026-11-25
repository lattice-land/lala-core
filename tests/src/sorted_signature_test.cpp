// Copyright 2026 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;
using S = Sort<standard_allocator>;

namespace {
  F var(const char* name) { return F::make_lvar(UNTYPED, LVar<standard_allocator>(name)); }
  F exists(const char* name, S::Tag tag) { return F::make_exists(UNTYPED, LVar<standard_allocator>(name), S(tag)); }
  F real(double r) { return F::make_real(r, r); }
  F set_z(logic_int l, logic_int u) {
    logic_set<F> s(1);
    s[0] = battery::make_tuple(F::make_z(l), F::make_z(u));
    return F::make_set(std::move(s));
  }

  void expect_sort(const F& f, S::Tag tag) {
    auto s = f.sort();
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(s->tag, tag);
  }
}

TEST(SortedSignature, SortFromSymbol) {
  expect_sort(F::make_binary(var("x"), ZADD, var("y")), S::Int);
  expect_sort(F::make_binary(var("a"), RADD, var("r")), S::Real);
  expect_sort(F::make_unary(INT2REAL, var("x")), S::Real);
  expect_sort(F::make_unary(FLOOR, var("a")), S::Int);
  expect_sort(F::make_unary(BOOL2INT, var("b")), S::Int);
  expect_sort(F::make_unary(INT2BOOL, var("x")), S::Bool);
  expect_sort(F::make_binary(var("x"), ZLEQ, var("y")), S::Bool);
  expect_sort(F::make_binary(var("a"), RLT, var("r")), S::Bool);
  expect_sort(F::make_binary(var("s"), EQ, var("t")), S::Bool);
  expect_sort(F::make_unary(CARD, var("s")), S::Int);
  expect_sort(F::make_binary(var("x"), ZEMOD, var("y")), S::Int);
  expect_sort(F::make_unary(RSIN, var("a")), S::Real);
  expect_sort(F::make_binary(var("x"), AND, var("y")), S::Bool);
}

TEST(SortedSignature, SortFromArguments) {
  auto u = F::make_binary(set_z(1, 1), UNION, set_z(2, 2)).sort();
  ASSERT_TRUE(u.has_value());
  EXPECT_TRUE(*u == S(S::Set, S(S::Int)));
  EXPECT_FALSE(F::make_binary(var("s"), UNION, var("t")).sort().has_value());
  F::Sequence ite;
  ite.push_back(var("c"));
  ite.push_back(F::make_z(1));
  ite.push_back(var("x"));
  expect_sort(F::make_nary(ITE, std::move(ite)), S::Int);
}

TEST(SortedSignature, SortUnchanged) {
  EXPECT_FALSE(var("x").sort().has_value());
  EXPECT_FALSE(F::make_nary(LVar<standard_allocator>("search"), F::Sequence()).sort().has_value());
  expect_sort(exists("x", S::Real), S::Real); // the sort of the quantified variable.
  expect_sort(F::make_z(1), S::Int);
  expect_sort(real(1.5), S::Real);
  expect_sort(F::make_bool(true), S::Bool);
}

TEST(SortedSignature, EvalConversions) {
  EXPECT_EQ(eval(F::make_unary(BOOL2INT, F::make_bool(true))), F::make_z(1));
  EXPECT_EQ(eval(F::make_unary(BOOL2INT, F::make_bool(false))), F::make_z(0));
  EXPECT_EQ(eval(F::make_unary(INT2BOOL, F::make_z(0))), F::make_false());
  EXPECT_EQ(eval(F::make_unary(INT2BOOL, F::make_z(-3))), F::make_true());
  EXPECT_EQ(eval(F::make_binary(F::make_z(2), ZADD, F::make_z(3))), F::make_z(5));
}

TEST(SortedSignature, Negation) {
  auto f = negate(F::make_binary(var("x"), ZLEQ, var("y")));
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(f->sig(), ZGT);
  auto g = negate(F::make_binary(var("a"), REQ, var("r")));
  ASSERT_TRUE(g.has_value());
  EXPECT_EQ(g->sig(), RNEQ);
  auto h = negate(F::make_binary(var("s"), LT, var("t")));
  ASSERT_TRUE(h.has_value());
  EXPECT_EQ(h->sig(), GEQ);
}
