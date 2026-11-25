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
  F empty_set() { return F::make_set(logic_set<F>()); }

  /** `var x:Z, var y:Z, var a:R, var r:R, var b:B, var s:S(Z), var t:S(Z)` and the formula `f`. */
  F with_decls(F f) {
    F::Sequence seq;
    seq.push_back(exists("x", S::Int));
    seq.push_back(exists("y", S::Int));
    seq.push_back(exists("a", S::Real));
    seq.push_back(exists("r", S::Real));
    seq.push_back(exists("b", S::Bool));
    seq.push_back(F::make_exists(UNTYPED, LVar<standard_allocator>("s"), S(S::Set, S(S::Int))));
    seq.push_back(F::make_exists(UNTYPED, LVar<standard_allocator>("t"), S(S::Set, S(S::Int))));
    seq.push_back(std::move(f));
    return F::make_nary(AND, std::move(seq));
  }

  void expect_sort(const F& f, S::Tag tag) {
    auto s = f.sort();
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(s->tag, tag);
  }

  void expect_wellformed(const F& f) {
    WFDiagnostic<standard_allocator> diag;
    bool res = is_wellformed(with_decls(f), VarEnv<standard_allocator>(), diag);
    if(!res) { diag.print(); }
    EXPECT_TRUE(res);
  }

  /** `f` is ill-formed, and the diagnostic message contains `msg`. */
  void expect_illformed(const F& f, const std::string& msg) {
    WFDiagnostic<standard_allocator> diag;
    EXPECT_FALSE(is_wellformed(with_decls(f), VarEnv<standard_allocator>(), diag));
    EXPECT_NE(diag.message.find(msg), std::string::npos) << "message: " << diag.message;
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

TEST(SortedSignature, EqualityOfSort) {
  EXPECT_EQ(eq_of_sort(S(S::Bool)), EQUIV);
  EXPECT_EQ(eq_of_sort(S(S::Int)), ZEQ);
  EXPECT_EQ(eq_of_sort(S(S::Real)), REQ);
  EXPECT_EQ(eq_of_sort(S(S::Set, S(S::Int))), EQ);
  EXPECT_EQ(eq_of_result(ZADD), ZEQ);
  EXPECT_EQ(eq_of_result(RADD), REQ);
  EXPECT_EQ(eq_of_result(ZLEQ), EQUIV);
  EXPECT_EQ(eq_of_result(UNION), EQ); // the result sort depends on the arguments.
  EXPECT_EQ(eq_of_value(F::make_z(1)), ZEQ);
  EXPECT_EQ(eq_of_value(real(1.5)), REQ);
  EXPECT_EQ(eq_of_value(F::make_bool(true)), EQUIV);
  EXPECT_EQ(eq_of_value(set_z(1, 2)), EQ);
  EXPECT_EQ(eq_of_decl(exists("b", S::Bool)), EQUIV);
  EXPECT_EQ(eq_of_decl(F::make_binary(exists("x", S::Int), AND, var("x"))), ZEQ);
  EXPECT_EQ(eq_of_decl(var("x")), EQ);
}

TEST(SortedSignature, InstantiateOp) {
  // Constants.
  EXPECT_EQ(instantiate_op(GEQ, F::make_z(1)), ZGEQ);
  EXPECT_EQ(instantiate_op(LT, real(1.5)), RLT);
  EXPECT_EQ(instantiate_op(EQ, F::make_bool(true)), ZEQ);
  // Sets keep the polymorphic comparison (lexicographic order).
  EXPECT_EQ(instantiate_op(LEQ, set_z(1, 2)), LEQ);
  EXPECT_EQ(instantiate_op(EQ, set_z(1, 2)), EQ);
  EXPECT_EQ(instantiate_op(GEQ, F::make_binary(set_z(1, 1), UNION, set_z(2, 2))), GEQ);
  // The sort of an application comes from its head symbol.
  EXPECT_EQ(instantiate_op(NEQ, F::make_binary(var("x"), ZADD, var("y"))), ZNEQ);
  EXPECT_EQ(instantiate_op(GT, F::make_binary(var("a"), RADD, var("r"))), RGT);
  EXPECT_EQ(instantiate_op(LEQ, F::make_unary(INT2REAL, var("x"))), RLEQ);
  // Without an environment, the sort of a variable is unknown: integers by default.
  EXPECT_EQ(instantiate_op(EQ, var("x")), ZEQ);
}

TEST(SortedSignature, WellFormed) {
  // Integers.
  expect_wellformed(F::make_binary(F::make_binary(var("x"), ZADD, F::make_z(1)), ZLEQ, var("y")));
  expect_wellformed(F::make_binary(F::make_binary(var("x"), ZEMOD, var("y")), ZEQ, F::make_z(0)));
  // Real numbers and conversions.
  expect_wellformed(F::make_binary(F::make_binary(F::make_unary(INT2REAL, var("x")), RADD, var("r")), REQ, real(2.5)));
  expect_wellformed(F::make_binary(F::make_unary(FLOOR, var("a")), ZEQ, var("x")));
  expect_wellformed(F::make_binary(F::make_unary(RSQRT, var("a")), RLEQ, F::make_binary(var("r"), RNROOT, F::make_z(3))));
  // Reification: the Boolean is converted to an integer.
  expect_wellformed(F::make_binary(
    F::make_binary(F::make_unary(BOOL2INT, F::make_binary(var("x"), ZGT, F::make_z(4))), ZADD, F::make_unary(BOOL2INT, var("b"))),
    ZNEQ, F::make_z(2)));
  expect_wellformed(F::make_binary(var("b"), EQUIV, F::make_binary(var("x"), ZLEQ, var("y"))));
  expect_wellformed(F::make_binary(F::make_unary(INT2BOOL, var("x")), AND, var("b")));
  // Sets and polymorphic comparisons.
  expect_wellformed(F::make_binary(var("s"), EQ, var("t")));
  expect_wellformed(F::make_binary(var("s"), LEQ, F::make_binary(var("t"), UNION, set_z(1, 3))));
  expect_wellformed(F::make_binary(var("s"), SUBSETEQ, empty_set()));
  expect_wellformed(F::make_binary(var("x"), IN, set_z(0, 5)));
  expect_wellformed(F::make_binary(F::make_unary(CARD, var("s")), ZLEQ, F::make_z(2)));
  expect_wellformed(F::make_binary(var("b"), EQ, F::make_bool(true)));
  // Objective and extended symbols.
  expect_wellformed(F::make_unary(MINIMIZE, var("a")));
  expect_wellformed(F::make_nary(LVar<standard_allocator>("search"), F::Sequence()));
}

TEST(SortedSignature, IllFormed) {
  expect_illformed(F::make_binary(var("x"), RADD, var("r")), "int2real");
  expect_illformed(F::make_binary(var("z"), ZLEQ, F::make_z(1)), "Undeclared variable `z`");
  expect_illformed(F::make_binary(var("a"), ZLEQ, var("r")), "floor");
  expect_illformed(F::make_binary(var("b"), ZADD, var("x")), "bool2int");
  expect_illformed(F::make_binary(var("x"), AND, var("b")), "int2bool");
  expect_illformed(F::make_binary(var("s"), EQ, var("x")), "same sort");
  expect_illformed(F::make_unary(ZLEQ, var("x")), "expects 2 argument(s)");
  {
    logic_set<F> s(2);
    s[0] = battery::make_tuple(F::make_z(1), F::make_z(2));
    s[1] = battery::make_tuple(real(1.5), real(2.5));
    expect_illformed(F::make_binary(var("x"), IN, F::make_set(std::move(s))), "same sort");
  }
  expect_illformed(F::make_binary(var("a"), IN, set_z(0, 1)), "`in` expects an element of sort Int");
  expect_illformed(F::make_unary(MAXIMIZE, var("s")), "Int or Real");
  // Conflicting declarations.
  WFDiagnostic<standard_allocator> diag;
  EXPECT_FALSE(is_wellformed(F::make_binary(exists("x", S::Int), AND, exists("x", S::Real)), VarEnv<standard_allocator>(), diag));
  EXPECT_NE(diag.message.find("declared with the sorts"), std::string::npos);
}

TEST(SortedSignature, Environment) {
  VarEnv<standard_allocator> env;
  env.extends_vars(0, LVar<standard_allocator>("x"), S(S::Int));
  EXPECT_TRUE(is_wellformed(F::make_binary(var("x"), ZLEQ, F::make_z(1)), env));
  EXPECT_FALSE(is_wellformed(F::make_binary(var("x"), RLEQ, real(1.0)), env));
  // A declaration conflicting with the environment.
  EXPECT_FALSE(is_wellformed(exists("x", S::Bool), env));
  auto s = sort_of_term(F::make_unary(INT2REAL, var("x")), env);
  ASSERT_TRUE(s.has_value());
  EXPECT_TRUE(s->is_real());
  EXPECT_FALSE(sort_of_term(F::make_binary(var("x"), RADD, var("x")), env).has_value());
}
