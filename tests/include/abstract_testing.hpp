// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_GENERIC_UNIVERSE_TEST_HPP
#define LALA_CORE_GENERIC_UNIVERSE_TEST_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/interpretation.hpp"
#include "lala/flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

static LVar<standard_allocator> var_x = "x";
static LVar<standard_allocator> var_y = "y";

inline VarEnv<standard_allocator> env_with(const char* fzn) {
  VarEnv<standard_allocator> env;
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  IDiagnostics<F> diagnostics;
  if(f->is(F::Seq) && f->sig() == AND) {
    for(int i = 0; i < f->seq().size(); ++i) {
      AVar avar;
      EXPECT_TRUE(env.interpret(f->seq(i), avar, diagnostics));
    }
  }
  else {
    AVar avar;
    EXPECT_TRUE(env.interpret(*f, avar, diagnostics));
  }
  return std::move(env);
}

/** Initialize an environment with single integer variable named `x` in the abstract domain typed `0`. */
inline VarEnv<standard_allocator> env_with_x() {
  return env_with("var int: x :: abstract(0);");
}

template<IKind kind, class L>
void interpret_must_error(const char* fzn, VarEnv<standard_allocator> env = VarEnv<standard_allocator>{}) {
  static_assert(kind == IKind::TELL || L::is_abstract_universe);
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  IDiagnostics<F> diagnostics;
  L value = make_bot<L>(env);
  bool res;
  if constexpr(L::is_abstract_universe) {
    res = ginterpret_in<kind, true>(*f, env, value, diagnostics);
  }
  else {
    if constexpr(kind == IKind::TELL) {
      typename L::template tell_type<standard_allocator> tell;
      res = top_level_ginterpret_in<kind, true>(value, *f, env, tell, diagnostics);
    }
    else {
      typename L::template ask_type<standard_allocator> ask;
      res = top_level_ginterpret_in<kind, true>(value, *f, env, ask, diagnostics);
    }
  }
  if(res) {
    EXPECT_TRUE(false) << "The formula should not be interpretable: ";
    value.print();
    printf("\n");
  }
}

template<class L>
void both_interpret_must_error(const char* fzn, VarEnv<standard_allocator> env = VarEnv<standard_allocator>{}) {
  interpret_must_error<IKind::TELL, L>(fzn, env);
  interpret_must_error<IKind::ASK, L>(fzn, env);
}

template <IKind kind, class L>
void interpret_must_succeed(const char* fzn, L& value, VarEnv<standard_allocator>& env, bool has_warning = false) {
  static_assert(kind == IKind::TELL || L::is_abstract_universe);
  using F = TFormula<standard_allocator>;
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  IDiagnostics<F> diagnostics;
  bool res;
  if constexpr(L::is_abstract_universe) {
    res = ginterpret_in<kind, true>(*f, env, value, diagnostics);
  }
  else {
    if constexpr(kind == IKind::TELL) {
      res = interpret_and_tell<true>(*f, env, value, diagnostics);
    }
    else {
      typename L::template ask_type<standard_allocator> ask;
      res = top_level_ginterpret_in<kind, true>(value, *f, env, ask, diagnostics);
    }
  }
  if(!res) {
    diagnostics.print();
    EXPECT_TRUE(false) << "The formula should be interpretable: " << fzn;
  }
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
}

template <class L>
L create_and_interpret_and_tell(const char* fzn, VarEnv<standard_allocator>& env, bool has_warning = false) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  IDiagnostics<F> diagnostics;
  auto value = create_and_interpret_and_tell<L, true>(*f, env, diagnostics);
  if(diagnostics.is_fatal()) {
    diagnostics.print();
  }
  EXPECT_FALSE(diagnostics.is_fatal());
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
  EXPECT_TRUE(value.has_value());
  return std::move(value.value());
}

template <class L>
L create_and_interpret_and_tell(const char* fzn, bool has_warning = false) {
  VarEnv<standard_allocator> env;
  return create_and_interpret_and_tell<L>(fzn, env, has_warning);
}

template <IKind kind, class L>
void expect_interpret_equal_to(const char* fzn, const L& expect, VarEnv<standard_allocator> env = VarEnv<standard_allocator>{}, bool has_warning = false) {
  L value{L::bot()};
  interpret_must_succeed<kind>(fzn, value, env, has_warning);
  EXPECT_EQ(value, expect);
}

/** When we expect an exact interpretation. */
template <class L>
void expect_both_interpret_equal_to(const char* fzn, const L& expect, const VarEnv<standard_allocator>& env = VarEnv<standard_allocator>{}, bool has_warning = false) {
  expect_interpret_equal_to<IKind::TELL>(fzn, expect, env, has_warning);
  expect_interpret_equal_to<IKind::ASK>(fzn, expect, env, has_warning);
}

template <class L>
bool interpret_and_ask(const char* fzn, L& value, VarEnv<standard_allocator>& env, bool has_warning = false) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  IDiagnostics<F> diagnostics;
  typename L::template ask_type<standard_allocator> ask;
  if(!ginterpret_in<IKind::ASK, true>(value, *f, env, ask, diagnostics)) {
    diagnostics.print();
    EXPECT_TRUE(false) << "The formula should be (ask-)interpretable: " << fzn;
  }
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
  return value.ask(ask);
}

/** We must have `A::bot() < mid < A::top()`. */
template <class A>
void bot_top_test(const A& mid) {
  A bot = A::bot();
  A top = A::top();
  EXPECT_TRUE(bot.is_bot());
  EXPECT_TRUE(top.is_top());
  EXPECT_FALSE(top.is_bot());
  EXPECT_FALSE(bot.is_top());
  EXPECT_TRUE(bot <= top);
  EXPECT_TRUE(top >= bot);
  EXPECT_TRUE(bot < top);
  EXPECT_TRUE(top > bot);
  EXPECT_TRUE(bot == bot);
  EXPECT_TRUE(top == top);
  EXPECT_TRUE(top != bot);
  EXPECT_FALSE(top == bot);

  EXPECT_FALSE(mid.is_bot());
  EXPECT_FALSE(mid.is_top());
  EXPECT_TRUE(bot < mid) << bot << " " << mid;
  EXPECT_TRUE(mid < top);

  expect_interpret_equal_to<IKind::TELL, A>("constraint true;", bot);
  expect_interpret_equal_to<IKind::TELL, A>("constraint false;", top);
  if constexpr(A::is_abstract_universe) {
    expect_interpret_equal_to<IKind::ASK, A>("constraint true;", bot);
    expect_interpret_equal_to<IKind::ASK, A>("constraint false;", top);
  }
}

template <class A>
void join_one_test(const A& a, const A& b, const A& expect, bool has_changed_expect, bool test_tell = true) {
  local::BInc has_changed = local::BInc::bot();
  EXPECT_EQ(join(a, b), expect)  << "join(" << a << ", " << b << ")";;
  if(test_tell) {
    A c(a);
    EXPECT_EQ(c.tell(b, has_changed), expect) << a << ".tell(" << b << ") == " << expect;
    EXPECT_EQ(has_changed, has_changed_expect) << a << ".tell(" << b << ")";
  }
}

template <class A>
void meet_one_test(const A& a, const A& b, const A& expect, bool has_changed_expect, bool test_tell = true) {
  local::BInc has_changed = local::BInc::bot();
  EXPECT_EQ(meet(a, b), expect) << "meet(" << a << ", " << b << ")";
  if(test_tell) {
    A c(a);
    EXPECT_EQ(c.dtell(b, has_changed), expect) << c << ".dtell(" << b << ")";
    EXPECT_EQ(has_changed, has_changed_expect) << c << ".dtell(" << b << ")";
  }
}

// `a` and `b` are supposed ordered and `a <= b`.
template <class A>
void join_meet_generic_test(const A& a, const A& b, bool commutative_tell = true, bool test_tell_a_b = true) {
  // Reflexivity
  join_one_test(a, a, a, false);
  meet_one_test(a, a, a, false);
  join_one_test(b, b, b, false);
  meet_one_test(b, b, b, false);
  // Coherency of join/meet w.r.t. ordering
  join_one_test(a, b, b, a != b, test_tell_a_b);
  join_one_test(b, a, b, false, commutative_tell);
  // // Commutativity
  meet_one_test(a, b, a, false, test_tell_a_b);
  meet_one_test(b, a, a, a != b, commutative_tell);
  // // Absorbing
  meet_one_test(a, A::top(), a, false);
  meet_one_test(b, A::top(), b, false);
  join_one_test(a, A::top(), A::top(), !a.is_top());
  join_one_test(b, A::top(), A::top(), !b.is_top());
  meet_one_test(a, A::bot(), A::bot(), !a.is_bot());
  meet_one_test(b, A::bot(), A::bot(), !b.is_bot());
  join_one_test(a, A::bot(), a, false);
  join_one_test(b, A::bot(), b, false);
}

template <Sig sig, class A, class R = A>
void generic_unary_fun_test() {
  if constexpr(R::is_supported_fun(sig)) {
    EXPECT_EQ((R::template fun<sig>(A::bot())), R::bot());
    EXPECT_EQ((R::template fun<sig>(A::top())), R::top());
  }
}

template <class A>
void generic_abs_test() {
  A a;
  auto env = env_with_x();
  interpret_must_succeed<IKind::TELL>("constraint int_ge(x, 0);", a, env);
  EXPECT_EQ((A::template fun<ABS>(A::bot())), a);
}

template <Sig sig, class A, class R = A>
void generic_binary_fun_test(const A& a) {
  if constexpr(R::is_supported_fun(sig)) {
    battery::print(sig);
    EXPECT_EQ((R::template fun<sig>(A::bot(), A::bot())), R::bot());
    EXPECT_EQ((R::template fun<sig>(A::top(), A::bot())), R::top());
    EXPECT_EQ((R::template fun<sig>(A::bot(), A::top())), R::top());
    EXPECT_EQ((R::template fun<sig>(A::top(), a)), R::top());
    EXPECT_EQ((R::template fun<sig>(a, A::top())), R::top());
    EXPECT_EQ((R::template fun<sig>(A::bot(), a)), R::bot());
    EXPECT_EQ((R::template fun<sig>(a, A::bot())), R::bot());
  }
}

template <class A, class R = A>
void generic_arithmetic_fun_test(const A& a) {
  generic_unary_fun_test<NEG, A, R>();
  generic_binary_fun_test<ADD, A, R>(a);
  generic_binary_fun_test<SUB, A, R>(a);
  generic_binary_fun_test<MUL, A, R>(a);
  generic_binary_fun_test<TDIV, A, R>(a);
  generic_binary_fun_test<FDIV, A, R>(a);
  generic_binary_fun_test<CDIV, A, R>(a);
  generic_binary_fun_test<EDIV, A, R>(a);
  generic_binary_fun_test<TMOD, A, R>(a);
  generic_binary_fun_test<FMOD, A, R>(a);
  generic_binary_fun_test<CMOD, A, R>(a);
  generic_binary_fun_test<EMOD, A, R>(a);
  generic_binary_fun_test<POW, A, R>(a);
}

/** Check that \f$ \llbracket . \rrbracket = \llbracket . \rrbracket \circ \rrbacket . \llbracket \circ \llbracket . \rrbracket \f$ */
template <class L>
void check_interpret_idempotence(const char* fzn) {
  using F = TFormula<standard_allocator>;
  VarEnv<standard_allocator> env1, env2;
  L value1 = create_and_interpret_and_tell<L>(fzn, env1);
  F f1 = value1.deinterpret(env1);
  f1.print(true);
  printf("\n");
  L value2 = make_bot<L>(env2);
  IDiagnostics<F> diagnostics;
  EXPECT_TRUE(interpret_and_tell(f1, env2, value2, diagnostics));
  EXPECT_EQ(value1, value2);
  F f2 = value2.deinterpret(env2);
  f2.print(true);
  printf("\n");
  EXPECT_EQ(f1, f2);
}

#endif
