// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_GENERIC_UNIVERSE_TEST_HPP
#define LALA_CORE_GENERIC_UNIVERSE_TEST_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "lala/logic/logic.hpp"
#include "lala/logic/ternarize.hpp"
#include "lala/universes/arith_bound.hpp"
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
  IDiagnostics diagnostics;
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
inline VarEnv<standard_allocator> env_with_x(const char flag = 'I') {
  if (flag == 'I') {
    return env_with("var int: x :: abstract(0);");
  }
  else if (flag == 'F') {
    return env_with("var float: x :: abstract(0);");
  }
}

template<IKind kind, class L>
void interpret_must_error(const char* fzn, VarEnv<standard_allocator> env = VarEnv<standard_allocator>{}) {
  static_assert(kind == IKind::TELL || L::is_abstract_universe);
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  IDiagnostics diagnostics;
  L value = make_top<L>(env);
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

template <IKind kind, bool ternarize_formula = false, class L>
void interpret_must_succeed(const char* fzn, L& value, VarEnv<standard_allocator>& env, bool has_warning = false) {
  static_assert(kind == IKind::TELL || L::is_abstract_universe);
  using F = TFormula<standard_allocator>;
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  if(ternarize_formula) {
    *f = ternarize(*f, env);
    f->print(); printf("\n");
  }
  *f = normalize(*f);
  IDiagnostics diagnostics;
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
  if(diagnostics.has_warning() && !has_warning) {
    diagnostics.print();
    EXPECT_TRUE(false) << "The formula generates a warning but should not: " << fzn;
  }
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
}

template <class L, bool ternarize_formula = false, class Typing>
L create_and_interpret_and_type_and_tell(const char* fzn, VarEnv<standard_allocator>& env, Typing&& typing, bool has_warning = false) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  if(ternarize_formula) {
    *f = ternarize(*f, env);
    f->print(); printf("\n");
  }
  *f = normalize(*f);
  printf("normalized:\n"); f->print(); printf("\n");
  typing(*f);
  IDiagnostics diagnostics;
  auto value = create_and_interpret_and_tell<L, true>(*f, env, diagnostics);
  if(diagnostics.is_fatal()) {
    diagnostics.print();
  }
  EXPECT_FALSE(diagnostics.is_fatal());
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
  EXPECT_TRUE(value.has_value());
  return std::move(value.value());
}

template <class L, bool ternarize_formula = false>
L create_and_interpret_and_tell(const char* fzn, VarEnv<standard_allocator>& env, bool has_warning = false) {
  return create_and_interpret_and_type_and_tell<L, ternarize_formula>(fzn, env, [](const F&){}, has_warning);
}

template <class L, bool ternarize_formula = false>
L create_and_interpret_and_tell(const char* fzn, bool has_warning = false) {
  VarEnv<standard_allocator> env;
  return create_and_interpret_and_tell<L, ternarize_formula>(fzn, env, has_warning);
}

template <IKind kind, class L>
void expect_interpret_equal_to(const char* fzn, const L& expect, VarEnv<standard_allocator> env = VarEnv<standard_allocator>{}, bool has_warning = false) {
  L value{L::top()};
  interpret_must_succeed<kind>(fzn, value, env, has_warning);
  EXPECT_EQ(value, expect);
}

/** When we expect an exact interpretation. */
template <class L>
void expect_both_interpret_equal_to(const char* fzn, const L& expect, const VarEnv<standard_allocator>& env = VarEnv<standard_allocator>{}, bool has_warning = false) {
  expect_interpret_equal_to<IKind::TELL>(fzn, expect, env, has_warning);
  expect_interpret_equal_to<IKind::ASK>(fzn, expect, env, has_warning);
}

template <class L, bool ternarize_formula = false>
bool interpret_and_ask(const char* fzn, L& value, VarEnv<standard_allocator>& env, bool has_warning = false) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  if(ternarize_formula) {
    *f = ternarize(*f, env);
    f->print(); printf("\n");
  }
  *f = normalize(*f);
  printf("normalized:\n"); f->print(); printf("\n");
  IDiagnostics diagnostics;
  typename L::template ask_type<standard_allocator> ask;
  if(!ginterpret_in<IKind::ASK, true>(value, *f, env, ask, diagnostics)) {
    diagnostics.print();
    EXPECT_TRUE(false) << "The formula should be (ask-)interpretable: " << fzn;
  }
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
  return value.ask(ask);
}

template <class L, class A> 
L help_create_float_interval(A a) {
  if constexpr (std::is_convertible_v<A, std::string>) {
    std::string sa = static_cast<std::string>(a);
    lala::logic_real ltv = lala::impl::string_to_real(sa);

    double ltvlb = battery::get<0>(ltv);
    double ltvub = battery::get<1>(ltv);
    return L(ltvlb, ltvub);
  }
  else if constexpr (std::is_convertible_v<A, double>) {
    double v = static_cast<double>(a);
    return L(v, v);
  }
  else if constexpr (std::is_constructible_v<L, A, A>) {
    return L(a, a);
  }
  else {
    static_assert(!std::is_same_v<A, A>, "help_create_float_interval: unsupported parameter type; provide string, numeric or matching bound type");
  }
}

template <class L, class A, class B> 
L create_float_interval(A a, B b) {
  L new_a = help_create_float_interval<L>(a);
  L new_b = help_create_float_interval<L>(b); 

  // L new_c = fjoin(new_a, new_b);
  // new_a.print();
  // new_b.print();
  // std::cout << std::endl;
  // std::cout << " --------------------- join result " << std::endl;
  // new_c.print();
  // std::cout << std::endl;

  // return new_c;
  return L(new_a.lb(), new_b.ub());
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

  expect_interpret_equal_to<IKind::TELL, A>("constraint true;", top);
  expect_interpret_equal_to<IKind::TELL, A>("constraint false;", bot);
  if constexpr(A::is_abstract_universe) {
    expect_interpret_equal_to<IKind::ASK, A>("constraint true;", top);
    expect_interpret_equal_to<IKind::ASK, A>("constraint false;", bot);
  }
}

template <class A>
void join_one_test(const A& a, const A& b, const A& expect, bool has_changed_expect, bool test_tell = true) {
  EXPECT_EQ(fjoin(a, b), expect)  << "join(" << a << ", " << b << ")";;
  if(test_tell) {
    A c(a);
    EXPECT_EQ(c.join(b), has_changed_expect) << a << ".join(" << b << ") == " << expect;
    EXPECT_EQ(c, expect) << a << ".join(" << b << ")";
  }
}

template <class A>
void meet_one_test(const A& a, const A& b, const A& expect, bool has_changed_expect, bool test_tell = true) {
  EXPECT_EQ(fmeet(a, b), expect) << "meet(" << a << ", " << b << ")";
  if(test_tell) {
    A c(a);
    EXPECT_EQ(c.meet(b), has_changed_expect) << c << ".meet(" << b << ")";
    EXPECT_EQ(c, expect) << c << ".meet(" << b << ")";
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

template <class A, class R = A>
void generic_unary_fun_test(Sig fun) {
  R r{};
  r.project(fun, A::top());
  EXPECT_TRUE(r.is_top());
  EXPECT_FALSE(r.is_bot());
  r.project(fun, A::bot());
  EXPECT_TRUE(r.is_bot());
  EXPECT_FALSE(r.is_top());
}

template <class A>
void generic_abs_test(const char flag = 'I') {
  A a;
  auto env = env_with_x(flag);
  if (flag == 'I') {
    interpret_must_succeed<IKind::TELL>("constraint int_ge(x, 0);", a, env);
  }
  else if (flag == 'F') {
    interpret_must_succeed<IKind::TELL>("constraint float_ge(x, 0.0);", a, env);
  }
  A r{};
  r.project(ABS, A::top());
  EXPECT_EQ(r, a);
}

template <class A, class R = A>
void generic_binary_fun_test(Sig fun, const A& a) {
  battery::print(fun);
  EXPECT_EQ((project_fun<A, R>(fun, A::bot(), A::bot())), R::bot());
  EXPECT_EQ((project_fun<A, R>(fun, A::top(), A::top())), R::top());
  EXPECT_EQ((project_fun<A, R>(fun, A::top(), A::bot())), R::bot());
  EXPECT_EQ((project_fun<A, R>(fun, A::bot(), A::top())), R::bot());
  if(!is_division(fun)) {
    EXPECT_EQ((project_fun<A, R>(fun, A::top(), a)), R::top()) << A::top() << " " << string_of_sig(fun) << " " << a;
    EXPECT_EQ((project_fun<A, R>(fun, a, A::top())), R::top()) << a  << " " << string_of_sig(fun) << " " << A::top();
  }
  EXPECT_EQ((project_fun<A, R>(fun, A::bot(), a)), R::bot());
  EXPECT_EQ((project_fun<A, R>(fun, a, A::bot())), R::bot());
}

template <class A, class R = A>
void generic_arithmetic_fun_test(const A& a) {
  generic_unary_fun_test<A, R>(NEG);
  generic_binary_fun_test<A, R>(ADD, a);
  generic_binary_fun_test<A, R>(SUB, a);
  generic_binary_fun_test<A, R>(MUL, a);
  if constexpr (a.preserve_concrete_covers) {
    generic_binary_fun_test<A, R>(TDIV, a);
    generic_binary_fun_test<A, R>(FDIV, a);
    generic_binary_fun_test<A, R>(CDIV, a);
    generic_binary_fun_test<A, R>(EDIV, a);
    generic_binary_fun_test<A, R>(TMOD, a);
    generic_binary_fun_test<A, R>(FMOD, a);
    generic_binary_fun_test<A, R>(CMOD, a);
    generic_binary_fun_test<A, R>(EMOD, a);
    generic_binary_fun_test<A, R>(DIV, a);
  }
  else {
    generic_binary_fun_test<A, R>(DIV, a);
  }
  generic_binary_fun_test<A, R>(POW, a);
}

/** Check that \f$ \llbracket . \rrbracket = \llbracket . \rrbracket \circ \rrbacket . \llbracket \circ \llbracket . \rrbracket \f$ */
template <class L, bool ternarize_formula = false>
void check_interpret_idempotence(const char* fzn) {
  using F = TFormula<standard_allocator>;
  VarEnv<standard_allocator> env1, env2;
  L value1 = create_and_interpret_and_tell<L, ternarize_formula>(fzn, env1);
  F f1 = value1.deinterpret(env1);
  f1.print(true);
  printf("\n");
  L value2 = make_top<L>(env2);
  IDiagnostics diagnostics;
  EXPECT_TRUE(interpret_and_tell(f1, env2, value2, diagnostics));
  EXPECT_EQ(value1, value2);
  F f2 = value2.deinterpret(env2);
  f2.print(true);
  printf("\n");
  EXPECT_EQ(f1, f2);
}

#endif
