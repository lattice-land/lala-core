// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_GENERIC_UNIVERSE_TEST_HPP
#define LALA_CORE_GENERIC_UNIVERSE_TEST_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

static LVar<standard_allocator> var_x = "x";
static LVar<standard_allocator> var_y = "y";

inline VarEnv<standard_allocator> init_env() {
  VarEnv<standard_allocator> env;
  auto f = parse_flatzinc_str<standard_allocator>("var int: x :: abstract(0);");
  EXPECT_TRUE(f);
  AVar avar;
  IDiagnostics<F> diagnostics;
  EXPECT_TRUE(env.interpret(*f, avar, diagnostics));
  return std::move(env);
}

namespace impl {
  template <bool is_tell, class L>
  bool interpretation(const F& f, VarEnv<standard_allocator>& env, L& value, IDiagnostics<F>& diagnostics) {
    if constexpr(is_tell) {
      return L::template interpret_tell<true>(f, env, value, diagnostics);
    }
    else {
      return L::template interpret_ask<true>(f, env, value, diagnostics);
    }
  }

  template <bool is_tell, class L>
  L interpret_to(const std::string& fzn, VarEnv<standard_allocator>& env) {
    auto f = parse_flatzinc_str<standard_allocator>(fzn);
    EXPECT_TRUE(f);
    std::cout << (is_tell ? "tell" : "ask") << ": ";
    f->print(true);
    IDiagnostics<F> diagnostics;
    L value{L::bot()};
    if(!interpretation<is_tell, L>(*f, env, value, diagnostics)) {
      diagnostics.print();
      EXPECT_TRUE(false) << "The formula should be interpretable.";
    }
    return value;
  }
}

template <class L>
L interpret_tell_to(const std::string& fzn, VarEnv<standard_allocator>& env) {
  return ::impl::interpret_to<true, L>(fzn, env);
}

template <class L>
L interpret_ask_to(const std::string& fzn, VarEnv<standard_allocator>& env) {
  return ::impl::interpret_to<false, L>(fzn, env);
}

template <class L>
L interpret_to(const std::string& fzn, VarEnv<standard_allocator>& env) {
  auto tell = interpret_tell_to<L>(fzn, env);
  auto ask = interpret_ask_to<L>(fzn, env);
  EXPECT_EQ(tell, ask) << "The tell and ask interpretations should be equal.";
  return tell;
}

template <class L>
L interpret_to(const char* fzn) {
  using F = TFormula<standard_allocator>;
  VarEnv<standard_allocator> env = init_env();
  return interpret_to<L>(fzn, env);
}

template <class L>
L interpret_ask_to(const char* fzn) {
  using F = TFormula<standard_allocator>;
  VarEnv<standard_allocator> env = init_env();
  return interpret_ask_to<L>(fzn, env);
}

template <class L>
L interpret_tell_to(const char* fzn) {
  using F = TFormula<standard_allocator>;
  VarEnv<standard_allocator> env = init_env();
  return interpret_tell_to<L>(fzn, env);
}

template <class L>
L interpret_to2(const char* fzn) {
  VarEnv<standard_allocator> env;
  return interpret_to<L>(fzn, env);
}

template <class L>
L interpret_tell_to2(const char* fzn) {
  VarEnv<standard_allocator> env;
  return interpret_tell_to<L>(fzn, env);
}

template <class L>
L interpret_ask_to2(const char* fzn) {
  VarEnv<standard_allocator> env;
  return interpret_ask_to<L>(fzn, env);
}

template <class L>
void interpret_and_tell(L& a, const char* fzn, VarEnv<standard_allocator>& env) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  f->print(true);
  IDiagnostics<F> diagnostics;
  typename L::template tell_type<standard_allocator> tell;
  if(!a.template interpret_tell_in<true>(*f, env, tell, diagnostics)) {
    diagnostics.print();
  }
  local::BInc has_changed;
  a.tell(std::move(tell), has_changed);
  EXPECT_TRUE(has_changed);
}

template <class L>
void interpret_and_ask(L& a, const char* fzn, bool expect) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  f->print(true);
  auto env = init_env();
  IDiagnostics<F> diagnostics;
  typename L::template ask_type<standard_allocator> ask;
  if(!a.template interpret_ask_in<true>(*f, env, ask, diagnostics)) {
    diagnostics.print();
  }
  EXPECT_EQ(a.ask(std::move(ask)), expect);
}

namespace impl {
  template<bool is_tell, class L>
  void must_interpret_to(VarEnv<standard_allocator>& env, const char* fzn, const L& expect, bool has_warning = false) {
    using F = TFormula<standard_allocator>;
    auto f = parse_flatzinc_str<standard_allocator>(fzn);
    EXPECT_TRUE(f);
    std::cout << (is_tell ? "tell" : "ask") << ": ";
    f->print(true);
    std::cout << std::endl;
    IDiagnostics<F> diagnostics;
    L value{L::bot()};
    if(!interpretation<is_tell, L>(*f, env, value, diagnostics)) {
      diagnostics.print();
      EXPECT_TRUE(false) << "The formula should be interpretable: " << fzn;
    }
    EXPECT_EQ(diagnostics.has_warning(), has_warning);
    EXPECT_EQ(value, expect);
  }

  template<bool is_tell, class L>
  void must_interpret_to(const char* fzn, const L& expect, bool has_warning = false) {
    VarEnv<standard_allocator> env;
    must_interpret_to<is_tell>(env, fzn, expect, has_warning);
  }
}

template<class L>
void must_interpret_tell_to(VarEnv<standard_allocator>& env, const char* fzn, const L& expect, bool has_warning = false) {
  ::impl::must_interpret_to<true>(env, fzn, expect, has_warning);
}

template<class L>
void must_interpret_ask_to(VarEnv<standard_allocator>& env, const char* fzn, const L& expect, bool has_warning = false) {
  ::impl::must_interpret_to<false>(env, fzn, expect, has_warning);
}

template<class L>
void must_interpret_to(VarEnv<standard_allocator>& env, const char* fzn, const L& expect, bool has_warning = false) {
  must_interpret_tell_to(env, fzn, expect, has_warning);
  must_interpret_ask_to(env, fzn, expect, has_warning);
}

template<class L>
void must_interpret_tell_to(const char* fzn, const L& expect, bool has_warning = false) {
  ::impl::must_interpret_to<true>(fzn, expect, has_warning);
}

template<class L>
void must_interpret_ask_to(const char* fzn, const L& expect, bool has_warning = false) {
  ::impl::must_interpret_to<false>(fzn, expect, has_warning);
}

template<class L>
void must_interpret_to(const char* fzn, const L& expect, bool has_warning = false) {
  must_interpret_tell_to(fzn, expect, has_warning);
  must_interpret_ask_to(fzn, expect, has_warning);
}

namespace impl {
  template<bool is_tell, class L>
  void must_error(VarEnv<standard_allocator>& env, const char* fzn) {
    auto f = parse_flatzinc_str<standard_allocator>(fzn);
    EXPECT_TRUE(f);
    std::cout << (is_tell ? "tell" : "ask") << ": ";
    f->print(true);

    IDiagnostics<F> diagnostics;
    L value{L::bot()};
    if(interpretation<is_tell, L>(*f, env, value, diagnostics)) {
      EXPECT_TRUE(false) << "The formula should not be interpretable: ";
      value.print();
      printf("\n");
    }
  }

  template<bool is_tell, class L>
  void must_error(const char* fzn) {
    VarEnv<standard_allocator> env;
    must_error<is_tell, L>(env, fzn);
  }
}

template<class L>
void must_error_tell(VarEnv<standard_allocator>& env, const char* fzn) {
  ::impl::must_error<true, L>(env, fzn);
}

template<class L>
void must_error_ask(VarEnv<standard_allocator>& env, const char* fzn) {
  ::impl::must_error<false, L>(env, fzn);
}

template<class L>
void must_error(VarEnv<standard_allocator>& env, const char* fzn) {
  must_error_tell<L>(env, fzn);
  must_error_ask<L>(env, fzn);
}

template<class L>
void must_error_tell(const char* fzn) {
  ::impl::must_error<true, L>(fzn);
}

template<class L>
void must_error_ask(const char* fzn) {
  ::impl::must_error<false, L>(fzn);
}

template<class L>
void must_error(const char* fzn) {
  must_error_tell<L>(fzn);
  must_error_ask<L>(fzn);
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

  must_interpret_tell_to<A>("constraint true;", bot);
  must_interpret_tell_to<A>("constraint false;", top);
  if constexpr(A::is_abstract_universe) {
    must_interpret_ask_to<A>("constraint true;", bot);
    must_interpret_ask_to<A>("constraint false;", top);
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
  EXPECT_EQ((A::template fun<ABS>(A::bot())), interpret_to<A>("constraint int_ge(x, 0);"));
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

  template <bool diagnose = false, class F, class Env, class Alloc2 = allocator_type>
  CUDA NI static std::optional<this_type> interpret_tell(const F& f, Env& env, IDiagnostics<F>& diagnostics, allocator_type alloc = allocator_type(), Alloc2 alloc2 = Alloc2()) {
    auto snap = env.snapshot();
    size_t ty = env.extends_abstract_dom();
    this_type store(ty, alloc);
    tell_type<Alloc2> tell(alloc2);
    if(store.interpret_tell_in(f, env, tell, diagnostics)) {
      store.tell(tell);
      return {store};
    }
    else {
      env.restore(snap);
      return {};
    }
  }

/** Check that $\llbracket . \rrbracket = \llbracket . \rrbracket \circ \rrbacket . \llbracket \circ \llbracket . \rrbracket. */
template <class L>
void check_interpret_idempotence(const char* fzn) {
  VarEnv<standard_allocator> env1;
  using F = TFormula<standard_allocator>;
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  f->print(true);
  printf("\n");

  IDiagnostics<F> diagnostics;
  L value{L::bot()};
  if(!L::template interpret_tell<true>(*f, env1, value, diagnostics)) {
    diagnostics.print();
    EXPECT_TRUE(false) << "The formula should be interpretable.";
  }

  F f2 = value.deinterpret(env1);
  f2.print(true);
  printf("\n");
  VarEnv<standard_allocator> env2;
  IDiagnostics<F> diagnostics2;
  L value2{L::bot()};
  if(!L::interpret_tell(f2, env2, value2, diagnostics2)) {
    diagnostics2.print();
    EXPECT_TRUE(false) << "Deinterpreted formulas should be (re)-interpretable.";
  }
  EXPECT_EQ(value, value2);

  F f3 = value2.deinterpret(env2);
  f3.print(true);
  printf("\n");
  EXPECT_EQ(f2, f3);
}

#endif
