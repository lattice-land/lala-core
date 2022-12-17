// Copyright 2021 Pierre Talbot

#ifndef GENERIC_UNIVERSE_TEST_HPP
#define GENERIC_UNIVERSE_TEST_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "thrust/optional.h"
#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<StandardAllocator>;

static LVar<StandardAllocator> var_x = "x";
static LVar<StandardAllocator> var_y = "y";

inline VarEnv<StandardAllocator> init_env() {
  VarEnv<StandardAllocator> env;
  auto f = parse_flatzinc_str<StandardAllocator>("var int: x :: abstract(0);");
  EXPECT_TRUE(f);
  env.interpret(*f);
  return std::move(env);
}

/** `appx` is the approximation kind of the top-level conjunction. */
template <class L>
L interpret_to(const std::string& fzn, VarEnv<StandardAllocator>& env, Approx appx = EXACT) {
  auto f = parse_flatzinc_str<StandardAllocator>(fzn);
  EXPECT_TRUE(f);
  f->approx_as(appx);
  f->print(true, true);
  IResult<L, F> r = L::interpret(*f, env);
  if(!r.is_ok()) {
    r.print_diagnostics();
  }
  EXPECT_TRUE(r.is_ok());
  return std::move(r.value());
}

template <class L>
L interpret_to(const char* fzn, Approx appx = EXACT) {
  using F = TFormula<StandardAllocator>;
  VarEnv<StandardAllocator> env = init_env();
  return interpret_to<L>(fzn, env);
}

template <class L>
L interpret_to2(const char* fzn, Approx appx = EXACT) {
  VarEnv<StandardAllocator> env;
  return interpret_to<L>(fzn, env, appx);
}

template<class L>
void must_interpret_to(VarEnv<StandardAllocator>& env, const char* fzn, const L& expect, bool has_warning = false) {
  using F = TFormula<StandardAllocator>;
  auto f = parse_flatzinc_str<StandardAllocator>(fzn);
  EXPECT_TRUE(f);
  f->print(true, true);
  std::cout << std::endl;
  IResult<L, F> r = L::interpret(*f, env);
  std::cout << fzn << std::endl;
  if(!r.is_ok()) {
    r.print_diagnostics();
  }
  EXPECT_TRUE(r.is_ok());
  EXPECT_EQ(r.has_warning(), has_warning);
  EXPECT_EQ(r.value(), expect);
}

template<class L>
void must_interpret_to(const char* fzn, const L& expect, bool has_warning = false) {
  VarEnv<StandardAllocator> env;
  must_interpret_to(env, fzn, expect, has_warning);
}

template<class L>
void must_error(VarEnv<StandardAllocator>& env, const char* fzn) {
  using F = TFormula<StandardAllocator>;
  auto f = parse_flatzinc_str<StandardAllocator>(fzn);
  EXPECT_TRUE(f);
  f->print(true, true);
  IResult<L, F> r = L::interpret(*f, env);
  std::cout << fzn << std::endl;
  if(r.is_ok()) {
    std::cout << r.value() << std::endl;
  }
  EXPECT_FALSE(r.is_ok());
}

template<class L>
void must_error(const char* fzn) {
  VarEnv<StandardAllocator> env;
  must_error<L>(env, fzn);
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

  must_interpret_to<A>("constraint true;", bot);
  must_interpret_to<A>("constraint false;", top);
}

template <class A>
void join_one_test(const A& a, const A& b, const A& expect, bool has_changed_expect, bool test_tell = true) {
  local::BInc has_changed = local::BInc::bot();
  EXPECT_EQ(join(a, b), expect)  << "join(" << a << ", " << b << ")";;
  if(test_tell) {
    A c(a);
    EXPECT_EQ(c.tell(b, has_changed), expect) << c << ".tell(" << b << ")";
    EXPECT_EQ(has_changed, has_changed_expect) << c << ".tell(" << b << ")";
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

template <Approx appx, Sig sig, class A>
void generic_unary_fun_test() {
  if constexpr(A::is_supported_fun(appx, sig)) {
    if constexpr(sig == ABS) {
      if constexpr(appx == EXACT) {
        EXPECT_EQ((A::template fun<appx, sig>(A::bot())), interpret_to<A>("constraint int_ge(x, 0) :: exact;"));
      }
      else if constexpr(appx == UNDER) {
        EXPECT_TRUE((A::template fun<appx, sig>(A::bot()) >= interpret_to<A>("constraint int_ge(x, 0) :: under;")));
      }
      else {
        EXPECT_TRUE((A::template fun<appx, sig>(A::bot()) <= interpret_to<A>("constraint int_ge(x, 0) :: over;")));
      }
    }
    else {
      EXPECT_EQ((A::template fun<appx, sig>(A::bot())), A::bot());
      EXPECT_EQ((A::template fun<appx, sig>(A::top())), A::top());
    }
  }
}

template <Approx appx, Sig sig, class A>
void generic_binary_fun_test(const A& a) {
  if constexpr(A::is_supported_fun(appx, sig)) {
    battery::print(sig);
    EXPECT_EQ((A::template fun<appx, sig>(A::bot(), A::bot())), A::bot());
    EXPECT_EQ((A::template fun<appx, sig>(A::top(), A::bot())), A::top());
    EXPECT_EQ((A::template fun<appx, sig>(A::bot(), A::top())), A::top());
    EXPECT_EQ((A::template fun<appx, sig>(A::top(), a)), A::top());
    EXPECT_EQ((A::template fun<appx, sig>(a, A::top())), A::top());
    EXPECT_EQ((A::template fun<appx, sig>(A::bot(), a)), A::bot());
    EXPECT_EQ((A::template fun<appx, sig>(a, A::bot())), A::bot());
  }
}

template <Approx appx, class A>
void generic_arithmetic_fun_test(const A& a) {
  generic_unary_fun_test<appx, NEG, A>();
  generic_unary_fun_test<appx, ABS, A>();
  generic_binary_fun_test<appx, ADD>(a);
  generic_binary_fun_test<appx, SUB>(a);
  generic_binary_fun_test<appx, MUL>(a);
  generic_binary_fun_test<appx, TDIV>(a);
  generic_binary_fun_test<appx, FDIV>(a);
  generic_binary_fun_test<appx, CDIV>(a);
  generic_binary_fun_test<appx, EDIV>(a);
  generic_binary_fun_test<appx, TMOD>(a);
  generic_binary_fun_test<appx, FMOD>(a);
  generic_binary_fun_test<appx, CMOD>(a);
  generic_binary_fun_test<appx, EMOD>(a);
  generic_binary_fun_test<appx, POW>(a);
  generic_binary_fun_test<appx, MIN>(a);
  generic_binary_fun_test<appx, MAX>(a);
}

#endif
