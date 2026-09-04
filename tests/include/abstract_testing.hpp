// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_GENERIC_UNIVERSE_TEST_HPP
#define LALA_CORE_GENERIC_UNIVERSE_TEST_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "lala/logic/logic.hpp"

using namespace lala;
using namespace battery;

/** Apply the abstract projection `fun` and return the result, instead of meeting it in place. */
template <class A, class R = A>
R project_fun(Sig fun, const A& a, const A& b) {
  R r{};
  r.project(fun, a, b);
  return r;
}

template <class A, class R = A>
R project_fun(Sig fun, const A& a) {
  R r{};
  r.project(fun, a);
  return r;
}

using F = TFormula<standard_allocator>;

static LVar<standard_allocator> var_x = "x";
static LVar<standard_allocator> var_y = "y";

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
}

template <class A>
void join_one_test(const A& a, const A& b, const A& expect, bool has_changed_expect, bool test_tell = true) {
  EXPECT_EQ(join(a, b), expect)  << "join(" << a << ", " << b << ")";;
  if(test_tell) {
    A c(a);
    EXPECT_EQ(c.join(b), has_changed_expect) << a << ".join(" << b << ") == " << expect;
    EXPECT_EQ(c, expect) << a << ".join(" << b << ")";
  }
}

template <class A>
void meet_one_test(const A& a, const A& b, const A& expect, bool has_changed_expect, bool test_tell = true) {
  EXPECT_EQ(meet(a, b), expect) << "meet(" << a << ", " << b << ")";
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
  generic_binary_fun_test<A, R>(TDIV, a);
  generic_binary_fun_test<A, R>(FDIV, a);
  generic_binary_fun_test<A, R>(CDIV, a);
  generic_binary_fun_test<A, R>(EDIV, a);
  generic_binary_fun_test<A, R>(TMOD, a);
  generic_binary_fun_test<A, R>(FMOD, a);
  generic_binary_fun_test<A, R>(CMOD, a);
  generic_binary_fun_test<A, R>(EMOD, a);
  generic_binary_fun_test<A, R>(POW, a);
}

#endif
