// Copyright 2021 Pierre Talbot

#ifndef GENERIC_UNIVERSE_TEST_HPP
#define GENERIC_UNIVERSE_TEST_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "thrust/optional.h"
#include "ast.hpp"
#include "allocator.hpp"
#include "utility.hpp"

using namespace lala;

using F = TFormula<StandardAllocator>;

static LVar<StandardAllocator> var_x = "x";

/** We must have `A::bot() < mid < A::top()`. */
template <typename A>
void bot_top_test(A mid) {
  A bot = A::bot();
  A top = A::top();
  EXPECT_EQ(bot, A::bot());
  EXPECT_EQ(top, A::top());
  EXPECT_NE(top, bot);
  EXPECT_TRUE(top.is_top());
  EXPECT_TRUE(bot.is_bot());
  EXPECT_FALSE(top.is_bot());
  EXPECT_FALSE(bot.is_top());

  EXPECT_FALSE(mid.is_bot());
  EXPECT_FALSE(mid.is_top());
  EXPECT_NE(bot, mid);
  EXPECT_NE(top, mid);
}

template <typename A>
void join_one_test(A a, A b, A expect, bool has_changed_expect) {
  bool has_changed = false;
  EXPECT_EQ(a.clone().join(b), expect);
  EXPECT_EQ(a.tell(b, has_changed), expect);
  EXPECT_EQ(has_changed, has_changed_expect);
}

template <typename A>
void meet_one_test(A a, A b, A expect, bool has_changed_expect) {
  bool has_changed = false;
  EXPECT_EQ(a.clone().meet(b), expect);
  EXPECT_EQ(a.dtell(b, has_changed), expect);
  EXPECT_EQ(has_changed, has_changed_expect);
}

// `a` and `b` are supposed ordered and `a <= b`.
template <typename A>
void join_meet_generic_test(A a, A b) {
  // Reflexivity
  join_one_test(a, a, a, false);
  meet_one_test(a, a, a, false);
  join_one_test(b, b, b, false);
  meet_one_test(b, b, b, false);
  // Coherency of join/meet w.r.t. ordering
  join_one_test(a, b, b, a != b);
  join_one_test(b, a, b, false);
  // Commutativity
  meet_one_test(a, b, a, false);
  meet_one_test(b, a, a, a != b);
  // Absorbing
  meet_one_test(a, A::top(), a, false);
  meet_one_test(b, A::top(), b, false);
  join_one_test(a, A::top(), A::top(), a != A::top());
  join_one_test(b, A::top(), A::top(), b != A::top());
  meet_one_test(a, A::bot(), A::bot(), a != A::bot());
  meet_one_test(b, A::bot(), A::bot(), b != A::bot());
  join_one_test(a, A::bot(), a, false);
  join_one_test(b, A::bot(), b, false);
}

template<typename A>
void generic_order_test(A element) {
  EXPECT_EQ(element.order(A::top()), true);
  EXPECT_EQ(element.order(A::bot()), false);
  EXPECT_EQ(A::bot().order(A::bot()), true);
  EXPECT_EQ(A::top().order(A::top()), true);
  EXPECT_EQ(A::top().order(A::bot()), false);
  EXPECT_EQ(A::bot().order(A::top()), true);
  EXPECT_EQ(A::top().order(element), false);
}

template<typename A>
using SplitSeq = DArray<A, StandardAllocator>;

template<typename A>
SplitSeq<A> make_singleton(A x) {
  return SplitSeq<A>({x});
}

template<typename A>
SplitSeq<A> make_empty() {
  return SplitSeq<A>();
}

template<typename A>
void generic_split_test(A element) {
  EXPECT_EQ(element.split(), make_singleton(element));
  EXPECT_EQ(A::top().split(), make_empty<A>());
  EXPECT_EQ(A::bot().split(), make_singleton(A::bot()));
}

template<typename A>
void generic_deinterpret_test() {
  EXPECT_EQ(A::bot().deinterpret(var_x), F::make_true());
  EXPECT_EQ(A::top().deinterpret(var_x), F::make_false());
}

template<typename Universe>
void test_formula(const F& f, thrust::optional<Universe> expect) {
  thrust::optional<Universe> j = Universe::interpret(f);
  EXPECT_EQ(j.has_value(), expect.has_value());
  EXPECT_EQ(j, expect);
}

template<typename Universe>
void test_interpret(Sig sig, Approx appx, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  test_formula<Universe>(
    make_v_op_z(var_x, sig, elem, appx, standard_allocator),
    expect);
}

template<typename Universe>
void test_all_interpret(Sig sig, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  Approx appxs[3] = {EXACT, UNDER, OVER};
  for(int i = 0; i < 3; ++i) {
    test_interpret<Universe>(sig, appxs[i], elem, expect);
  }
}

template<typename Universe>
void test_exact_interpret(Sig sig, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, EXACT, elem, expect);
}

template<typename Universe>
void test_under_interpret(Sig sig, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, UNDER, elem, expect);
}

template<typename Universe>
void test_over_interpret(Sig sig, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, OVER, elem, expect);
}
#endif
