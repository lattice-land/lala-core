// Copyright 2021 Pierre Talbot

#ifndef GENERIC_UNIVERSE_TEST_HPP
#define GENERIC_UNIVERSE_TEST_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "thrust/optional.h"
#include "ast.hpp"
#include "allocator.hpp"
#include "utility.hpp"
#include "arithmetic.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<StandardAllocator>;

static LVar<StandardAllocator> var_x = "x";
static LVar<StandardAllocator> var_y = "y";

#define EXPECT_EQ2(a,b) EXPECT_EQ(unwrap(a), unwrap(b))
#define EXPECT_TRUE2(a) EXPECT_TRUE(unwrap(a))
#define EXPECT_FALSE2(a) EXPECT_FALSE(unwrap(a))

/** We must have `A::bot() < mid < A::top()`. */
template <class A>
void bot_top_test(A mid) {
  A bot = A::bot();
  A top = A::top();
  EXPECT_TRUE2(bot.is_bot());
  EXPECT_TRUE2(top.is_top());
  EXPECT_FALSE2(top.is_bot());
  EXPECT_FALSE2(bot.is_top());
  EXPECT_TRUE2(gt<A>(top, bot.dual()));

  EXPECT_FALSE2(mid.is_bot());
  EXPECT_FALSE2(mid.is_top());
  EXPECT_TRUE2(lt<A>(bot.dual(), mid));
  EXPECT_TRUE2(gt<A>(top.dual(), mid));
}

template <class A>
void join_one_test(A a, A b, A expect, bool has_changed_expect) {
  BInc has_changed = BInc::bot();
  EXPECT_EQ2(join(a, b), expect);
  EXPECT_EQ2(a.tell(b, has_changed), expect);
  EXPECT_EQ2(has_changed, has_changed_expect);
}

template <class A>
void meet_one_test(A a, A b, A expect, bool has_changed_expect) {
  BInc has_changed = BInc::bot();
  EXPECT_EQ2(meet(a, b), expect);
  EXPECT_EQ2(a.dtell(b, has_changed), expect);
  EXPECT_EQ2(has_changed, has_changed_expect);
}

// `a` and `b` are supposed ordered and `a <= b`.
template <class A>
void join_meet_generic_test(A a, A b) {
  // Reflexivity
  join_one_test(a, a, a, false);
  meet_one_test(a, a, a, false);
  join_one_test(b, b, b, false);
  meet_one_test(b, b, b, false);
  // Coherency of join/meet w.r.t. ordering
  join_one_test(a, b, b, a.value() != b.value());
  join_one_test(b, a, b, false);
  // Commutativity
  meet_one_test(a, b, a, false);
  meet_one_test(b, a, a, a.value() != b.value());
  // Absorbing
  meet_one_test(a, A::top(), a, false);
  meet_one_test(b, A::top(), b, false);
  join_one_test(a, A::top(), A::top(), !a.is_top().value());
  join_one_test(b, A::top(), A::top(), !b.is_top().value());
  meet_one_test(a, A::bot(), A::bot(), !a.is_bot().value());
  meet_one_test(b, A::bot(), A::bot(), !b.is_bot().value());
  join_one_test(a, A::bot(), a, false);
  join_one_test(b, A::bot(), b, false);
}

template<class A>
void generic_order_test(A element) {
  using B = typename A::dual_type;
  EXPECT_EQ2(leq<A>(element, B::bot()), true);
  EXPECT_EQ2(leq<A>(element, B::top()), false);
  EXPECT_EQ2(leq<A>(A::bot(), B::top()), true);
  EXPECT_EQ2(leq<A>(A::top(), B::bot()), true);
  EXPECT_EQ2(leq<A>(A::top(), B::top()), false);
  EXPECT_EQ2(leq<A>(A::top(), B::bot()), true);
  EXPECT_EQ2(leq<A>(A::bot(), B::bot()), true);
  EXPECT_EQ2(leq<B>(B::top(), element), false);
}

template<class A>
using SplitSeq = vector<A, StandardAllocator>;

template<class A>
SplitSeq<A> make_singleton(A x) {
  return SplitSeq<A>({x});
}

template<class A>
SplitSeq<A> make_empty() {
  return SplitSeq<A>();
}

template<class A>
SplitSeq<A> split(const A& a) {
  return a.template split<StandardAllocator>();
}

template<class A>
void generic_split_test(A element) {
  EXPECT_EQ2(split(element)[0], element);
  EXPECT_EQ(split(element).size(), 1);
  EXPECT_EQ(split(A::top()).size(), 0);
  EXPECT_TRUE2(split(A::bot())[0].is_bot());
  EXPECT_EQ(split(A::bot()).size(), 1);
}

template<class A>
void generic_deinterpret_test() {
  EXPECT_EQ(A::bot().deinterpret(var_x), F::make_true());
  EXPECT_EQ(A::top().deinterpret(var_x), F::make_false());
}

template<class Universe>
void test_formula(const F& f, thrust::optional<Universe> expect) {
  thrust::optional<Universe> j = Universe::interpret(f);
  EXPECT_EQ(j.has_value(), expect.has_value());
  if(j.has_value() && expect.has_value()) {
    EXPECT_EQ2(j.value(), expect.value());
  }
}

template<class Universe, class K>
void test_interpret(Sig sig, Approx appx, K elem, thrust::optional<Universe> expect) {
  StandardAllocator standard_allocator;
  test_formula<Universe>(
    make_v_op_z(var_x, sig, elem, UNTYPED, appx, standard_allocator),
    expect);
}

template<class Universe, class K>
void test_all_interpret(Sig sig, K elem, thrust::optional<Universe> expect) {
  Approx appxs[3] = {EXACT, UNDER, OVER};
  for(int i = 0; i < 3; ++i) {
    test_interpret<Universe>(sig, appxs[i], elem, expect);
  }
}

template<class Universe, class K>
void test_exact_interpret(Sig sig, K elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, EXACT, elem, expect);
}

template<class Universe, class K>
void test_under_interpret(Sig sig, K elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, UNDER, elem, expect);
}

template<class Universe, class K>
void test_over_interpret(Sig sig, K elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, OVER, elem, expect);
}
#endif
