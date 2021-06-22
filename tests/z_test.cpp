// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "thrust/optional.h"
#include "ast.hpp"
#include "z.hpp"
#include "allocator.hpp"
#include "utility.hpp"

using namespace lala;

typedef ZInc<int, StandardAllocator> zi;
typedef Formula<StandardAllocator> F;

void test_formula(Approx appx, const F& f, thrust::optional<zi> expect) {
  thrust::optional<zi> j = zi::bot().interpret(appx, f);
  EXPECT_EQ(j.has_value(), expect.has_value());
  EXPECT_EQ(j, expect);
}

TEST(ZDeathTest, BadConstruction) {
  ASSERT_DEATH(zi(Limits<int>::bot()), "");
  ASSERT_DEATH(zi(Limits<int>::top()), "");
}

TEST(ZTest, ValidInterpret) {
  test_formula(
    EXACT,
    make_x_op_i(F::GEQ, 0, 10, standard_allocator),
    zi(10));
  test_formula(
    EXACT,
    make_x_op_i(F::GT, 0, 10, standard_allocator),
    zi(11));
  test_formula(
    UNDER,
    make_x_op_i(F::NEQ, 0, 10, standard_allocator),
    zi(11));
  test_formula(
    OVER,
    make_x_op_i(F::EQ, 0, 10, standard_allocator),
    zi(10));
}

TEST(ZTest, NoInterpret) {
  test_formula(
    EXACT,
    make_x_op_i(F::NEQ, 0, 10, standard_allocator),
    {});
  test_formula(
    EXACT,
    make_x_op_i(F::EQ, 0, 10, standard_allocator),
    {});
  Approx appxs[3] = {EXACT, UNDER, OVER};
  for(int i = 0; i < 3; ++i) {
    test_formula(
      appxs[i],
      make_x_op_i(F::LEQ, 0, 10, standard_allocator),
      {});
    test_formula(
      appxs[i],
      make_x_op_i(F::LT, 0, 10, standard_allocator),
      {});
  }
}

// `a` and `b` are supposed ordered and `a < b`.
template <typename A>
void join_meet_generic_test(A a, A b) {
  // Reflexivity
  EXPECT_EQ(a.join(a), a);
  EXPECT_EQ(a.meet(a), a);
  EXPECT_EQ(b.join(b), b);
  EXPECT_EQ(b.meet(b), b);
  // Coherency of join/meet w.r.t. ordering
  EXPECT_EQ(a.join(b), b);
  EXPECT_EQ(b.join(a), b);
  // Commutativity
  EXPECT_EQ(a.meet(b), a);
  EXPECT_EQ(b.meet(a), a);
  // Absorbing
  EXPECT_EQ(a.meet(A::top()), a);
  EXPECT_EQ(b.meet(A::top()), b);
  EXPECT_EQ(a.join(A::top()), A::top());
  EXPECT_EQ(b.join(A::top()), A::top());
  EXPECT_EQ(a.meet(A::bot()), A::bot());
  EXPECT_EQ(b.meet(A::bot()), A::bot());
  EXPECT_EQ(a.join(A::bot()), a);
  EXPECT_EQ(b.join(A::bot()), b);
}

TEST(ZTest, JoinMeet) {
  join_meet_generic_test(zi::bot(), zi::top());
  join_meet_generic_test(zi(0), zi(1));
  join_meet_generic_test(zi(-10), zi(10));
  join_meet_generic_test(zi(Limits<int>::top() - 1), zi::top());
}

TEST(ZTest, Refine) {
  EXPECT_EQ(zi(0).refine(), false);
  EXPECT_EQ(zi::top().refine(), false);
  EXPECT_EQ(zi::bot().refine(), false);
}

TEST(ZTest, Entailment) {
  EXPECT_EQ(zi(0).entailment(zi(0)), true);
  EXPECT_EQ(zi(0).entailment(zi(1)), false);
  EXPECT_EQ(zi(0).entailment(zi::top()), false);
  EXPECT_EQ(zi(0).entailment(zi::bot()), true);
  EXPECT_EQ(zi(0).entailment(zi(-1)), true);
  EXPECT_EQ(zi::bot().entailment(zi::bot()), true);
  EXPECT_EQ(zi::top().entailment(zi::top()), true);
  EXPECT_EQ(zi::top().entailment(zi::bot()), true);
  EXPECT_EQ(zi::top().entailment(zi(0)), true);
}

typedef DArray<zi, StandardAllocator> SplitSeq;

SplitSeq make_singleton(zi x) {
  return SplitSeq({x});
}

SplitSeq make_empty() {
  return SplitSeq();
}

TEST(ZTest, Split) {
  EXPECT_EQ(zi(0).split(), make_singleton(zi(0)));
  EXPECT_EQ(zi::top().split(), make_empty());
  EXPECT_EQ(zi::bot().split(), make_singleton(zi::bot()));
}

TEST(ZTest, Deinterpret) {
  F f10 = make_x_op_i(F::GEQ, 0, 10, standard_allocator);
  zi z10 = zi::bot().interpret(EXACT, f10).value();
  F f10_bis = z10.deinterpret();
  EXPECT_EQ(f10, f10_bis);
  F f9 = make_x_op_i(F::GT, 0, 9, standard_allocator);
  zi z9 = zi::bot().interpret(EXACT, f9).value();
  F f9_bis = z9.deinterpret();
  EXPECT_EQ(f10, f9_bis);
  EXPECT_EQ(zi::bot().deinterpret(), F::make_true());
  EXPECT_EQ(zi::top().deinterpret(), F::make_false());
}