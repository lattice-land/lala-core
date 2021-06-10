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

void test_formula(Approx appx, const Formula& f, thrust::optional<zi> expect) {
  thrust::optional<zi> j = zi::bot().interpret(appx, f);
  EXPECT_EQ(j, expect);
}

TEST(ZDeathTest, BadConstruction) {
  ASSERT_DEATH(zi(Limits<int>::bot()), "");
  ASSERT_DEATH(zi(Limits<int>::top()), "");
}

TEST(ZTest, ValidInterpret) {
  test_formula(
    EXACT,
    make_x_op_i(standard_allocator, Formula::GEQ, 0, 10),
    zi(10));
  test_formula(
    EXACT,
    make_x_op_i(standard_allocator, Formula::GT, 0, 10),
    zi(11));
  test_formula(
    UNDER,
    make_x_op_i(standard_allocator, Formula::NEQ, 0, 10),
    zi(11));
  test_formula(
    OVER,
    make_x_op_i(standard_allocator, Formula::EQ, 0, 10),
    zi(10));
}

TEST(ZTest, NoInterpret) {
  test_formula(
    EXACT,
    make_x_op_i(standard_allocator, Formula::NEQ, 0, 10),
    {});
  test_formula(
    EXACT,
    make_x_op_i(standard_allocator, Formula::EQ, 0, 10),
    {});
  Approx appxs[3] = {EXACT, UNDER, OVER};
  for(int i = 0; i < 3; ++i) {
    test_formula(
      appxs[i],
      make_x_op_i(standard_allocator, Formula::LEQ, 0, 10),
      {});
    test_formula(
      appxs[i],
      make_x_op_i(standard_allocator, Formula::LT, 0, 10),
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
  EXPECT_EQ(zi(0).refine(), true);
  EXPECT_EQ(zi::top().refine(), true);
  EXPECT_EQ(zi::bot().refine(), true);
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

// TEST(ZTest, Split) {
//   EXPECT_EQ(zi(0).split(), {zi(0)});
//   EXPECT_EQ(zi::top().split(), {});
//   EXPECT_EQ(zi::bot().split(), {zi::bot()});
// }
