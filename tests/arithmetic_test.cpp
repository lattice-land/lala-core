// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "z.hpp"
#include "arithmetic.hpp"
#include "generic_universe_test.hpp"

using zi = ZInc<int>;
using zd = ZDec<int>;

template<typename A>
void test_exact_op(A one) {
  A zero = sub(one, one);
  EXPECT_EQ2(add<UNDER>(zero, one), one);
  EXPECT_EQ2(add<UNDER>(zero, one), add<OVER>(zero, one));
  EXPECT_EQ2(add<UNDER>(zero, one), add(zero, one));
  EXPECT_EQ2(sub<UNDER>(one, one), zero);
  EXPECT_EQ2(sub<UNDER>(one, one), sub<OVER>(one, one));
  EXPECT_EQ2(sub<UNDER>(one, one), sub(one, one));
  EXPECT_EQ2(mul<UNDER>(zero, one), zero);
  EXPECT_EQ2(mul<UNDER>(one, one), one);
  EXPECT_EQ2(mul<UNDER>(one, one), mul<OVER>(one, one));
  EXPECT_EQ2(mul<UNDER>(one, one), mul(one, one));
  EXPECT_EQ2(neg<UNDER>(one), sub(zero, one));
  EXPECT_EQ2(neg<UNDER>(one), neg<OVER>(one));
  A two = add(one, one);
  A three = add(two, one);
  EXPECT_EQ2((div<UNDER, zi>(three, two)), two);
  EXPECT_EQ2((div<OVER, zd>(three, two)), two);
  EXPECT_EQ2((div<OVER, zi>(three, two)), one);
  EXPECT_EQ2((div<UNDER, zd>(three, two)), one);
  EXPECT_EQ2((div<UNDER, zi>(3, 2)), 2);
  EXPECT_EQ2((div<OVER, zd>(3, 2)), 2);
  EXPECT_EQ2((div<OVER, zi>(3, 2)), 1);
  EXPECT_EQ2((div<UNDER, zd>(3, 2)), 1);
  A mtwo = neg(two);
  A mthree = neg(three);
  EXPECT_EQ2((div<UNDER, zi>(mthree, mtwo)), two);
  EXPECT_EQ2((div<OVER, zi>(mthree, mtwo)), one);
  EXPECT_EQ2((div<UNDER, zi>(mthree, two)), neg(one));
  EXPECT_EQ2((div<OVER, zi>(mthree, two)), neg(two));
  EXPECT_EQ2((div<UNDER, zi>(three, mtwo)), neg(one));
  EXPECT_EQ2((div<OVER, zi>(three, mtwo)), neg(two));
  // Exact division
  EXPECT_EQ2((div<OVER, zi>(two, mtwo)), sub(zero, one));
  EXPECT_EQ2((div<OVER, zi>(two, mtwo)), (div<UNDER, zi>(two, mtwo)));
  EXPECT_EQ2((div<OVER, zi>(two, two)), one);
  EXPECT_EQ2((div<OVER, zi>(two, two)), (div<UNDER, zi>(two, two)));
  // Assert on division by zero.
  ASSERT_DEATH((div<UNDER, zi>(two, zero)), "");
  ASSERT_DEATH((div<OVER, zi>(two, zero)), "");
  ASSERT_DEATH((div<OVER, zi>(zero, zero)), "");
  ASSERT_DEATH((div<UNDER, zi>(zero, zero)), "");
}

TEST(ArithZTest, ExactOp) {
  test_exact_op(1);
}

TEST(ArithZTest, AddOp) {
  ZPInc<int> a(10);
  ZNInc<int> b(-5);
  ZInc<int> r = add(a, b);
  ZPDec<int> c(10);

  // Compile-time fail because it is not a monotone operation.
  // add(c, b);
  // add(c, a);
  // add(a, c);
  // add(b, c);
}
