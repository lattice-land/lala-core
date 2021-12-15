// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "z.hpp"
#include "arithmetic.hpp"

using namespace lala;

using zi = ZInc<int, StandardAllocator>;
using zd = ZDec<int, StandardAllocator>;

template<typename A>
void test_exact_op(A one) {
  A zero = sub(one, one);
  EXPECT_EQ(add_up(zero, one), one);
  EXPECT_EQ(add_up(zero, one), add_down(zero, one));
  EXPECT_EQ(add_up(zero, one), add(zero, one));
  EXPECT_EQ(sub_up(one, one), zero);
  EXPECT_EQ(sub_up(one, one), sub_down(one, one));
  EXPECT_EQ(sub_up(one, one), sub(one, one));
  EXPECT_EQ(mul_up(zero, one), zero);
  EXPECT_EQ(mul_up(one, one), one);
  EXPECT_EQ(mul_up(one, one), mul_down(one, one));
  EXPECT_EQ(mul_up(one, one), mul(one, one));
  EXPECT_EQ(neg_up(one), sub(zero, one));
  EXPECT_EQ(neg_up(one), neg_down(one));
  A two = add(one, one);
  A three = add(two, one);
  EXPECT_EQ(div_up(three, two), two);
  EXPECT_EQ(div_down(three, two), one);
  A mtwo = neg(two);
  A mthree = neg(three);
  EXPECT_EQ(div_up(mthree, mtwo), two);
  EXPECT_EQ(div_down(mthree, mtwo), one);
  EXPECT_EQ(div_up(mthree, two), neg(one));
  EXPECT_EQ(div_down(mthree, two), neg(two));
  EXPECT_EQ(div_up(three, mtwo), neg(one));
  EXPECT_EQ(div_down(three, mtwo), neg(two));
  // Exact division
  EXPECT_EQ(div_down(two, mtwo), sub(zero, one));
  EXPECT_EQ(div_down(two, mtwo), div_up(two, mtwo));
  EXPECT_EQ(div_down(two, two), one);
  EXPECT_EQ(div_down(two, two), div_up(two, two));
  // Assert on division by zero.
  ASSERT_DEATH(div_up(two, zero), "");
  ASSERT_DEATH(div_down(two, zero), "");
  ASSERT_DEATH(div_down(zero, zero), "");
  ASSERT_DEATH(div_up(zero, zero), "");
}

TEST(ArithZTest, ExactOp) {
  test_exact_op<zi>(zi(1));
  test_exact_op(zd(1));
}
