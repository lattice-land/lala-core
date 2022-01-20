// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "z.hpp"
#include "arithmetic.hpp"

using namespace lala;

template<typename A>
void test_exact_op(A one) {
  A zero = sub(one, one);
  EXPECT_EQ(add<UNDER>(zero, one), one);
  EXPECT_EQ(add<UNDER>(zero, one), add<OVER>(zero, one));
  EXPECT_EQ(add<UNDER>(zero, one), add(zero, one));
  EXPECT_EQ(sub<UNDER>(one, one), zero);
  EXPECT_EQ(sub<UNDER>(one, one), sub<OVER>(one, one));
  EXPECT_EQ(sub<UNDER>(one, one), sub(one, one));
  EXPECT_EQ(mul<UNDER>(zero, one), zero);
  EXPECT_EQ(mul<UNDER>(one, one), one);
  EXPECT_EQ(mul<UNDER>(one, one), mul<OVER>(one, one));
  EXPECT_EQ(mul<UNDER>(one, one), mul(one, one));
  EXPECT_EQ(neg<UNDER>(one), sub(zero, one));
  EXPECT_EQ(neg<UNDER>(one), neg<OVER>(one));
  A two = add(one, one);
  A three = add(two, one);
  EXPECT_EQ(div<UNDER>(three, two), two);
  EXPECT_EQ(div<OVER>(three, two), one);
  A mtwo = neg(two);
  A mthree = neg(three);
  EXPECT_EQ(div<UNDER>(mthree, mtwo), two);
  EXPECT_EQ(div<OVER>(mthree, mtwo), one);
  EXPECT_EQ(div<UNDER>(mthree, two), neg(one));
  EXPECT_EQ(div<OVER>(mthree, two), neg(two));
  EXPECT_EQ(div<UNDER>(three, mtwo), neg(one));
  EXPECT_EQ(div<OVER>(three, mtwo), neg(two));
  // Exact division
  EXPECT_EQ(div<OVER>(two, mtwo), sub(zero, one));
  EXPECT_EQ(div<OVER>(two, mtwo), div<UNDER>(two, mtwo));
  EXPECT_EQ(div<OVER>(two, two), one);
  EXPECT_EQ(div<OVER>(two, two), div<UNDER>(two, two));
  // Assert on division by zero.
  ASSERT_DEATH(div<UNDER>(two, zero), "");
  ASSERT_DEATH(div<OVER>(two, zero), "");
  ASSERT_DEATH(div<OVER>(zero, zero), "");
  ASSERT_DEATH(div<UNDER>(zero, zero), "");
}

TEST(ArithZTest, ExactOp) {
  test_exact_op(1);
}
