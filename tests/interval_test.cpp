// Copyright 2021 Pierre Talbot

#include "z.hpp"
#include "arithmetic.hpp"
#include "interval.hpp"
#include "generic_universe_test.hpp"

using zi = ZInc<int>;
using Itv = Interval<zi>;

using zpi = ZPInc<int>;
using PItv = Interval<zpi>;

TEST(IntervalTest, AddTest) {
  Itv top(0, -1);
  Itv i1_4(1,4);
  Itv i1_3(1,3);
  EXPECT_EQ2(add(i1_4, i1_3), Itv(2,7));
  EXPECT_EQ2(add(i1_3, i1_4), Itv(2,7));
  EXPECT_EQ2(add(i1_4, 4), Itv(5,8));
  EXPECT_EQ2(add(4, i1_4), Itv(5,8));
  EXPECT_EQ2(add(top, 4), top);
  EXPECT_EQ2(add(4, top), top);
  EXPECT_EQ2(add(i1_4, top), top);
  EXPECT_EQ2(add(top, i1_4), top);
  EXPECT_EQ2(add(top, top), top);
}

TEST(IntervalTest, SubTest) {
  Itv top(0, -1);
  Itv i1_4(1,4);
  Itv i1_3(1,3);
  EXPECT_EQ2(sub(i1_4, i1_3), Itv(-2,3));
  EXPECT_EQ2(sub(i1_3, i1_4), Itv(-3,2));
  EXPECT_EQ2(sub(i1_4, 4), Itv(-3,0));
  EXPECT_EQ2(sub(4, i1_4), Itv(0,3));
  EXPECT_EQ2(sub(Itv(-4,-2), Itv(1,2)), Itv(-6,-3));
  EXPECT_EQ2(sub(Itv(1,2), Itv(-4,-2)), Itv(3,6));
  EXPECT_EQ2(sub(Itv(-2,-1), Itv(-4,-2)), Itv(0,3));
  EXPECT_EQ2(sub(top, 4), top);
  EXPECT_EQ2(sub(4, top), top);
  EXPECT_EQ2(sub(i1_4, top), top);
  EXPECT_EQ2(sub(top, i1_4), top);
  EXPECT_EQ2(sub(top, top), top);
}

TEST(IntervalTest, MulTest) {
  Itv top(0, -1);
  Itv zero(0,0);
  EXPECT_EQ2(mul(zero,zero), zero);
  EXPECT_EQ2(mul(Itv(1,2), Itv(1,2)), Itv(1,4));
  EXPECT_TRUE2(mul(top, top).is_top());
  EXPECT_TRUE2(mul(Itv::top(), Itv(0,0)).is_top());
  EXPECT_TRUE2(mul(Itv::top(), Itv(1,2)).is_top());
  EXPECT_TRUE2(mul(Itv::top(), Itv(0,10)).is_top());
  EXPECT_EQ2(mul(Itv(0,0), Itv(0,10)), Itv(0,0));
  EXPECT_EQ2(mul(Itv(1,1), Itv(0,10)), Itv(0,10));
  EXPECT_EQ2(mul(Itv(1,2), Itv(0,10)), Itv(0,20));
  EXPECT_EQ2(mul(Itv(-5,10), Itv(0,10)), Itv(-50,100));
  EXPECT_EQ2(mul(Itv(0,10), Itv(-5,10)), Itv(-50,100));
  EXPECT_EQ2(mul(Itv(-5,10), Itv(-30,-20)), Itv(-300,150));
  EXPECT_EQ2(mul(Itv(-30,-20), Itv(-5,10)), Itv(-300,150));
  EXPECT_EQ2(mul(Itv(-10,-5), Itv(-30,-20)), Itv(100,300));

  EXPECT_EQ2(mul(Itv(1,2), 2), Itv(2,4));
  EXPECT_EQ2(mul(Itv(0,0), 10), Itv(0,0));
  EXPECT_EQ2(mul(top, 10), top);

  EXPECT_EQ2(mul(2, Itv(1,2)), Itv(2,4));
  EXPECT_EQ2(mul(10, Itv(0,0)), Itv(0,0));
  EXPECT_EQ2(mul(10, top), top);
}

TEST(IntervalTest, DivTest) {
  PItv top(1, 0);
  PItv zero(0,0);
  EXPECT_TRUE2(div(zero,zero).is_top());
  EXPECT_EQ2(div(PItv(1,2), PItv(1,2)), PItv(1,2));
  EXPECT_EQ2(div(zero, PItv(1,2)), zero);
  EXPECT_EQ2(div(PItv(10,20), PItv(2,3)), PItv(4,10));
  EXPECT_EQ2(div(PItv(10,20), PItv(3,6)), PItv(2,6));
  EXPECT_EQ2(div(Itv(10,20), Itv(3,6)), Itv(2,6));
}

TEST(IntervalTest, JoinMeetTest) {
  EXPECT_EQ2(join(Itv(10, 20), Itv(4,14)), Itv(10,14));
  EXPECT_EQ2(meet(Itv(10, 20), Itv(4,14)), Itv(4,20));
}