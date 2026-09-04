// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "abstract_testing.hpp"
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/arith_bound.hpp"
#include "lala/universes/flat_universe.hpp"

using namespace lala;
using namespace battery;

TEST(ArithBoundTest, BotTopTest) {
  bot_top_test(local::ZLB(0));
  bot_top_test(local::ZUB(0));
}

TEST(ArithBoundTest, OrderTest) {
  EXPECT_TRUE(local::ZLB(10) < local::ZLB::top());
  EXPECT_TRUE(local::ZLB(10) < local::ZLB(0));
  EXPECT_TRUE(local::ZLB(10) > local::ZLB::bot());
  EXPECT_TRUE(local::ZUB(10) < local::ZUB::top());
  EXPECT_TRUE(local::ZUB(10) > local::ZUB(0));
  EXPECT_TRUE(local::ZUB(10) > local::ZUB::bot());
}

TEST(ArithBoundTest, JoinMeetTest) {
  join_meet_generic_test(local::ZLB::bot(), local::ZLB::top());
  join_meet_generic_test(local::ZLB(0), local::ZLB(0));
  join_meet_generic_test(local::ZLB(5), local::ZLB(0));
  join_meet_generic_test(local::ZLB(-5), local::ZLB(-10));

  join_meet_generic_test(local::ZUB::bot(), local::ZUB::top());
  join_meet_generic_test(local::ZUB(0), local::ZUB(0));
  join_meet_generic_test(local::ZUB(0), local::ZUB(5));
  join_meet_generic_test(local::ZUB(-10), local::ZUB(-5));
}

template <class L>
void test_z_arithmetic() {
  using F = L::template flat_type<battery::local_memory>;

  generic_arithmetic_fun_test<F, L>(F(0));

  EXPECT_EQ((project_fun<F, L>(NEG, F(L::bot()))), L::bot());

  EXPECT_EQ((project_fun<F, L>(ADD, F(0), F(1))), L(1));
  EXPECT_EQ((project_fun<F, L>(ADD, F(-10), F(0))), L(-10));
  EXPECT_EQ((project_fun<F, L>(ADD, F(-10), F(-5))), L(-15));
  EXPECT_EQ((project_fun<F, L>(ADD, F(10), F(-5))), L(5));
  EXPECT_EQ((project_fun<F, L>(ADD, F(10), F(5))), L(15));
}

TEST(ArithBoundTest, ArithmeticTest) {
  test_z_arithmetic<local::ZLB>();
  test_z_arithmetic<local::ZUB>();
  using zlb = local::ZLB;
  using zub = local::ZUB;
  EXPECT_EQ((project_fun(MIN, zlb::top(), zlb(10))), zlb::top());
  EXPECT_EQ((project_fun(MIN, zlb(10), zlb::top())), zlb::top());
  EXPECT_EQ((project_fun(MAX, zlb::top(), zlb(10))), zlb(10));
  EXPECT_EQ((project_fun(MIN, zub::top(), zub(10))), zub(10));
  EXPECT_EQ((project_fun(MIN, zub(10), zub::top())), zub(10));
  EXPECT_EQ((project_fun(MAX, zlb(10), zlb::top())), zlb(10));
  EXPECT_EQ((project_fun(MAX, zub::top(), zub(10))), zub::top());
  EXPECT_EQ((project_fun(MAX, zub(10), zub::top())), zub::top());

  EXPECT_EQ((project_fun(MIN, zlb::bot(), zlb(10))), zlb::bot());
  EXPECT_EQ((project_fun(MAX, zlb::bot(), zlb(10))), zlb::bot());
  EXPECT_EQ((project_fun(MIN, zub::bot(), zub(10))), zub::bot());
  EXPECT_EQ((project_fun(MAX, zub::bot(), zub(10))), zub::bot());
}
