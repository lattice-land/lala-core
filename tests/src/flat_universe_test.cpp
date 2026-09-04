// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "lala/logic/logic.hpp"
#include "lala/universes/flat_universe.hpp"
#include "battery/allocator.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;

using ZF = local::ZFlat;

TEST(FlatUniverseTest, BotTopTest) {
  bot_top_test(ZF(0));
}

TEST(FlatUniverseTest, JoinMeetTest) {
  join_meet_generic_test(ZF::bot(), ZF(0));
  join_meet_generic_test(ZF(1), ZF::top());
  join_one_test(ZF(0), ZF(1), ZF::top(), true);
  meet_one_test(ZF(0), ZF(1), ZF::bot(), true);
}

TEST(FlatUniverseTest, ArithmeticTest) {
  generic_arithmetic_fun_test(ZF(0));
  EXPECT_EQ(project_fun(ADD, ZF(0), ZF(1)), ZF(1));
  EXPECT_EQ(project_fun(ADD, ZF(-10), ZF(-5)), ZF(-15));
}

TEST(FlatUniverseTest, ConversionUpset) {
  EXPECT_EQ((ZF(local::ZLB::top())), ZF::top());
  EXPECT_EQ((ZF(local::ZUB::top())), ZF::top());
  EXPECT_EQ((ZF(local::ZLB::bot())), ZF::bot());
  EXPECT_EQ((ZF(local::ZUB::bot())), ZF::bot());
}
