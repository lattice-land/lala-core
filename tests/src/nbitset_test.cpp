// Copyright 2021 Pierre Talbot

#include "lala/universes/nbitset.hpp"
#include "abstract_testing.hpp"

using NBit = NBitset<128, battery::local_memory, unsigned long long>;

TEST(NBitsetTest, BotTopTests) {
  bot_top_test(NBit(0, 1));
  bot_top_test(NBit(0));
  bot_top_test(NBit(-1, -1));
  bot_top_test(NBit(-1, 10));
  bot_top_test(NBit(0, 10));
  bot_top_test(NBit(5, 15));
  bot_top_test(NBit(0, 1000));
  bot_top_test(NBit(1000, 1000));
}



TEST(NBitsetTest, JoinMeetTest) {
  join_meet_generic_test(NBit::bot(), NBit::top());
  join_meet_generic_test(NBit(0), NBit(0));
  join_meet_generic_test(NBit(0,1), NBit(0,1));
  join_meet_generic_test(NBit(0,5), NBit(0,10));
  join_meet_generic_test(NBit(5,5), NBit(0,10));
  join_meet_generic_test(NBit(0,0), NBit(0,1));
  join_meet_generic_test(NBit(1,1), NBit(0,1));

  join_meet_generic_test(NBit(-1,0), NBit(-1,1));
  join_meet_generic_test(NBit(-1,-1), NBit(-1,1));
  join_meet_generic_test(NBit(-1,10), NBit(-1,100));
  join_meet_generic_test(NBit(10,1000), NBit(0,1000));

  join_meet_generic_test(NBit::from_set({0,5}), NBit::from_set({0,5}));
  join_meet_generic_test(NBit::from_set({0,10}), NBit::from_set({0,5,10}));
  join_meet_generic_test(NBit::from_set({0,1000}), NBit::from_set({0,5,10,1000}));
  join_meet_generic_test(NBit::from_set({1000}), NBit::from_set({0,5,10,1000}));
  join_meet_generic_test(NBit::from_set({1000}), NBit::from_set({-1,5,10,1000}));
  join_meet_generic_test(NBit::from_set({-1}), NBit::from_set({-1,5,10,1000}));
  join_meet_generic_test(NBit::from_set({5}), NBit::from_set({-1,5,10,1000}));
  join_meet_generic_test(NBit::from_set({}), NBit::from_set({-1,5,10,1000}));
}

TEST(NBitsetTest, OrderTest) {
  EXPECT_FALSE(NBit(10, 20) <= NBit(8, 12));
  EXPECT_TRUE(NBit(8, 12) <= NBit(8, 12));
  EXPECT_FALSE(NBit(7, 13) <= NBit(8, 12));
  EXPECT_TRUE(NBit(7, 13) >= NBit(8, 12));
  EXPECT_TRUE(NBit(10, 12) <= NBit(8, 12));

  EXPECT_FALSE(NBit(8, 12) <= NBit(10, 20));
  EXPECT_TRUE(NBit(8, 12) <= NBit(8, 12));
  EXPECT_TRUE(NBit(8, 12) <= NBit(7, 13));
  EXPECT_FALSE(NBit(8, 12) <= NBit(10, 12));

  EXPECT_TRUE(NBit::from_set({-2, 10, 100}) >= NBit::from_set({10, 100}));
  EXPECT_TRUE(NBit::from_set({-2, 10, 100}) >= NBit::from_set({10}));
  EXPECT_TRUE(NBit::from_set({-2, 1000}) >= NBit::from_set({1000}));
  EXPECT_TRUE(NBit::from_set({-2, 1000}) >= NBit::from_set({-2}));
}


TEST(NBitsetTest, Negation) {
  EXPECT_EQ((project_fun(NEG, NBit(5, 10))), NBit(-1));
  EXPECT_EQ((project_fun(NEG, NBit(-10, 10))), NBit::top());
  EXPECT_EQ((project_fun(NEG, NBit(-10, -1))), NBit(0,1000));
  EXPECT_EQ((project_fun(NEG, NBit(0, 1000))), NBit(-1));
}

TEST(NBitsetTest, Absolute) {
  EXPECT_EQ((project_fun(ABS, NBit(5, 10))), NBit(5, 10));
  EXPECT_EQ((project_fun(ABS, NBit(-10, 10))), NBit(0, 1000));
  EXPECT_EQ((project_fun(ABS, NBit(0,1000))), NBit(0, 1000));
  EXPECT_EQ((project_fun(ABS, NBit(-1))), NBit(0, 1000));
  EXPECT_EQ((project_fun(ABS, NBit(1000))), NBit(1000));
  EXPECT_EQ((project_fun(ABS, NBit(-10, -5))), NBit(0, 1000));
}

TEST(NBitsetTest, Width) {
  EXPECT_EQ(NBit(0,0).width(), NBit(1));
  EXPECT_EQ(NBit(-10, 10).width(), NBit::top());
  EXPECT_EQ(NBit(0, 1000).width(), NBit::top());
  EXPECT_EQ(NBit(0, 10).width(), NBit(11));
  EXPECT_EQ(NBit(5, 10).width(), NBit(6));
  EXPECT_EQ(NBit::top().width(), NBit::top());
  EXPECT_EQ(NBit::bot().width(), NBit(0));
}

TEST(NBitsetTest, Projections) {
  using LB = NBit::LB;
  using UB = NBit::UB;
  EXPECT_EQ(NBit(0,0).lb(), LB(0));
  EXPECT_EQ(NBit(0,0).ub(), UB(0));
  EXPECT_EQ(NBit(-10, 10).lb(), LB::top());
  EXPECT_EQ(NBit(-10, 10).ub(), UB(10));
  EXPECT_EQ(NBit(0, 1000).lb(), LB(0));
  EXPECT_EQ(NBit(0, 1000).ub(), UB::top());
  EXPECT_EQ(NBit(5, 10).lb(), LB(5));
  EXPECT_EQ(NBit(5, 10).ub(), UB(10));
  EXPECT_EQ(NBit::top().lb(), LB::top());
  EXPECT_EQ(NBit::top().ub(), UB::top());
  EXPECT_EQ(NBit::bot().lb(), LB::bot());
  EXPECT_EQ(NBit::bot().ub(), UB::bot());
  EXPECT_EQ(NBit(1000, 1000).lb(), LB(126));
  EXPECT_EQ(NBit(1000, 1000).ub(), UB::top());
  EXPECT_EQ(NBit(-1, -1).lb(), LB::top());
  EXPECT_EQ(NBit(-1, -1).ub(), UB(-1));
}

TEST(NBitsetTest, GenericFunTests) {
  generic_unary_fun_test<NBit>(NEG);
}
