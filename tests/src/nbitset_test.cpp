// Copyright 2021 Pierre Talbot

#include "abstract_testing.hpp"
#include "lala/universes/nbitset.hpp"

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

TEST(NBitsetTest, TellInterpretation) {
  VarEnv<standard_allocator> env;
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, -100);", NBit(-1), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, -1);", NBit(-1), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", NBit(0), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 10);", NBit(10), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 125);", NBit(125), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 126);", NBit(1000), env, true);

  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0); constraint int_eq(x, 10);", NBit::from_set({}), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint nbool_or(int_eq(x, 0), int_eq(x, 10));", NBit::from_set({0, 10}), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint set_in(x, {0, 10});", NBit::from_set({0, 10}), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint nbool_or(int_eq(x, -1), int_eq(x, 100), int_eq(x, 126));", NBit::from_set({-1, 100, 1000}), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint set_in(x, {-1, 1000});", NBit::from_set({-1, 1000}), env, true);

  expect_interpret_equal_to<IKind::TELL>("var 0..32: x;", NBit(0,32), env, false);
  expect_interpret_equal_to<IKind::TELL>("var {0,32}: x;", NBit::from_set({0,32}), env, false);

  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, 0);", NBit(0).complement(), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, 0); constraint int_ne(x, 10);", NBit::from_set({0, 10}).complement(), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, -1);", NBit(), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, 1000);", NBit(), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, -1); constraint int_ne(x, 0);", NBit(0).complement(), env, true);

  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, -1);", NBit(), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 0);", NBit(0,1000), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 10);", NBit(10,1000), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_gt(x, 10);", NBit(11,1000), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 126);", NBit(1000), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 1000);", NBit(1000), env, true);

  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, -2);", NBit(-100), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, -1);", NBit(-100), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 0);", NBit(-1,0), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 10);", NBit(-1, 10), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_lt(x, 10);", NBit(-1, 9), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 1000);", NBit(), env, true);

  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 1000); constraint int_ge(x, 0);", NBit(0, 200), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 100); constraint int_ge(x, 0);", NBit(0, 100), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 1); constraint int_ge(x, 0);", NBit(0, 1), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 0); constraint int_ge(x, 0);", NBit(0, 0), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 100); constraint int_ge(x, -100);", NBit(-1, 100), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint nbool_or(int_le(x, 1), int_ge(x, 0));", NBit(), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint nbool_or(int_le(x, 0), int_ge(x, 0));", NBit(), env, false);
}

TEST(NBitsetTest, AskInterpretation) {
  VarEnv<standard_allocator> env;
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, -100);", NBit::top(), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, -1);", NBit::top(), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 0);", NBit(0), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 10);", NBit(10), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 125);", NBit(125), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 126);", NBit::top(), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 0); constraint int_eq(x, 10);", NBit::top(), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint nbool_or(int_eq(x, 0), int_eq(x, 10));", NBit::from_set({0, 10}), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint set_in(x, {0, 10});", NBit::from_set({0, 10}), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint nbool_or(int_eq(x, -1), int_eq(x, 100), int_eq(x, 126));", NBit(100), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint set_in(x, {-1, 1000});", NBit::top(), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 0);", NBit(0).complement(), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 0); constraint int_ne(x, 10);", NBit::from_set({0, 10}).complement(), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, -1);", NBit(0,1000), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1000);", NBit(-1,125), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, -1); constraint int_ne(x, 0);", NBit(1, 1000), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, -1);", NBit(0, 1000), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, 0);", NBit(0,1000), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, 10);", NBit(10,1000), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_gt(x, 10);", NBit(11,1000), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, 126);", NBit(126), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, 1000);", NBit::top(), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, -2);", NBit::top(), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, -1);", NBit(-1), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 0);", NBit(-1,0), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 10);", NBit(-1, 10), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_lt(x, 10);", NBit(-1, 9), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 1000);", NBit(-1, 125), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 1000); constraint int_ge(x, 0);", NBit(0, 125), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 100); constraint int_ge(x, 0);", NBit(0, 100), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 1); constraint int_ge(x, 0);", NBit(0, 1), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 0); constraint int_ge(x, 0);", NBit(0, 0), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 100); constraint int_ge(x, -100);", NBit(0, 100), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint nbool_or(int_le(x, 1), int_ge(x, 0));", NBit(), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint nbool_or(int_le(x, 0), int_ge(x, 0));", NBit(), env, false);
}

TEST(NBitsetTest, JoinMeetTest) {
  join_meet_generic_test(NBit::bot(), NBit::top());
  join_meet_generic_test(NBit(0), NBit(0));
  join_meet_generic_test(NBit(0,1), NBit(0,1));
  join_meet_generic_test(NBit(0,10), NBit(0,5));
  join_meet_generic_test(NBit(0,10), NBit(5,5));
  join_meet_generic_test(NBit(0,1), NBit(0,0));
  join_meet_generic_test(NBit(0,1), NBit(1,1));

  join_meet_generic_test(NBit(-1,1), NBit(-1,0));
  join_meet_generic_test(NBit(-1,1), NBit(-1,-1));
  join_meet_generic_test(NBit(-1,100), NBit(-1,10));
  join_meet_generic_test(NBit(0,1000), NBit(10, 1000));

  join_meet_generic_test(NBit::from_set({0,5}), NBit::from_set({0,5}));
  join_meet_generic_test(NBit::from_set({0,5,10}), NBit::from_set({0,10}));
  join_meet_generic_test(NBit::from_set({0,5,10,1000}), NBit::from_set({0,1000}));
  join_meet_generic_test(NBit::from_set({0,5,10,1000}), NBit::from_set({1000}));
  join_meet_generic_test(NBit::from_set({-1,5,10,1000}), NBit::from_set({1000}));
  join_meet_generic_test(NBit::from_set({-1,5,10,1000}), NBit::from_set({-1}));
  join_meet_generic_test(NBit::from_set({-1,5,10,1000}), NBit::from_set({5}));
  join_meet_generic_test(NBit::from_set({-1,5,10,1000}), NBit::from_set({}));
}

TEST(NBitsetTest, OrderTest) {
  EXPECT_FALSE(NBit(10, 20) <= NBit(8, 12));
  EXPECT_TRUE(NBit(8, 12) <= NBit(8, 12));
  EXPECT_TRUE(NBit(7, 13) <= NBit(8, 12));
  EXPECT_FALSE(NBit(10, 12) <= NBit(8, 12));

  EXPECT_FALSE(NBit(8, 12) <= NBit(10, 20));
  EXPECT_TRUE(NBit(8, 12) <= NBit(8, 12));
  EXPECT_FALSE(NBit(8, 12) <= NBit(7, 13));
  EXPECT_TRUE(NBit(8, 12) <= NBit(10, 12));

  EXPECT_TRUE(NBit::from_set({-2, 10, 100}) <= NBit::from_set({10, 100}));
  EXPECT_TRUE(NBit::from_set({-2, 10, 100}) <= NBit::from_set({10}));
  EXPECT_TRUE(NBit::from_set({-2, 1000}) <= NBit::from_set({1000}));
  EXPECT_TRUE(NBit::from_set({-2, 1000}) <= NBit::from_set({-2}));
}

TEST(NBitsetTest, GenericFunTests) {
  generic_unary_fun_test<NEG, NBit>();
  generic_abs_test<NBit>();
}

TEST(NBitsetTest, Negation) {
  EXPECT_EQ((NBit::fun<NEG>(NBit(5, 10))), NBit(-1));
  EXPECT_EQ((NBit::fun<NEG>(NBit(-10, 10))), NBit::bot());
  EXPECT_EQ((NBit::fun<NEG>(NBit(-10, -1))), NBit(0,1000));
  EXPECT_EQ((NBit::fun<NEG>(NBit(0, 1000))), NBit(-1));
}

TEST(NBitsetTest, Absolute) {
  EXPECT_EQ((NBit::fun<ABS>(NBit(5, 10))), NBit(5, 10));
  EXPECT_EQ((NBit::fun<ABS>(NBit(-10, 10))), NBit(0, 1000));
  EXPECT_EQ((NBit::fun<ABS>(NBit(0,1000))), NBit(0, 1000));
  EXPECT_EQ((NBit::fun<ABS>(NBit(-1))), NBit(0, 1000));
  EXPECT_EQ((NBit::fun<ABS>(NBit(1000))), NBit(1000));
  EXPECT_EQ((NBit::fun<ABS>(NBit(-10, -5))), NBit(0, 1000));
}

TEST(NBitsetTest, Width) {
  EXPECT_EQ(NBit(0,0).width(), NBit(1));
  EXPECT_EQ(NBit(-10, 10).width(), NBit::bot());
  EXPECT_EQ(NBit(0, 1000).width(), NBit::bot());
  EXPECT_EQ(NBit(0, 10).width(), NBit(11));
  EXPECT_EQ(NBit(5, 10).width(), NBit(6));
  EXPECT_EQ(NBit::bot().width(), NBit::bot());
  EXPECT_EQ(NBit::top().width(), NBit(0));
}
