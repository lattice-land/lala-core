// Copyright 2021 Pierre Talbot

#include "interval.hpp"
#include "abstract_testing.hpp"

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;

TEST(IntervalTest, BotTopTests) {
  bot_top_test(Itv(-1, 1));
  bot_top_test(Itv(zi::bot(), zd(0)));
  bot_top_test(Itv(zi(0), zd::bot()));
}

TEST(IntervalTest, NoInterpret) {
  VarEnv<StandardAllocator> env = init_env();
  must_error<Itv>(env, "constraint int_eq(x, 10) :: under;");
  must_error<Itv>(env, "constraint int_ne(x, 10) :: over;");
  must_error<Itv>(env, "constraint int_ne(x, 10) :: exact;");
}

TEST(IntervalTest, ValidInterpret) {
  // VarEnv<StandardAllocator> env = init_env();
  VarEnv<StandardAllocator> env;
  must_interpret_to(env, "constraint int_eq(x, 10);", Itv(10, 10), false);
  must_interpret_to(env, "constraint int_eq(x, 10) :: over;", Itv(10, 10), false);
  must_interpret_to(env, "constraint int_ne(x, 10) :: under;", Itv(zi(11), zd::bot()), false);
}

TEST(IntervalTest, JoinMeetTest) {
  join_meet_generic_test(Itv::bot(), Itv::top());
  join_meet_generic_test(Itv(0,0), Itv(0,0));
  join_meet_generic_test(Itv(0,1), Itv(0,1));
  join_meet_generic_test(Itv(0,10), Itv(0,5));
  join_meet_generic_test(Itv(0,10), Itv(5,5));
  join_meet_generic_test(Itv(0,1), Itv(0,0));
  join_meet_generic_test(Itv(0,1), Itv(1,1));

  EXPECT_EQ(join(Itv(10, 20), Itv(4,14)), Itv(10,14));
  EXPECT_EQ(meet(Itv(10, 20), Itv(4,14)), Itv(4,20));
}

TEST(IntervalTest, OrderTest) {
  EXPECT_FALSE(Itv(10, 20) <= Itv(8, 12));
  EXPECT_TRUE(Itv(8, 12) <= Itv(8, 12));
  EXPECT_TRUE(Itv(7, 13) <= Itv(8, 12));
  EXPECT_FALSE(Itv(10, 12) <= Itv(8, 12));

  EXPECT_FALSE(Itv(8, 12) <= Itv(10, 20));
  EXPECT_TRUE(Itv(8, 12) <= Itv(8, 12));
  EXPECT_FALSE(Itv(8, 12) <= Itv(7, 13));
  EXPECT_TRUE(Itv(8, 12) <= Itv(10, 12));
}

TEST(IntervalTest, GenericFunTests) {
  generic_unary_fun_test<EXACT, NEG, Itv>();
  generic_abs_test<Itv>();
  generic_binary_fun_test<EXACT, ADD>(Itv(0,10));
  generic_binary_fun_test<EXACT, SUB>(Itv(0,10));
  generic_binary_fun_test<EXACT, MIN>(Itv(0,10));
  generic_binary_fun_test<EXACT, MAX>(Itv(0,10));
  generic_arithmetic_fun_test<OVER>(Itv(0, 10));
  generic_arithmetic_fun_test<OVER>(Itv(1, 10));
  generic_arithmetic_fun_test<OVER>(Itv(-10, 10));
  generic_arithmetic_fun_test<OVER>(Itv(-10, -1));
  generic_arithmetic_fun_test<OVER>(Itv(-10, 0));
}

TEST(IntervalTest, Negation) {
  EXPECT_EQ((Itv::fun<EXACT, NEG>(Itv(5, 10))), Itv(-10, -5));
  EXPECT_EQ((Itv::fun<EXACT, NEG>(Itv(-10, 10))), Itv(-10, 10));
  EXPECT_EQ((Itv::fun<EXACT, NEG>(Itv(10, -10))), Itv(10, -10));
  EXPECT_EQ((Itv::fun<EXACT, NEG>(Itv(-10, -5))), Itv(5, 10));
}

TEST(IntervalTest, Absolute) {
  EXPECT_EQ((Itv::fun<EXACT, ABS>(Itv(5, 10))), Itv(5, 10));
  EXPECT_EQ((Itv::fun<EXACT, ABS>(Itv(-10, 10))), Itv(0, 10));
  EXPECT_EQ((Itv::fun<EXACT, ABS>(Itv(10, -10))), Itv(10, 10));
  EXPECT_EQ((Itv::fun<EXACT, ABS>(Itv(-10, -5))), Itv(5, 10));
}

TEST(IntervalTest, Addition) {
  EXPECT_EQ((Itv::fun<EXACT, ADD>(Itv(-10, -10), Itv(-10, -10))), Itv(-20, -20));
  EXPECT_EQ((Itv::fun<EXACT, ADD>(Itv(-10, -10), Itv(0, 0))), Itv(-10, -10));
  EXPECT_EQ((Itv::fun<EXACT, ADD>(Itv(0, 0), Itv(-10, -10))), Itv(-10, -10));
  EXPECT_EQ((Itv::fun<EXACT, ADD>(Itv(1, 10), Itv(1, 10))), Itv(2, 20));
  EXPECT_EQ((Itv::fun<EXACT, ADD>(Itv(-1, 10), Itv(1, 10))), Itv(0, 20));
  EXPECT_EQ((Itv::fun<EXACT, ADD>(Itv(-1, 10), Itv(-1, 10))), Itv(-2, 20));
  EXPECT_EQ((Itv::fun<EXACT, ADD>(Itv(-10, -1), Itv(1, 10))), Itv(-9, 9));
}

TEST(IntervalTest, Subtraction) {
  EXPECT_EQ((Itv::fun<EXACT, SUB>(Itv(-10, -10), Itv(-10, -10))), Itv(0, 0));
  EXPECT_EQ((Itv::fun<EXACT, SUB>(Itv(-10, -10), Itv(0, 0))), Itv(-10, -10));
  EXPECT_EQ((Itv::fun<EXACT, SUB>(Itv(0, 0), Itv(-10, -10))), Itv(10, 10));
  EXPECT_EQ((Itv::fun<EXACT, SUB>(Itv(1, 10), Itv(1, 10))), Itv(-9, 9));
  EXPECT_EQ((Itv::fun<EXACT, SUB>(Itv(-1, 10), Itv(1, 10))), Itv(-11, 9));
  EXPECT_EQ((Itv::fun<EXACT, SUB>(Itv(-1, 10), Itv(-1, 10))), Itv(-11, 11));
  EXPECT_EQ((Itv::fun<EXACT, SUB>(Itv(-10, -1), Itv(1, 10))), Itv(-20, -2));
}

TEST(IntervalTest, Multiplication) {
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(-10, -2), Itv(-9, -3))), Itv(6, 90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(-10, -2), Itv(3, 9))), Itv(-90, -6));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(-10, -2), Itv(-9, 9))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(-10, -2), Itv(9, -9))), Itv(90, -90));

  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(2, 10), Itv(-9, -3))), Itv(-90, -6));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(2, 10), Itv(3, 9))), Itv(6, 90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(2, 10), Itv(-9, 9))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(2, 10), Itv(9, -9))), Itv(90, -90));

  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(-10, 10), Itv(-9, -3))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(-10, 10), Itv(3, 9))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(-10, 10), Itv(-9, 9))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(-10, 10), Itv(9, -9))), Itv::eq_zero());

  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(10, -10), Itv(-9, -3))), Itv(90, -90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(10, -10), Itv(3, 9))), Itv(90, -90));
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(10, -10), Itv(-9, 9))), Itv::eq_zero());
  EXPECT_EQ((Itv::fun<OVER, MUL>(Itv(10, -10), Itv(9, -9))), Itv(90, -90));
}

// TEST(IntervalTest, Division) {
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(-10, -2), Itv(-9, -3))), Itv(6, 90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(-10, -2), Itv(3, 9))), Itv(-90, -6));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(-10, -2), Itv(-9, 9))), Itv(-90, 90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(-10, -2), Itv(9, -9))), Itv(90, -90));

//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(2, 10), Itv(-9, -3))), Itv(-90, -6));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(2, 10), Itv(3, 9))), Itv(6, 90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(2, 10), Itv(-9, 9))), Itv(-90, 90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(2, 10), Itv(9, -9))), Itv(90, -90));

//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(-10, 10), Itv(-9, -3))), Itv(-90, 90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(-10, 10), Itv(3, 9))), Itv(-90, 90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(-10, 10), Itv(-9, 9))), Itv(-90, 90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(-10, 10), Itv(9, -9))), Itv::eq_zero());

//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(10, -10), Itv(-9, -3))), Itv(90, -90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(10, -10), Itv(3, 9))), Itv(90, -90));
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(10, -10), Itv(-9, 9))), Itv::eq_zero());
//   EXPECT_EQ((Itv::fun<OVER, EDIV>(Itv(10, -10), Itv(9, -9))), Itv(90, -90));
// }
