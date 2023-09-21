// Copyright 2021 Pierre Talbot

#include "abstract_testing.hpp"
#include "lala/interval.hpp"

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;

TEST(IntervalTest, BotTopTests) {
  bot_top_test(Itv(-1, 1));
  bot_top_test(Itv(zi::bot(), zd(0)));
  bot_top_test(Itv(zi(0), zd::bot()));
}

TEST(IntervalTest, NoInterpret) {
  VarEnv<standard_allocator> env = init_env();
  must_error_ask<Itv>(env, "constraint float_eq(x, 1111111111.0000000000001);");
  must_error_tell<Itv>(env, "constraint int_ne(x, 10);");
}

TEST(IntervalTest, ValidInterpret) {
  // VarEnv<standard_allocator> env = init_env();
  VarEnv<standard_allocator> env;
  must_interpret_tell_to(env, "constraint int_eq(x, 10);", Itv(10, 10), false);
  must_interpret_ask_to(env, "constraint int_ne(x, 10);", Itv(zi(11), zd::bot()), false);
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
  generic_unary_fun_test<NEG, Itv>();
  generic_abs_test<Itv>();
  generic_binary_fun_test<ADD>(Itv(0,10));
  generic_binary_fun_test<SUB>(Itv(0,10));
  generic_arithmetic_fun_test(Itv(0, 10));
  generic_arithmetic_fun_test(Itv(1, 10));
  generic_arithmetic_fun_test(Itv(-10, 10));
  generic_arithmetic_fun_test(Itv(-10, -1));
  generic_arithmetic_fun_test(Itv(-10, 0));
}

TEST(IntervalTest, MinMax) {
  EXPECT_EQ((Itv::template fun<MIN>(Itv::bot(), Itv(-10, 10))), Itv(zi::bot(), zd(10)));
  EXPECT_EQ((Itv::template fun<MIN>(Itv(-10, 10), Itv::bot())), Itv(zi::bot(), zd(10)));
  EXPECT_EQ((Itv::template fun<MAX>(Itv::bot(), Itv(-10, 10))), Itv(zi(-10), zd::bot()));
  EXPECT_EQ((Itv::template fun<MAX>(Itv(-10, 10), Itv::bot())), Itv(zi(-10), zd::bot()));
}

TEST(IntervalTest, Negation) {
  EXPECT_EQ((Itv::fun<NEG>(Itv(5, 10))), Itv(-10, -5));
  EXPECT_EQ((Itv::fun<NEG>(Itv(-10, 10))), Itv(-10, 10));
  EXPECT_EQ((Itv::fun<NEG>(Itv(10, -10))), Itv(10, -10));
  EXPECT_EQ((Itv::fun<NEG>(Itv(-10, -5))), Itv(5, 10));
}

TEST(IntervalTest, Absolute) {
  EXPECT_EQ((Itv::fun<ABS>(Itv(5, 10))), Itv(5, 10));
  EXPECT_EQ((Itv::fun<ABS>(Itv(-10, 10))), Itv(0, 10));
  EXPECT_EQ((Itv::fun<ABS>(Itv(10, -10))), Itv(10, 0));
  EXPECT_EQ((Itv::fun<ABS>(Itv(-10, -5))), Itv(5, 10));
  EXPECT_EQ((Itv::fun<ABS>(Itv(-15, 5))), Itv(0, 15));
}

TEST(IntervalTest, Addition) {
  EXPECT_EQ((Itv::fun<ADD>(Itv(-10, -10), Itv(-10, -10))), Itv(-20, -20));
  EXPECT_EQ((Itv::fun<ADD>(Itv(-10, -10), Itv(0, 0))), Itv(-10, -10));
  EXPECT_EQ((Itv::fun<ADD>(Itv(0, 0), Itv(-10, -10))), Itv(-10, -10));
  EXPECT_EQ((Itv::fun<ADD>(Itv(1, 10), Itv(1, 10))), Itv(2, 20));
  EXPECT_EQ((Itv::fun<ADD>(Itv(-1, 10), Itv(1, 10))), Itv(0, 20));
  EXPECT_EQ((Itv::fun<ADD>(Itv(-1, 10), Itv(-1, 10))), Itv(-2, 20));
  EXPECT_EQ((Itv::fun<ADD>(Itv(-10, -1), Itv(1, 10))), Itv(-9, 9));
}

TEST(IntervalTest, Subtraction) {
  EXPECT_EQ((Itv::fun<SUB>(Itv(-10, -10), Itv(-10, -10))), Itv(0, 0));
  EXPECT_EQ((Itv::fun<SUB>(Itv(-10, -10), Itv(0, 0))), Itv(-10, -10));
  EXPECT_EQ((Itv::fun<SUB>(Itv(0, 0), Itv(-10, -10))), Itv(10, 10));
  EXPECT_EQ((Itv::fun<SUB>(Itv(1, 10), Itv(1, 10))), Itv(-9, 9));
  EXPECT_EQ((Itv::fun<SUB>(Itv(-1, 10), Itv(1, 10))), Itv(-11, 9));
  EXPECT_EQ((Itv::fun<SUB>(Itv(-1, 10), Itv(-1, 10))), Itv(-11, 11));
  EXPECT_EQ((Itv::fun<SUB>(Itv(-10, -1), Itv(1, 10))), Itv(-20, -2));
}

TEST(IntervalTest, Multiplication) {
  EXPECT_EQ((Itv::fun<MUL>(Itv(-10, -2), Itv(-9, -3))), Itv(6, 90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(-10, -2), Itv(3, 9))), Itv(-90, -6));
  EXPECT_EQ((Itv::fun<MUL>(Itv(-10, -2), Itv(-9, 9))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(-10, -2), Itv(9, -9))), Itv(90, -90));

  EXPECT_EQ((Itv::fun<MUL>(Itv(2, 10), Itv(-9, -3))), Itv(-90, -6));
  EXPECT_EQ((Itv::fun<MUL>(Itv(2, 10), Itv(3, 9))), Itv(6, 90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(2, 10), Itv(-9, 9))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(2, 10), Itv(9, -9))), Itv(90, -90));

  EXPECT_EQ((Itv::fun<MUL>(Itv(-10, 10), Itv(-9, -3))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(-10, 10), Itv(3, 9))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(-10, 10), Itv(-9, 9))), Itv(-90, 90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(-10, 10), Itv(9, -9))), Itv::eq_zero());

  EXPECT_EQ((Itv::fun<MUL>(Itv(10, -10), Itv(-9, -3))), Itv(90, -90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(10, -10), Itv(3, 9))), Itv(90, -90));
  EXPECT_EQ((Itv::fun<MUL>(Itv(10, -10), Itv(-9, 9))), Itv::eq_zero());
  EXPECT_EQ((Itv::fun<MUL>(Itv(10, -10), Itv(9, -9))), Itv(90, -90));
}

// Based on the table provided in (Leijen D. (2003). Division and Modulus for Computer Scientists).
TEST(IntervalTest, GroundDivisionModulo) {
  // Eucliden Division and Modulo

  // a, b, qT, rT, qF, rF, qE, rE, qC, rC
  std::vector<std::vector<int>> div_mod = {
    {8, 3, 2, 2, 2, 2, 2, 2, 3, -1},
    {8, -3, -2, 2, -3, -1, -2, 2, -2, 2},
    {-8, 3, -2, -2, -3, 1, -3, 1, -2, -2},
    {-8, -3, 2, -2, 2, -2, 3, 1, 3, 1},
    {1, 2, 0, 1, 0, 1, 0, 1, 1, -1},
    {1, -2, 0, 1, -1, -1, 0, 1, 0, 1},
    {-1, 2, 0, -1, -1, 1, -1, 1, 0, -1},
    {-1, -2, 0, -1, 0, -1, 1, 1, 1, 1}
  };

  for(int i = 0; i < div_mod.size(); ++i) {
    Itv a(div_mod[i][0]);
    Itv b(div_mod[i][1]);
    EXPECT_EQ((Itv::fun<TDIV>(a, b)), (Itv(div_mod[i][2]))) << i;
    EXPECT_EQ((Itv::fun<TMOD>(a, b)), (Itv(div_mod[i][3]))) << i;
    EXPECT_EQ((Itv::fun<FDIV>(a, b)), (Itv(div_mod[i][4]))) << i;
    EXPECT_EQ((Itv::fun<FMOD>(a, b)), (Itv(div_mod[i][5]))) << i;
    EXPECT_EQ((Itv::fun<EDIV>(a, b)), (Itv(div_mod[i][6]))) << i;
    EXPECT_EQ((Itv::fun<EMOD>(a, b)), (Itv(div_mod[i][7]))) << i;
    EXPECT_EQ((Itv::fun<CDIV>(a, b)), (Itv(div_mod[i][8]))) << i;
    EXPECT_EQ((Itv::fun<CMOD>(a, b)), (Itv(div_mod[i][9]))) << i;
  }

  // std::vector<int> a = {1, 8, -1, -8};
  // std::vector<int> b = {2, 3, -2, -3};
  // for(int x : b) { printf("%d ", x); }
  // printf("\n");
  // for(int i = 0; i < a.size(); ++i) {
  //   printf("%d | ", a[i]);
  //   for(int j = 0; j < b.size(); ++j) {
  //     printf("%d ", Itv::fun<EDIV>(Itv(a[i]), Itv(b[j])).lb().value());
  //   }
  //   printf("\n");
  // }
}

TEST(IntervalTest, EuclideanDivision) {
  EXPECT_EQ((Itv::fun<EDIV>(Itv(1, 8), Itv(2, 3))), Itv(0, 4));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(1, 8), Itv(-3, 2))), Itv(-2, 4));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(1, 8), Itv(-2, 3))), Itv(-4, 2));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(1, 8), Itv(-3, -2))), Itv(-4, 0));

  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 8), Itv(2, 3))), Itv(-1, 4));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 8), Itv(-3, 2))), Itv(-2, 4));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 8), Itv(-2, 3))), Itv(-4, 2));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 8), Itv(-3, -2))), Itv(-4, 1));

  EXPECT_EQ((Itv::fun<EDIV>(Itv(-8, 1), Itv(2, 3))), Itv(-4, 0));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-8, 1), Itv(-3, 2))), Itv(-4, 3));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-8, 1), Itv(-2, 3))), Itv(-3, 4));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-8, 1), Itv(-3, -2))), Itv(0, 4));

  EXPECT_EQ((Itv::fun<EDIV>(Itv(-8, -1), Itv(2, 3))), Itv(-4, -1));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-8, -1), Itv(-3, 2))), Itv(-4, 3));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-8, -1), Itv(-2, 3))), Itv(-3, 4));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-8, -1), Itv(-3, -2))), Itv(1, 4));

  EXPECT_EQ((Itv::fun<EDIV>(Itv(0, 1), Itv(0, 1))), Itv(0, 1));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(0, 1), Itv(0, 0))), Itv::top());
  EXPECT_EQ((Itv::fun<EDIV>(Itv(0, 1), Itv(1, 1))), Itv(0, 1));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(0, 1), Itv(-1, 1))), Itv(-1, 1));

  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 1), Itv(0, 1))), Itv(-1, 1));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 1), Itv(0, 0))), Itv::top());
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 1), Itv(1, 1))), Itv(-1, 1));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 1), Itv(-1, 1))), Itv(-1, 1));

  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 0), Itv(0, 1))), Itv(-1, 0));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 0), Itv(0, 0))), Itv::top());
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 0), Itv(1, 1))), Itv(-1, 0));
  EXPECT_EQ((Itv::fun<EDIV>(Itv(-1, 0), Itv(-1, 1))), Itv(-1, 1));
}

TEST(IntervalTest, Width) {
  EXPECT_EQ(Itv(0,0).width(), Itv(0,0));
  EXPECT_EQ(Itv(-10, 10).width(), Itv(20,20));
  EXPECT_EQ(Itv(zi::bot(), zd(10)).width(), Itv::bot());
  EXPECT_EQ(Itv(zi(10), zd::bot()).width(), Itv::bot());
  EXPECT_EQ(Itv::bot().width(), Itv::bot());
  EXPECT_TRUE(Itv::top().width().is_top());
}

TEST(IntervalTest, Median) {
  EXPECT_EQ(Itv(0, 0).median(), Itv(0, 0));
  EXPECT_EQ(Itv(-10, 10).median(), Itv(0, 0));
  EXPECT_EQ(Itv(-9, 10).median(), Itv(0, 1));
  EXPECT_EQ(Itv(zi::bot(), zd(10)).median(), Itv::bot());
  EXPECT_EQ(Itv(zi(10), zd::bot()).median(), Itv::bot());
  EXPECT_EQ(Itv::bot().median(), Itv::bot());
  EXPECT_TRUE(Itv::top().median().is_top());
}
