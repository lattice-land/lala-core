// Copyright 2021 Pierre Talbot

#include "abstract_testing.hpp"
#include "lala/interval.hpp"

using zlb = local::ZLB;
using zub = local::ZUB;
using Itv = Interval<zlb>;

TEST(IntervalTest, BotTopTests) {
  bot_top_test(Itv(-1, 1));
  bot_top_test(Itv(zlb::top(), zub(0)));
  bot_top_test(Itv(zlb(0), zub::top()));
}

TEST(IntervalTest, NoInterpret) {
  VarEnv<standard_allocator> env = env_with_x();
  interpret_must_error<IKind::TELL, Itv>("constraint int_ne(x, 10);", env);
  interpret_must_error<IKind::ASK, Itv>("constraint float_eq(x, 1111111111.0000000000001);", env);
}

TEST(IntervalTest, ValidInterpret) {
  VarEnv<standard_allocator> env;
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 10);", Itv(10, 10), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 10);", Itv(zlb(11), zub::top()), env, false);
}

TEST(IntervalTest, JoinMeetTest) {
  join_meet_generic_test(Itv::bot(), Itv::top());
  join_meet_generic_test(Itv(0,0), Itv(0,0));
  join_meet_generic_test(Itv(0,1), Itv(0,1));
  join_meet_generic_test(Itv(0,5), Itv(0,10));
  join_meet_generic_test(Itv(5,5), Itv(0,10));
  join_meet_generic_test(Itv(0,0), Itv(0,1));
  join_meet_generic_test(Itv(1,1), Itv(0,1));

  EXPECT_EQ(fmeet(Itv(10, 20), Itv(4,14)), Itv(10,14));
  EXPECT_EQ(fjoin(Itv(10, 20), Itv(4,14)), Itv(4,20));

  EXPECT_EQ(fjoin(Itv(1, 9), Itv(11,10)), Itv(1, 9));
  EXPECT_EQ(fmeet(Itv(1, 9), Itv(11,10)), Itv::bot());
}

TEST(IntervalTest, OrderTest) {
  EXPECT_FALSE(Itv(10, 20) <= Itv(8, 12));
  EXPECT_TRUE(Itv(8, 12) <= Itv(8, 12));
  EXPECT_FALSE(Itv(7, 13) <= Itv(8, 12));
  EXPECT_TRUE(Itv(7, 13) >= Itv(8, 12));
  EXPECT_TRUE(Itv(10, 12) <= Itv(8, 12));

  EXPECT_FALSE(Itv(8, 12) <= Itv(10, 20));
  EXPECT_TRUE(Itv(8, 12) <= Itv(8, 12));
  EXPECT_TRUE(Itv(8, 12) <= Itv(7, 13));
  EXPECT_FALSE(Itv(8, 12) <= Itv(10, 12));
}

TEST(IntervalTest, GenericFunTests) {
  generic_unary_fun_test<Itv>(NEG);
  generic_abs_test<Itv>();
  generic_binary_fun_test(ADD, Itv(0,10));
  generic_binary_fun_test(SUB, Itv(0,10));
  generic_arithmetic_fun_test(Itv(0, 10));
  generic_arithmetic_fun_test(Itv(1, 10));
  generic_arithmetic_fun_test(Itv(-10, 10));
  generic_arithmetic_fun_test(Itv(-10, -1));
  generic_arithmetic_fun_test(Itv(-10, 0));
}

TEST(IntervalTest, MinMax) {
  EXPECT_EQ((project_fun(MIN, Itv::top(), Itv(-10, 10))), Itv(zlb::top(), zub(10)));
  EXPECT_EQ((project_fun(MIN, Itv(-10, 10), Itv::top())), Itv(zlb::top(), zub(10)));
  EXPECT_EQ((project_fun(MAX, Itv::top(), Itv(-10, 10))), Itv(zlb(-10), zub::top()));
  EXPECT_EQ((project_fun(MAX, Itv(-10, 10), Itv::top())), Itv(zlb(-10), zub::top()));
}

TEST(IntervalTest, Negation) {
  EXPECT_EQ((project_fun(NEG, Itv(5, 10))), Itv(-10, -5));
  EXPECT_EQ((project_fun(NEG, Itv(-10, 10))), Itv(-10, 10));
  EXPECT_EQ((project_fun(NEG, Itv(10, -10))), Itv(10, -10));
  EXPECT_EQ((project_fun(NEG, Itv(-10, -5))), Itv(5, 10));
}

TEST(IntervalTest, Absolute) {
  EXPECT_EQ((project_fun(ABS, Itv(5, 10))), Itv(5, 10));
  EXPECT_EQ((project_fun(ABS, Itv(-10, 10))), Itv(0, 10));
  EXPECT_EQ((project_fun(ABS, Itv(10, -10))), Itv(10, 0));
  EXPECT_EQ((project_fun(ABS, Itv(-10, -5))), Itv(5, 10));
  EXPECT_EQ((project_fun(ABS, Itv(-15, 5))), Itv(0, 15));
}

TEST(IntervalTest, Addition) {
  EXPECT_EQ((project_fun(ADD, Itv(-10, -10), Itv(-10, -10))), Itv(-20, -20));
  EXPECT_EQ((project_fun(ADD, Itv(-10, -10), Itv(0, 0))), Itv(-10, -10));
  EXPECT_EQ((project_fun(ADD, Itv(0, 0), Itv(-10, -10))), Itv(-10, -10));
  EXPECT_EQ((project_fun(ADD, Itv(1, 10), Itv(1, 10))), Itv(2, 20));
  EXPECT_EQ((project_fun(ADD, Itv(-1, 10), Itv(1, 10))), Itv(0, 20));
  EXPECT_EQ((project_fun(ADD, Itv(-1, 10), Itv(-1, 10))), Itv(-2, 20));
  EXPECT_EQ((project_fun(ADD, Itv(-10, -1), Itv(1, 10))), Itv(-9, 9));
}

TEST(IntervalTest, Subtraction) {
  EXPECT_EQ((project_fun(SUB, Itv(-10, -10), Itv(-10, -10))), Itv(0, 0));
  EXPECT_EQ((project_fun(SUB, Itv(-10, -10), Itv(0, 0))), Itv(-10, -10));
  EXPECT_EQ((project_fun(SUB, Itv(0, 0), Itv(-10, -10))), Itv(10, 10));
  EXPECT_EQ((project_fun(SUB, Itv(1, 10), Itv(1, 10))), Itv(-9, 9));
  EXPECT_EQ((project_fun(SUB, Itv(-1, 10), Itv(1, 10))), Itv(-11, 9));
  EXPECT_EQ((project_fun(SUB, Itv(-1, 10), Itv(-1, 10))), Itv(-11, 11));
  EXPECT_EQ((project_fun(SUB, Itv(-10, -1), Itv(1, 10))), Itv(-20, -2));
}

TEST(IntervalTest, Multiplication) {
  EXPECT_EQ((project_fun(MUL, Itv(-10, -2), Itv(-9, -3))), Itv(6, 90));
  EXPECT_EQ((project_fun(MUL, Itv(-10, -2), Itv(3, 9))), Itv(-90, -6));
  EXPECT_EQ((project_fun(MUL, Itv(-10, -2), Itv(-9, 9))), Itv(-90, 90));

  EXPECT_EQ((project_fun(MUL, Itv(2, 10), Itv(-9, -3))), Itv(-90, -6));
  EXPECT_EQ((project_fun(MUL, Itv(2, 10), Itv(3, 9))), Itv(6, 90));
  EXPECT_EQ((project_fun(MUL, Itv(2, 10), Itv(-9, 9))), Itv(-90, 90));

  EXPECT_EQ((project_fun(MUL, Itv(-10, 10), Itv(-9, -3))), Itv(-90, 90));
  EXPECT_EQ((project_fun(MUL, Itv(-10, 10), Itv(3, 9))), Itv(-90, 90));
  EXPECT_EQ((project_fun(MUL, Itv(-10, 10), Itv(-9, 9))), Itv(-90, 90));

  EXPECT_EQ((project_fun(MUL, Itv(-10, 10), Itv(9, -9))), Itv::bot());
  EXPECT_EQ((project_fun(MUL, Itv(9, -9), Itv(-10, 10))), Itv::bot());

  EXPECT_EQ((project_fun(MUL, Itv(zlb::top(), 2), Itv(2, 2))), Itv(zlb::top(), 4));
  EXPECT_EQ((project_fun(MUL, Itv(-2, -2), Itv(2, zub::top()))), Itv(zlb::top(), -4));
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
    EXPECT_EQ((project_fun(TDIV, a, b)), (Itv(div_mod[i][2]))) << i;
    EXPECT_EQ((project_fun(TMOD, a, b)), (Itv(div_mod[i][3]))) << i;
    EXPECT_EQ((project_fun(FDIV, a, b)), (Itv(div_mod[i][4]))) << i;
    EXPECT_EQ((project_fun(FMOD, a, b)), (Itv(div_mod[i][5]))) << i;
    EXPECT_EQ((project_fun(EDIV, a, b)), (Itv(div_mod[i][6]))) << i;
    EXPECT_EQ((project_fun(EMOD, a, b)), (Itv(div_mod[i][7]))) << i;
    EXPECT_EQ((project_fun(CDIV, a, b)), (Itv(div_mod[i][8]))) << i;
    EXPECT_EQ((project_fun(CMOD, a, b)), (Itv(div_mod[i][9]))) << i;
  }

  // std::vector<int> a = {1, 8, -1, -8};
  // std::vector<int> b = {2, 3, -2, -3};
  // for(int x : b) { printf("%d ", x); }
  // printf("\n");
  // for(int i = 0; i < a.size(); ++i) {
  //   printf("%d | ", a[i]);
  //   for(int j = 0; j < b.size(); ++j) {
  //     printf("%d ", project_fun(EDIV, Itv(a[i]), Itv(b[j])).lb().value());
  //   }
  //   printf("\n");
  // }
}

TEST(IntervalTest, EuclideanDivision) {
  EXPECT_EQ((project_fun(EDIV, Itv(1, 8), Itv(2, 3))), Itv(0, 4));
  EXPECT_EQ((project_fun(EDIV, Itv(1, 8), Itv(-3, 2))), Itv(-8, 8));
  EXPECT_EQ((project_fun(EDIV, Itv(1, 8), Itv(-2, 3))), Itv(-8, 8));
  EXPECT_EQ((project_fun(EDIV, Itv(1, 8), Itv(-3, -2))), Itv(-4, 0));

  EXPECT_EQ((project_fun(EDIV, Itv(-1, 8), Itv(2, 3))), Itv(-1, 4));
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 8), Itv(-3, 2))), Itv(-8, 8));
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 8), Itv(-2, 3))), Itv(-8, 8));
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 8), Itv(-3, -2))), Itv(-4, 1));

  EXPECT_EQ((project_fun(EDIV, Itv(-8, 1), Itv(2, 3))), Itv(-4, 0));
  EXPECT_EQ((project_fun(EDIV, Itv(-8, 1), Itv(-3, 2))), Itv(-8, 8));
  EXPECT_EQ((project_fun(EDIV, Itv(-8, 1), Itv(-2, 3))), Itv(-8, 8));
  EXPECT_EQ((project_fun(EDIV, Itv(-8, 1), Itv(-3, -2))), Itv(0, 4));

  EXPECT_EQ((project_fun(EDIV, Itv(-8, -1), Itv(2, 3))), Itv(-4, -1));
  EXPECT_EQ((project_fun(EDIV, Itv(-8, -1), Itv(-3, 2))), Itv(-8, 8));
  EXPECT_EQ((project_fun(EDIV, Itv(-8, -1), Itv(-2, 3))), Itv(-8, 8));
  EXPECT_EQ((project_fun(EDIV, Itv(-8, -1), Itv(-3, -2))), Itv(1, 4));

  EXPECT_EQ((project_fun(EDIV, Itv(0, 1), Itv(0, 1))), Itv(0, 1));
  EXPECT_EQ((project_fun(EDIV, Itv(0, 1), Itv(0, 0))), Itv::bot());
  EXPECT_EQ((project_fun(EDIV, Itv(0, 1), Itv(1, 1))), Itv(0, 1));
  EXPECT_EQ((project_fun(EDIV, Itv(0, 1), Itv(-1, 1))), Itv(-1, 1));

  EXPECT_EQ((project_fun(EDIV, Itv(-1, 1), Itv(0, 1))), Itv(-1, 1));
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 1), Itv(0, 0))), Itv::bot());
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 1), Itv(1, 1))), Itv(-1, 1));
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 1), Itv(-1, 1))), Itv(-1, 1));

  EXPECT_EQ((project_fun(EDIV, Itv(-1, 0), Itv(0, 1))), Itv(-1, 0));
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 0), Itv(0, 0))), Itv::bot());
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 0), Itv(1, 1))), Itv(-1, 0));
  EXPECT_EQ((project_fun(EDIV, Itv(-1, 0), Itv(-1, 1))), Itv(-1, 1));

  EXPECT_EQ((project_fun(EDIV, Itv(zlb::top(), 2), Itv(2, 2))), Itv(zlb::top(), 1));
  EXPECT_EQ((project_fun(EDIV, Itv::top(), Itv::top())), Itv::top());

  EXPECT_EQ((project_fun(EDIV, Itv(0, 1), Itv(2, 2))), Itv(0, 0));
  EXPECT_EQ((project_fun(EDIV, Itv(1, 1), Itv(2, 2))), Itv(0, 0));

  EXPECT_EQ((project_fun(EDIV, Itv(zlb::top(), 2), Itv(zlb::top(), 2))), Itv(zlb::top(), 2));
  // EXPECT_EQ((project_fun(EDIV, Itv(zlb::top(), -2), Itv(2, zub::top()))), Itv(zlb::top(), 0));

  EXPECT_EQ((project_fun(EDIV, Itv(-10, 10), Itv::top())), Itv(-10, 10));
  EXPECT_EQ((project_fun(EDIV, Itv(0, 10), Itv::top())), Itv(-10, 10));
  EXPECT_EQ((project_fun(EDIV, Itv(-10, 0), Itv::top())), Itv(-10, 10));
}

TEST(IntervalTest, Width) {
  EXPECT_EQ(Itv(0,0).width(), Itv(0,0));
  EXPECT_EQ(Itv(-10, 10).width(), Itv(20,20));
  EXPECT_EQ(Itv(zlb::top(), zub(10)).width(), Itv::top());
  EXPECT_EQ(Itv(zlb(10), zub::top()).width(), Itv::top());
  EXPECT_EQ(Itv::top().width(), Itv::top());
  EXPECT_TRUE(Itv::bot().width().is_bot());
}

TEST(IntervalTest, Median) {
  EXPECT_EQ(Itv(0, 0).median(), Itv(0, 0));
  EXPECT_EQ(Itv(-10, 10).median(), Itv(0, 0));
  EXPECT_EQ(Itv(-9, 10).median(), Itv(0, 1));
  EXPECT_EQ(Itv(zlb::top(), zub(10)).median(), Itv::top());
  EXPECT_EQ(Itv(zlb(10), zub::top()).median(), Itv::top());
  EXPECT_EQ(Itv::top().median(), Itv::top());
  EXPECT_TRUE(Itv::bot().median().is_bot());
}
