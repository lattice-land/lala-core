// Copyright 2025 Yi-Nung Tsao 

#include "abstract_testing.hpp"
#include "lala/interval.hpp"

using flb = local::FLB;
using fub = local::FUB;
using Itv = Interval<flb>;

TEST(IntervalFloatingTest, BotTopTests) {
  bot_top_test(create_float_interval<Itv>("-1.0", "1.0"));
  bot_top_test(create_float_interval<Itv>(flb::top(), "0.0"));
  bot_top_test(create_float_interval<Itv>("0.0", fub::top()));
}

TEST(IntervalFloatingTest, NoInterpret) {
  VarEnv<standard_allocator> env = env_with_x();
  interpret_must_error<IKind::TELL, Itv>("constraint int_ne(x, 10);", env);
  interpret_must_error<IKind::ASK, Itv>("constraint float_eq(x, 1111111111.0000000000001);", env);
}

TEST(IntervalFloatingTest, ValidInterpret) {
  VarEnv<standard_allocator> env; 
  expect_interpret_equal_to<IKind::TELL>("constraint float_eq(x, 10.0);", create_float_interval<Itv>(10.0, 10.0), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint float_ne(x, 10.0);", create_float_interval<Itv>(10.000000000000002, fub::top()), env, false);
}

TEST(IntervalFloatingTest, JoinMeetTest) {
  join_meet_generic_test(Itv::bot(), Itv::top());
  join_meet_generic_test(create_float_interval<Itv>("0.5", "0.5"), create_float_interval<Itv>("0.5", "0.5"));
  join_meet_generic_test(create_float_interval<Itv>("0.0", "1.010101"), create_float_interval<Itv>("0.0", "1.010101"));
  join_meet_generic_test(create_float_interval<Itv>("0.99999", "5.123456"), create_float_interval<Itv>("0.99999", "10.87654"));
  join_meet_generic_test(create_float_interval<Itv>("5.123456", "5.123456"), create_float_interval<Itv>("0.999999", "10.87654"));
  join_meet_generic_test(create_float_interval<Itv>("0.99999", "0.99999"), create_float_interval<Itv>("0.99999", "1.010101"));
  join_meet_generic_test(create_float_interval<Itv>("1.010101", "1.010101"), create_float_interval<Itv>("0.99999", "1.010101"));

  EXPECT_EQ(fmeet(create_float_interval<Itv>("-10.54321", "20.099999"), create_float_interval<Itv>("-4.1010222", "14.654321")), create_float_interval<Itv>("-4.1010222", "14.654321"));
  EXPECT_EQ(fjoin(create_float_interval<Itv>("-10.54321", "20.099999"), create_float_interval<Itv>("-4.1010222", "14.654321")), create_float_interval<Itv>("-10.54321", "20.099999"));

  EXPECT_EQ(fjoin(create_float_interval<Itv>("1.3333333", "9.188888"), create_float_interval<Itv>("11.9877777", "10.3322111")), create_float_interval<Itv>("1.3333333", "9.188888")); 
  EXPECT_EQ(fmeet(create_float_interval<Itv>("1.555555", "9.188888"), create_float_interval<Itv>("11.77771122", "10.3322111")), Itv::bot());
}

TEST(IntervalFloatingTest, OrderTest) {
  EXPECT_FALSE(create_float_interval<Itv>("10.0000001", "20.59999999") <= create_float_interval<Itv>("10.0000002", "20.50000001"));
  EXPECT_TRUE(create_float_interval<Itv>("8.000001", "12.000002") <= create_float_interval<Itv>("8.000001", "12.000002"));
  EXPECT_FALSE(create_float_interval<Itv>("7.0", "13.0") <= create_float_interval<Itv>("7.0000001", "12.587841"));
  EXPECT_TRUE(create_float_interval<Itv>("7.0000002", "13.77777") >= create_float_interval<Itv>("8.11111111", "12.666666"));
  EXPECT_TRUE(create_float_interval<Itv>("10.0000001", "12.66666") <= create_float_interval<Itv>("8.00000001", "12.66666"));

  EXPECT_FALSE(create_float_interval<Itv>("8.000001", "12.555555") <= create_float_interval<Itv>("10.0000001", "20.3333333"));
  EXPECT_TRUE(create_float_interval<Itv>("8.1111111", "12.0000001") <= create_float_interval<Itv>("8.111111", "12.0000001"));
  EXPECT_TRUE(create_float_interval<Itv>("8.11033325", "12.11111112") <= create_float_interval<Itv>("7.0512", "13.0000001"));
  EXPECT_FALSE(create_float_interval<Itv>("8.0000003", "12.000001") <= create_float_interval<Itv>("10.000002", "12.000001"));
}

TEST(IntervalFloatingTest, GenericFunTests) {
  generic_unary_fun_test<Itv>(NEG);
  generic_abs_test<Itv>('F');
  generic_binary_fun_test(ADD, create_float_interval<Itv>("0.0199999", "10.78328"));
  generic_binary_fun_test(SUB, create_float_interval<Itv>("0.0199999", "10.78328"));
  
  generic_arithmetic_fun_test(create_float_interval<Itv>("0.0199999", "10.78328"));
  generic_arithmetic_fun_test(create_float_interval<Itv>("1.0199999", "10.78328"));
  generic_arithmetic_fun_test(create_float_interval<Itv>("-10.78328", "10.78328"));
  generic_arithmetic_fun_test(create_float_interval<Itv>("-10.78328", "-1.0199999"));
  generic_arithmetic_fun_test(create_float_interval<Itv>("-10.78328", "0.78328"));
}

TEST(IntervalFloatingTest, MinMax) {
  EXPECT_EQ((project_fun(MIN, Itv::top(), create_float_interval<Itv>("-10.0111111111", "10.0123132121"))), create_float_interval<Itv>(flb::top(), "10.0123132121"));
  EXPECT_EQ((project_fun(MIN, create_float_interval<Itv>("-10.0111111111", "10.0123132121"), Itv::top())), create_float_interval<Itv>(flb::top(), "10.0123132121"));
  EXPECT_EQ((project_fun(MAX, Itv::top(), create_float_interval<Itv>("-10.0111111111", "10.0123132121"))), create_float_interval<Itv>("-10.0111111111", fub::top()));
  EXPECT_EQ((project_fun(MAX, create_float_interval<Itv>("-10.0111111111", "10.0123132121"), Itv::top())), create_float_interval<Itv>("-10.0111111111", fub::top()));
}

TEST(IntervalFloatingTest, Negation) {
  EXPECT_EQ((project_fun(NEG, create_float_interval<Itv>("5.000001111", "10.099999992222"))), create_float_interval<Itv>("-10.099999992222", "-5.000001111"));
  EXPECT_EQ((project_fun(NEG, create_float_interval<Itv>("-10.099999992222", "10.099999992222"))), create_float_interval<Itv>("-10.099999992222", "10.099999992222"));
  EXPECT_EQ((project_fun(NEG, create_float_interval<Itv>("10.099999992222", "-10.099999992222"))), create_float_interval<Itv>("10.099999992222", "-10.099999992222"));
  EXPECT_EQ((project_fun(NEG, create_float_interval<Itv>("-10.099999992222", "-5.000001111"))), create_float_interval<Itv>("5.000001111", "10.099999992222"));
}

TEST(IntervalFloatingTest, Absolute) {
  EXPECT_EQ((project_fun(ABS, create_float_interval<Itv>("5.000001111", "10.099999992222"))), create_float_interval<Itv>("5.000001111", "10.099999992222"));
  EXPECT_EQ((project_fun(ABS, create_float_interval<Itv>("-10.099999992222", "10.099999992222"))), create_float_interval<Itv>("0.0", "10.099999992222"));
  EXPECT_EQ((project_fun(ABS, create_float_interval<Itv>("10.099999992222", "-10.099999992222"))), create_float_interval<Itv>("10.099999992222", "0.0"));
  EXPECT_EQ((project_fun(ABS, create_float_interval<Itv>("-10.099999992222", "-5.000001111"))), create_float_interval<Itv>("5.000001111", "10.099999992222"));
  EXPECT_EQ((project_fun(ABS, create_float_interval<Itv>("-15.123123", "5.000001111"))), create_float_interval<Itv>("0.0", "15.123123"));
}

TEST(IntervalFloatingTest, Addition) {
  EXPECT_EQ((project_fun(ADD, create_float_interval<Itv>("-10.01010101011", "-10.01010101011"), create_float_interval<Itv>("-10.01010101011", "-10.01010101011"))), create_float_interval<Itv>("-20.02020202022", "-20.02020202022"));

  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("-10.99999", "-10.99999"), create_float_interval<Itv>("-10.00001", "-10.00001")).lb()), -21.0);
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("-10.99999", "-10.99999"), create_float_interval<Itv>("-10.00001", "-10.00001")).ub()), -20.999999999999996);

  EXPECT_EQ((project_fun(ADD, create_float_interval<Itv>("-10.0000000000001", "-10.0000000000001"), create_float_interval<Itv>("0.00000000000001", "0.00000000000001"))), create_float_interval<Itv>(-10.000000000000092, -10.000000000000088));
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("0.00000000000001", "0.00000000000001"), create_float_interval<Itv>("-10.00000000000001", "-10.00000000000001")).lb()), -10.0);
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("0.00000000000001", "0.00000000000001"), create_float_interval<Itv>("-10.00000000000001", "-10.00000000000001")).ub()), -9.9999999999999982);
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("-10.000001", "-10.000001"), create_float_interval<Itv>("0.000001", "0.000001")).lb()), -10.0);
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("-10.000001", "-10.000001"), create_float_interval<Itv>("0.000001", "0.000001")).ub()), -9.9999999999999982);

  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("1.000000000000001", "10.000000000000001"), create_float_interval<Itv>("1.999999999999999", "10.999999999999999")).lb()), 3.0);
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("1.000000000000001", "10.000000000000001"), create_float_interval<Itv>("1.999999999999999", "10.999999999999999")).ub()), 21.000000000000004);
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("-1.0000000000000001", "10.0000000000000001"), create_float_interval<Itv>("1.0000000000000001", "10.9999999999999999")).lb()), -2.220446049250313e-16);
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("-1.0000000000000001", "10.0000000000000001"), create_float_interval<Itv>("1.0000000000000001", "10.9999999999999999")).ub()), 21.000000000000004);

  EXPECT_EQ((project_fun(ADD, create_float_interval<Itv>("-1.5", "10.9"), create_float_interval<Itv>("-1.3", "10.7"))), create_float_interval<Itv>("-2.8", "21.6"));
  EXPECT_EQ((project_fun(ADD, create_float_interval<Itv>("-10.2", "-1.7"), create_float_interval<Itv>("1.1", "10.666"))), create_float_interval<Itv>("-9.1", "8.966"));
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("-10.1", "-1.1"), create_float_interval<Itv>("1.1", "10.1")).lb()), -9.0);
  EXPECT_DOUBLE_EQ((project_fun(ADD, create_float_interval<Itv>("-10.1", "-1.1"), create_float_interval<Itv>("1.1", "10.1")).ub()), 9.0000000000000018);
}

TEST(IntervalFloatingTest, Subtraction) {
  EXPECT_DOUBLE_EQ((project_fun(SUB, create_float_interval<Itv>("-10.001", "-10.001"), create_float_interval<Itv>("-10.001", "-10.001")).lb()), -1.7763568394002504e-15);
  EXPECT_DOUBLE_EQ((project_fun(SUB, create_float_interval<Itv>("-10.001", "-10.001"), create_float_interval<Itv>("-10.001", "-10.001")).ub()), 1.7763568394002505e-15);
  EXPECT_EQ((project_fun(SUB, create_float_interval<Itv>("-10.99999", "-10.99999"), create_float_interval<Itv>("0.0", "0.0"))), create_float_interval<Itv>("-10.99999", "-10.99999"));
  EXPECT_EQ((project_fun(SUB, create_float_interval<Itv>("0.0", "0.0"), create_float_interval<Itv>("-10.123456", "-10.123456"))), create_float_interval<Itv>("10.123456", "10.123456"));
  EXPECT_DOUBLE_EQ((project_fun(SUB, create_float_interval<Itv>("1.5432", "10.3456"), create_float_interval<Itv>("1.5432", "10.3456")).lb()), -8.8024000000000022);
  EXPECT_DOUBLE_EQ((project_fun(SUB, create_float_interval<Itv>("1.5432", "10.3456"), create_float_interval<Itv>("1.5432", "10.3456")).ub()), 8.8024000000000023);

  EXPECT_EQ((project_fun(SUB, create_float_interval<Itv>("-1.0", "10.0"), create_float_interval<Itv>("1.0", "10.0"))), create_float_interval<Itv>("-11.0", "9.0"));
  EXPECT_EQ((project_fun(SUB, create_float_interval<Itv>("-1.0", "10.0"), create_float_interval<Itv>("-1.0", "10.0"))), create_float_interval<Itv>("-11.0", "11.0"));
  EXPECT_EQ((project_fun(SUB, create_float_interval<Itv>("-10.0", "-1.0"), create_float_interval<Itv>("1.0", "10.0"))), create_float_interval<Itv>("-20.0", "-2.0"));
}

TEST(IntervalFloatingTest, Multiplication) {
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "-2.0101"), create_float_interval<Itv>("-9.1234", "-3.9999")).lb()), 8.04019899);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "-2.0101"), create_float_interval<Itv>("-9.1234", "-3.9999")).ub()), 96.09494752);

  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "-2.0101"), create_float_interval<Itv>("3.9999", "9.1234")).lb()), -96.09494752);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "-2.0101"), create_float_interval<Itv>("3.9999", "9.1234")).ub()), -8.04019899);

  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "-2.0101"), create_float_interval<Itv>("-9.1234", "9.1234")).lb()), -96.09494752);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "-2.0101"), create_float_interval<Itv>("-9.1234", "9.1234")).ub()), 96.09494752);

  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("2.0101", "10.5328"), create_float_interval<Itv>("-9.1234", "-3.9999")).lb()), -96.09494752);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("2.0101", "10.5328"), create_float_interval<Itv>("-9.1234", "-3.9999")).ub()), -8.04019899);
  
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("2.0101", "10.5328"), create_float_interval<Itv>("3.9999", "9.1234")).lb()), 8.04019899);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("2.0101", "10.5328"), create_float_interval<Itv>("3.9999", "9.1234")).ub()), 96.09494752);

  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("2.0101", "10.5328"), create_float_interval<Itv>("-9.1234", "9.1234")).lb()), -96.09494752);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("2.0101", "10.5328"), create_float_interval<Itv>("-9.1234", "9.1234")).ub()), 96.09494752);

  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "10.5328"), create_float_interval<Itv>("-9.1234", "-3.9999")).lb()), -96.09494752);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "10.5328"), create_float_interval<Itv>("-9.1234", "-3.9999")).ub()), 96.09494752);

  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "10.5328"), create_float_interval<Itv>("3.9999", "9.1234")).lb()), -96.09494752);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "10.5328"), create_float_interval<Itv>("3.9999", "9.1234")).ub()), 96.09494752);

  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "10.5328"), create_float_interval<Itv>("-9.1234", "9.1234")).lb()), -96.09494752);
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>("-10.5328", "10.5328"), create_float_interval<Itv>("-9.1234", "9.1234")).ub()), 96.09494752);

  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>(flb::top(), "2.0101"), create_float_interval<Itv>("2.0101", "2.0101")).lb()), flb::top());
  EXPECT_DOUBLE_EQ((project_fun(MUL, create_float_interval<Itv>(flb::top(), "2.0101"), create_float_interval<Itv>("2.0101", "2.0101")).ub()), 4.04050201);

  EXPECT_EQ((project_fun(MUL, create_float_interval<Itv>("-2.0101", "-2.0101"), create_float_interval<Itv>("2.0101", fub::top()))), create_float_interval<Itv>(flb::top(), "-4.04050201"));
}

TEST(IntervalFloatingTest, Width) {
  EXPECT_EQ(create_float_interval<Itv>("0.0", "0.0").width(), create_float_interval<Itv>("0.0", "0.0"));
  EXPECT_EQ(create_float_interval<Itv>("-10.0", "10.0").width(), create_float_interval<Itv>("20.0", "20.0"));
  EXPECT_EQ(create_float_interval<Itv>(flb::top(), "10.0").width(), Itv::top());
  EXPECT_EQ(create_float_interval<Itv>("10.0", fub::top()).width(), Itv::top());
  EXPECT_EQ(Itv::top().width(), Itv::top());
  EXPECT_TRUE(Itv::bot().width().is_bot());
}

// TEST(IntervalFloatingTest, Median) {
//   EXPECT_EQ(Itv(0, 0).median(), Itv(0, 0));
//   EXPECT_EQ(Itv(-10, 10).median(), Itv(0, 0));
//   EXPECT_EQ(Itv(-9, 10).median(), Itv(0, 1));
//   EXPECT_EQ(Itv(flb::top(), fub(10)).median(), Itv::top());
//   EXPECT_EQ(Itv(flb(10), fub::top()).median(), Itv::top());
//   EXPECT_EQ(Itv::top().median(), Itv::top());
//   EXPECT_TRUE(Itv::bot().median().is_bot());
// }
