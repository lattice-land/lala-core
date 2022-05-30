// Copyright 2021 Pierre Talbot

#include "z.hpp"
#include "generic_universe_test.hpp"

using zi = ZInc<int>;
using zd = ZDec<int>;

TEST(ZDeathTest, BadConstruction) {
  ASSERT_DEATH(zi(Limits<int>::bot()), "");
  ASSERT_DEATH(zi(Limits<int>::top()), "");
  // Dual
  ASSERT_DEATH(zd(Limits<int>::bot()), "");
  ASSERT_DEATH(zd(Limits<int>::top()), "");
}

TEST(ZTest, TopBot) {
  bot_top_test<zi>(zi(0));
  bot_top_test<zd>(zd(0));
}

TEST(ZTest, ValidInterpret) {
  test_all_interpret<zi>(GEQ, 10, zi(10));
  test_all_interpret<zi>(GT, 10, zi(11));
  test_under_interpret<zi>(NEQ, 10, zi(11));
  test_over_interpret<zi>(EQ, 10, zi(10));
  // Dual
  test_all_interpret<zd>(LEQ, 10, zd(10));
  test_all_interpret<zd>(LT, 10, zd(9));
  test_under_interpret<zd>(NEQ, 10, zd(9));
  test_over_interpret<zd>(EQ, 10, zd(10));
}

TEST(ZTest, NoInterpret) {
  test_exact_interpret<zi>(NEQ, 10, {});
  test_exact_interpret<zi>(EQ, 10, {});
  test_all_interpret<zi>(LEQ, 10, {});
  test_all_interpret<zi>(LT, 10, {});
  // Dual
  test_exact_interpret<zd>(NEQ, 10, {});
  test_exact_interpret<zd>(EQ, 10, {});
  test_all_interpret<zd>(GEQ, 10, {});
  test_all_interpret<zd>(GT, 10, {});
}

TEST(ZTest, ExistentialInterpret) {
  Approx appxs[3] = {EXACT, UNDER, OVER};
  for(int i = 0; i < 3; ++i) {
    auto e = F::make_exists(0, var_x, Int, appxs[i]);
    zi z = zi::bot();
    ASSERT_TRUE(z.interpret(e).has_value());
    ASSERT_TRUE(z.interpret(e)->is_bot().value());
  }
  auto e1 = F::make_exists(0, var_x, Real, EXACT);
  ASSERT_FALSE(zi::bot().interpret(e1).has_value());
  auto e2 = F::make_exists(0, var_x, Real, OVER);
  ASSERT_FALSE(zi::bot().interpret(e2).has_value());
  auto e3 = F::make_exists(0, var_x, Real, UNDER);
  ASSERT_TRUE(zi::bot().interpret(e3).has_value());
}

TEST(ZTest, JoinMeet) {
  join_meet_generic_test(zi::bot(), zi::top());
  join_meet_generic_test(zi(0), zi(1));
  join_meet_generic_test(zi(-10), zi(10));
  join_meet_generic_test(zi(Limits<int>::top() - 1), zi::top());
  // Dual
  join_meet_generic_test(zd::bot(), zd::top());
  join_meet_generic_test(zd(1), zd(0));
  join_meet_generic_test(zd(10), zd(-10));
  join_meet_generic_test(zd(Limits<int>::bot() + 1), zd::top());
}

TEST(ZTest, Order) {
  EXPECT_EQ2((leq<zi>(zi(0), zd(0))), true);
  EXPECT_EQ2((leq<zi>(zi(1), zd(0))), false);
  EXPECT_EQ2((leq<zi>(zi(0), zd(1))), true);
  EXPECT_EQ2((leq<zi>(zi(0), zd(-1))), false);
  EXPECT_EQ2((leq<zi>(zi(-1), zd(0))), true);
  generic_order_test(zi(0));
  // Dual
  EXPECT_EQ2((leq<zd>(zd(0), zi(0))), true);
  EXPECT_EQ2((leq<zd>(zd(1), zi(0))), true);
  EXPECT_EQ2((leq<zd>(zd(0), zi(1))), false);
  EXPECT_EQ2((leq<zd>(zd(0), zi(-1))), true);
  EXPECT_EQ2((leq<zd>(zd(-1), zi(0))), false);
  generic_order_test(zd(0));
}

TEST(ZTest, Split) {
  generic_split_test(zi(0));
  generic_split_test(zd(0));
}

TEST(ZTest, Deinterpret) {
  F f10 = make_v_op_z(var_x, GEQ, 10, UNTYPED, EXACT, standard_allocator);
  zi z10 = zi::interpret(f10).value();
  F f10_bis = z10.deinterpret(var_x);
  EXPECT_EQ(f10, f10_bis);
  F f9 = make_v_op_z(var_x, GT, 9, UNTYPED, EXACT, standard_allocator);
  zi z9 = zi::interpret(f9).value();
  F f9_bis = z9.deinterpret(var_x);
  EXPECT_EQ(f10, f9_bis);
  generic_deinterpret_test<zi>();
  // Dual
  F f10_d = make_v_op_z(var_x, LEQ, 10, UNTYPED, EXACT, standard_allocator);
  zd z10_d = zd::interpret(f10_d).value();
  F f10_bis_d = z10_d.deinterpret(var_x);
  EXPECT_EQ(f10_d, f10_bis_d);
  F f11_d = make_v_op_z(var_x, LT, 11, UNTYPED, EXACT, standard_allocator);
  zd z11_d = zd::interpret(f11_d).value();
  F f11_bis_d = z11_d.deinterpret(var_x);
  EXPECT_EQ(f10_d, f11_bis_d);
  generic_deinterpret_test<zd>();
}
