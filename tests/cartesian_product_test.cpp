// Copyright 2021 Pierre Talbot

#include "z.hpp"
#include "cartesian_product.hpp"
#include "generic_universe_test.hpp"

using zi = ZInc<int>;
using zd = ZDec<int>;
using Itv = CartesianProduct<zi, zd>;

TEST(CPTest, BotTopTests) {
  Itv itv1_2(zi(1), zd(2));
  bot_top_test(itv1_2);
}

TEST(CPTest, NoInterpret) {
  test_exact_interpret<Itv>(NEQ, 10, {});
  test_exact_interpret<Itv>(EQ, 10, {});
}

TEST(CPTest, ValidInterpret) {
  // First component.
  test_all_interpret<Itv>(GEQ, 10, join<0>(Itv::bot(), zi(10)));
  test_all_interpret<Itv>(GT, 10, join<0>(Itv::bot(), zi(11)));
  // Second component.
  test_all_interpret<Itv>(LEQ, 10, join<1>(Itv::bot(), zd(10)));
  test_all_interpret<Itv>(LT, 10, join<1>(Itv::bot(), zd(9)));
  // Both.
  test_under_interpret<Itv>(NEQ, 10, Itv(11, 9));
  test_over_interpret<Itv>(EQ, 10, Itv(10, 10));
}

TEST(CPTest, InterpretTwo) {
  auto geq_1 = make_v_op_z(var_x, GEQ, 1, EXACT, standard_allocator);
  auto leq_2 = make_v_op_z(var_x, LEQ, 2, EXACT, standard_allocator);
  auto geq_1_leq_2 = F::make_binary(geq_1, AND, leq_2);
  auto f1_opt = Itv::interpret(geq_1);
  EXPECT_TRUE(f1_opt.has_value());
  auto f2_opt = Itv::interpret(leq_2);
  EXPECT_TRUE(f2_opt.has_value());

  auto f1 = f1_opt.value();
  auto f2 = f2_opt.value();
  EXPECT_EQ2(f1, join<0>(Itv::bot(), zi(1)));
  EXPECT_EQ2(f2, join<1>(Itv::bot(), zd(2)));
  Itv itv2 = join(f1, f2);
  EXPECT_EQ2(itv2, Itv(1,2));
  EXPECT_EQ2(itv2.deinterpret(var_x), geq_1_leq_2);
}

TEST(CPTest, JoinMeetTest) {
  join_meet_generic_test(Itv::bot(), Itv::top());
  join_meet_generic_test(Itv(0,0), Itv(0,0));
  join_meet_generic_test(Itv(0,1), Itv(0,1));
  join_meet_generic_test(Itv(0,10), Itv(0,5));
  join_meet_generic_test(Itv(0,10), Itv(5,5));
  join_meet_generic_test(Itv(0,1), Itv(0,0));
  join_meet_generic_test(Itv(0,1), Itv(1,1));

  // Unordered intervals [1..2] and [2..3].
  Itv itv1 = Itv(1,2);
  Itv itv2 = Itv(2,3);
  Itv itv3 = Itv(2,2);
  Itv itv4 = Itv(1,3);
  join_one_test(itv1, itv2, itv3, true);
  meet_one_test(itv1, itv2, itv4, true);

  // Unordered intervals [1..2] and [3..4].
  Itv itv1_b = Itv(1,2);
  Itv itv2_b = Itv(3,4);
  Itv itv3_b = Itv(3,2);
  Itv itv4_b = Itv(1,4);
  join_one_test(itv1_b, itv2_b, itv3_b, true);
  meet_one_test(itv1_b, itv2_b, itv4_b, true);

  // Join/Meet a single component
  BInc b = BInc::bot();
  Itv itv5 = Itv(1,2);
  itv5.tell<0>(zi::top(), b);
  EXPECT_EQ2(itv5, Itv(zi::top(), zd(2)));
  itv5.tell<1>(zd::top(), b);
  EXPECT_EQ2(itv5, Itv::top());

  Itv itv6 = Itv(1,2);
  BInc has_changed = BInc::bot();
  itv6.dtell<0>(zi::bot(), has_changed);
  EXPECT_TRUE2(has_changed);
  EXPECT_EQ2(itv6, Itv(zi::bot(),zd(2)));
  BInc has_changed2 = BInc::bot();
  itv6.dtell<1>(zd::bot(), has_changed2);
  EXPECT_TRUE2(has_changed2);
  EXPECT_EQ2(itv6, Itv::bot());
}

TEST(CPTest, CPOrder) {
  generic_order_test(Itv(0,0));
  generic_order_test(Itv(0,1));

  Itv i1_ = join<0>(Itv::bot(), zi(1));
  Itv i_2 = join<1>(Itv::bot(), zd(2));
  Itv i1_2 = Itv(1,2);

  EXPECT_FALSE2(leq<Itv>(i1_2, i1_.dual()));     // [1..2] <= [1..]
  EXPECT_FALSE2(leq<Itv>(i1_2, i_2.dual()));     // [1..2] <= [..2]
  EXPECT_TRUE2(leq<Itv>(i1_, i1_2.dual()));      // [1..] <= [1..2]
  EXPECT_TRUE2(leq<Itv>(i_2, i1_2.dual()));      // [..2] <= [1..2]
  EXPECT_FALSE2(leq<Itv>(i1_, i_2.dual()));      // [1..] <= [..2]
  EXPECT_FALSE2(leq<Itv>(i_2, i1_.dual()));       // [..2] <= [1..]
}

TEST(CPTest, CPSplit) {
  auto split_top = split(Itv::top());
  auto split_bot = split(Itv::bot());
  auto split1 = split(Itv(1,1));
  auto split2 = split(Itv(1,2));
  EXPECT_EQ(split_top.size(), 0);
  EXPECT_EQ(split_bot.size(), 1);
  EXPECT_EQ(split1.size(), 1);
  EXPECT_EQ(split2.size(), 1);
}

TEST(CPTest, CPClone) {
  Itv bot_itv = Itv::bot();
  Itv top_itv = Itv::top();
  Itv itv2 = Itv(1,2);
  Itv itv3 = top_itv;
  EXPECT_TRUE2(bot_itv.clone().is_bot());
  EXPECT_TRUE2(top_itv.clone().is_top());
  EXPECT_EQ2(itv2.clone(), itv2);
}
