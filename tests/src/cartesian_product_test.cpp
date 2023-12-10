// Copyright 2021 Pierre Talbot

#include "abstract_testing.hpp"
#include "lala/cartesian_product.hpp"

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = CartesianProduct<zi, zd>;

TEST(CPTest, BotTopTests) {
  Itv itv1_2(zi(1), zd(2));
  bot_top_test(itv1_2);
}

TEST(CPTest, CPOrder) {
  Itv i1_ = Itv(zi(1), zd::bot());
  Itv i_2 = Itv(zi::bot(), zd(2));
  Itv i1_2 = Itv(1,2);

  EXPECT_FALSE(i1_2 <= i1_);     // [1..2] <= [1..]
  EXPECT_FALSE(i1_2 <= i_2);     // [1..2] <= [..2]
  EXPECT_TRUE(i1_ <= i1_2);      // [1..] <= [1..2]
  EXPECT_TRUE(i_2 <= i1_2);      // [..2] <= [1..2]
  EXPECT_FALSE(i1_ <= i_2);      // [1..] <= [..2]
  EXPECT_FALSE(i_2 <= i1_);      // [..2] <= [1..]
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
  local::BInc b = local::BInc::bot();
  Itv itv5 = Itv(1,2);
  itv5.tell<0>(zi::top(), b);
  EXPECT_EQ(itv5, Itv(zi::top(), zd(2)));
  itv5.tell<1>(zd::top(), b);
  EXPECT_EQ(itv5, Itv::top());

  Itv itv6 = Itv(1,2);
  local::BInc has_changed = local::BInc::bot();
  itv6.dtell<0>(zi::bot(), has_changed);
  EXPECT_TRUE(has_changed);
  EXPECT_EQ(itv6, Itv(zi::bot(),zd(2)));
  local::BInc has_changed2 = local::BInc::bot();
  itv6.dtell<1>(zd::bot(), has_changed2);
  EXPECT_TRUE(has_changed2);
  EXPECT_EQ(itv6, Itv::bot());
}

TEST(CPTest, NoInterpret) {
  interpret_must_error<IKind::TELL, Itv>("constraint int_ne(x, 10);", env_with_x());
  interpret_must_error<IKind::ASK, Itv>("constraint int_eq(x, 10);", env_with_x());
}

TEST(CPTest, ValidInterpret) {
  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_ge(x, 10);", Itv(zi(10), zd::bot()), env, false);
  expect_both_interpret_equal_to("constraint int_gt(x, 10);", Itv(zi(11), zd::bot()), env, false);
  expect_both_interpret_equal_to("constraint int_le(x, 10);", Itv(zi::bot(), zd(10)), env, false);
  expect_both_interpret_equal_to("constraint int_lt(x, 10);", Itv(zi::bot(), zd(9)), env, false);
  expect_both_interpret_equal_to("constraint int_ge(x, 10);\
                          constraint int_le(x, 20);", Itv(zi(10), zd(20)), env, false);
  expect_both_interpret_equal_to("constraint int_ge(x, 10);\
                          constraint int_le(x, 20);\
                          constraint int_le(x, 15);\
                          constraint int_ge(x, 5);", Itv(zi(10), zd(15)), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 10);", Itv(zi(10), zd(10)), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 10);", Itv(zi(11), zd(9)), env, false);
}
