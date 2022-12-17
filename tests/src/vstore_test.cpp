// Copyright 2021 Pierre Talbot

#include "vstore.hpp"
#include "cartesian_product.hpp"
#include "interval.hpp"
#include "generic_universe_test.hpp"

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;
using CP = CartesianProduct<zi, zd>;
using ZStore = VStore<zi, StandardAllocator>;
using CPStore = VStore<CP, StandardAllocator>;
using IStore = VStore<Itv, StandardAllocator>;

TEST(VStoreTest, BotTopTests) {
  ZStore one = interpret_to2<ZStore>("var int: x; constraint int_ge(x, 1);");
  ZStore two = interpret_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  IStore istore = interpret_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  bot_top_test(one);
  bot_top_test(two);
  bot_top_test(istore);
}

TEST(VStoreTest, JoinMeetTest) {
  ZStore one = interpret_to2<ZStore>("var int: x; constraint int_ge(x, 1);");
  ZStore two = interpret_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, -1); constraint int_ge(y, 10);");
  ZStore joined = interpret_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  ZStore met = interpret_to2<ZStore>("var int: x; constraint int_ge(x, -1);");

  std::cout << one << "\n" << two << "\n" << joined << "\n" << met << std::endl;

  join_meet_generic_test(ZStore::bot(), ZStore::top());
  join_meet_generic_test(met, met);
  join_meet_generic_test(met, one);
  // tell is not commutative when stores have a different number of variables.
  join_meet_generic_test(met, two, true, false);
  join_meet_generic_test(one, joined, true, false);
  join_meet_generic_test(two, joined);
  join_meet_generic_test(joined, joined);
}

TEST(VStoreTest, CopyConstructor) {
  ZStore vstore = interpret_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
  ZStore copy(vstore, AbstractDeps<>());
  EXPECT_EQ(vstore.vars(), copy.vars());
  for(int i = 0; i < vstore.vars(); ++i) {
    EXPECT_EQ(vstore[i], copy[i]);
  }
}

TEST(VStoreTest, SnapshotRestore) {
  ZStore vstore = interpret_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
  ZStore::snapshot_type<> snap = vstore.snapshot();
  EXPECT_EQ(vstore[0], zi(1));
  EXPECT_EQ(vstore[1], zi(1));
  for(int j = 0; j < 3; ++j) {
    local::BInc has_changed = local::BInc::bot();
    vstore.tell(0, zi(2), has_changed);
    EXPECT_EQ(vstore[0], zi(2));
    EXPECT_TRUE(has_changed);
    vstore.restore(snap);
    EXPECT_EQ(vstore[0], zi(1));
  }
  // Test restore after reaching top.
  local::BInc has_changed = local::BInc::bot();
  EXPECT_FALSE(vstore.is_top());
  vstore.tell(1, zi::top(), has_changed);
  EXPECT_TRUE(vstore.is_top());
  EXPECT_EQ(vstore[1], zi::top());
  EXPECT_TRUE(has_changed);
  vstore.restore(snap);
  EXPECT_EQ(vstore[1], zi(1));
  EXPECT_FALSE(vstore.is_top());
}

TEST(VStoreTest, Extract) {
  ZStore vstore = interpret_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
  ZStore copy(vstore, AbstractDeps<>());
  local::BInc has_changed = local::BInc::bot();
  copy
    .tell(0, zi(2), has_changed)
    .tell(1, zi::top(), has_changed);
  EXPECT_TRUE(vstore.extract(copy));
  for(int i = 0; i < 2; ++i) {
    EXPECT_EQ(copy[i], vstore[i]);
  }
}

/** `appx` is the approximation kind of the top-level conjunction. */
template<class L>
L interpret_and_test(const char* fzn, const vector<typename L::universe_type>& expect, Approx appx = EXACT) {
  L s = interpret_to2<L>(fzn, appx);
  EXPECT_EQ(s.vars(), expect.size());
  for(int i = 0; i < expect.size(); ++i) {
    EXPECT_EQ(s[i], expect[i]);
  }
  return std::move(s);
}

TEST(VStoreTest, InterpretationZStore) {
  must_error<ZStore>("constraint int_gt(x, 4);"); // (undeclared variable)
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4);", {zi(5)});
  must_error<ZStore>("constraint int_gt(x, 4); var int: x;"); // (declaration after usage)
  must_error<ZStore>("var int: x; constraint int_lt(x, 4);"); // (x < 4 not supported in increasing integers abstract universe).
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4); constraint int_gt(x, 5);", {zi(6)});
}

TEST(VStoreTest, InterpretationCPStore) {
  interpret_and_test<CPStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 6);", {CP(zi(5), zd(5))});
  interpret_and_test<CPStore>("var int: x; var int: y; constraint int_gt(x, 4); constraint int_lt(x, 6); constraint int_le(y, 1);", {CP(zi(5), zd(5)), CP(zi::bot(), zd(1))});
  must_error<CPStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 6); constraint int_le(y, 1);");
}

TEST(VStoreTest, InterpretationIStore) {
  IStore s1 = interpret_and_test<IStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 4);", {Itv(5, 3)});
  EXPECT_TRUE(s1.is_top());
  interpret_and_test<IStore>("var int: x; constraint int_ge(x, 4); constraint int_le(x, 4);", {Itv(4, 4)});
  IStore s2 = interpret_and_test<IStore>("var int: x; var int: y; constraint int_gt(x, 4); constraint int_lt(x, 4); constraint int_lt(y, 2);", {Itv(5, 3), Itv(zi::bot(), zd(1))});
  EXPECT_TRUE(s2.is_top());
  interpret_and_test<IStore>("var int: x; constraint int_eq(x, 4);", {Itv(4, 4)});
  must_error<IStore>("var int: x; constraint int_eq(x, 4)::under;");
  interpret_and_test<IStore>("var int: x; constraint int_eq(x, 4)::under;", {Itv::bot()}, OVER);
  interpret_and_test<IStore>("var int: x; constraint int_eq(x, 4)::over;", {Itv(4, 4)});
  IStore s3 = interpret_and_test<IStore>("var int: x; constraint int_ne(x, 4)::under;", {Itv(zi(5), zd::bot())});
  EXPECT_FALSE(s3.is_top());
}

TEST(VStoreTest, UnderOverInterpretation) {
  interpret_and_test<ZStore>("var int: x; constraint int_eq(x, 4)::under;", {zi::bot()}, OVER);
  interpret_and_test<ZStore>("var int: x; constraint int_eq(x, 4)::over;", {zi(4)}, OVER);
  interpret_and_test<ZStore>("var int: x; var int: y; constraint int_eq(x, 4)::under; constraint int_gt(y, 1); constraint int_lt(y, 10);", {zi::bot(), zi(2)}, OVER);
  interpret_and_test<ZStore>("var int: x; var int: y; constraint int_eq(x, 4)::over; constraint int_gt(y, 1); constraint int_lt(y, 10);", {zi(4), zi(2)}, OVER);

  interpret_and_test<CPStore>("var int: x; var int: y; constraint int_eq(x, 4)::under; constraint int_gt(y, 1); constraint int_lt(y, 10);", {CP::bot(), CP(zi(2), zd(9))}, OVER);
  must_error<CPStore>("var int: x; constraint int_eq(x, 4)::under;");
}
