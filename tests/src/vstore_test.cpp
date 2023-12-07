// Copyright 2021 Pierre Talbot

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "abstract_testing.hpp"

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;
using CP = CartesianProduct<zi, zd>;
using ZStore = VStore<zi, standard_allocator>;
using CPStore = VStore<CP, standard_allocator>;
using IStore = VStore<Itv, standard_allocator>;

TEST(VStoreTest, BotTopTests) {
  ZStore one = interpret_tell_to2<ZStore>("var int: x; constraint int_ge(x, 1);");
  ZStore two = interpret_tell_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  IStore istore = interpret_tell_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  bot_top_test(one);
  bot_top_test(two);
  bot_top_test(istore);
}

TEST(VStoreTest, JoinMeetTest) {
  ZStore one = interpret_tell_to2<ZStore>("var int: x; constraint int_ge(x, 1);");
  ZStore two = interpret_tell_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, -1); constraint int_ge(y, 10);");
  ZStore joined = interpret_tell_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  ZStore met = interpret_tell_to2<ZStore>("var int: x; constraint int_ge(x, -1);");

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
  ZStore vstore = interpret_tell_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
  ZStore copy(vstore, AbstractDeps<standard_allocator>(standard_allocator{}));
  EXPECT_EQ(vstore.vars(), copy.vars());
  for(int i = 0; i < vstore.vars(); ++i) {
    EXPECT_EQ(vstore[i], copy[i]);
  }
}

TEST(VStoreTest, SnapshotRestore) {
  ZStore vstore = interpret_tell_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
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
  ZStore vstore = interpret_tell_to2<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
  ZStore copy(vstore, AbstractDeps<standard_allocator>(standard_allocator{}));
  local::BInc has_changed = local::BInc::bot();
  copy
    .tell(0, zi(2), has_changed)
    .tell(1, zi::top(), has_changed);
  EXPECT_TRUE(vstore.is_extractable());
  vstore.extract(copy);
  for(int i = 0; i < 2; ++i) {
    EXPECT_EQ(copy[i], vstore[i]);
  }
}

template<class L>
L interpret_and_test(const char* fzn, const vector<typename L::universe_type>& expect) {
  L s = interpret_tell_to2<L>(fzn);
  EXPECT_EQ(s.vars(), expect.size());
  for(int i = 0; i < expect.size(); ++i) {
    EXPECT_EQ(s[i], expect[i]);
  }
  return std::move(s);
}

TEST(VStoreTest, InterpretationZStore) {
  must_error_tell<ZStore>("constraint int_gt(x, 4);"); // (undeclared variable)
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4);", {zi(5)});
  must_error_tell<ZStore>("constraint int_gt(x, 4); var int: x;"); // (declaration after usage)
  must_error_tell<ZStore>("var int: x; constraint int_lt(x, 4);"); // (x < 4 not supported in increasing integers abstract universe).
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4); constraint int_gt(x, 5);", {zi(6)});
}

TEST(VStoreTest, InterpretationCPStore) {
  interpret_and_test<CPStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 6);", {CP(zi(5), zd(5))});
  interpret_and_test<CPStore>("var int: x; var int: y; constraint int_gt(x, 4); constraint int_lt(x, 6); constraint int_le(y, 1);", {CP(zi(5), zd(5)), CP(zi::bot(), zd(1))});
  must_error_tell<CPStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 6); constraint int_le(y, 1);");
}

TEST(VStoreTest, InterpretationIStore) {
  IStore s1 = interpret_and_test<IStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 4);", {Itv(5, 3)});
  EXPECT_TRUE(s1.is_top());
  IStore s2 = interpret_and_test<IStore>("var int: x; var int: y; constraint int_gt(x, 4); constraint int_lt(x, 4); constraint int_lt(y, 2);", {Itv(5, 3), Itv(zi::bot(), zd(1))});
  EXPECT_TRUE(s2.is_top());
  IStore s3 = interpret_and_test<IStore>("var int: x; constraint int_ge(x, 4); constraint int_le(x, 4);", {Itv(4, 4)});
  interpret_and_test<IStore>("var int: x; constraint int_eq(x, 4);", {Itv(4, 4)});
  IStore s4 = interpret_and_test<IStore>("var 1..10: x;", {Itv(1, 10)});
  interpret_and_test<IStore>("var 5..10: x; var -5..5: y;", {Itv(5, 10), Itv(-5, 5)});
  interpret_and_ask(s1, "constraint int_eq(x, 4);", true);
  interpret_and_ask(s2, "constraint int_eq(x, 4);", true);
  interpret_and_ask(s3, "constraint int_eq(x, 4);", true);
  interpret_and_ask(s4, "constraint int_eq(x, 4);", false);
  interpret_and_ask(s1, "constraint int_ne(x, 4);", true);
  interpret_and_ask(s2, "constraint int_ne(x, 4);", true);
  interpret_and_ask(s3, "constraint int_ne(x, 4);", false);
  interpret_and_ask(s4, "constraint int_ne(x, 4);", false);
}

TEST(VStoreTest, AskOperation) {
  VarEnv<standard_allocator> env;
  ZStore store = interpret_tell_to<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);", env);
  IDiagnostics<F> diagnostics;
  ZStore::ask_type<standard_allocator> ask1, ask2, ask3, ask4, ask5;
  EXPECT_TRUE(store.interpret_ask_in(*parse_flatzinc_str<standard_allocator>("constraint int_ge(x, 0); constraint int_ge(y, 1);"), env, ask1, diagnostics));
  EXPECT_TRUE(store.interpret_ask_in(*parse_flatzinc_str<standard_allocator>("constraint int_ge(y, -1);"), env, ask2, diagnostics));
  EXPECT_TRUE(store.interpret_ask_in(*parse_flatzinc_str<standard_allocator>("constraint int_ge(x, 0); constraint int_ge(y, 2);"), env, ask3, diagnostics));
  EXPECT_TRUE(store.interpret_ask_in(*parse_flatzinc_str<standard_allocator>("constraint int_ge(x, 10); constraint int_ge(y, 2);"), env, ask4, diagnostics));
  EXPECT_TRUE(store.interpret_ask_in(*parse_flatzinc_str<standard_allocator>("constraint int_ge(x, 10);"), env, ask5, diagnostics));
  EXPECT_TRUE(store.ask(ask1));
  EXPECT_TRUE(store.ask(ask2));
  EXPECT_FALSE(store.ask(ask3));
  EXPECT_FALSE(store.ask(ask4));
  EXPECT_FALSE(store.ask(ask5));
}

TEST(VStoreTest, CopyAndAllocator) {
  IStore vstore = interpret_tell_to2<IStore>("array[1..10] of var int: x;");
  using stat_alloc = statistics_allocator<standard_allocator>;
  using IStore2 = VStore<Itv, stat_alloc>;
  IStore2 copy(vstore, AbstractDeps<stat_alloc>(stat_alloc{}));
}

TEST(VStoreTest, Idempotence) {
  check_interpret_idempotence<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  check_interpret_idempotence<CPStore>("var int: x; var int: y; constraint int_gt(x, 4); constraint int_lt(x, 6); constraint int_le(y, 1);");
  check_interpret_idempotence<IStore>("array[1..10] of var int: x;");
  check_interpret_idempotence<IStore>("array[1..10] of var 1..10: x;");
}
