// Copyright 2021 Pierre Talbot

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "abstract_testing.hpp"

using zlb = local::ZLB;
using zub = local::ZUB;
using Itv = Interval<zlb>;
using CP = CartesianProduct<zlb, zub>;
using ZStore = VStore<zlb, standard_allocator>;
using CPStore = VStore<CP, standard_allocator>;
using IStore = VStore<Itv, standard_allocator>;

TEST(VStoreTest, BotTopTests) {
  ZStore one = create_and_interpret_and_tell<ZStore>("var int: x; constraint int_ge(x, 1);");
  ZStore two = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  IStore istore = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  bot_top_test(one);
  bot_top_test(two);
  bot_top_test(istore);
}

TEST(VStoreTest, JoinMeetTest) {
  ZStore one = create_and_interpret_and_tell<ZStore>("var int: x; constraint int_ge(x, 1);");
  ZStore two = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, -1); constraint int_ge(y, 10);");
  ZStore met = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  ZStore joined = create_and_interpret_and_tell<ZStore>("var int: x; constraint int_ge(x, -1);");

  std::cout << one << "\n" << two << "\n" << joined << "\n" << met << std::endl;

  join_meet_generic_test(ZStore::bot(), ZStore::top());
  join_meet_generic_test(met, met);
  join_meet_generic_test(met, two);
  join_meet_generic_test(one, joined);
  join_meet_generic_test(joined, joined);
  // join and meet are not commutative when stores have a different number of variables.
  join_meet_generic_test(met, one, false, false);
  join_meet_generic_test(two, joined, false, false);
}

TEST(VStoreTest, CopyConstructor) {
  ZStore vstore = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
  ZStore copy(vstore, AbstractDeps<standard_allocator>(standard_allocator{}));
  EXPECT_EQ(vstore.vars(), copy.vars());
  for(int i = 0; i < vstore.vars(); ++i) {
    EXPECT_EQ(vstore[i], copy[i]);
  }
}

TEST(VStoreTest, SnapshotRestore) {
  ZStore vstore = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
  ZStore::snapshot_type<> snap = vstore.snapshot();
  EXPECT_EQ(vstore[0], zlb(1));
  EXPECT_EQ(vstore[1], zlb(1));
  for(int j = 0; j < 3; ++j) {
    EXPECT_TRUE(vstore.embed(0, zlb(2)));
    EXPECT_EQ(vstore[0], zlb(2));
    vstore.restore(snap);
    EXPECT_EQ(vstore[0], zlb(1));
  }
  // Test restore after reaching bot.
  EXPECT_FALSE(vstore.is_bot());
  EXPECT_TRUE(vstore.embed(1, zlb::bot()));
  EXPECT_TRUE(vstore.is_bot());
  EXPECT_EQ(vstore[1], zlb::bot());
  vstore.restore(snap);
  EXPECT_EQ(vstore[1], zlb(1));
  EXPECT_FALSE(vstore.is_bot());
}

TEST(VStoreTest, Extract) {
  ZStore vstore = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);");
  ZStore copy(vstore, AbstractDeps<standard_allocator>(standard_allocator{}));
  copy.embed(0, zlb(2));
  copy.embed(1, zlb::bot());
  EXPECT_TRUE(vstore.is_extractable());
  vstore.extract(copy);
  for(int i = 0; i < 2; ++i) {
    EXPECT_EQ(copy[i], vstore[i]);
  }
}

template<class L>
L interpret_and_test(const char* fzn, const vector<typename L::universe_type>& expect) {
  L s = create_and_interpret_and_tell<L>(fzn);
  EXPECT_EQ(s.vars(), expect.size());
  for(int i = 0; i < expect.size(); ++i) {
    EXPECT_EQ(s[i], expect[i]);
  }
  return std::move(s);
}

TEST(VStoreTest, InterpretationZStore) {
  interpret_must_error<IKind::TELL, ZStore>("constraint int_gt(x, 4);"); // (undeclared variable)
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4);", {zlb(5)});
  interpret_must_error<IKind::TELL, ZStore>("constraint int_gt(x, 4); var int: x;"); // (declaration after usage)
  interpret_must_error<IKind::TELL, ZStore>("var int: x; constraint int_lt(x, 4);"); // (x < 4 not supported in increasing integers abstract universe).
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4); constraint int_gt(x, 5);", {zlb(6)});
}

TEST(VStoreTest, InterpretationCPStore) {
  interpret_and_test<CPStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 6);", {CP(zlb(5), zub(5))});
  interpret_and_test<CPStore>("var int: x; var int: y; constraint int_gt(x, 4); constraint int_lt(x, 6); constraint int_le(y, 1);", {CP(zlb(5), zub(5)), CP(zlb::top(), zub(1))});
  interpret_must_error<IKind::TELL, CPStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 6); constraint int_le(y, 1);");
}

TEST(VStoreTest, InterpretationIStore) {
  IStore s1 = interpret_and_test<IStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 4);", {Itv(5, 3)});
  EXPECT_TRUE(s1.is_bot());
  IStore s2 = interpret_and_test<IStore>("var int: x; var int: y; constraint int_gt(x, 4); constraint int_lt(x, 4); constraint int_lt(y, 2);", {Itv::bot(), Itv(zlb::top(), zub(1))});
  EXPECT_TRUE(s2.is_bot());
  IStore s3 = interpret_and_test<IStore>("var int: x; constraint int_ge(x, 4); constraint int_le(x, 4);", {Itv(4, 4)});
  interpret_and_test<IStore>("var int: x; constraint int_eq(x, 4);", {Itv(4, 4)});
  IStore s4 = interpret_and_test<IStore>("var 1..10: x;", {Itv(1, 10)});
  interpret_and_test<IStore>("var 5..10: x; var -5..5: y;", {Itv(5, 10), Itv(-5, 5)});
  VarEnv<standard_allocator> env = env_with("var int: x :: abstract(0); var int: y :: abstract(0);");
  EXPECT_TRUE(interpret_and_ask("constraint int_eq(x, 4);", s1, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_eq(x, 4);", s2, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_eq(x, 4);", s3, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_eq(x, 4);", s4, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_ne(x, 4);", s1, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_ne(x, 4);", s2, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ne(x, 4);", s3, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ne(x, 4);", s4, env));
}

TEST(VStoreTest, AskOperation) {
  VarEnv<standard_allocator> env;
  ZStore store = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);", env);
  EXPECT_TRUE(interpret_and_ask("constraint int_ge(x, 0); constraint int_ge(y, 1);", store, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_ge(y, -1);", store, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ge(x, 0); constraint int_ge(y, 2);", store, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ge(x, 10); constraint int_ge(y, 2);", store, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ge(x, 10);", store, env));
}

TEST(VStoreTest, AskOperationInfiniteDom) {
  VarEnv<standard_allocator> env;
  IStore store = create_and_interpret_and_tell<IStore>("var int: x;", env);
  EXPECT_FALSE(interpret_and_ask("constraint int_le(x, 5);", store, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_gt(x, 5);", store, env));
}

TEST(VStoreTest, CopyAndAllocator) {
  IStore vstore = create_and_interpret_and_tell<IStore>("array[1..10] of var int: x;");
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
