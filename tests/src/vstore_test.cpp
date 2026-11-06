// Copyright 2021 Pierre Talbot

#include "lala/vstore.hpp"
#include "lala/zinterval.hpp"
#include "abstract_testing.hpp"

using zlb = LB<int>;
using zub = UB<int>;
using Itv = ZInterval<int>;
using ZStore = VStore<zlb, standard_allocator>;
using IStore = VStore<Itv, standard_allocator>;

/** Build a store of `doms.size()` variables. */
template <class Store, class U>
Store make_store(std::initializer_list<U> doms) {
  Store store(static_cast<int>(doms.size()), standard_allocator{});
  int i = 0;
  for(const U& u : doms) {
    store.embed(i++, u);
  }
  return std::move(store);
}

TEST(VStoreTest, BotTopTests) {
  ZStore one = make_store<ZStore, zlb>({zlb(1)});
  ZStore two = make_store<ZStore, zlb>({zlb(1), zlb(10)});
  IStore istore(two);
  bot_top_test(one);
  bot_top_test(two);
  bot_top_test(istore);
}

TEST(VStoreTest, JoinMeetTest) {
  ZStore one = make_store<ZStore, zlb>({zlb(1)});
  ZStore two = make_store<ZStore, zlb>({zlb(-1), zlb(10)});
  ZStore met = make_store<ZStore, zlb>({zlb(1), zlb(10)});
  ZStore joined = make_store<ZStore, zlb>({zlb(-1)});

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
  ZStore vstore = make_store<ZStore, zlb>({zlb(1), zlb(1)});
  ZStore copy(vstore, AbstractDeps<standard_allocator>(standard_allocator{}));
  EXPECT_EQ(vstore.vars(), copy.vars());
  for(int i = 0; i < vstore.vars(); ++i) {
    EXPECT_EQ(vstore[i], copy[i]);
  }
}

TEST(VStoreTest, SnapshotRestore) {
  ZStore vstore = make_store<ZStore, zlb>({zlb(1), zlb(1)});
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
  ZStore vstore = make_store<ZStore, zlb>({zlb(1), zlb(1)});
  ZStore copy(vstore, AbstractDeps<standard_allocator>(standard_allocator{}));
  copy.embed(0, zlb(2));
  copy.embed(1, zlb::bot());
  EXPECT_TRUE(vstore.is_extractable());
  vstore.extract(copy);
  for(int i = 0; i < 2; ++i) {
    EXPECT_EQ(copy[i], vstore[i]);
  }
}

TEST(VStoreTest, CopyAndAllocator) {
  IStore vstore(10, standard_allocator{});
  using stat_alloc = statistics_allocator<standard_allocator>;
  using IStore2 = VStore<Itv, stat_alloc>;
  IStore2 copy(vstore, AbstractDeps<stat_alloc>(stat_alloc{}));
}

/** A store does not need an environment nor an abstract type: it can be used as a plain array of lattice elements.
 * Here, we compute the minimum of an array of integers as the meet of upper bounds. */
TEST(VStoreTest, PlainArrayMin) {
  using UStore = VStore<zub, standard_allocator>;
  UStore store(5);
  EXPECT_EQ(store.vars(), 5);
  EXPECT_TRUE(store.is_top());
  int values[] = {7, 3, 9, 4, 8};
  for(int i = 0; i < 5; ++i) {
    EXPECT_TRUE(store.embed(i, zub(values[i])));
  }
  zub min = zub::top();
  for(int i = 0; i < store.vars(); ++i) {
    store.project(i, min);
  }
  EXPECT_EQ(min, zub(3));
}

TEST(VStoreTest, DeduceTell) {
  ZStore store = ZStore::top();
  EXPECT_EQ(store.vars(), 0);
  ZStore::tell_type<standard_allocator> tell;
  tell.push_back(ZStore::var_dom<standard_allocator>(2, zlb(5)));
  EXPECT_TRUE(store.deduce(tell));
  // The store grows to hold the element at index 2.
  EXPECT_EQ(store.vars(), 3);
  EXPECT_EQ(store[2], zlb(5));
  EXPECT_TRUE(store[0].is_top());
  EXPECT_TRUE(store.ask(tell));
  // The index `-1` represents `false`.
  ZStore::tell_type<standard_allocator> tell_false;
  tell_false.push_back(ZStore::var_dom<standard_allocator>());
  EXPECT_TRUE(store.deduce(tell_false));
  EXPECT_TRUE(store.is_bot());
}
