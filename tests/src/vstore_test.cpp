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

const static AType ty = 0;

// template <typename U>
// void check_project(VStore<U, StandardAllocator>& store, const LVar<StandardAllocator>& x, U v) {
//   auto avar_opt = store.environment().to_avar(x);
//   EXPECT_TRUE(avar_opt.has_value());
//   AVar avar = *avar_opt;
//   EXPECT_EQ(store.project(avar), v);
// }

// template <typename U, typename F>
// void tell_store(VStore<U, StandardAllocator>& store, F f, const LVar<StandardAllocator>& x, U v) {
//   auto cons = store.interpret(f);
//   EXPECT_TRUE(cons.has_value());
//   local::BInc has_changed = local::BInc::bot();
//   store.tell(*cons, has_changed);
//   EXPECT_TRUE(has_changed);
//   check_project(store, x, v);
// }

// void populate_zstore_10_vars(ZStore& store, int v) {
//   for(int i = 0; i < 10; ++i) {
//     LVar<StandardAllocator> x = "x ";
//     x[1] = '0' + i;
//     EXPECT_TRUE(store.interpret(F::make_exists(ty, x, Int)).has_value());
//     tell_store(store, make_v_op_z(x, GEQ, v), x, zi(v));
//   }
// }

// void populate_istore_10_vars(CPStore& store, int l, int u) {
//   for(int i = 0; i < 10; ++i) {
//     LVar<StandardAllocator> x = "x ";
//     x[1] = '0' + i;
//     EXPECT_TRUE(store.interpret(F::make_exists(ty, x, Int)).has_value());
//     tell_store(store, make_v_op_z(x, GEQ, l), x, CP(zi(l), zd::bot()));
//     tell_store(store, make_v_op_z(x, LEQ, u), x, CP(zi(l), zd(u)));
//   }
// }

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


// // // I. With integer variables and exact interpretation.

// template <typename A>
// void check_failed_interpret(const thrust::optional<typename A::TellType>& tell, const A& store) {
//   EXPECT_FALSE(tell.has_value());
//   EXPECT_EQ(store.vars(), 0);
//   EXPECT_TRUE(store.is_bot());
// }

template<class L>
void interpret_and_test(const char* fzn, const vector<typename L::universe_type>& expect) {
  L s = interpret_to2<L>(fzn);
  EXPECT_EQ(s.vars(), expect.size());
  for(int i = 0; i < expect.size(); ++i) {
    EXPECT_EQ(s[i], expect[i]);
  }
}

TEST(VStoreTest, InterpretationZStore) {
  must_error<ZStore>("constraint int_gt(x, 4);"); // (undeclared variable)
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4);", {zi(5)});
  must_error<ZStore>("constraint int_gt(x, 4); var int: x;"); // (declaration after usage)
  must_error<ZStore>("var int: x; constraint int_lt(x, 4);"); // (x < 4 not supported in increasing integers abstract universe).
}


// // ∃x /\ x > 4 /\ x > 5 should succeed.
// TEST(VStoreTest, Interpret5) {
//   ZStore zstore = ZStore::bot(ty);
//   F::Sequence conjunction(3);
//   conjunction[0] = F::make_exists(ty, var_x, Int);
//   conjunction[1] = make_v_op_z(var_x, GT, 4);
//   conjunction[2] = make_v_op_z(var_x, GT, 5);
//   auto f = F::make_nary(AND, conjunction, ty);
//   tell_store(zstore, f, var_x, zi(6));
//   EXPECT_EQ(zstore.vars(), 1);
// }

// // ∃x /\ x > 4 /\ x < 6 should succeed, with size of tell element == 1.
// TEST(VStoreTest, Interpret6) {
//   CPStore cpstore = CPStore::bot(ty);
//   F::Sequence conjunction(3);
//   conjunction[0] = F::make_exists(ty, var_x, Int);
//   conjunction[1] = make_v_op_z(var_x, GT, 4);
//   conjunction[2] = make_v_op_z(var_x, LT, 6);
//   auto f = F::make_nary(AND, conjunction, ty);
//   tell_store(cpstore, f, var_x, CP(zi(5), zd(5)));
//   EXPECT_EQ(cpstore.vars(), 1);
// }

// // ∃x /\ ∃y /\ x > 4 /\ x < 6 /\ y < 2 should succeed, with size of tell element == 2.
// TEST(VStoreTest, Interpret7) {
//   CPStore cpstore = CPStore::bot(ty);
//   F::Sequence conjunction(5);
//   conjunction[0] = F::make_exists(ty, var_x, Int);
//   conjunction[1] = F::make_exists(ty, var_y, Int);
//   conjunction[2] = make_v_op_z(var_x, GT, 4);
//   conjunction[3] = make_v_op_z(var_x, LT, 6);
//   conjunction[4] = make_v_op_z(var_y, LT, 2);
//   auto f = F::make_nary(AND, conjunction, ty);
//   tell_store(cpstore, f, var_x, CP(zi(5), zd(5)));
//   EXPECT_EQ(cpstore.vars(), 2);
//   check_project(cpstore, var_y, CP(zi::bot(), zd(1)));
// }

// // ∃x /\ x > 4 /\ x < 4 /\ y < 2 should fail (undeclared variable).
// TEST(VStoreTest, Interpret8) {
//   CPStore cpstore = CPStore::bot(ty);
//   F::Sequence conjunction(4);
//   conjunction[0] = F::make_exists(ty, var_x, Int);
//   conjunction[1] = make_v_op_z(var_x, GT, 4);
//   conjunction[2] = make_v_op_z(var_x, LT, 6);
//   conjunction[3] = make_v_op_z(var_y, LT, 2);
//   auto tell = cpstore.interpret(F::make_nary(AND, conjunction, ty));
//   check_failed_interpret(tell, cpstore);
// }

// // II. With integer variables and under- and over-approximations.

// // ∃x /\_o x ==_u 4 should succeed in ZStore, with size of tell element == 0 (the formula `x == 4` is not interpreted because the underlying abstract universe does not support under-approximation of ==).
// TEST(VStoreTest, Interpret9) {
//   ZStore zstore = ZStore::bot(ty);
//   auto exists_x = F::make_exists(ty, var_x, Int);
//   auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNTYPED, UNDER);
//   auto tell = zstore.interpret(F::make_binary(exists_x, AND, x_eq_4, ty, OVER));
//   EXPECT_TRUE(tell.has_value());
//   EXPECT_EQ(zstore.vars(), 1);
//   EXPECT_EQ((*tell).size(), 0);
// }

// // ∃x /\_o x ==_o 4 should succeed in ZStore, with size of tell element == 1.
// TEST(VStoreTest, Interpret10) {
//   ZStore zstore = ZStore::bot(ty);
//   auto exists_x = F::make_exists(ty, var_x, Int);
//   auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNTYPED, OVER);
//   auto f = F::make_binary(exists_x, AND, x_eq_4, ty, OVER);
//   tell_store(zstore, f, var_x, zi(4));
//   EXPECT_EQ(zstore.vars(), 1);
// }

// // ∃x /\ x ==_u 4 fail in CPStore because x ==_u 4 is not interpretable in zi, neither in zd.
// TEST(VStoreTest, Interpret11) {
//   CPStore cpstore = CPStore::bot(ty);
//   auto exists_x = F::make_exists(ty, var_x, Int);
//   auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNTYPED, UNDER);
//   auto tell = cpstore.interpret(F::make_binary(exists_x, AND, x_eq_4, ty, EXACT));
//   check_failed_interpret(tell, cpstore);
// }

// // ∃x /\ ∃y /\_o x ==_u 4 /\_o y > 1 /\_o y < 10 should succeed in ZStore, with size of tell element == 1 (the formula `x == 4` is not interpreted because the underlying abstract universe does not support under-approximation of ==).
// TEST(VStoreTest, Interpret12) {
//   ZStore zstore = ZStore::bot(ty);
//   F::Sequence conjunction(5);
//   conjunction[0] = F::make_exists(ty, var_x, Int);
//   conjunction[1] = F::make_exists(ty, var_y, Int);
//   conjunction[2] = make_v_op_z(var_x, EQ, 4, UNTYPED, UNDER);
//   conjunction[3] = make_v_op_z(var_y, GT, 1);
//   conjunction[4] = make_v_op_z(var_y, LT, 10);
//   auto f = F::make_nary(AND, conjunction, ty, OVER);
//   tell_store(zstore, f, var_y, zi(2));
//   EXPECT_EQ(zstore.vars(), 2);
// }

// // ∃x /\ ∃y /\_o x ==_u 4 /\_o y > 1 /\_o y < 10 should succeed in CPStore, with size of tell element == 1.
// TEST(VStoreTest, Interpret13) {
//   CPStore cpstore = CPStore::bot(ty);
//   F::Sequence conjunction(5);
//   conjunction[0] = F::make_exists(ty, var_x, Int);
//   conjunction[1] = F::make_exists(ty, var_y, Int);
//   conjunction[2] = make_v_op_z(var_x, EQ, 4, UNTYPED, UNDER);
//   conjunction[3] = make_v_op_z(var_y, GT, 1);
//   conjunction[4] = make_v_op_z(var_y, LT, 10);
//   auto f = F::make_nary(AND, conjunction, ty, OVER);
//   tell_store(cpstore, f, var_y, CP(zi(2), zd(9)));
//   EXPECT_EQ(cpstore.vars(), 2);
// }

// // III. With interval, to test `is_top` mostly.

// // ∃x /\ x > 4 /\ x < 4 should succeed, with size of tell element == 1 and equal to top.

// TEST(VStoreTest, Interpret14) {
//   IStore istore = IStore::bot(ty);
//   F::Sequence conjunction(3);
//   conjunction[0] = F::make_exists(ty, var_x, Int);
//   conjunction[1] = make_v_op_z(var_x, GT, 4);
//   conjunction[2] = make_v_op_z(var_x, LT, 4);
//   auto f = F::make_nary(AND, conjunction, ty);
//   tell_store(istore, f, var_x, Itv(zi(5), zd(3)));
//   EXPECT_EQ(istore.vars(), 1);
//   EXPECT_TRUE(istore.is_top());
//   EXPECT_FALSE2(istore.is_bot());
// }

// // // ∃x /\ ∃y /\ x > 4 /\ x < 4 /\ y < 2 should succeed, with size of tell element == 1 and equal to top.
// TEST(VStoreTest, Interpret15) {
//   IStore istore = IStore::bot(ty);
//   F::Sequence conjunction(5);
//   conjunction[0] = F::make_exists(ty, var_x, Int);
//   conjunction[1] = F::make_exists(ty, var_y, Int);
//   conjunction[2] = make_v_op_z(var_x, GT, 4);
//   conjunction[3] = make_v_op_z(var_x, LT, 4);
//   conjunction[4] = make_v_op_z(var_y, LT, 2);
//   auto f = F::make_nary(AND, conjunction, ty);
//   tell_store(istore, f, var_x, Itv(zi(5), zd(3)));
//   check_project(istore, var_y, Itv(zi::bot(), zd(1)));
//   EXPECT_EQ(istore.vars(), 2);
//   EXPECT_TRUE(istore.is_top());
//   EXPECT_FALSE2(istore.is_bot());
// }

// // // ∃x /\ x ==_u 4 should succeed in IStore, with size of tell element == 1.
// TEST(VStoreTest, Interpret16) {
//   IStore istore = IStore::bot(ty);
//   auto exists_x = F::make_exists(ty, var_x, Int);
//   auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNTYPED, UNDER);
//   auto f = F::make_binary(exists_x, AND, x_eq_4, ty);
//   tell_store(istore, f, var_x, Itv(zi(4), zd(4)));
//   EXPECT_EQ(istore.vars(), 1);
// }

// // // ∃x /\ x ==_o 4 should succeed in IStore, with size of tell element == 1.
// TEST(VStoreTest, Interpret17) {
//   IStore istore = IStore::bot(ty);
//   auto exists_x = F::make_exists(ty, var_x, Int);
//   auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNTYPED, OVER);
//   auto f = F::make_binary(exists_x, AND, x_eq_4, ty);
//   tell_store(istore, f, var_x, Itv(zi(4), zd(4)));
//   EXPECT_EQ(istore.vars(), 1);
// }

// // // ∃x /\ x !=_u 4 should succeed in IStore, with size of tell element == 1, also it should not be equal to top.
// TEST(VStoreTest, Interpret18) {
//   IStore istore = IStore::bot(ty);
//   auto exists_x = F::make_exists(ty, var_x, Int);
//   auto x_neq_4 = make_v_op_z(var_x, NEQ, 4, UNTYPED, UNDER);
//   auto f = F::make_binary(exists_x, AND, x_neq_4, ty);
//   tell_store(istore, f, var_x, Itv(zi(5), zd::bot()));
//   EXPECT_EQ(istore.vars(), 1);
//   EXPECT_FALSE2(istore.is_top());
// }
