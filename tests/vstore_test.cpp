// Copyright 2021 Pierre Talbot

#include "z.hpp"
#include "cartesian_product.hpp"
#include "interval.hpp"
#include "vstore.hpp"
#include "generic_universe_test.hpp"

using zi = ZInc<int>;
using zd = ZDec<int>;
using CP = CartesianProduct<zi, zd>;
using ZStore = VStore<zi, StandardAllocator>;
using CPStore = VStore<CP, StandardAllocator>;

const static AType ty = 0;

template <typename U>
void check_project(VStore<U, StandardAllocator>& store, const LVar<StandardAllocator>& x, U v) {
  auto avar_opt = store.environment().to_avar(x);
  EXPECT_TRUE(avar_opt.has_value());
  AVar avar = *avar_opt;
  EXPECT_EQ2(store.project(avar), v);
}

template <typename U, typename F>
void tell_store(VStore<U, StandardAllocator>& store, F f, const LVar<StandardAllocator>& x, U v) {
  auto cons = store.interpret(f);
  EXPECT_TRUE(cons.has_value());
  BInc has_changed = BInc::bot();
  store.tell(*cons, has_changed);
  EXPECT_TRUE2(has_changed);
  check_project(store, x, v);
}

void populate_zstore_10_vars(ZStore& store, int v) {
  for(int i = 0; i < 10; ++i) {
    LVar<StandardAllocator> x = "x ";
    x[1] = '0' + i;
    EXPECT_TRUE(store.interpret(F::make_exists(ty, x, Int)).has_value());
    tell_store(store, make_v_op_z(x, GEQ, v), x, zi(v));
  }
}

void populate_istore_10_vars(CPStore& store, int l, int u) {
  for(int i = 0; i < 10; ++i) {
    LVar<StandardAllocator> x = "x ";
    x[1] = '0' + i;
    EXPECT_TRUE(store.interpret(F::make_exists(ty, x, Int)).has_value());
    tell_store(store, make_v_op_z(x, GEQ, l), x, CP(zi(l), zd::bot()));
    tell_store(store, make_v_op_z(x, LEQ, u), x, CP(zi(l), zd(u)));
  }
}

TEST(VStoreTest, TopBot) {
  ZStore ones(ty, 10);
  populate_zstore_10_vars(ones, 1);
  EXPECT_EQ2(ones.vars(), 10);
  CPStore bools(ty, 10);
  populate_istore_10_vars(bools, 0, 1);
  EXPECT_EQ2(bools.vars(), 10);
  // bot_top_test<ZStore>(ones);
  // bot_top_test<CPStore>(bools);
}

TEST(VStoreTest, CopyConstructor) {
  ZStore ones(ty, 10);
  populate_zstore_10_vars(ones, 1);
  ZStore copy(ones, AbstractDeps<>());
  EXPECT_EQ2(ones.vars(), copy.vars());
  for(int i = 0; i < ones.vars().value(); ++i) {
    auto av = make_var(ty, i);
    EXPECT_EQ2(ones.project(av), copy.project(av));
  }
}

// // I. With integer variables and exact interpretation.

template <typename A>
void check_failed_interpret(const thrust::optional<typename A::TellType>& tell, const A& store) {
  EXPECT_FALSE(tell.has_value());
  EXPECT_EQ2(store.vars(), 0);
  EXPECT_TRUE2(store.is_bot());
}

// x > 4 should fail (undeclared variable)
TEST(VStoreTest, Interpret1) {
  ZStore zstore = ZStore::bot(ty);
  auto x_gt_4 = make_v_op_z(var_x, GT, 4);
  auto tell = zstore.interpret(x_gt_4);
  check_failed_interpret(tell, zstore);
}

// ∃x /\ x > 4 should succeed.
TEST(VStoreTest, Interpret2) {
  ZStore zstore = ZStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_gt_4 = make_v_op_z(var_x, GT, 4);
  auto f = F::make_binary(exists_x, AND, x_gt_4, ty);
  tell_store(zstore, f, var_x, zi(5));
  EXPECT_EQ2(zstore.vars(), 1);
}

// x > 4 /\ ∃x should fail.
TEST(VStoreTest, Interpret3) {
  ZStore zstore = ZStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_gt_4 = make_v_op_z(var_x, GT, 4);
  auto f = F::make_binary(x_gt_4, AND, exists_x, ty);
  auto tell = zstore.interpret(f);
  check_failed_interpret(tell, zstore);
}

// ∃x /\ x < 4 should fail (x < 4 not supported in increasing integers abstract universe).
TEST(VStoreTest, Interpret4) {
  ZStore zstore = ZStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_lt_4 = make_v_op_z(var_x, LT, 4);
  auto tell = zstore.interpret(F::make_binary(exists_x, AND, x_lt_4, ty));
  check_failed_interpret(tell, zstore);
}

// ∃x /\ x > 4 /\ x > 5 should succeed.
TEST(VStoreTest, Interpret5) {
  ZStore zstore = ZStore::bot(ty);
  F::Sequence conjunction(3);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = make_v_op_z(var_x, GT, 4);
  conjunction[2] = make_v_op_z(var_x, GT, 5);
  auto f = F::make_nary(AND, conjunction, ty);
  tell_store(zstore, f, var_x, zi(6));
  EXPECT_EQ2(zstore.vars(), 1);
}

// ∃x /\ x > 4 /\ x < 6 should succeed, with size of tell element == 1.
TEST(VStoreTest, Interpret6) {
  CPStore cpstore = CPStore::bot(ty);
  F::Sequence conjunction(3);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = make_v_op_z(var_x, GT, 4);
  conjunction[2] = make_v_op_z(var_x, LT, 6);
  auto f = F::make_nary(AND, conjunction, ty);
  tell_store(cpstore, f, var_x, CP(zi(5), zd(5)));
  EXPECT_EQ2(cpstore.vars(), 1);
}

// ∃x /\ ∃y /\ x > 4 /\ x < 6 /\ y < 2 should succeed, with size of tell element == 2.
TEST(VStoreTest, Interpret7) {
  CPStore cpstore = CPStore::bot(ty);
  F::Sequence conjunction(5);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = F::make_exists(ty, var_y, Int);
  conjunction[2] = make_v_op_z(var_x, GT, 4);
  conjunction[3] = make_v_op_z(var_x, LT, 6);
  conjunction[4] = make_v_op_z(var_y, LT, 2);
  auto f = F::make_nary(AND, conjunction, ty);
  tell_store(cpstore, f, var_x, CP(zi(5), zd(5)));
  EXPECT_EQ2(cpstore.vars(), 2);
  check_project(cpstore, var_y, CP(zi::bot(), zd(1)));
}

// ∃x /\ x > 4 /\ x < 4 /\ y < 2 should fail (undeclared variable).
TEST(VStoreTest, Interpret8) {
  CPStore cpstore = CPStore::bot(ty);
  F::Sequence conjunction(4);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = make_v_op_z(var_x, GT, 4);
  conjunction[2] = make_v_op_z(var_x, LT, 6);
  conjunction[3] = make_v_op_z(var_y, LT, 2);
  auto tell = cpstore.interpret(F::make_nary(AND, conjunction, ty));
  check_failed_interpret(tell, cpstore);
}

// II. With integer variables and under- and over-approximations.

// ∃x /\_o x ==_u 4 should succeed in ZStore, with size of tell element == 0 (the formula `x == 4` is not interpreted because the underlying abstract universe does not support under-approximation of ==).
TEST(VStoreTest, Interpret9) {
  ZStore zstore = ZStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNDER);
  auto tell = zstore.interpret(F::make_binary(exists_x, AND, x_eq_4, ty, OVER));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ2(zstore.vars(), 1);
  EXPECT_EQ2((*tell).size(), 0);
}

// ∃x /\_o x ==_o 4 should succeed in ZStore, with size of tell element == 1.
TEST(VStoreTest, Interpret10) {
  ZStore zstore = ZStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_eq_4 = make_v_op_z(var_x, EQ, 4, OVER);
  auto f = F::make_binary(exists_x, AND, x_eq_4, ty, OVER);
  tell_store(zstore, f, var_x, zi(4));
  EXPECT_EQ2(zstore.vars(), 1);
}

// ∃x /\ x ==_u 4 fail in CPStore because x ==_u 4 is not interpretable in zi, neither in zd.
TEST(VStoreTest, Interpret11) {
  CPStore cpstore = CPStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNDER);
  auto tell = cpstore.interpret(F::make_binary(exists_x, AND, x_eq_4, ty, EXACT));
  check_failed_interpret(tell, cpstore);
}

// ∃x /\ ∃y /\_o x ==_u 4 /\_o y > 1 /\_o y < 10 should succeed in ZStore, with size of tell element == 1 (the formula `x == 4` is not interpreted because the underlying abstract universe does not support under-approximation of ==).
TEST(VStoreTest, Interpret12) {
  ZStore zstore = ZStore::bot(ty);
  F::Sequence conjunction(5);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = F::make_exists(ty, var_y, Int);
  conjunction[2] = make_v_op_z(var_x, EQ, 4, UNDER);
  conjunction[3] = make_v_op_z(var_y, GT, 1);
  conjunction[4] = make_v_op_z(var_y, LT, 10);
  auto f = F::make_nary(AND, conjunction, ty, OVER);
  tell_store(zstore, f, var_y, zi(2));
  EXPECT_EQ2(zstore.vars(), 2);
}

// ∃x /\ ∃y /\_o x ==_u 4 /\_o y > 1 /\_o y < 10 should succeed in CPStore, with size of tell element == 1.
TEST(VStoreTest, Interpret13) {
  CPStore cpstore = CPStore::bot(ty);
  F::Sequence conjunction(5);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = F::make_exists(ty, var_y, Int);
  conjunction[2] = make_v_op_z(var_x, EQ, 4, UNDER);
  conjunction[3] = make_v_op_z(var_y, GT, 1);
  conjunction[4] = make_v_op_z(var_y, LT, 10);
  auto f = F::make_nary(AND, conjunction, ty, OVER);
  tell_store(cpstore, f, var_y, CP(zi(2), zd(9)));
  EXPECT_EQ2(cpstore.vars(), 2);
}

// III. With interval, to test `is_top` mostly.

using Itv = Interval<zi>;
using IStore = VStore<Itv, StandardAllocator>;

// ∃x /\ x > 4 /\ x < 4 should succeed, with size of tell element == 1 and equal to top.

TEST(VStoreTest, Interpret14) {
  IStore istore = IStore::bot(ty);
  F::Sequence conjunction(3);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = make_v_op_z(var_x, GT, 4);
  conjunction[2] = make_v_op_z(var_x, LT, 4);
  auto f = F::make_nary(AND, conjunction, ty);
  tell_store(istore, f, var_x, Itv(zi(5), zd(3)));
  EXPECT_EQ2(istore.vars(), 1);
  EXPECT_TRUE2(istore.is_top());
  EXPECT_FALSE2(istore.is_bot());
}

// // ∃x /\ ∃y /\ x > 4 /\ x < 4 /\ y < 2 should succeed, with size of tell element == 1 and equal to top.
TEST(VStoreTest, Interpret15) {
  IStore istore = IStore::bot(ty);
  F::Sequence conjunction(5);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = F::make_exists(ty, var_y, Int);
  conjunction[2] = make_v_op_z(var_x, GT, 4);
  conjunction[3] = make_v_op_z(var_x, LT, 4);
  conjunction[4] = make_v_op_z(var_y, LT, 2);
  auto f = F::make_nary(AND, conjunction, ty);
  tell_store(istore, f, var_x, Itv(zi(5), zd(3)));
  check_project(istore, var_y, Itv(zi::bot(), zd(1)));
  EXPECT_EQ2(istore.vars(), 2);
  EXPECT_TRUE2(istore.is_top());
  EXPECT_FALSE2(istore.is_bot());
}

// // ∃x /\ x ==_u 4 should succeed in IStore, with size of tell element == 1.
TEST(VStoreTest, Interpret16) {
  IStore istore = IStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNDER);
  auto f = F::make_binary(exists_x, AND, x_eq_4, ty);
  tell_store(istore, f, var_x, Itv(zi(4), zd(4)));
  EXPECT_EQ2(istore.vars(), 1);
}

// // ∃x /\ x ==_o 4 should succeed in IStore, with size of tell element == 1.
TEST(VStoreTest, Interpret17) {
  IStore istore = IStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_eq_4 = make_v_op_z(var_x, EQ, 4, OVER);
  auto f = F::make_binary(exists_x, AND, x_eq_4, ty);
  tell_store(istore, f, var_x, Itv(zi(4), zd(4)));
  EXPECT_EQ2(istore.vars(), 1);
}

// // ∃x /\ x !=_u 4 should succeed in IStore, with size of tell element == 1, also it should not be equal to top.
TEST(VStoreTest, Interpret18) {
  IStore istore = IStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_neq_4 = make_v_op_z(var_x, NEQ, 4, UNDER);
  auto f = F::make_binary(exists_x, AND, x_neq_4, ty);
  tell_store(istore, f, var_x, Itv(zi(5), zd::bot()));
  EXPECT_EQ2(istore.vars(), 1);
  EXPECT_FALSE2(istore.is_top());
}
