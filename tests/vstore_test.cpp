// Copyright 2021 Pierre Talbot

#include "z.hpp"
#include "cartesian_product.hpp"
#include "vstore.hpp"
#include "generic_universe_test.hpp"

using zi = ZInc<int, StandardAllocator>;
using zd = ZDec<int, StandardAllocator>;
using Itv = CartesianProduct<zi, zd>;
using ZStore = VStore<zi, StandardAllocator>;
using IStore = VStore<Itv, StandardAllocator>;

const static AType ty = 0;

void populate_zstore_10_vars(ZStore& store, int v) {
  for(int i = 0; i < 10; ++i) {
    LVar<StandardAllocator> x = "x ";
    x[1] = '0' + i;
    EXPECT_TRUE(store.interpret(F::make_exists(ty, x, Int)).has_value());
    EXPECT_TRUE(store.interpret(make_v_op_z(x, GEQ, v)).has_value());
  }
}

void populate_istore_10_vars(IStore& store, int l, int u) {
  for(int i = 0; i < 10; ++i) {
    LVar<StandardAllocator> x = "x ";
    x[1] = '0' + i;
    EXPECT_TRUE(store.interpret(F::make_exists(ty, x, Int)).has_value());
    EXPECT_TRUE(store.interpret(make_v_op_z(x, GEQ, l)).has_value());
    EXPECT_TRUE(store.interpret(make_v_op_z(x, LEQ, u)).has_value());
  }
}

TEST(VStoreTest, TopBot) {
  ZStore ones(ty, 10);
  populate_zstore_10_vars(ones, 1);
  EXPECT_EQ(ones.vars(), 10);
  IStore bools(ty, 10);
  populate_istore_10_vars(bools, 0, 1);
  EXPECT_EQ(bools.vars(), 10);
  bot_top_test<ZStore>(ones);
  bot_top_test<IStore>(bools);
}

// I. With integer variables and exact interpretation.

template <typename A>
void check_failed_interpret(const thrust::optional<typename A::TellType>& tell, const A& store) {
  EXPECT_FALSE(tell.has_value());
  EXPECT_EQ(store.vars(), 0);
  EXPECT_EQ(store, A::bot());
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
  auto tell = zstore.interpret(F::make_binary(exists_x, AND, x_gt_4, ty));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ(zstore.vars(), 1);
  EXPECT_EQ((*tell).size(), 1);
  EXPECT_EQ(get<1>((*tell)[0]), zi(5));
}

// x > 4 /\ ∃x should fail.
TEST(VStoreTest, Interpret3) {
  ZStore zstore = ZStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_gt_4 = make_v_op_z(var_x, GT, 4);
  auto tell = zstore.interpret(F::make_binary(x_gt_4, AND, exists_x, ty));
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
  auto tell = zstore.interpret(F::make_nary(AND, conjunction, ty));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ(zstore.vars(), 1);
  EXPECT_EQ((*tell).size(), 1);
  EXPECT_EQ(get<1>((*tell)[0]), zi(6));
}

// ∃x /\ x > 4 /\ x < 6 should succeed, with size of tell element == 1.
TEST(VStoreTest, Interpret6) {
  IStore istore = IStore::bot(ty);
  F::Sequence conjunction(3);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = make_v_op_z(var_x, GT, 4);
  conjunction[2] = make_v_op_z(var_x, LT, 6);
  auto tell = istore.interpret(F::make_nary(AND, conjunction, ty));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ(istore.vars(), 1);
  EXPECT_EQ((*tell).size(), 1);
  EXPECT_EQ(get<1>((*tell)[0]), Itv(zi(5), zd(5)));
}


// ∃x /\ ∃y /\ x > 4 /\ x < 6 /\ y < 2 should succeed, with size of tell element == 2.
TEST(VStoreTest, Interpret7) {
  IStore istore = IStore::bot(ty);
  F::Sequence conjunction(5);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = F::make_exists(ty, var_y, Int);
  conjunction[2] = make_v_op_z(var_x, GT, 4);
  conjunction[3] = make_v_op_z(var_x, LT, 6);
  conjunction[4] = make_v_op_z(var_y, LT, 2);
  auto tell = istore.interpret(F::make_nary(AND, conjunction, ty));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ(istore.vars(), 2);
  EXPECT_EQ((*tell).size(), 2);
  EXPECT_EQ(get<1>((*tell)[0]), Itv(zi(5), zd(5)));
  EXPECT_EQ(get<1>((*tell)[1]), Itv(zi::bot(), zd(1)));
}

// ∃x /\ x > 4 /\ x < 4 /\ y < 2 should fail (undeclared variable).
TEST(VStoreTest, Interpret8) {
  IStore istore = IStore::bot(ty);
  F::Sequence conjunction(4);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = make_v_op_z(var_x, GT, 4);
  conjunction[2] = make_v_op_z(var_x, LT, 6);
  conjunction[3] = make_v_op_z(var_y, LT, 2);
  auto tell = istore.interpret(F::make_nary(AND, conjunction, ty));
  check_failed_interpret(tell, istore);
}

// II. With integer variables and under- and over-approximations.

// ∃x /\_o x ==_u 4 should succeed in ZStore, with size of tell element == 0 (the formula `x == 4` is not interpreted because the underlying abstract universe does not support under-approximation of ==).
TEST(VStoreTest, Interpret9) {
  ZStore zstore = ZStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNDER);
  auto tell = zstore.interpret(F::make_binary(exists_x, AND, x_eq_4, ty, OVER));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ(zstore.vars(), 1);
  EXPECT_EQ((*tell).size(), 0);
}

// ∃x /\_o x ==_o 4 should succeed in ZStore, with size of tell element == 1.
TEST(VStoreTest, Interpret10) {
  ZStore zstore = ZStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_eq_4 = make_v_op_z(var_x, EQ, 4, OVER);
  auto tell = zstore.interpret(F::make_binary(exists_x, AND, x_eq_4, ty, OVER));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ(zstore.vars(), 1);
  EXPECT_EQ((*tell).size(), 1);
  EXPECT_EQ(get<1>((*tell)[0]), zi(4));
}

// ∃x /\ x ==_u 4 fail in IStore because x ==_u 4 is not interpretable in zi, neither in zd.
TEST(VStoreTest, Interpret11) {
  IStore istore = IStore::bot(ty);
  auto exists_x = F::make_exists(ty, var_x, Int);
  auto x_eq_4 = make_v_op_z(var_x, EQ, 4, UNDER);
  auto tell = istore.interpret(F::make_binary(exists_x, AND, x_eq_4, ty, EXACT));
  check_failed_interpret(tell, istore);
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
  auto tell = zstore.interpret(F::make_nary(AND, conjunction, ty, OVER));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ(zstore.vars(), 2);
  EXPECT_EQ((*tell).size(), 1);
  EXPECT_EQ(get<1>((*tell)[0]), zi(2));
}

// ∃x /\ ∃y /\_o x ==_u 4 /\_o y > 1 /\_o y < 10 should succeed in IStore, with size of tell element == 1.
TEST(VStoreTest, Interpret13) {
  IStore istore = IStore::bot(ty);
  F::Sequence conjunction(5);
  conjunction[0] = F::make_exists(ty, var_x, Int);
  conjunction[1] = F::make_exists(ty, var_y, Int);
  conjunction[2] = make_v_op_z(var_x, EQ, 4, UNDER);
  conjunction[3] = make_v_op_z(var_y, GT, 1);
  conjunction[4] = make_v_op_z(var_y, LT, 10);
  auto tell = istore.interpret(F::make_nary(AND, conjunction, ty, OVER));
  EXPECT_TRUE(tell.has_value());
  EXPECT_EQ(istore.vars(), 2);
  EXPECT_EQ((*tell).size(), 1);
  EXPECT_EQ(get<1>((*tell)[0]), Itv(zi(2), zd(9)));
}

// TODO in a store of "real" intervals (not Cartesian product).

// ∃x /\ x > 4 /\ x < 4 should succeed, with size of tell element == 1.
// ∃x /\ ∃y /\ x > 4 /\ x < 4 /\ y < 2 should succeed, with size of tell element == 1 and equal to top.
// ∃x /\ x ==_u 4 should succeed in IStore, with size of tell element == 1.

// ∃x /\ ∃y /\ ∃z /\ x > 1 /\ x < 2 /\ y > 0 /\ z < 10 should succeed in IStore, with size of tell element == 3 and equals to top.
// ∃x /\ ∃y /\ ∃z /\ x > 1 /\ x < 1 /\ y > 0 /\ z < 10 should succeed in IStore, with size of tell element == 1 and equals to top.