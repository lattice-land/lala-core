// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "thrust/optional.h"
#include "ast.hpp"
#include "z.hpp"
#include "allocator.hpp"
#include "utility.hpp"

using namespace lala;

typedef ZInc<int, StandardAllocator> zi;
typedef ZDec<int, StandardAllocator> zd;
typedef TFormula<StandardAllocator> F;

static AVar var_x = 0;

TEST(ZDeathTest, BadConstruction) {
  ASSERT_DEATH(zi(Limits<int>::bot()), "");
  ASSERT_DEATH(zi(Limits<int>::top()), "");
  // Dual
  ASSERT_DEATH(zd(Limits<int>::bot()), "");
  ASSERT_DEATH(zd(Limits<int>::top()), "");
}

template<typename Universe>
void test_formula(Approx appx, const F& f, thrust::optional<Universe> expect) {
  thrust::optional<Universe> j = Universe::interpret(appx, f);
  EXPECT_EQ(j.has_value(), expect.has_value());
  EXPECT_EQ(j, expect);
}

template<typename Universe>
void test_interpret(Sig sig, Approx appx, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  test_formula<Universe>(
    appx,
    make_v_op_z(0, sig, elem, standard_allocator),
    expect);
}

template<typename Universe>
void test_all_interpret(Sig sig, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  Approx appxs[3] = {EXACT, UNDER, OVER};
  for(int i = 0; i < 3; ++i) {
    test_interpret<Universe>(sig, appxs[i], elem, expect);
  }
}

template<typename Universe>
void test_exact_interpret(Sig sig, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, EXACT, elem, expect);
}

template<typename Universe>
void test_under_interpret(Sig sig, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, UNDER, elem, expect);
}

template<typename Universe>
void test_over_interpret(Sig sig, typename Universe::ValueType elem, thrust::optional<Universe> expect) {
  test_interpret<Universe>(sig, OVER, elem, expect);
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

// `a` and `b` are supposed ordered and `a <= b`.
template <typename A>
void join_meet_generic_test(A a, A b) {
  // Reflexivity
  EXPECT_EQ(a.join(a), a);
  EXPECT_EQ(a.meet(a), a);
  EXPECT_EQ(b.join(b), b);
  EXPECT_EQ(b.meet(b), b);
  // Coherency of join/meet w.r.t. ordering
  EXPECT_EQ(a.join(b), b);
  EXPECT_EQ(b.join(a), b);
  // Commutativity
  EXPECT_EQ(a.meet(b), a);
  EXPECT_EQ(b.meet(a), a);
  // Absorbing
  EXPECT_EQ(a.meet(A::top()), a);
  EXPECT_EQ(b.meet(A::top()), b);
  EXPECT_EQ(a.join(A::top()), A::top());
  EXPECT_EQ(b.join(A::top()), A::top());
  EXPECT_EQ(a.meet(A::bot()), A::bot());
  EXPECT_EQ(b.meet(A::bot()), A::bot());
  EXPECT_EQ(a.join(A::bot()), a);
  EXPECT_EQ(b.join(A::bot()), b);
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

template<typename A>
void generic_order_test(A element) {
  EXPECT_EQ(element.order(A::top()), true);
  EXPECT_EQ(element.order(A::bot()), false);
  EXPECT_EQ(A::bot().order(A::bot()), true);
  EXPECT_EQ(A::top().order(A::top()), true);
  EXPECT_EQ(A::top().order(A::bot()), false);
  EXPECT_EQ(A::bot().order(A::top()), true);
  EXPECT_EQ(A::top().order(element), false);
}

TEST(ZTest, Order) {
  EXPECT_EQ(zi(0).order(zi(0)), true);
  EXPECT_EQ(zi(1).order(zi(0)), false);
  EXPECT_EQ(zi(0).order(zi(1)), true);
  EXPECT_EQ(zi(0).order(zi(-1)), false);
  EXPECT_EQ(zi(-1).order(zi(0)), true);
  generic_order_test(zi(0));
  // Dual
  EXPECT_EQ(zd(0).order(zd(0)), true);
  EXPECT_EQ(zd(1).order(zd(0)), true);
  EXPECT_EQ(zd(0).order(zd(1)), false);
  EXPECT_EQ(zd(0).order(zd(-1)), true);
  EXPECT_EQ(zd(-1).order(zd(0)), false);
  generic_order_test(zd(0));
}

template<typename A>
using SplitSeq = DArray<A, StandardAllocator>;

template<typename A>
SplitSeq<A> make_singleton(A x) {
  return SplitSeq<A>({x});
}

template<typename A>
SplitSeq<A> make_empty() {
  return SplitSeq<A>();
}

template<typename A>
void generic_split_test(A element) {
  EXPECT_EQ(element.split(), make_singleton(element));
  EXPECT_EQ(A::top().split(), make_empty<A>());
  EXPECT_EQ(A::bot().split(), make_singleton(A::bot()));
}

TEST(ZTest, Split) {
  generic_split_test(zi(0));
  generic_split_test(zd(0));
}

template<typename A>
void generic_deinterpret_test() {
  EXPECT_EQ(A::bot().deinterpret(var_x), F::make_true());
  EXPECT_EQ(A::top().deinterpret(var_x), F::make_false());
}

TEST(ZTest, Deinterpret) {
  F f10 = make_v_op_z(0, GEQ, 10, standard_allocator);
  zi z10 = zi::interpret(EXACT, f10).value();
  F f10_bis = z10.deinterpret(var_x);
  EXPECT_EQ(f10, f10_bis);
  F f9 = make_v_op_z(0, GT, 9, standard_allocator);
  zi z9 = zi::interpret(EXACT, f9).value();
  F f9_bis = z9.deinterpret(var_x);
  EXPECT_EQ(f10, f9_bis);
  generic_deinterpret_test<zi>();
  // Dual
  F f10_d = make_v_op_z(0, LEQ, 10, standard_allocator);
  zd z10_d = zd::interpret(EXACT, f10_d).value();
  F f10_bis_d = z10_d.deinterpret(var_x);
  EXPECT_EQ(f10_d, f10_bis_d);
  F f11_d = make_v_op_z(0, LT, 11, standard_allocator);
  zd z11_d = zd::interpret(EXACT, f11_d).value();
  F f11_bis_d = z11_d.deinterpret(var_x);
  EXPECT_EQ(f10_d, f11_bis_d);
  generic_deinterpret_test<zd>();
}
