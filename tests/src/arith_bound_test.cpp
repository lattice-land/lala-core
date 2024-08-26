// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "abstract_testing.hpp"
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/arith_bound.hpp"
#include "lala/universes/flat_universe.hpp"

using namespace lala;
using namespace battery;

TEST(ArithBoundTest, BotTopTest) {
  bot_top_test(local::ZLB(0));
  bot_top_test(local::ZUB(0));
}

TEST(ArithBoundTest, OrderTest) {
  EXPECT_TRUE(local::ZLB(10) < local::ZLB::top());
  EXPECT_TRUE(local::ZLB(10) < local::ZLB(0));
  EXPECT_TRUE(local::ZLB(10) > local::ZLB::bot());
  EXPECT_TRUE(local::ZUB(10) < local::ZUB::top());
  EXPECT_TRUE(local::ZUB(10) > local::ZUB(0));
  EXPECT_TRUE(local::ZUB(10) > local::ZUB::bot());
}

TEST(ArithBoundTest, JoinMeetTest) {
  join_meet_generic_test(local::ZLB::bot(), local::ZLB::top());
  join_meet_generic_test(local::ZLB(0), local::ZLB(0));
  join_meet_generic_test(local::ZLB(5), local::ZLB(0));
  join_meet_generic_test(local::ZLB(-5), local::ZLB(-10));

  join_meet_generic_test(local::ZUB::bot(), local::ZUB::top());
  join_meet_generic_test(local::ZUB(0), local::ZUB(0));
  join_meet_generic_test(local::ZUB(0), local::ZUB(5));
  join_meet_generic_test(local::ZUB(-10), local::ZUB(-5));
}

template <class L>
void test_z_arithmetic() {
  using F = L::template flat_type<battery::local_memory>;

  generic_arithmetic_fun_test<F, L>(F(0));

  EXPECT_EQ((project_fun<F, L>(NEG, F(L::bot()))), L::bot());

  EXPECT_EQ((project_fun<F, L>(ADD, F(0), F(1))), L(1));
  EXPECT_EQ((project_fun<F, L>(ADD, F(-10), F(0))), L(-10));
  EXPECT_EQ((project_fun<F, L>(ADD, F(-10), F(-5))), L(-15));
  EXPECT_EQ((project_fun<F, L>(ADD, F(10), F(-5))), L(5));
  EXPECT_EQ((project_fun<F, L>(ADD, F(10), F(5))), L(15));
}

TEST(ArithBoundTest, ArithmeticTest) {
  test_z_arithmetic<local::ZLB>();
  test_z_arithmetic<local::ZUB>();
  using zlb = local::ZLB;
  using zub = local::ZUB;
  EXPECT_EQ((project_fun(MIN, zlb::top(), zlb(10))), zlb::top());
  EXPECT_EQ((project_fun(MIN, zlb(10), zlb::top())), zlb::top());
  EXPECT_EQ((project_fun(MAX, zlb::top(), zlb(10))), zlb(10));
  EXPECT_EQ((project_fun(MIN, zub::top(), zub(10))), zub(10));
  EXPECT_EQ((project_fun(MIN, zub(10), zub::top())), zub(10));
  EXPECT_EQ((project_fun(MAX, zlb(10), zlb::top())), zlb(10));
  EXPECT_EQ((project_fun(MAX, zub::top(), zub(10))), zub::top());
  EXPECT_EQ((project_fun(MAX, zub(10), zub::top())), zub::top());

  EXPECT_EQ((project_fun(MIN, zlb::bot(), zlb(10))), zlb::bot());
  EXPECT_EQ((project_fun(MAX, zlb::bot(), zlb(10))), zlb::bot());
  EXPECT_EQ((project_fun(MIN, zub::bot(), zub(10))), zub::bot());
  EXPECT_EQ((project_fun(MAX, zub::bot(), zub(10))), zub::bot());
}

template<class Z, class F>
void interpret_integer_type() {
  std::cout << "Z ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", Z::top());
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", F::top(), VarEnv<standard_allocator>{}, true);
}

TEST(ArithBoundTest, InterpretIntegerType) {
  interpret_integer_type<local::ZLB, local::FLB>();
  interpret_integer_type<local::ZUB, local::FUB>();
}

template<class Z, class F>
void interpret_real_type() {
  std::cout << "Z ";
  both_interpret_must_error<Z>("var real: x;");
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var real: x;", F::top());
}

TEST(ArithBoundTest, InterpretRealType) {
  interpret_real_type<local::ZLB, local::FLB>();
  interpret_real_type<local::ZUB, local::FUB>();
}

template<class Z, class F>
void interpret_bool_type() {
  std::cout << "Z ";
  both_interpret_must_error<Z>("var bool: x;");
  std::cout << "F ";
  both_interpret_must_error<F>("var bool: x;");
}

TEST(ArithBoundTest, InterpretBoolType) {
  interpret_bool_type<local::ZLB, local::FLB>();
  interpret_bool_type<local::ZUB, local::FUB>();
}

TEST(ArithBoundTest, ZLBInterpretation) {
  using zlb = local::ZLB;
  expect_both_interpret_equal_to("constraint true;", zlb::top());
  expect_both_interpret_equal_to("constraint false;", zlb::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_ge(x, 0);", zlb(0), env);
  expect_both_interpret_equal_to("constraint int_ge(x, -10);", zlb(-10), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 10);", zlb(10), env);

  expect_both_interpret_equal_to("constraint int_gt(x, 0);", zlb(1), env);
  expect_both_interpret_equal_to("constraint int_gt(x, -10);", zlb(-9), env);
  expect_both_interpret_equal_to("constraint int_gt(x, 10);", zlb(11), env);

  interpret_must_error<IKind::ASK, zlb>("constraint int_eq(x, 0);", env);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", zlb(0), env);

  interpret_must_error<IKind::TELL, zlb>("constraint int_ne(x, 1);", env);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1);", zlb(2), env);

  both_interpret_must_error<zlb>("constraint int_le(x, 10);", env);
  both_interpret_must_error<zlb>("constraint int_lt(x, 10);", env);
}

TEST(ArithBoundTest, ZUBInterpretation) {
  using zub = local::ZUB;
  expect_both_interpret_equal_to("constraint true;", zub::top());
  expect_both_interpret_equal_to("constraint false;", zub::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_le(x, 0);", zub(0), env);
  expect_both_interpret_equal_to("constraint int_le(x, -10);", zub(-10), env);
  expect_both_interpret_equal_to("constraint int_le(x, 10);", zub(10), env);

  expect_both_interpret_equal_to("constraint int_lt(x, 0);", zub(-1), env);
  expect_both_interpret_equal_to("constraint int_lt(x, -10);", zub(-11), env);
  expect_both_interpret_equal_to("constraint int_lt(x, 10);", zub(9), env);

  interpret_must_error<IKind::ASK, zub>("constraint int_eq(x, 0);", env);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", zub(0), env);

  interpret_must_error<IKind::TELL, zub>("constraint int_ne(x, 1);", env);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1);", zub(0), env);

  both_interpret_must_error<zub>("constraint int_ge(x, 10);", env);
  both_interpret_must_error<zub>("constraint int_gt(x, 10);", env);
}

TEST(ArithBoundTest, ConjunctionDisjunction) {
  using zlb = local::ZLB;
  expect_both_interpret_equal_to("constraint true; constraint false;", zlb::bot());
  expect_both_interpret_equal_to("constraint false; constraint true;", zlb::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_ge(x, 0); constraint int_ge(x, -2); constraint int_ge(x, 2);", zlb(2), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 0); constraint int_ge(x, 2); constraint int_ge(x, -2);", zlb(2), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 2); constraint int_ge(x, -2); constraint int_ge(x, 0);", zlb(2), env);

  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", zlb(-2), env);
  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", zlb(-2), env);
  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", zlb(-2), env);
}

