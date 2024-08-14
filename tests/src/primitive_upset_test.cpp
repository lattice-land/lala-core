// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "abstract_testing.hpp"
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/universes/flat_universe.hpp"

using namespace lala;
using namespace battery;

TEST(PrimitiveUpsetTest, BotTopTest) {
  bot_top_test(local::ZInc(0));
  bot_top_test(local::ZDec(0));
}

TEST(PrimitiveUpsetTest, JoinMeetTest) {
  join_meet_generic_test(local::ZInc::bot(), local::ZInc::top());
  join_meet_generic_test(local::ZInc(0), local::ZInc(0));
  join_meet_generic_test(local::ZInc(0), local::ZInc(5));
  join_meet_generic_test(local::ZInc(-10), local::ZInc(-5));

  join_meet_generic_test(local::ZDec::bot(), local::ZDec::top());
  join_meet_generic_test(local::ZDec(0), local::ZDec(0));
  join_meet_generic_test(local::ZDec(5), local::ZDec(0));
  join_meet_generic_test(local::ZDec(-5), local::ZDec(-10));
}

template <class L>
void test_z_arithmetic() {
  using F = L::template flat_type<battery::local_memory>;

  generic_arithmetic_fun_test<F, L>(F(0));

  EXPECT_EQ((project_fun<F, L>(NEG, F(L::top()))), L::top());

  EXPECT_EQ((project_fun<F, L>(ADD, F(0), F(1))), L(1));
  EXPECT_EQ((project_fun<F, L>(ADD, F(-10), F(0))), L(-10));
  EXPECT_EQ((project_fun<F, L>(ADD, F(-10), F(-5))), L(-15));
  EXPECT_EQ((project_fun<F, L>(ADD, F(10), F(-5))), L(5));
  EXPECT_EQ((project_fun<F, L>(ADD, F(10), F(5))), L(15));
}

TEST(PrimitiveUpsetTest, ArithmeticTest) {
  test_z_arithmetic<local::ZInc>();
  test_z_arithmetic<local::ZDec>();
  using ZI = local::ZInc;
  using ZD = local::ZDec;
  EXPECT_EQ((project_fun(MIN, ZI::bot(), ZI(10))), ZI::bot());
  EXPECT_EQ((project_fun(MIN, ZI(10), ZI::bot())), ZI::bot());
  EXPECT_EQ((project_fun(MAX, ZI::bot(), ZI(10))), ZI(10));
  EXPECT_EQ((project_fun(MIN, ZD::bot(), ZD(10))), ZD(10));
  EXPECT_EQ((project_fun(MIN, ZD(10), ZD::bot())), ZD(10));
  EXPECT_EQ((project_fun(MAX, ZI(10), ZI::bot())), ZI(10));
  EXPECT_EQ((project_fun(MAX, ZD::bot(), ZD(10))), ZD::bot());
  EXPECT_EQ((project_fun(MAX, ZD(10), ZD::bot())), ZD::bot());

  EXPECT_EQ((project_fun(MIN, ZI::top(), ZI(10))), ZI::top());
  EXPECT_EQ((project_fun(MAX, ZI::top(), ZI(10))), ZI::top());
  EXPECT_EQ((project_fun(MIN, ZD::top(), ZD(10))), ZD::top());
  EXPECT_EQ((project_fun(MAX, ZD::top(), ZD(10))), ZD::top());
}

template<class Z, class F, class B>
void interpret_integer_type() {
  std::cout << "Z ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", Z::bot());
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", F::bot(), VarEnv<standard_allocator>{}, true);
  std::cout << "B ";
  both_interpret_must_error<B>("var int: x;");
}

TEST(PrimitiveUpsetTest, InterpretIntegerType) {
  interpret_integer_type<local::ZInc, local::FInc, local::BInc>();
  interpret_integer_type<local::ZDec, local::FDec, local::BDec>();
}

template<class Z, class F, class B>
void interpret_real_type() {
  std::cout << "Z ";
  both_interpret_must_error<Z>("var real: x;");
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var real: x;", F::bot());
  std::cout << "B ";
  both_interpret_must_error<B>("var real: x;");
}

TEST(PrimitiveUpsetTest, InterpretRealType) {
  interpret_real_type<local::ZInc, local::FInc, local::BInc>();
  interpret_real_type<local::ZDec, local::FDec, local::BDec>();
}

template<class Z, class F, class B>
void interpret_bool_type() {
  std::cout << "Z ";
  both_interpret_must_error<Z>("var bool: x;");
  std::cout << "F ";
  both_interpret_must_error<F>("var bool: x;");
  std::cout << "B ";
  expect_interpret_equal_to<IKind::TELL>("var bool: x;", B::bot());
}

TEST(PrimitiveUpsetTest, InterpretBoolType) {
  interpret_bool_type<local::ZInc, local::FInc, local::BInc>();
  interpret_bool_type<local::ZDec, local::FDec, local::BDec>();
}

TEST(PrimitiveUpsetTest, ZIncInterpretation) {
  using ZI = local::ZInc;
  expect_both_interpret_equal_to("constraint true;", ZI::bot());
  expect_both_interpret_equal_to("constraint false;", ZI::top());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_ge(x, 0);", ZI(0), env);
  expect_both_interpret_equal_to("constraint int_ge(x, -10);", ZI(-10), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 10);", ZI(10), env);

  expect_both_interpret_equal_to("constraint int_gt(x, 0);", ZI(1), env);
  expect_both_interpret_equal_to("constraint int_gt(x, -10);", ZI(-9), env);
  expect_both_interpret_equal_to("constraint int_gt(x, 10);", ZI(11), env);

  interpret_must_error<IKind::ASK, ZI>("constraint int_eq(x, 0);", env);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", ZI(0), env);

  interpret_must_error<IKind::TELL, ZI>("constraint int_ne(x, 1);", env);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1);", ZI(2), env);

  both_interpret_must_error<ZI>("constraint int_le(x, 10);", env);
  both_interpret_must_error<ZI>("constraint int_lt(x, 10);", env);
}

TEST(PrimitiveUpsetTest, ZDecInterpretation) {
  using ZD = local::ZDec;
  expect_both_interpret_equal_to("constraint true;", ZD::bot());
  expect_both_interpret_equal_to("constraint false;", ZD::top());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_le(x, 0);", ZD(0), env);
  expect_both_interpret_equal_to("constraint int_le(x, -10);", ZD(-10), env);
  expect_both_interpret_equal_to("constraint int_le(x, 10);", ZD(10), env);

  expect_both_interpret_equal_to("constraint int_lt(x, 0);", ZD(-1), env);
  expect_both_interpret_equal_to("constraint int_lt(x, -10);", ZD(-11), env);
  expect_both_interpret_equal_to("constraint int_lt(x, 10);", ZD(9), env);

  interpret_must_error<IKind::ASK, ZD>("constraint int_eq(x, 0);", env);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", ZD(0), env);

  interpret_must_error<IKind::TELL, ZD>("constraint int_ne(x, 1);", env);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1);", ZD(0), env);

  both_interpret_must_error<ZD>("constraint int_ge(x, 10);", env);
  both_interpret_must_error<ZD>("constraint int_gt(x, 10);", env);
}

TEST(PrimitiveUpsetTest, ConjunctionDisjunction) {
  using ZI = local::ZInc;
  expect_both_interpret_equal_to("constraint true; constraint false;", ZI::top());
  expect_both_interpret_equal_to("constraint false; constraint true;", ZI::top());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_ge(x, 0); constraint int_ge(x, -2); constraint int_ge(x, 2);", ZI(2), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 0); constraint int_ge(x, 2); constraint int_ge(x, -2);", ZI(2), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 2); constraint int_ge(x, -2); constraint int_ge(x, 0);", ZI(2), env);

  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", ZI(-2), env);
  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", ZI(-2), env);
  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", ZI(-2), env);
}

