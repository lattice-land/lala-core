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

  EXPECT_EQ((L::template fun<NEG>(F(L::top()))), L::top());

  EXPECT_EQ((L::template fun<ADD>(F(0), F(1))), L(1));
  EXPECT_EQ((L::template fun<ADD>(F(-10), F(0))), L(-10));
  EXPECT_EQ((L::template fun<ADD>(F(-10), F(-5))), L(-15));
  EXPECT_EQ((L::template fun<ADD>(F(10), F(-5))), L(5));
  EXPECT_EQ((L::template fun<ADD>(F(10), F(5))), L(15));
}

TEST(PrimitiveUpsetTest, ArithmeticTest) {
  test_z_arithmetic<local::ZInc>();
  test_z_arithmetic<local::ZDec>();
  using ZI = local::ZInc;
  using ZD = local::ZDec;
  EXPECT_EQ((ZI::template fun<MIN>(ZI::bot(), ZI(10))), ZI::bot());
  EXPECT_EQ((ZI::template fun<MIN>(ZI(10), ZI::bot())), ZI::bot());
  EXPECT_EQ((ZI::template fun<MAX>(ZI::bot(), ZI(10))), ZI(10));
  EXPECT_EQ((ZD::template fun<MIN>(ZD::bot(), ZD(10))), ZD(10));
  EXPECT_EQ((ZD::template fun<MIN>(ZD(10), ZD::bot())), ZD(10));
  EXPECT_EQ((ZI::template fun<MAX>(ZI(10), ZI::bot())), ZI(10));
  EXPECT_EQ((ZD::template fun<MAX>(ZD::bot(), ZD(10))), ZD::bot());
  EXPECT_EQ((ZD::template fun<MAX>(ZD(10), ZD::bot())), ZD::bot());

  EXPECT_EQ((ZI::template fun<MIN>(ZI::top(), ZI(10))), ZI::top());
  EXPECT_EQ((ZI::template fun<MAX>(ZI::top(), ZI(10))), ZI::top());
  EXPECT_EQ((ZD::template fun<MIN>(ZD::top(), ZD(10))), ZD::top());
  EXPECT_EQ((ZD::template fun<MAX>(ZD::top(), ZD(10))), ZD::top());
}

template<class Z, class F, class B>
void interpret_integer_type() {
  std::cout << "Z ";
  must_interpret_tell_to("var int: x;", Z::bot());
  std::cout << "F ";
  must_interpret_tell_to("var int: x;", F::bot(), true);
  std::cout << "B ";
  must_error<B>("var int: x;");
}

TEST(PrimitiveUpsetTest, InterpretIntegerType) {
  interpret_integer_type<local::ZInc, local::FInc, local::BInc>();
  interpret_integer_type<local::ZDec, local::FDec, local::BDec>();
}

template<class Z, class F, class B>
void interpret_real_type() {
  std::cout << "Z ";
  must_error<Z>("var real: x;");
  std::cout << "F ";
  must_interpret_tell_to("var real: x;", F::bot());
  std::cout << "B ";
  must_error<B>("var real: x;");
}

TEST(PrimitiveUpsetTest, InterpretRealType) {
  interpret_real_type<local::ZInc, local::FInc, local::BInc>();
  interpret_real_type<local::ZDec, local::FDec, local::BDec>();
}

template<class Z, class F, class B>
void interpret_bool_type() {
  std::cout << "Z ";
  must_error<Z>("var bool: x;");
  std::cout << "F ";
  must_error<F>("var bool: x;");
  std::cout << "B ";
  must_interpret_tell_to("var bool: x;", B::bot());
}

TEST(PrimitiveUpsetTest, InterpretBoolType) {
  interpret_bool_type<local::ZInc, local::FInc, local::BInc>();
  interpret_bool_type<local::ZDec, local::FDec, local::BDec>();
}

TEST(PrimitiveUpsetTest, ZIncInterpretation) {
  using ZI = local::ZInc;
  must_interpret_to("constraint true;", ZI::bot());
  must_interpret_to("constraint false;", ZI::top());

  VarEnv<standard_allocator> env;
  auto f = parse_flatzinc_str<standard_allocator>("var int: x :: abstract(0);");
  EXPECT_TRUE(f);
  AVar avar;
  IDiagnostics<F> diagnostics;
  EXPECT_TRUE(env.interpret(*f, avar, diagnostics));
  must_interpret_to(env, "constraint int_ge(x, 0);", ZI(0));
  must_interpret_to(env, "constraint int_ge(x, -10);", ZI(-10));
  must_interpret_to(env, "constraint int_ge(x, 10);", ZI(10));

  must_interpret_to(env, "constraint int_gt(x, 0);", ZI(1));
  must_interpret_to(env, "constraint int_gt(x, -10);", ZI(-9));
  must_interpret_to(env, "constraint int_gt(x, 10);", ZI(11));

  must_error_ask<ZI>(env, "constraint int_eq(x, 0);");
  must_interpret_tell_to(env, "constraint int_eq(x, 0);", ZI(0));

  must_error_tell<ZI>(env, "constraint int_ne(x, 1);");
  must_interpret_ask_to(env, "constraint int_ne(x, 1);", ZI(2));

  must_error<ZI>(env, "constraint int_le(x, 10);");
  must_error<ZI>(env, "constraint int_lt(x, 10);");
}

TEST(PrimitiveUpsetTest, ZDecInterpretation) {
  using ZD = local::ZDec;
  must_interpret_to("constraint true;", ZD::bot());
  must_interpret_to("constraint false;", ZD::top());

  VarEnv<standard_allocator> env;
  auto f = parse_flatzinc_str<standard_allocator>("var int: x :: abstract(0);");
  EXPECT_TRUE(f);
  AVar avar;
  IDiagnostics<F> diagnostics;
  EXPECT_TRUE(env.interpret(*f, avar, diagnostics));
  must_interpret_to(env, "constraint int_le(x, 0);", ZD(0));
  must_interpret_to(env, "constraint int_le(x, -10);", ZD(-10));
  must_interpret_to(env, "constraint int_le(x, 10);", ZD(10));

  must_interpret_to(env, "constraint int_lt(x, 0);", ZD(-1));
  must_interpret_to(env, "constraint int_lt(x, -10);", ZD(-11));
  must_interpret_to(env, "constraint int_lt(x, 10);", ZD(9));

  must_error_ask<ZD>(env, "constraint int_eq(x, 0);");
  must_interpret_tell_to(env, "constraint int_eq(x, 0);", ZD(0));

  must_error_tell<ZD>(env, "constraint int_ne(x, 1);");
  must_interpret_ask_to(env, "constraint int_ne(x, 1);", ZD(0));

  must_error<ZD>(env, "constraint int_ge(x, 10);");
  must_error<ZD>(env, "constraint int_gt(x, 10);");
}
