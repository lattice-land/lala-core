// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "universes/flat_universe.hpp"
#include "allocator.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;

TEST(UpsetUniverseTest, BotTopTest) {
  bot_top_test(local::ZInc(0));
  bot_top_test(local::ZDec(0));
}

TEST(UpsetUniverseTest, JoinMeetTest) {
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
  using F = typename L::flat_type<battery::LocalMemory>;

  generic_arithmetic_fun_test<F, L>(F(0));

  EXPECT_EQ((L::template fun<NEG>(F(L::top()))), L::top());

  EXPECT_EQ((L::template fun<ADD>(F(0), F(1))), L(1));
  EXPECT_EQ((L::template fun<ADD>(F(-10), F(0))), L(-10));
  EXPECT_EQ((L::template fun<ADD>(F(-10), F(-5))), L(-15));
  EXPECT_EQ((L::template fun<ADD>(F(10), F(-5))), L(5));
  EXPECT_EQ((L::template fun<ADD>(F(10), F(5))), L(15));
}

TEST(UpsetUniverseTest, ArithmeticTest) {
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
  must_interpret_to("var int: x :: exact;", Z::bot());
  must_interpret_to("var int: x :: under;", Z::bot());
  must_interpret_to("var int: x :: over;", Z::bot());

  std::cout << "F ";
  must_error<F>("var int: x :: exact;");
  must_error<F>("var int: x :: under;");
  must_interpret_to("var int: x :: over;", F::bot(), true);

  std::cout << "B ";
  must_error<B>("var int: x :: exact;");
  must_interpret_to("var int: x :: under;", B::bot(), true);
  must_error<B>("var int: x :: over;");
}

TEST(UpsetUniverseTest, InterpretIntegerType) {
  interpret_integer_type<local::ZInc, local::FInc, local::BInc>();
  interpret_integer_type<local::ZDec, local::FDec, local::BDec>();
}

template<class Z, class F, class B>
void interpret_real_type() {
  std::cout << "Z ";
  must_error<Z>("var real: x :: exact;");
  must_interpret_to("var real: x :: under;", Z::bot(), true);
  must_error<Z>("var real: x :: over;");

  std::cout << "F ";
  must_interpret_to("var real: x :: exact;", F::bot());
  must_interpret_to("var real: x :: under;", F::bot());
  must_interpret_to("var real: x :: over;", F::bot());

  std::cout << "B ";
  must_error<B>("var real: x :: exact;");
  must_interpret_to("var real: x :: under;", B::bot(), true);
  must_error<B>("var real: x :: over;");
}

TEST(UpsetUniverseTest, InterpretRealType) {
  interpret_real_type<local::ZInc, local::FInc, local::BInc>();
  interpret_real_type<local::ZDec, local::FDec, local::BDec>();
}

template<class Z, class F, class B>
void interpret_bool_type() {
  std::cout << "Z ";
  must_error<Z>("var bool: x :: exact;");
  must_error<Z>("var bool: x :: under;");
  must_interpret_to("var bool: x :: over;", Z::bot(), true);

  std::cout << "F ";
  must_error<F>("var bool: x :: exact;");
  must_error<F>("var bool: x :: under;");
  must_interpret_to("var bool: x :: over;", F::bot(), true);

  std::cout << "B ";
  must_interpret_to("var bool: x :: exact;", B::bot());
  must_interpret_to("var bool: x :: under;", B::bot());
  must_interpret_to("var bool: x :: over;", B::bot());
}

TEST(UpsetUniverseTest, InterpretBoolType) {
  interpret_bool_type<local::ZInc, local::FInc, local::BInc>();
  interpret_bool_type<local::ZDec, local::FDec, local::BDec>();
}

TEST(UpsetUniverseTest, ZIncInterpretation) {
  using ZI = local::ZInc;
  must_interpret_to("constraint true :: exact;", ZI::bot());
  must_interpret_to("constraint true :: over;", ZI::bot());
  must_interpret_to("constraint true :: under;", ZI::bot());

  must_interpret_to("constraint false :: exact;", ZI::top());
  must_interpret_to("constraint false :: over;", ZI::top());
  must_interpret_to("constraint false :: under;", ZI::top());

  VarEnv<StandardAllocator> env;
  auto f = parse_flatzinc_str<StandardAllocator>("var int: x :: abstract(0);");
  EXPECT_TRUE(f);
  env.interpret(*f);
  must_interpret_to(env, "constraint int_ge(x, 0) :: exact;", ZI(0));
  must_interpret_to(env, "constraint int_ge(x, -10) :: over;", ZI(-10));
  must_interpret_to(env, "constraint int_ge(x, 10) :: under;", ZI(10));

  must_interpret_to(env, "constraint int_gt(x, 0) :: exact;", ZI(1));
  must_interpret_to(env, "constraint int_gt(x, -10) :: over;", ZI(-9));
  must_interpret_to(env, "constraint int_gt(x, 10) :: under;", ZI(11));

  must_error<ZI>(env, "constraint int_eq(x, 0) :: exact;");
  must_interpret_to(env, "constraint int_eq(x, 0) :: over;", ZI(0));
  must_error<ZI>(env, "constraint int_eq(x, 0) :: under;");

  must_error<ZI>(env, "constraint int_ne(x, 1) :: exact;");
  must_error<ZI>(env, "constraint int_ne(x, 1) :: over;");
  must_interpret_to(env, "constraint int_ne(x, 1) :: under;", ZI(2));

  must_error<ZI>(env, "constraint int_le(x, 10) :: exact;");
  must_error<ZI>(env, "constraint int_le(x, 10) :: under;");
  must_error<ZI>(env, "constraint int_le(x, 10) :: over;");
  must_error<ZI>(env, "constraint int_lt(x, 10) :: exact;");
  must_error<ZI>(env, "constraint int_lt(x, 10) :: under;");
  must_error<ZI>(env, "constraint int_lt(x, 10) :: over;");

  // Under-approximating a floating-point constant in an integer.
  // must_interpret_to("constraint float_ge(x, 0.) :: exact;", ZI(0));
  // must_interpret_to("constraint float_ge(x, -10.) :: over;", ZI(-10));
  // must_interpret_to("constraint float_ge(x, 10.) :: under;", ZI(10));

  // must_interpret_to("constraint float_gt(x, 0.) :: exact;", ZI(1));
  // must_interpret_to("constraint float_gt(x, -10.) :: over;", ZI(-9));
  // must_interpret_to("constraint float_gt(x, 10.) :: under;", ZI(11));

  // must_error<ZI>("constraint float_eq(x, 0.) :: exact;");
  // must_interpret_to("constraint float_eq(x, 0.) :: over;", ZI(0));
  // must_error<ZI>("constraint float_eq(x, 0.) :: under;");

  // must_error<ZI>("constraint float_ne(x, 1.) :: exact;");
  // must_error<ZI>("constraint float_ne(x, 1.) :: over;");
  // must_interpret_to("constraint float_ne(x, 1.) :: under;", ZI(2));
}


TEST(UpsetUniverseTest, ZDecInterpretation) {
  using ZD = local::ZDec;
  must_interpret_to("constraint true :: exact;", ZD::bot());
  must_interpret_to("constraint true :: over;", ZD::bot());
  must_interpret_to("constraint true :: under;", ZD::bot());

  must_interpret_to("constraint false :: exact;", ZD::top());
  must_interpret_to("constraint false :: over;", ZD::top());
  must_interpret_to("constraint false :: under;", ZD::top());

  VarEnv<StandardAllocator> env;
  auto f = parse_flatzinc_str<StandardAllocator>("var int: x :: abstract(0);");
  EXPECT_TRUE(f);
  env.interpret(*f);
  must_interpret_to(env, "constraint int_le(x, 0) :: exact;", ZD(0));
  must_interpret_to(env, "constraint int_le(x, -10) :: over;", ZD(-10));
  must_interpret_to(env, "constraint int_le(x, 10) :: under;", ZD(10));

  must_interpret_to(env, "constraint int_lt(x, 0) :: exact;", ZD(-1));
  must_interpret_to(env, "constraint int_lt(x, -10) :: over;", ZD(-11));
  must_interpret_to(env, "constraint int_lt(x, 10) :: under;", ZD(9));

  must_error<ZD>(env, "constraint int_eq(x, 0) :: exact;");
  must_interpret_to(env, "constraint int_eq(x, 0) :: over;", ZD(0));
  must_error<ZD>(env, "constraint int_eq(x, 0) :: under;");

  must_error<ZD>(env, "constraint int_ne(x, 1) :: exact;");
  must_error<ZD>(env, "constraint int_ne(x, 1) :: over;");
  must_interpret_to(env, "constraint int_ne(x, 1) :: under;", ZD(0));

  must_error<ZD>(env, "constraint int_ge(x, 10) :: exact;");
  must_error<ZD>(env, "constraint int_ge(x, 10) :: under;");
  must_error<ZD>(env, "constraint int_ge(x, 10) :: over;");
  must_error<ZD>(env, "constraint int_gt(x, 10) :: exact;");
  must_error<ZD>(env, "constraint int_gt(x, 10) :: under;");
  must_error<ZD>(env, "constraint int_gt(x, 10) :: over;");
}
