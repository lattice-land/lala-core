// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "lala/logic/logic.hpp"
#include "lala/universes/flat_universe.hpp"
#include "battery/allocator.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;

using ZF = local::ZFlat;

TEST(FlatUniverseTest, BotTopTest) {
  bot_top_test(ZF(0));
}

TEST(FlatUniverseTest, JoinMeetTest) {
  join_meet_generic_test(ZF::bot(), ZF(0));
  join_meet_generic_test(ZF(1), ZF::top());
  join_one_test(ZF(0), ZF(1), ZF::top(), true);
}

TEST(FlatUniverseTest, ArithmeticTest) {
  generic_arithmetic_fun_test(ZF(0));

  EXPECT_EQ((ZF::template fun<ADD>(ZF(0), ZF(1))), ZF(1));
  EXPECT_EQ((ZF::template fun<ADD>(ZF(-10), ZF(-5))), ZF(-15));
}

TEST(FlatUniverseTest, ConversionUpset) {
  EXPECT_EQ((ZF(local::ZInc::top())), ZF::top());
  EXPECT_EQ((ZF(local::ZDec::top())), ZF::top());
  EXPECT_EQ((ZF(local::ZInc::bot())), ZF::bot());
  EXPECT_EQ((ZF(local::ZDec::bot())), ZF::bot());
}

TEST(FlatUniverseTest, InterpretIntegerType) {
  std::cout << "Z ";
  must_interpret_tell_to("var int: x;", ZF::bot());
  std::cout << "F ";
  must_interpret_tell_to("var int: x;", local::FFlat::bot(), true);
}

TEST(FlatUniverseTest, InterpretRealType) {
  std::cout << "Z ";
  must_error_tell<ZF>("var real: x;");
  std::cout << "F ";
  must_interpret_tell_to("var real: x;", local::FFlat::bot());
}

TEST(FlatUniverseTest, InterpretBoolType) {
  std::cout << "Z ";
  must_error_tell<ZF>("var bool: x;");
  std::cout << "F ";
  must_error_tell<local::FFlat>("var bool: x;");
}

TEST(FlatUniverseTest, ZFlatInterpretation) {
  must_interpret_to("constraint true;", ZF::bot());
  must_interpret_to("constraint false;", ZF::top());

  VarEnv<standard_allocator> env;
  auto f = parse_flatzinc_str<standard_allocator>("var int: x :: abstract(0);");
  EXPECT_TRUE(f);
  AVar avar;
  IDiagnostics<F> diagnostics;
  EXPECT_TRUE(env.interpret(*f, avar, diagnostics));
  must_interpret_to(env, "constraint int_eq(x, 0);", ZF(0));
  must_error<ZF>(env, "constraint int_ne(x, 1);");
}
