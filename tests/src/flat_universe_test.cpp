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
  EXPECT_EQ(project_fun(ADD, ZF(0), ZF(1)), ZF(1));
  EXPECT_EQ(project_fun(ADD, ZF(-10), ZF(-5)), ZF(-15));
}

TEST(FlatUniverseTest, ConversionUpset) {
  EXPECT_EQ((ZF(local::ZInc::top())), ZF::top());
  EXPECT_EQ((ZF(local::ZDec::top())), ZF::top());
  EXPECT_EQ((ZF(local::ZInc::bot())), ZF::bot());
  EXPECT_EQ((ZF(local::ZDec::bot())), ZF::bot());
}

TEST(FlatUniverseTest, InterpretIntegerType) {
  std::cout << "Z ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", ZF::bot());
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", local::FFlat::bot(), VarEnv<standard_allocator>{}, true);
}

TEST(FlatUniverseTest, InterpretRealType) {
  std::cout << "Z ";
  interpret_must_error<IKind::TELL, ZF>("var real: x;");
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var real: x;", local::FFlat::bot());
}

TEST(FlatUniverseTest, InterpretBoolType) {
  std::cout << "Z ";
  interpret_must_error<IKind::TELL, ZF>("var bool: x;");
  std::cout << "F ";
  interpret_must_error<IKind::TELL, local::FFlat>("var bool: x;");
}

TEST(FlatUniverseTest, ZFlatInterpretation) {
  expect_both_interpret_equal_to("constraint true;", ZF::bot());
  expect_both_interpret_equal_to("constraint false;", ZF::top());

  VarEnv<standard_allocator> env = env_with_x();
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", ZF(0), env);
  both_interpret_must_error<ZF>("constraint int_ne(x, 1);", env);
}
