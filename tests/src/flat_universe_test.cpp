// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "logic/logic.hpp"
#include "universes/flat_universe.hpp"
#include "allocator.hpp"
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
  generic_arithmetic_fun_test<EXACT>(ZF(0));
  generic_arithmetic_fun_test<UNDER>(ZF(0));
  generic_arithmetic_fun_test<OVER>(ZF(0));

  EXPECT_EQ((ZF::template fun<EXACT, ADD>(ZF(0), ZF(1))), ZF(1));
  EXPECT_EQ((ZF::template fun<EXACT, ADD>(ZF(-10), ZF(-5))), ZF(-15));
}

TEST(FlatUniverseTest, InterpretIntegerType) {
  std::cout << "Z ";
  must_interpret_to("var int: x :: exact;", ZF::bot());
  must_interpret_to("var int: x :: under;", ZF::bot());
  must_interpret_to("var int: x :: over;", ZF::bot());

  std::cout << "F ";
  must_error<local::FFlat>("var int: x :: exact;");
  must_error<local::FFlat>("var int: x :: under;");
  must_interpret_to("var int: x :: over;", local::FFlat::bot(), true);
}

TEST(FlatUniverseTest, InterpretRealType) {
  std::cout << "Z ";
  must_error<ZF>("var real: x :: exact;");
  must_interpret_to("var real: x :: under;", ZF::bot(), true);
  must_error<ZF>("var real: x :: over;");

  std::cout << "F ";
  must_interpret_to("var real: x :: exact;", local::FFlat::bot());
  must_interpret_to("var real: x :: under;", local::FFlat::bot());
  must_interpret_to("var real: x :: over;", local::FFlat::bot());
}

TEST(FlatUniverseTest, InterpretBoolType) {
  std::cout << "Z ";
  must_error<ZF>("var bool: x :: exact;");
  must_error<ZF>("var bool: x :: under;");
  must_interpret_to("var bool: x :: over;", ZF::bot(), true);

  std::cout << "F ";
  must_error<local::FFlat>("var bool: x :: exact;");
  must_error<local::FFlat>("var bool: x :: under;");
  must_interpret_to("var bool: x :: over;", local::FFlat::bot(), true);
}

TEST(FlatUniverseTest, ZFlatInterpretation) {
  must_interpret_to("constraint true :: exact;", ZF::bot());
  must_interpret_to("constraint true :: over;", ZF::bot());
  must_interpret_to("constraint true :: under;", ZF::bot());

  must_interpret_to("constraint false :: exact;", ZF::top());
  must_interpret_to("constraint false :: over;", ZF::top());
  must_interpret_to("constraint false :: under;", ZF::top());

  VarEnv<StandardAllocator> env;
  auto f = parse_flatzinc_str<StandardAllocator>("var int: x :: abstract(0);");
  EXPECT_TRUE(f);
  env.interpret(*f);
  must_interpret_to(env, "constraint int_eq(x, 0) :: exact;", ZF(0));
  must_error<ZF>(env, "constraint int_eq(x, -10) :: over;");
  must_error<ZF>(env, "constraint int_eq(x, 10) :: under;");

  must_error<ZF>(env, "constraint int_ne(x, 1) :: exact;");
  must_error<ZF>(env, "constraint int_ne(x, 1) :: over;");
  must_error<ZF>(env, "constraint int_ne(x, 1) :: under;");
}
