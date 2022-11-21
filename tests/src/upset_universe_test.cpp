// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "allocator.hpp"
#include "flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

template<class L>
void must_interpret_to(const char* fzn, L expect, bool has_warning = false) {
  using F = TFormula<StandardAllocator>;
  auto f = parse_flatzinc_str<StandardAllocator>(fzn);
  EXPECT_TRUE(f);
  IResult<L, F> r = L::interpret(*f);
  std::cout << fzn << std::endl;
  EXPECT_TRUE(r.is_ok());
  EXPECT_EQ(r.has_warning(), has_warning);
  EXPECT_EQ(r.value(), expect);
}

template<class L>
void must_error(const char* fzn) {
  using F = TFormula<StandardAllocator>;
  auto f = parse_flatzinc_str<StandardAllocator>(fzn);
  EXPECT_TRUE(f);
  f->print(true, true);
  IResult<L, F> r = L::interpret(*f);
  std::cout << fzn << std::endl;
  EXPECT_FALSE(r.is_ok());
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

TEST(UniverseTest, InterpretIntegerType) {
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

TEST(UniverseTest, InterpretRealType) {
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

TEST(UniverseTest, InterpretBoolType) {
  interpret_bool_type<local::ZInc, local::FInc, local::BInc>();
  interpret_bool_type<local::ZDec, local::FDec, local::BDec>();
}


// TEST(UniverseTest, IntegerInterpretation) {
//   must_interpret_to("x >= 0 :: exact;")
// }
