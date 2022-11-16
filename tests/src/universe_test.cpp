// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "allocator.hpp"
#include "flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

template<class L>
void must_interpret_to(const char* fzn, L expect) {
  using F = TFormula<StandardAllocator>;
  auto f = parse_flatzinc_str<StandardAllocator>(fzn);
  EXPECT_TRUE(f);
  IResult<L, F> r = L::interpret(f->formula());
  EXPECT_TRUE(r.is_ok());
  EXPECT_FALSE(r.has_warning());
  EXPECT_EQ(r.value(), expect);
}

template<class L>
void must_error(const char* fzn) {
  using F = TFormula<StandardAllocator>;
  auto f = parse_flatzinc_str<StandardAllocator>(fzn);
  EXPECT_TRUE(f);
  IResult<L, F> r = L::interpret(f->formula());
  EXPECT_FALSE(r.is_ok());
}

TEST(UniverseTest, InterpretIntegerType) {
  must_interpret_to("var int: x :: exact;", local::ZInc::bot());
  must_interpret_to("var int: x :: under;", local::ZInc::bot());
  must_interpret_to("var int: x :: over;", local::ZInc::bot());

  must_error<local::FInc>("var int: x :: exact;");
  must_error<local::FInc>("var int: x :: under;");
  must_interpret_to("var int: x :: over;", local::FInc::bot());

  must_error<local::BInc>("var int: x :: exact;");
  must_error<local::BInc>("var int: x :: under;");
  must_interpret_to("var int: x :: over;", local::BInc::bot());
}
