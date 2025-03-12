// Copyright 2025 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/flatzinc_parser.hpp"

#include <optional>

using namespace lala;
using namespace battery;

void test_rewriting(const char* input, const char* expected, bool can_rewrite = true) {
  auto f = parse_flatzinc_str<standard_allocator>(input);
  EXPECT_TRUE(f);
  std::map<std::string, std::vector<std::string>> set2bool_vars;
  auto rewritten = decompose_set_constraints(*f, set2bool_vars);
  EXPECT_EQ(rewritten.has_value(), can_rewrite);
  auto expect = parse_flatzinc_str<standard_allocator>(expected);
  EXPECT_TRUE(expect);
  EXPECT_EQ(*rewritten, *expect);
}

// TEST(AST, SetRewritingDomain1) {
//   test_rewriting(
//     "var set of 1..2: S;",

//     "var bool: __S_contains_1;\
//      var bool: __S_contains_2;"
//   );
// }

// TEST(AST, SetRewritingDomain2) {
//   test_rewriting(
//     "var set of {-1, 1, 3}: S;",

//     "var bool: __S_contains_m1;\
//      var bool: __S_contains_1;\
//      var bool: __S_contains_3;"
//   );
// }

// TEST(AST, SetRewritingMembership) {
//   test_rewriting(
//     "var int: x;\
//      var set of {1, 2}: S;\
//      constraint set_in(x, S);",

//     "var int: x;\
//      var bool: __S_contains_1;\
//      var bool: __S_contains_2;\
//      constraint bool_imply(int_eq(x, 1), bool_eq(__S_contains_1, true));\
//      constraint bool_imply(int_eq(x, 2), bool_eq(__S_contains_2, true));"
//   );
//   /** NOTE: `bool_imply(int_eq(x, 1), bool_eq(__S_contains_1, true))` represents the implication constraint:
//    * `x = 1 => __S_contains_1 = true`.
//    */
// }

