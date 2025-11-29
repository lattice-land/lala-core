// Copyright 2025 Pierre Talbot

#include "battery/allocator.hpp"
#include "lala/flatzinc_parser.hpp"
#include "lala/logic/logic.hpp"
#include <gtest/gtest.h>

#include <optional>

using namespace lala;
using namespace battery;

void test_rewriting(const char *input, const char *expected,
                    bool can_rewrite = true) {
  auto f = parse_flatzinc_str<standard_allocator>(input);
  EXPECT_TRUE(f);
  std::map<std::string, std::vector<std::string>> set2bool_vars;
  auto rewritten = decompose_set_constraints(*f, set2bool_vars);
  EXPECT_EQ(rewritten.has_value(), can_rewrite);
  auto expect = parse_flatzinc_str<standard_allocator>(expected);
  EXPECT_TRUE(expect);
  EXPECT_EQ(*rewritten, *expect);
}

TEST(AST, SetRewritingDomain1) {
  test_rewriting("var set of 1..2: S;",

                 "var bool: __S_contains_1;\
     var bool: __S_contains_2;");
}

TEST(AST, SetRewritingDomain2) {
  test_rewriting("var set of {-1, 1, 3}: S;",

                 "var bool: __S_contains_m1;\
      var bool: __S_contains_1;\
      var bool: __S_contains_3;");
}

TEST(AST, SetRewritingDomain3) {
  test_rewriting("var set of -3..3: S;",

                 "var bool: __S_contains_m3;\
     var bool: __S_contains_m2;\
     var bool: __S_contains_m1;\
     var bool: __S_contains_0;\
     var bool: __S_contains_1;\
     var bool: __S_contains_2;\
     var bool: __S_contains_3;");
}

TEST(AST, SetRewritingSubseteq) {
  test_rewriting("var set of {1, 2}: S;\
     var set of {1, 2, 3}: T;\
     constraint set_subset(S, T);",

                 "var bool: __S_contains_1;\
     var bool: __S_contains_2;\
     var bool: __T_contains_1;\
     var bool: __T_contains_2;\
     var bool: __T_contains_3;\
     constraint bool_imply(bool_eq(__S_contains_1, true), bool_eq(__T_contains_1, true));\
     constraint bool_imply(bool_eq(__S_contains_2, true), bool_eq(__T_contains_2, true));");
}

TEST(AST, SetRewritingMembership1) {
  test_rewriting("var int: x;\
     var set of {1, 2}: S;\
     constraint set_in(x, S);",

                 "var int: x;\
     var bool: __S_contains_1;\
     var bool: __S_contains_2;\
     constraint bool_imply(int_eq(x, 1), bool_eq(__S_contains_1, true));\
     constraint bool_imply(int_eq(x, 2), bool_eq(__S_contains_2, true));");
  /** NOTE: `bool_imply(int_eq(x, 1), bool_eq(__S_contains_1, true))`
   *
   * represents
   * the implication constraint: `x = 1 => __S_contains_1 =
   *
   * true`.
   */
}

TEST(AST, SetRewritingMembership2) {
  test_rewriting("var int: x;\
     var set of {1, 2, 4, 5, 6, 9, 11, 12, 13}: S;\
     constraint set_in(x, S);",

                 "var int: x;\
     var bool: __S_contains_1;\
     var bool: __S_contains_2;\
     var bool: __S_contains_4;\
     var bool: __S_contains_5;\
     var bool: __S_contains_6;\
     var bool: __S_contains_9;\
     var bool: __S_contains_11;\
     var bool: __S_contains_12;\
     var bool: __S_contains_13;\
     constraint bool_imply(int_eq(x, 1), bool_eq(__S_contains_1, true));\
     constraint bool_imply(int_eq(x, 2), bool_eq(__S_contains_2, true));\
     constraint bool_imply(int_eq(x, 4), bool_eq(__S_contains_4, true));\
     constraint bool_imply(int_eq(x, 5), bool_eq(__S_contains_5, true));\
     constraint bool_imply(int_eq(x, 6), bool_eq(__S_contains_6, true));\
     constraint bool_imply(int_eq(x, 9), bool_eq(__S_contains_9, true));\
     constraint bool_imply(int_eq(x, 11), bool_eq(__S_contains_11, true));\
     constraint bool_imply(int_eq(x, 12), bool_eq(__S_contains_12, true));\
     constraint bool_imply(int_eq(x, 13), bool_eq(__S_contains_13, true));");
  /** NOTE: `bool_imply(int_eq(x, 1), bool_eq(__S_contains_1, true))`
   *
   * represents
   * the implication constraint: `x = 1 => __S_contains_1 =
   *
   * true`.
   */
}

TEST(AST, SetRewritingCardinality1) {
  test_rewriting("var set of 1..2: S;\
    constraint set_card(S, 1);",

                 "var bool: __S_contains_1;\
     var bool: __S_contains_2;\
     constraint bool_imply(true, int_plus(__S_contains_1, __S_contains_2, 1));");

}
