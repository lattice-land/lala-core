// Copyright 2021 Pierre Talbot

#include "lala/vstore.hpp"
#include "lala/simplifier.hpp"
#include "lala/interval.hpp"
#include "lala/fixpoint.hpp"
#include "abstract_testing.hpp"

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;
using IStore = VStore<Itv, standard_allocator>;

void test_simplification(
  const char* store_formula,
  const char* simplifier_formula,
  const char* expected_simplified_formula)
{
  VarEnv<standard_allocator> env;

  // Can be interpreted in IStore.
  auto f1 = *parse_flatzinc_str<standard_allocator>(store_formula);
  // Cannot be interpreted in IStore, but after applying Simplifier, it can be interpreted.
  auto f2 = *parse_flatzinc_str<standard_allocator>(simplifier_formula);

  f2.print();

  IDiagnostics diagnostics;
  auto istore = battery::make_shared<IStore, standard_allocator>(create_and_interpret_and_tell<IStore>(f1, env, diagnostics).value());

  using simplifier_type = Simplifier<IStore, standard_allocator>;
  simplifier_type simplifier{
    env.extends_abstract_dom(),
    istore
  };

  simplifier_type::tell_type<standard_allocator> tell;
  EXPECT_TRUE((ginterpret_in<IKind::TELL, true>(simplifier, f2, env, tell, diagnostics)));
  simplifier.tell(std::move(tell));
  local::BInc has_changed = GaussSeidelIteration{}.fixpoint(simplifier);
  EXPECT_TRUE(has_changed);

  printf("fixed point reached\n");

  auto f3 = *parse_flatzinc_str<standard_allocator>(expected_simplified_formula);
  f3.print();
  auto f4 = simplifier.deinterpret();
  f4.print();
  EXPECT_EQ(f3, f4);
}

TEST(Simplifier, SimplificationGlobalTest) {
  test_simplification(
    "var 0..8: x; var 2..10: y; var 5..5: z; var 0..10: w;",
    "var 0..8: x; var 2..10: y; var 5..5: z; var 0..10: w; constraint int_eq(x, y); constraint int_ge(y, z); constraint int_ge(y, w);",
    "var 2..8: x; var 0..10: w; constraint int_ge(x, 5); constraint int_ge(x, w);"
  );

  test_simplification(
    "var 0..8: x; var 2..10: y; var 5..5: z; var 0..10: w;",
    "var 0..8: x; var 2..10: y; var 5..5: z; var 0..10: w; constraint int_eq(x, y); constraint int_eq(y, w); constraint int_eq(w, z);",
    "var 5..5: x;"
  );

  test_simplification(
    "var 0..8: x; var 2..2: y; var 5..5: z;",
    "var 0..8: x; var 2..2: y; var 5..5: z; constraint int_eq(x, int_plus(y, z));",
    "var 0..8: x; constraint int_eq(x, int_plus(2, 5));"
  );
}
