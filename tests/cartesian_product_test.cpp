// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "thrust/optional.h"
#include "ast.hpp"
#include "z.hpp"
#include "allocator.hpp"
#include "utility.hpp"
#include "cartesian_product.hpp"

using namespace lala;

typedef ZInc<int, StandardAllocator> zi;
typedef ZDec<int, StandardAllocator> zd;
typedef CartesianProduct<zi, zd> Itv;
typedef TFormula<StandardAllocator> F;

TEST(CPTest, IntervalAsCartesianProduct) {
  Itv itv1_2(zi(1), zd(2));
  Itv bot_itv = Itv::bot();
  Itv top_itv = Itv::top();
  EXPECT_FALSE(itv1_2.is_bot());
  EXPECT_FALSE(itv1_2.is_top());
  EXPECT_TRUE(bot_itv.is_bot());
  EXPECT_FALSE(bot_itv.is_top());
  EXPECT_FALSE(top_itv.is_bot());
  EXPECT_TRUE(top_itv.is_top());

  auto geq_1 = make_v_op_z(0, GEQ, 1, standard_allocator);
  auto leq_2 = make_v_op_z(0, LEQ, 2, standard_allocator);
  auto geq_1_leq_2 = F::make_binary(geq_1, AND, leq_2);
  auto f1_opt = bot_itv.interpret(EXACT, geq_1);
  EXPECT_TRUE(f1_opt.has_value());
  auto f2_opt = bot_itv.interpret(EXACT, leq_2);
  EXPECT_TRUE(f2_opt.has_value());
  auto f1 = f1_opt.value();
  auto f2 = f2_opt.value();

  Itv itv2 = Itv::bot();
  itv2.join(f1);
  EXPECT_NE(itv2, bot_itv);
  EXPECT_NE(itv2, top_itv);
  EXPECT_NE(itv2, itv1_2);
  Itv itv3 = itv2;
  itv2.join(f2);
  EXPECT_NE(itv2, bot_itv);
  EXPECT_NE(itv2, top_itv);
  EXPECT_EQ(itv2, itv1_2);
  EXPECT_EQ(itv2.deinterpret(), geq_1_leq_2);
  itv2.meet<1>(zd::bot());
  EXPECT_NE(itv2, bot_itv);
  EXPECT_NE(itv2, top_itv);
  EXPECT_NE(itv2, itv1_2);
  EXPECT_EQ(itv2, itv3);
  itv2.meet<0>(zi::bot());
  EXPECT_EQ(itv2, bot_itv);
  EXPECT_NE(itv2, top_itv);
  EXPECT_NE(itv2, itv1_2);
  EXPECT_NE(itv2, itv3);

  EXPECT_FALSE(bot_itv.refine());
  EXPECT_FALSE(top_itv.refine());
  EXPECT_FALSE(itv2.refine());
  EXPECT_FALSE(itv3.refine());

  EXPECT_EQ(bot_itv, bot_itv.clone());
  EXPECT_EQ(top_itv, top_itv.clone());
  EXPECT_EQ(itv2, itv2.clone());

  itv3.reset(bot_itv);
  EXPECT_EQ(itv3, bot_itv);
  EXPECT_EQ(itv3, itv2);

  auto f1_opt_2 = bot_itv.interpret_one<0>(EXACT, geq_1);
  EXPECT_TRUE(f1_opt_2.has_value());
  auto f2_opt_2 = bot_itv.interpret_one<1>(EXACT, leq_2);
  EXPECT_TRUE(f2_opt.has_value());
  auto f1_opt_3 = bot_itv.interpret_one<1>(EXACT, geq_1);
  EXPECT_FALSE(f1_opt_3.has_value());
  auto itv4 = bot_itv.join(f1).join(f2);
  auto itv5 = bot_itv.join(f1_opt_2.value()).join(f2_opt_2.value());
  EXPECT_EQ(itv4, itv5);

  EXPECT_TRUE(itv4.entailment(f1_opt_2.value()));
  EXPECT_TRUE(itv4.entailment(f2_opt_2.value()));
  EXPECT_TRUE(itv4.entailment(f1));
  EXPECT_TRUE(itv4.entailment(f2));

  auto split_top = top_itv.split();
  auto split_bot = bot_itv.split();
  auto split4 = itv4.split();
  EXPECT_EQ(split_top.size(), 0);
  EXPECT_EQ(split_bot.size(), 1);
  EXPECT_EQ(split4.size(), 1);
}
