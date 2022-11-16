// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "copy_dag_helper.hpp"
#include "allocator.hpp"

using namespace lala;
using namespace battery;

TEST(AST, AbstractDeps) {
  struct FakeAD {
    int uid_;
    FakeAD(): uid_(0) {}
    FakeAD(const FakeAD&, AbstractDeps<>&): uid_(0) {}
    int uid() const { return uid_; }
  };
  StandardAllocator alloc;
  shared_ptr<FakeAD> a(new(alloc) FakeAD, alloc);
  AbstractDeps deps;
  EXPECT_EQ(deps.size(), 0);
  shared_ptr<FakeAD> b = deps.clone(a);
  EXPECT_EQ(deps.size(), 1);
}
