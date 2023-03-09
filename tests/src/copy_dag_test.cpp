// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "copy_dag_helper.hpp"
#include "allocator.hpp"

using namespace lala;
using namespace battery;

TEST(AST, AbstractDeps) {
  struct FakeAD {
    int atype;
    FakeAD(): atype(0) {}
    FakeAD(const FakeAD&, AbstractDeps<>&): atype(0) {}
    int aty() const { return atype; }
  };
  StandardAllocator alloc;
  shared_ptr<FakeAD> a(new(alloc) FakeAD, alloc);
  AbstractDeps deps;
  EXPECT_EQ(deps.size(), 0);
  shared_ptr<FakeAD> b = deps.template clone<FakeAD>(a);
  EXPECT_EQ(deps.size(), 1);
}
