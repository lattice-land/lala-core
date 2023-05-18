// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "lala/abstract_deps.hpp"
#include "battery/allocator.hpp"

using namespace lala;
using namespace battery;

TEST(AST, AbstractDeps) {
  struct FakeAD {
    using allocator_type = standard_allocator;
    allocator_type get_allocator() const { return allocator_type{}; }
    int atype;
    FakeAD(): atype(0) {}
    FakeAD(const FakeAD&, AbstractDeps<allocator_type>&): atype(0) {}
    int aty() const { return atype; }
  };
  standard_allocator alloc;
  shared_ptr<FakeAD> a(new(alloc) FakeAD, alloc);
  AbstractDeps<standard_allocator> deps(alloc);
  EXPECT_EQ(deps.size(), 0);
  shared_ptr<FakeAD> b = deps.template clone<FakeAD>(a);
  EXPECT_EQ(deps.size(), 1);
}
