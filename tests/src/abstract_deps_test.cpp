// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "lala/abstract_deps.hpp"
#include "battery/allocator.hpp"

using namespace lala;
using namespace battery;

/** A leaf of the hierarchy. It has no abstract type (`aty()`), which is not required to be copied. */
struct FakeChild {
  using allocator_type = standard_allocator;
  allocator_type get_allocator() const { return allocator_type{}; }
  int value;
  FakeChild(int value = 0): value(value) {}
  FakeChild(const FakeChild& other, AbstractDeps<allocator_type>&): value(other.value) {}
};

/** A node of the hierarchy holding a sub-element, possibly shared with other nodes. */
struct FakeParent {
  using allocator_type = standard_allocator;
  allocator_type get_allocator() const { return allocator_type{}; }
  abstract_ptr<FakeChild> sub;
  FakeParent(abstract_ptr<FakeChild> sub): sub(sub) {}
  FakeParent(const FakeParent& other, AbstractDeps<allocator_type>& deps)
    : sub(deps.template clone<FakeChild>(other.sub)) {}
};

template <class A, class... Args>
abstract_ptr<A> make_abstract(Args&&... args) {
  standard_allocator alloc;
  return abstract_ptr<A>(new(alloc) A(std::forward<Args>(args)...), alloc);
}

TEST(AbstractDeps, CloneWithoutAType) {
  auto a = make_abstract<FakeChild>(7);
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  EXPECT_EQ(deps.size(), 0);
  auto b = deps.template clone<FakeChild>(a);
  EXPECT_EQ(deps.size(), 1);
  EXPECT_NE(a.get(), b.get());
  EXPECT_EQ(b->value, 7);
}

TEST(AbstractDeps, CloneTwiceReturnsSameCopy) {
  auto a = make_abstract<FakeChild>(1);
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  auto b1 = deps.template clone<FakeChild>(a);
  auto b2 = deps.template clone<FakeChild>(a);
  EXPECT_EQ(b1.get(), b2.get());
  EXPECT_EQ(deps.size(), 1);
}

TEST(AbstractDeps, CloneNull) {
  abstract_ptr<FakeChild> a{standard_allocator{}};
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  auto b = deps.template clone<FakeChild>(a);
  EXPECT_FALSE(static_cast<bool>(b));
  EXPECT_EQ(deps.size(), 0);
}

/** Two parents share the same child: the child must be copied once, and both copies must share it. */
TEST(AbstractDeps, SharedChildIsCopiedOnce) {
  auto child = make_abstract<FakeChild>(3);
  auto p1 = make_abstract<FakeParent>(child);
  auto p2 = make_abstract<FakeParent>(child);
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  auto q1 = deps.template clone<FakeParent>(p1);
  auto q2 = deps.template clone<FakeParent>(p2);
  EXPECT_EQ(deps.size(), 3);
  EXPECT_EQ(q1->sub.get(), q2->sub.get());
  EXPECT_NE(q1->sub.get(), child.get());
  EXPECT_EQ(q1->sub->value, 3);
}

/** The order used in Turbo: the store is copied first, then the propagators sharing this store. */
TEST(AbstractDeps, ChildClonedBeforeParent) {
  auto child = make_abstract<FakeChild>(5);
  auto parent = make_abstract<FakeParent>(child);
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  auto child2 = deps.template clone<FakeChild>(child);
  auto parent2 = deps.template clone<FakeParent>(parent);
  EXPECT_EQ(deps.size(), 2);
  EXPECT_EQ(parent2->sub.get(), child2.get());
}

/** Two distinct elements are copied separately, even when they have the same type. */
TEST(AbstractDeps, DistinctElementsAreNotMerged) {
  auto c1 = make_abstract<FakeChild>(1);
  auto c2 = make_abstract<FakeChild>(2);
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  auto d1 = deps.template clone<FakeChild>(c1);
  auto d2 = deps.template clone<FakeChild>(c2);
  EXPECT_EQ(deps.size(), 2);
  EXPECT_NE(d1.get(), d2.get());
  EXPECT_EQ(d1->value, 1);
  EXPECT_EQ(d2->value, 2);
}
