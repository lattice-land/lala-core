// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_TYPES_HPP
#define LALA_CORE_TYPES_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "battery/unique_ptr.hpp"

namespace lala {

/** Each abstract domain is uniquely identified by an UID.
    We call it an _abstract type_.
    Each formula (and recursively, its subformulas) is assigned to an abstract type indicating in what abstract domain this formula should be interpreted. */
using AType = int;

/** This value means a formula is not typed in a particular abstract domain and its type should be inferred. */
#define UNTYPED (-1)

/** The concrete type of variables, called `sort`, introduced by existential quantification.
    More concrete types could be added later. */
template <class Allocator>
struct Sort {
  enum Tag {
    Bool,
    Int,
    Real,
    Set
  };

  using allocator_type = Allocator;
  using this_type = Sort<allocator_type>;

  Tag tag;
  battery::unique_ptr<this_type, allocator_type> sub;

  CUDA Sort(Tag tag): tag(tag) {
    assert(tag != Set);
  }

  CUDA NI Sort(Tag tag, Sort&& sub_ty, const allocator_type& alloc = allocator_type())
   : tag(tag), sub(battery::allocate_unique<this_type>(alloc, std::move(sub_ty)))
  {
    assert(tag == Set);
  }

  template<class Alloc2>
  CUDA NI Sort(const Sort<Alloc2>& other, const allocator_type& alloc = allocator_type())
   : tag(static_cast<Tag>(other.tag))
  {
    if(other.sub) {
      this_type s = this_type(*other.sub, alloc);
      sub = battery::allocate_unique<this_type>(alloc, std::move(s));
    }
  }

  CUDA Sort(const this_type& other): Sort(other, other.sub.get_allocator()) {}

  Sort& operator=(const this_type& other) = default;
  Sort(Sort&&) = default;
  Sort& operator=(Sort&&) = default;

  CUDA bool is_bool() const { return tag == Bool; }
  CUDA bool is_int() const { return tag == Int; }
  CUDA bool is_real() const { return tag == Real; }
  CUDA bool is_set() const { return tag == Set; }

  CUDA NI void print() const {
    switch(tag) {
      case Bool: printf("B"); break;
      case Int: printf("Z"); break;
      case Real: printf("R"); break;
      case Set: printf("S("); sub->print(); printf(")"); break;
      default: assert(false); // "Sort: Unknown type".
    }
  }
};

template <class Alloc1, class Alloc2>
CUDA NI inline bool operator==(const Sort<Alloc1>& lhs, const Sort<Alloc2>& rhs) {
  if(lhs.tag == rhs.tag) {
    if(lhs.tag == Sort<Alloc1>::Set) {
      return *(lhs.sub) == *(rhs.sub);
    }
    return true;
  }
  return false;
}

/** The type of Boolean used in logic formulas. */
using logic_bool = bool;

/** The type of integers used in logic formulas.
    Integers are represented by the set \f$ \{-\infty, \infty\} \cup Z (\text{ with} Z \subset \mathbb{Z}) \f$.
    The minimal and maximal values of `logic_int` represents \f$ -\infty \f$ and \f$ \infty \f$ respectively. */
using logic_int = long long int;

/** The type of real numbers used in logic formulas.
    Real numbers are approximated by the set \f$ \mathbb{F} \times \mathbb{F} \f$.
    When a real number \f$ r \in \mathbb{R} \f$ is also a floating-point number, then it is represented by \f$ (r, r) \f$, otherwise it is represented by \f$ (\lfloor r \rfloor, \lceil r \rceil) \f$ such that \f$ \lfloor r \rfloor < r < \lceil r \rceil \f$ and there is no floating-point number \f$ f \f$ such that \f$ \lfloor r \rfloor < f < \lceil r \rceil \f$. */
using logic_real = battery::tuple<double, double>;

/** A set is parametric in a universe of discourse.
    For instance, `logic_set<F>`, with F representing an integer constant, is a set of integers.
    Sets are defined in extension: we explicitly list the values belonging to the set.
    To avoid using too much memory with large sets, we use an interval representation, e.g., \f$ \{1..3, 5..5, 10..12\} = \{1, 2, 3, 5, 10, 11, 12\} \f$.
    When sets occur in intervals, they are ordered by set inclusion, e.g., \f$ \{\{1..2\}..\{1..4\}\} = \{\{1,2\}, \{1,2,3\}, \{1,2,4\}, \{1,2,3,4\}\} \f$. */
template<class F>
using logic_set = battery::vector<battery::tuple<F, F>, typename F::allocator_type>;

}

#endif
