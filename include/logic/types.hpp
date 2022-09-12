// Copyright 2022 Pierre Talbot

#ifndef TYPES_HPP
#define TYPES_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "string.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "thrust/optional.h"

namespace lala {

/** Each abstract domain is uniquely identified by an UID.
    We call it an _abstract type_.
    Each formula (and recursively, its subformulas) is assigned to an abstract type indicating in what abstract domain this formula should be interpreted. */
using AType = int;

/** This value means a formula is not typed in a particular abstract domain and its type should be inferred. */
#define UNTYPED (-1)

/** The concrete type of variables introduced by existential quantification.
    More concrete types could be added later. */
struct CType {
  enum Tag {
    Int,
    Real,
    Set
  };

  Tag tag;
  thrust::optional<CType> sub;

  CType(Tag tag): tag(tag) {
    assert(tag != Set);
  }

  CType(Tag tag, CType sub): tag(tag), sub(sub) {
    assert(tag == Set);
    assert(sub.has_value());
  }

  CType(const CType&) = default;
  CType(CType&&) = default;

  void print() const {
    switch(tag) {
      case Int: printf("Z"); break;
      case Real: printf("R"); break;
      case Set: printf("S("); sub.print(); print(")"); break;
    }
  }
};

/** The type of integers used in logic formulas.
    Integers are represented by the set \f$ \{-\infty, \infty\} \cup Z (\text{ with} Z \subset \mathbb{Z}) \f$.
    The minimal and maximal values of `logic_int` represents \f$ -\infty \f$ and \f$ \infty \f$ respectively. */
using logic_int = long long int;

/** The type of real numbers used in logic formulas.
    Real numbers are approximated by the set \f$ \mathbb{F} \times \mathbb{F} \f$.
    When a real number \f$ r \in \mathbb{R} \f$ is also a floating-point number, then it is represented by \f$ (r, r) \f$, otherwise it is represented by \f$ (\lfloor r \rfloor, \lceil r \rceil) \f$ such that \f$ \lfloor r \rfloor < r < \lceil r \rceil \f$ and there is no floating-point number \f$ f \f$ such that \f$ \lfloor r \rfloor < f < \lceil r \rceil \f$. */
using logic_real = battery::tuple<double, double>;

/** A set is parametric in a universe of discourse.
    For instance, `logic_set<logic_int>` is a set of integers.
    Sets are defined in extension: we explicitly list the values belonging to the set.
    To avoid using too much memory with large sets, we use an interval representation, e.g., \f$ \{1..3, 5..5, 10..12\} = \{1, 2, 3, 5, 10, 11, 12\} \f$.
    When sets occur in intervals, they are ordered by set inclusion, e.g., \f$ \{\{1..2\}..\{1..4\}\} = \{\{1,2\}, \{1,2,3\}, \{1,2,4\}, \{1,2,3,4\}\} \f$. */
template<class F, class Allocator>
using logic_set = battery::vector<battery::tuple<F, F>, Allocator>;

}

#endif
