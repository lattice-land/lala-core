// Copyright 2022 Pierre Talbot

#ifndef TYPES_HPP
#define TYPES_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "string.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "unique_ptr.hpp"

namespace lala {

/** Each abstract domain is uniquely identified by an UID.
    We call it an _abstract type_.
    Each formula (and recursively, its subformulas) is assigned to an abstract type indicating in what abstract domain this formula should be interpreted. */
using AType = int;

/** This value means a formula is not typed in a particular abstract domain and its type should be inferred. */
#define UNTYPED (-1)

/** The approximation of a formula in an abstract domain w.r.t. the concrete domain. */
enum Approx {
  UNDER, ///< An under-approximating element contains only solutions but not necessarily all.
  OVER, ///< An over-approximating element contains all solutions but not necessarily only solutions.
  EXACT ///< An exact element is both under- and over-approximating; it exactly represents the set of solutions.
};

static inline void print_approx(Approx appx) {
  switch(appx) {
    case UNDER: printf("under"); break;
    case OVER: printf("over"); break;
    case EXACT: printf("exact"); break;
  }
}

static constexpr Approx dapprox(Approx appx) {
  return appx == EXACT ? EXACT : (appx == UNDER ? OVER : UNDER);
}

/** The concrete type of variables introduced by existential quantification.
    More concrete types could be added later. */
template <class Allocator>
struct CType {
  enum Tag {
    Bool,
    Int,
    Real,
    Set
  };

  using allocator_type = Allocator;
  using this_type = CType<allocator_type>;

  Tag tag;
  battery::unique_ptr<this_type, allocator_type> sub;

  CUDA CType(Tag tag): tag(tag) {
    assert(tag != Set);
  }

  CUDA CType(Tag tag, CType&& sub_ty, const allocator_type& alloc = allocator_type())
   : tag(tag), sub(battery::allocate_unique<this_type>(alloc, std::move(sub_ty)))
  {
    assert(tag == Set);
  }

  template<class Alloc2>
  CUDA CType(const CType<Alloc2>& other, const allocator_type& alloc = allocator_type())
   : tag(other.tag)
  {
    if(other.sub) {
      this_type s = this_type(*other.sub, alloc);
      sub = battery::allocate_unique<this_type>(alloc, std::move(s));
    }
  }

  CUDA CType(const this_type& other): CType(other, other.sub.get_allocator()) {}

  CUDA CType(CType&&) = default;

  CUDA Approx default_approx() const {
    switch(tag) {
      case Bool: return EXACT;
      case Int: return EXACT;
      case Real: return OVER;
      case Set: return sub->default_approx();
      default: assert(false); // "CType: Unknown type".
    }
  }

  CUDA bool is_bool() const { return tag == Bool; }
  CUDA bool is_int() const { return tag == Int; }
  CUDA bool is_real() const { return tag == Real; }
  CUDA bool is_set() const { return tag == Set; }

  CUDA void print() const {
    switch(tag) {
      case Bool: printf("B"); break;
      case Int: printf("Z"); break;
      case Real: printf("R"); break;
      case Set: printf("S("); sub->print(); printf(")"); break;
      default: assert(false); // "CType: Unknown type".
    }
  }
};

template <class Alloc1, class Alloc2>
inline bool operator==(const CType<Alloc1>& lhs, const CType<Alloc2>& rhs) {
  if(lhs.tag == rhs.tag) {
    if(lhs.tag == CType<Alloc1>::Set) {
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
    For instance, `logic_set<logic_int>` is a set of integers.
    Sets are defined in extension: we explicitly list the values belonging to the set.
    To avoid using too much memory with large sets, we use an interval representation, e.g., \f$ \{1..3, 5..5, 10..12\} = \{1, 2, 3, 5, 10, 11, 12\} \f$.
    When sets occur in intervals, they are ordered by set inclusion, e.g., \f$ \{\{1..2\}..\{1..4\}\} = \{\{1,2\}, \{1,2,3\}, \{1,2,4\}, \{1,2,3,4\}\} \f$. */
template<class F, class Allocator>
using logic_set = battery::vector<battery::tuple<F, F>, Allocator>;

}

#endif
