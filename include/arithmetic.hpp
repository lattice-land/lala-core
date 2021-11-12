// Copyright 2021 Pierre Talbot

#ifndef ARITHMETIC_HPP
#define ARITHMETIC_HPP

#include <type_traits>

/** \file arithmetic.hpp
We provide a collection of arithmetic operators for abstract domains encapsulating an arithmetic value such as `ZInc` or `ZDec`.
These operators can be specialized on different domain.
Each operator comes in two version, e.g., `add_up` and `add_down`, which might be the same for some types (such as `int`), but different on non-exact types such as `double`.
Admitting the exact result of an operation is `r`, the "up" version of an operator returns an approximation \f$ r_{up} \f$ of `r` such that \f$ r_{up} \geq r \f$, and dually for the "down" version.
Therefore, the exact result is always comprised between the down and up bound, i.e., \f$ r \in [r_{\mathit{down}}..r_{\mathit{up}}] \f$.

Precondition: For efficiency purposes, all operators suppose their arguments to be different from infinity or any other special value (generally bottom or top).
*/

namespace lala {

template<typename A>
using IsInteger = std::enable_if_t<std::numeric_limits<typename A::ValueType>::is_integer, bool>;

template<typename A>
using VT = typename A::ValueType;

template<typename A> CUDA A neg(A a) { return A(-VT<A>(a)); }
template<typename A> CUDA A add(A a, A b) { return A(VT<A>(a) + VT<A>(b)); }
template<typename A> CUDA A sub(A a, A b) { return A(VT<A>(a) - VT<A>(b)); }
template<typename A> CUDA A mul(A a, A b) { return A(VT<A>(a) * VT<A>(b)); }

template<typename A, IsInteger<A> = true> CUDA A neg_up(A a) { return neg(a); }
template<typename A, IsInteger<A> = true> CUDA A neg_down(A a) { return neg(a); }

/** This macro create up rounding and down rounding implementations of operations on exact types. */
#define EXACT_BINOP_UP_DOWN(name) \
template<typename A, IsInteger<A> = true> CUDA A name##_up(A a, A b) { return name(a, b); } \
template<typename A, IsInteger<A> = true> CUDA A name##_down(A a, A b) { return name(a, b); }

EXACT_BINOP_UP_DOWN(add)
EXACT_BINOP_UP_DOWN(sub)
EXACT_BINOP_UP_DOWN(mul)

template<typename A, IsInteger<A> = true>
CUDA A div_up(A a, A b) {
  typedef typename A::ValueType VT;
  VT i = VT(a);
  VT j = VT(b);
  assert(j != 0);
  VT r = i / j;
  // division is rounded towards zero.
  // We add one only if `r` was truncated and `i, j` are of equal sign (so the division operated in the positive numbers).
  // Inspired by https://stackoverflow.com/questions/921180/how-can-i-ensure-that-a-division-of-integers-is-always-rounded-up/926806#926806
  return A((i % j != 0 && i > 0 == j > 0) ? r + 1 : r);
}

template<typename A, IsInteger<A> = true>
CUDA A div_down(A a, A b) {
  typedef typename A::ValueType VT;
  VT i = VT(a);
  VT j = VT(b);
  assert(j != 0);
  VT r = i / j;
  return A((i % j != 0 && i > 0 != j > 0) ? r - 1 : r);
}

}

#endif
